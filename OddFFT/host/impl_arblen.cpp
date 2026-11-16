// host/impl_arblen.cpp  (OddFFT)
//
// Arbitrary-length transforms: Bluestein for any N, Rader for prime N.
// Both reduce the transform to a convolution evaluated by a power-of-two
// inner transform, which here is this project's own plan and kernels rather
// than a separate implementation. See host/arb_common.h for what that
// replaces and why it matters.
//
// The inner radix is a VARIANT AXIS, not a choice made here: `_r2_` and `_r4_`
// differ in nothing but which radix the inner transform uses, so the gain from
// radix 4 -- which the power-of-two matrix settled on 46 of 48 comparisons --
// is measured in this setting too rather than assumed to carry over.
#include "arb_common.h"
#include "plan.h"
#include "butterflies.h"
#include "host_fft.h"
#include "host_twiddles.h"
#include "host_prefix_fast.h"
#include "bench_harness.h"
#include "registry.h"
#include "../kernels/oddfft_tiling.h"
#include "../kernels/kernel_limits.h"
#include "../kernels/kernel_table.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>

#ifndef ASCENDC_CPU_DEBUG
#include "acl/acl.h"
#include "platform_cores.h"
extern void arb_cmul_do(uint32_t blockDim, void *stream, void *ar, void *ai,
                        void *br, void *bi, uint32_t n, uint32_t tile, uint32_t blk_num);
extern void arb_cscale_do(uint32_t blockDim, void *stream, void *ar, void *ai,
                          uint32_t n, float s, float sign, uint32_t tile, uint32_t blk_num);
#define CHECK_ACL2(x) do { \
    auto _rc = (x); \
    if (_rc != ACL_SUCCESS) fprintf(stderr, "[ERROR] ACL call failed: %s (rc=%d) at %s:%d\n", \
                                     #x, (int)_rc, __FILE__, __LINE__); \
} while (0)
#endif

using namespace oddfft;

static constexpr double ARB_PI = 3.14159265358979323846;

// ===========================================================================
// Host-side reference inner transform, used in CPU-debug and to prepare the
// filter spectra before the clock starts.
// ===========================================================================
static void inner_fft_host(const InnerFFT &f, float *xr, float *xi, bool inverse)
{
    if (inverse) for (uint32_t i = 0; i < f.M; ++i) xi[i] = -xi[i];
    std::vector<float> zr(f.M), zi(f.M);
    apply_perm(f, xr, xi, zr.data(), zi.data());
    for (size_t s = 0; s < f.plan.stages.size(); ++s)
        host_stage_fast(f.plan, s, zr.data(), zi.data());
    const float sc = inverse ? (1.0f / (float)f.M) : 1.0f;
    for (uint32_t i = 0; i < f.M; ++i) {
        xr[i] = zr[i] * sc;
        xi[i] = inverse ? -zi[i] * sc : zi[i];
    }
}

#ifndef ASCENDC_CPU_DEBUG
// ===========================================================================
// One inner transform on the device.
//
// The permutation and any sub-floor stages happen on the HOST BUFFER that
// already holds the data for the permutation, so they cost no extra transfer:
// one device-to-host, the host work, one host-to-device, then the device
// stages. The inverse is the forward pass between two scale kernels.
// ===========================================================================
struct DevCtx {
    aclrtStream stream;
    void   *xrDev, *xiDev, *twDev;
    float  *xrHost, *xiHost;
    uint32_t blockDim, tile;
    // Host compute accumulated by inner_fft_device: the permutation and any
    // sub-floor stages. The caller zeroes it per run and adds it to cpu_us, so
    // the split stays honest without a second clock per stage.
    //
    // NO DEFAULT MEMBER INITIALIZER HERE, deliberately. A member initializer
    // makes the type a non-aggregate under C++11, so `DevCtx ctx{...}` stops
    // compiling -- which is exactly what the device build did, even though the
    // target asks for -std=c++17: something in the toolkit's own flags puts an
    // earlier -std after it, and the last one wins. Rather than chase that
    // flag, the field is simply initialised at the call site, which is correct
    // under every standard.
    double cpu_us;
};

static void inner_fft_device(const InnerFFT &f, DevCtx &c, bool inverse)
{
    const size_t bytes = (size_t)f.M * sizeof(float);
    if (inverse) {
        arb_cscale_do(c.blockDim, c.stream, c.xrDev, c.xiDev, f.M, 1.0f, -1.0f,
                      c.tile, c.blockDim);
        CHECK_ACL2(aclrtSynchronizeStream(c.stream));
    }

    // EVERY launch is followed by a stream synchronisation, and every
    // host-visible copy is preceded by one. aclrtMemcpy is synchronous with
    // respect to the HOST; it does not order against work queued on an
    // explicitly created stream. run_odd never meets this because it syncs
    // after each stage launch, so the habit was invisible rather than
    // deliberate -- and the first device version of this file copied the data
    // back while the kernel that produced it was still running. The CPU-debug
    // path cannot show that, which is why it passed.
    CHECK_ACL2(aclrtMemcpy(c.xrHost, bytes, c.xrDev, bytes, ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK_ACL2(aclrtMemcpy(c.xiHost, bytes, c.xiDev, bytes, ACL_MEMCPY_DEVICE_TO_HOST));
    {
        auto h0 = std::chrono::high_resolution_clock::now();
        std::vector<float> zr(f.M), zi(f.M);
        apply_perm(f, c.xrHost, c.xiHost, zr.data(), zi.data());
        // Any stage below the threshold runs here, on data already in host
        // memory, so it costs nothing extra in transfers.
        for (size_t s = 0; s < f.plan.stages.size(); ++s) {
            if (f.plan.stages[s].on_npu) break;
            host_stage_fast(f.plan, s, zr.data(), zi.data());
        }
        std::memcpy(c.xrHost, zr.data(), bytes);
        std::memcpy(c.xiHost, zi.data(), bytes);
        auto h1 = std::chrono::high_resolution_clock::now();
        c.cpu_us += std::chrono::duration<double, std::micro>(h1 - h0).count();
    }
    CHECK_ACL2(aclrtMemcpy(c.xrDev, bytes, c.xrHost, bytes, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL2(aclrtMemcpy(c.xiDev, bytes, c.xiHost, bytes, ACL_MEMCPY_HOST_TO_DEVICE));

    for (const StageInfo &st : f.plan.stages) {
        if (!st.on_npu) continue;
        const OddKernelEntry *ke = nullptr;
        for (uint32_t i = 0; i < ODD_KERNEL_COUNT; ++i)
            if (ODD_KERNELS[i].radix == st.radix &&
                std::string("direct") == ODD_KERNELS[i].butterfly) ke = &ODD_KERNELS[i];
        if (!ke) continue;
        const uint32_t rowLen = ((st.t + 7u) / 8u) * 8u;
        const uint32_t nShift = ODDFFT_SHIFT_COUNT(st.t, c.tile);
        const uint32_t nTiles = st.nBlk * nShift;
        const uint32_t bd = (nTiles < c.blockDim) ? nTiles : c.blockDim;
        ke->launch(bd, c.stream, c.xrDev, c.xiDev, c.twDev,
                   f.M, st.t, st.block, st.wOffset, rowLen, c.tile, bd);
        CHECK_ACL2(aclrtSynchronizeStream(c.stream));
    }

    if (inverse) {
        arb_cscale_do(c.blockDim, c.stream, c.xrDev, c.xiDev, f.M,
                      1.0f / (float)f.M, -1.0f, c.tile, c.blockDim);
        CHECK_ACL2(aclrtSynchronizeStream(c.stream));
    }
}
#endif

// ===========================================================================
// Bluestein
// ===========================================================================
struct BluePrep {
    uint32_t N = 0, M = 0;
    std::vector<float> wcr, wci;   // the chirp, used by BOTH the pre- and the
                                   // post-multiply, so only one table exists
    std::vector<float> fbr, fbi;   // the transformed convolution kernel
    InnerFFT inner;
};

static bool blue_prepare(uint32_t N, uint32_t radix, uint32_t min_t,
                         BluePrep &p, std::string *err)
{
    p.N = N;
    p.M = next_pow2(2u * N - 1u);
    if (!build_inner(p.M, radix, min_t, p.inner, err)) return false;

    p.wcr.assign(p.M, 0.0f); p.wci.assign(p.M, 0.0f);
    for (uint32_t n = 0; n < N; ++n) {
        const double a = -ARB_PI * (double)n * (double)n / (double)N;
        p.wcr[n] = (float)std::cos(a);
        p.wci[n] = (float)std::sin(a);
    }
    // The convolution kernel carries the conjugate chirp, extended so that a
    // cyclic convolution of length M equals the linear one we want:
    // b[-n] = b[n], because the chirp depends on n squared.
    p.fbr.assign(p.M, 0.0f); p.fbi.assign(p.M, 0.0f);
    for (uint32_t n = 0; n < N; ++n) {
        const double a = +ARB_PI * (double)n * (double)n / (double)N;
        const float re = (float)std::cos(a), im = (float)std::sin(a);
        p.fbr[n] = re; p.fbi[n] = im;
        if (n > 0) { p.fbr[p.M - n] = re; p.fbi[p.M - n] = im; }
    }
    // Transformed once, before any timing.
    inner_fft_host(p.inner, p.fbr.data(), p.fbi.data(), false);
    return true;
}

// ===========================================================================
// Rader (prime N only)
// ===========================================================================
struct RaderPrep {
    uint32_t N = 0, L = 0, M = 0, g = 0, ginv = 0;
    std::vector<uint32_t> idxIn, idxOut;
    std::vector<float> fbr, fbi;
    InnerFFT inner;
};

static bool rader_prepare(uint32_t N, uint32_t radix, uint32_t min_t,
                          RaderPrep &p, std::string *err)
{
    // N >= 3: at N = 2 the convolution length L is 1 and M collapses to 1, so
    // the fold below has no second half to add. For every prime N >= 3,
    // L = N-1 >= 2 makes 2L-1 an odd number above 1, which is never a power of
    // two, so M = next_pow2(2L-1) >= 2L and c[p+L] is always in range.
    if (N < 3u) {
        if (err) *err = "Rader's algorithm needs a prime N >= 3";
        return false;
    }
    if (!is_prime(N)) {
        if (err) *err = "Rader's algorithm needs a PRIME N; " + std::to_string(N) +
                        " is composite. Bluestein takes any N.";
        return false;
    }
    p.N = N;
    p.L = N - 1u;
    p.M = next_pow2(2u * p.L - 1u);
    p.g = primitive_root(N);
    if (p.g == 0) { if (err) *err = "no primitive root found"; return false; }
    p.ginv = (uint32_t)mod_pow(p.g, N - 2u, N);
    if (!build_inner(p.M, radix, min_t, p.inner, err)) return false;

    p.idxIn.resize(p.L);
    p.idxOut.resize(p.L);
    for (uint32_t q = 0; q < p.L; ++q) {
        p.idxIn[q]  = (uint32_t)mod_pow(p.g, q, N);
        p.idxOut[q] = (uint32_t)mod_pow(p.ginv, q, N);
    }
    // b'[q] = W_N^(g^-q). Zero-padded, not periodically extended: the cyclic
    // convolution of length L is recovered after the transform by folding
    // c[p] + c[p+L], which is exact for any M >= 2L-1 and does not require M
    // to be a multiple of L.
    p.fbr.assign(p.M, 0.0f); p.fbi.assign(p.M, 0.0f);
    for (uint32_t q = 0; q < p.L; ++q) {
        const double a = -2.0 * ARB_PI * (double)p.idxOut[q] / (double)N;
        p.fbr[q] = (float)std::cos(a);
        p.fbi[q] = (float)std::sin(a);
    }
    inner_fft_host(p.inner, p.fbr.data(), p.fbi.data(), false);
    return true;
}


// ===========================================================================
// WHAT IS INSIDE THE CLOCK HERE, AND WHY IT DIFFERS FROM run_odd
//
// run_odd() times the host prefix and the kernel launches, and leaves the two
// initial host-to-device copies outside both timers. That is internally
// consistent -- every cell of the 900-cell matrix is measured the same way --
// but it would be the wrong convention for THIS file, because the criticism
// that motivated rewriting Bluestein and Rader was precisely the number of
// PCIe transfers. Not timing them would reproduce the defect being fixed.
//
// So here ONE wall clock spans the whole transform, and the split reported is
//     cpu_us = the host compute (the permutation, any sub-floor stages,
//              the input gather and the output scatter)
//     npu_us = everything else inside that clock, that is, the transfers
//              AND the kernel launches
// This makes npu_us here not directly comparable with npu_us in the odd-radix
// rows; total_us is the honest column for both, and for the arbitrary-length
// rows it is honest about rather more. For scale: the copies run_odd leaves
// untimed are 2*N*4 bytes, 806 KB at N = 100800, so of order 80 us against a
// 6201 us total -- about 1.3%, which is worth stating rather than discovering.
// ===========================================================================

// stages reported per INNER transform; a call does two of them.
static void arb_stage_counts(const InnerFFT &f, uint32_t &npu, uint32_t &cpu)
{
    cpu = f.plan.cpu_first;
    npu = (uint32_t)f.plan.stages.size() - cpu;
}

// ---------------------------------------------------------------------------
// Bluestein
// ---------------------------------------------------------------------------
static void run_blue(uint32_t N, uint32_t m, const std::vector<float> &input,
                     uint32_t warmup, uint32_t bench_runs,
                     const std::string &variant_name,
                     uint32_t radix, bool multicore)
{
    (void)m;
    BluePrep pr;
    std::string err;
    if (!blue_prepare(N, radix, g_MIN_T, pr, &err)) {
        fprintf(stderr, "[ERROR ] %s\n", err.c_str());
        return;                      // no output_x.bin -> main() reports NO_OUTPUT
    }
    const uint32_t M = pr.M;
    const uint32_t tile = ODDFFT_CLAMP_TILE(g_TILE_SIZE);
    g_cell.tile = tile;
    uint32_t npuStages, cpuStages;
    arb_stage_counts(pr.inner, npuStages, cpuStages);
    g_cell.cpu_first = cpuStages;
    printf("[BLUE  ] N=%u  M=%u (=next_pow2(2N-1))  inner radix=%u  "
           "inner stages=%u (npu=%u cpu=%u) x2 per call\n",
           N, M, radix, (uint32_t)pr.inner.plan.stages.size(), npuStages, cpuStages);

    std::vector<float> output(2 * (size_t)N);

#ifdef ASCENDC_CPU_DEBUG
    (void)multicore;
    auto run_one = [&](bool) -> RunResult {
        RunResult res{0, 0, 0, npuStages, cpuStages};
        std::vector<float> ar(M, 0.0f), ai(M, 0.0f);
        for (uint32_t n = 0; n < N; ++n) {
            const float xr = input[2*n], xi = input[2*n+1];
            ar[n] = xr * pr.wcr[n] - xi * pr.wci[n];
            ai[n] = xr * pr.wci[n] + xi * pr.wcr[n];
        }
        inner_fft_host(pr.inner, ar.data(), ai.data(), false);
        for (uint32_t j = 0; j < M; ++j) {
            const float t = ar[j] * pr.fbr[j] - ai[j] * pr.fbi[j];
            ai[j]         = ar[j] * pr.fbi[j] + ai[j] * pr.fbr[j];
            ar[j]         = t;
        }
        inner_fft_host(pr.inner, ar.data(), ai.data(), true);
        for (uint32_t n = 0; n < N; ++n) {
            output[2*n]   = ar[n] * pr.wcr[n] - ai[n] * pr.wci[n];
            output[2*n+1] = ar[n] * pr.wci[n] + ai[n] * pr.wcr[n];
        }
        return res;
    };
    BenchOutcome __out = run_benchmark(warmup, bench_runs, run_one);
    append_csv_row(variant_name, N, tile, __out);
#else
    const size_t mb = (size_t)M * sizeof(float);
    float *xrHost, *xiHost;
    CHECK_ACL2(aclrtMallocHost((void **)&xrHost, mb));
    CHECK_ACL2(aclrtMallocHost((void **)&xiHost, mb));
    void *xrDev, *xiDev, *wcrDev, *wciDev, *fbrDev, *fbiDev, *twDev;
    CHECK_ACL2(aclrtMalloc(&xrDev,  mb, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL2(aclrtMalloc(&xiDev,  mb, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL2(aclrtMalloc(&wcrDev, mb, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL2(aclrtMalloc(&wciDev, mb, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL2(aclrtMalloc(&fbrDev, mb, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL2(aclrtMalloc(&fbiDev, mb, ACL_MEM_MALLOC_HUGE_FIRST));
    const size_t twb = (pr.inner.tw.empty() ? 1u : pr.inner.tw.size()) * sizeof(float);
    CHECK_ACL2(aclrtMalloc(&twDev, twb, ACL_MEM_MALLOC_HUGE_FIRST));
    aclrtStream stream;
    CHECK_ACL2(aclrtCreateStream(&stream));

    // Everything constant is uploaded ONCE, here, outside the timing: the
    // chirp, the transformed kernel, and the twiddles for every inner stage in
    // a single buffer. The implementation this replaces allocated and freed a
    // twiddle buffer per stage per call.
    CHECK_ACL2(aclrtMemcpy(wcrDev, mb, pr.wcr.data(), mb, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL2(aclrtMemcpy(wciDev, mb, pr.wci.data(), mb, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL2(aclrtMemcpy(fbrDev, mb, pr.fbr.data(), mb, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL2(aclrtMemcpy(fbiDev, mb, pr.fbi.data(), mb, ACL_MEMCPY_HOST_TO_DEVICE));
    if (!pr.inner.tw.empty())
        CHECK_ACL2(aclrtMemcpy(twDev, twb, pr.inner.tw.data(), twb, ACL_MEMCPY_HOST_TO_DEVICE));

    const uint32_t bd = multicore ? GetAivCoreNum(SOC_VERSION_STR) : 1u;
    DevCtx ctx{stream, xrDev, xiDev, twDev, xrHost, xiHost, bd, tile, 0.0};

    auto run_one = [&](bool) -> RunResult {
        RunResult res{0, 0, 0, npuStages, cpuStages};
        auto T0 = std::chrono::high_resolution_clock::now();

        auto c0 = std::chrono::high_resolution_clock::now();
        std::memset(xrHost, 0, mb);
        std::memset(xiHost, 0, mb);
        for (uint32_t n = 0; n < N; ++n) { xrHost[n] = input[2*n]; xiHost[n] = input[2*n+1]; }
        auto c1 = std::chrono::high_resolution_clock::now();
        double cpu = std::chrono::duration<double, std::micro>(c1 - c0).count();

        // The full M floats go up, not just the first N: the tail must be zero
        // before the forward transform, and after the first iteration it holds
        // the previous call's inverse-transform output.
        CHECK_ACL2(aclrtMemcpy(xrDev, mb, xrHost, mb, ACL_MEMCPY_HOST_TO_DEVICE));
        CHECK_ACL2(aclrtMemcpy(xiDev, mb, xiHost, mb, ACL_MEMCPY_HOST_TO_DEVICE));

        ctx.cpu_us = 0.0;
        // Pre-multiply over all M. The chirp table is zero beyond N, so this
        // also re-establishes the zero tail without a second pass -- and the
        // tail is a genuine zero rather than uninitialised device memory,
        // which matters because 0 * NaN is NaN.
        arb_cmul_do(bd, stream, xrDev, xiDev, wcrDev, wciDev, M, tile, bd);
        CHECK_ACL2(aclrtSynchronizeStream(stream));
        inner_fft_device(pr.inner, ctx, false);
        arb_cmul_do(bd, stream, xrDev, xiDev, fbrDev, fbiDev, M, tile, bd);
        CHECK_ACL2(aclrtSynchronizeStream(stream));
        inner_fft_device(pr.inner, ctx, true);
        // Post-multiply over the first N only: the rest is discarded.
        arb_cmul_do(bd, stream, xrDev, xiDev, wcrDev, wciDev, N, tile, bd);
        CHECK_ACL2(aclrtSynchronizeStream(stream));
        cpu += ctx.cpu_us;

        const size_t nb = (size_t)N * sizeof(float);
        CHECK_ACL2(aclrtMemcpy(xrHost, nb, xrDev, nb, ACL_MEMCPY_DEVICE_TO_HOST));
        CHECK_ACL2(aclrtMemcpy(xiHost, nb, xiDev, nb, ACL_MEMCPY_DEVICE_TO_HOST));
        auto c2 = std::chrono::high_resolution_clock::now();
        for (uint32_t n = 0; n < N; ++n) { output[2*n] = xrHost[n]; output[2*n+1] = xiHost[n]; }
        auto c3 = std::chrono::high_resolution_clock::now();
        cpu += std::chrono::duration<double, std::micro>(c3 - c2).count();

        auto T1 = std::chrono::high_resolution_clock::now();
        res.total_us = std::chrono::duration<double, std::micro>(T1 - T0).count();
        res.cpu_us   = cpu;
        res.npu_us   = res.total_us - cpu;
        return res;
    };
    BenchOutcome __out = run_benchmark(warmup, bench_runs, run_one);
    append_csv_row(variant_name, N, tile, __out);

    aclrtFree(xrDev); aclrtFree(xiDev); aclrtFree(twDev);
    aclrtFree(wcrDev); aclrtFree(wciDev); aclrtFree(fbrDev); aclrtFree(fbiDev);
    aclrtFreeHost(xrHost); aclrtFreeHost(xiHost);
    aclrtDestroyStream(stream);
#endif
    write_bin("./output/output_x.bin", output.data(), 2 * (size_t)N * sizeof(float));
}

// ---------------------------------------------------------------------------
// Rader
//
// X[g^-p] = x[0] + sum_q x[g^q] * W^(g^(q-p)), a CORRELATION of length L=N-1.
// Written as a convolution with b'[q] = W^(g^-q) it becomes (a * b')[p], and
// the cyclic convolution of length L is recovered from the linear one by
// folding c[p] + c[p+L]. The fold is why the kernel is zero-padded rather
// than periodically extended: extending b' with period L inside a buffer of
// length M is only equivalent to the cyclic convolution when L divides M,
// which it does not -- M is a power of two and L = N-1 is not.
// ---------------------------------------------------------------------------
static void run_rader(uint32_t N, uint32_t m, const std::vector<float> &input,
                      uint32_t warmup, uint32_t bench_runs,
                      const std::string &variant_name,
                      uint32_t radix, bool multicore)
{
    (void)m;
    RaderPrep pr;
    std::string err;
    if (!rader_prepare(N, radix, g_MIN_T, pr, &err)) {
        fprintf(stderr, "[ERROR ] %s\n", err.c_str());
        return;
    }
    const uint32_t M = pr.M, L = pr.L;
    const uint32_t tile = ODDFFT_CLAMP_TILE(g_TILE_SIZE);
    g_cell.tile = tile;
    uint32_t npuStages, cpuStages;
    arb_stage_counts(pr.inner, npuStages, cpuStages);
    g_cell.cpu_first = cpuStages;
    printf("[RADER ] N=%u (prime)  L=%u  M=%u  g=%u  g^-1=%u  inner radix=%u  "
           "inner stages=%u (npu=%u cpu=%u) x2 per call\n",
           N, L, M, pr.g, pr.ginv, radix,
           (uint32_t)pr.inner.plan.stages.size(), npuStages, cpuStages);

    std::vector<float> output(2 * (size_t)N);

    // The two host halves, shared by both paths so there is one definition of
    // the index algebra rather than two that can disagree.
    auto gather = [&](float *ar, float *ai) {
        for (uint32_t q = 0; q < L; ++q) {
            const uint32_t n = pr.idxIn[q];
            ar[q] = input[2*n]; ai[q] = input[2*n+1];
        }
        for (uint32_t q = L; q < M; ++q) { ar[q] = 0.0f; ai[q] = 0.0f; }
    };
    auto scatter = [&](const float *cr, const float *ci) {
        double sr = 0.0, si = 0.0;
        for (uint32_t n = 0; n < N; ++n) { sr += input[2*n]; si += input[2*n+1]; }
        output[0] = (float)sr; output[1] = (float)si;
        const float x0r = input[0], x0i = input[1];
        for (uint32_t pp = 0; pp < L; ++pp) {
            const uint32_t k = pr.idxOut[pp];
            output[2*k]   = x0r + cr[pp] + cr[pp + L];
            output[2*k+1] = x0i + ci[pp] + ci[pp + L];
        }
    };

#ifdef ASCENDC_CPU_DEBUG
    (void)multicore;
    auto run_one = [&](bool) -> RunResult {
        RunResult res{0, 0, 0, npuStages, cpuStages};
        std::vector<float> ar(M), ai(M);
        gather(ar.data(), ai.data());
        inner_fft_host(pr.inner, ar.data(), ai.data(), false);
        for (uint32_t j = 0; j < M; ++j) {
            const float t = ar[j] * pr.fbr[j] - ai[j] * pr.fbi[j];
            ai[j]         = ar[j] * pr.fbi[j] + ai[j] * pr.fbr[j];
            ar[j]         = t;
        }
        inner_fft_host(pr.inner, ar.data(), ai.data(), true);
        scatter(ar.data(), ai.data());
        return res;
    };
    BenchOutcome __out = run_benchmark(warmup, bench_runs, run_one);
    append_csv_row(variant_name, N, tile, __out);
#else
    const size_t mb = (size_t)M * sizeof(float);
    float *xrHost, *xiHost;
    CHECK_ACL2(aclrtMallocHost((void **)&xrHost, mb));
    CHECK_ACL2(aclrtMallocHost((void **)&xiHost, mb));
    void *xrDev, *xiDev, *fbrDev, *fbiDev, *twDev;
    CHECK_ACL2(aclrtMalloc(&xrDev,  mb, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL2(aclrtMalloc(&xiDev,  mb, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL2(aclrtMalloc(&fbrDev, mb, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL2(aclrtMalloc(&fbiDev, mb, ACL_MEM_MALLOC_HUGE_FIRST));
    const size_t twb = (pr.inner.tw.empty() ? 1u : pr.inner.tw.size()) * sizeof(float);
    CHECK_ACL2(aclrtMalloc(&twDev, twb, ACL_MEM_MALLOC_HUGE_FIRST));
    aclrtStream stream;
    CHECK_ACL2(aclrtCreateStream(&stream));

    CHECK_ACL2(aclrtMemcpy(fbrDev, mb, pr.fbr.data(), mb, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL2(aclrtMemcpy(fbiDev, mb, pr.fbi.data(), mb, ACL_MEMCPY_HOST_TO_DEVICE));
    if (!pr.inner.tw.empty())
        CHECK_ACL2(aclrtMemcpy(twDev, twb, pr.inner.tw.data(), twb, ACL_MEMCPY_HOST_TO_DEVICE));

    const uint32_t bd = multicore ? GetAivCoreNum(SOC_VERSION_STR) : 1u;
    DevCtx ctx{stream, xrDev, xiDev, twDev, xrHost, xiHost, bd, tile, 0.0};

    auto run_one = [&](bool) -> RunResult {
        RunResult res{0, 0, 0, npuStages, cpuStages};
        auto T0 = std::chrono::high_resolution_clock::now();

        auto c0 = std::chrono::high_resolution_clock::now();
        gather(xrHost, xiHost);
        auto c1 = std::chrono::high_resolution_clock::now();
        double cpu = std::chrono::duration<double, std::micro>(c1 - c0).count();

        CHECK_ACL2(aclrtMemcpy(xrDev, mb, xrHost, mb, ACL_MEMCPY_HOST_TO_DEVICE));
        CHECK_ACL2(aclrtMemcpy(xiDev, mb, xiHost, mb, ACL_MEMCPY_HOST_TO_DEVICE));

        ctx.cpu_us = 0.0;
        inner_fft_device(pr.inner, ctx, false);
        arb_cmul_do(bd, stream, xrDev, xiDev, fbrDev, fbiDev, M, tile, bd);
        CHECK_ACL2(aclrtSynchronizeStream(stream));
        inner_fft_device(pr.inner, ctx, true);
        CHECK_ACL2(aclrtSynchronizeStream(stream));
        cpu += ctx.cpu_us;

        // All M floats come back: the fold reads c[p+L] for p < L, that is up
        // to index 2L-1.
        CHECK_ACL2(aclrtMemcpy(xrHost, mb, xrDev, mb, ACL_MEMCPY_DEVICE_TO_HOST));
        CHECK_ACL2(aclrtMemcpy(xiHost, mb, xiDev, mb, ACL_MEMCPY_DEVICE_TO_HOST));
        auto c2 = std::chrono::high_resolution_clock::now();
        scatter(xrHost, xiHost);
        auto c3 = std::chrono::high_resolution_clock::now();
        cpu += std::chrono::duration<double, std::micro>(c3 - c2).count();

        auto T1 = std::chrono::high_resolution_clock::now();
        res.total_us = std::chrono::duration<double, std::micro>(T1 - T0).count();
        res.cpu_us   = cpu;
        res.npu_us   = res.total_us - cpu;
        return res;
    };
    BenchOutcome __out = run_benchmark(warmup, bench_runs, run_one);
    append_csv_row(variant_name, N, tile, __out);

    aclrtFree(xrDev); aclrtFree(xiDev); aclrtFree(twDev);
    aclrtFree(fbrDev); aclrtFree(fbiDev);
    aclrtFreeHost(xrHost); aclrtFreeHost(xiHost);
    aclrtDestroyStream(stream);
#endif
    write_bin("./output/output_x.bin", output.data(), 2 * (size_t)N * sizeof(float));
}

// ===========================================================================
// DEVICE DIAGNOSTIC -- not a benchmark variant
//
// Every cell failing verification says nothing about WHERE. The CPU-debug path
// is already proven against a naive DFT at 38 sizes, so the defect is in the
// device path, and the device path has exactly three new pieces: the cmul
// kernel, the cscale kernel, and the inner transform's stage launches. This
// compares each of them against the host arithmetic that is known correct,
// stage by stage, and names the FIRST divergence.
//
// It writes no output file on purpose, so main() reports NO_OUTPUT (exit 2)
// and run_matrix.sh records SKIP rather than a result. Run it directly:
//
//     ./build/oddfft chk/inner_r4_xs 64 0 1 1
//     ./build/oddfft chk/inner_r2_xs 64 0 1 8
//
// The last argument is MIN_T, which decides how many stages go to the device.
// ===========================================================================
#ifndef ASCENDC_CPU_DEBUG
static double max_rel_diff(const float *ar, const float *ai,
                           const float *br, const float *bi, uint32_t n,
                           uint32_t *worst, bool *any_nan)
{
    double norm = 0.0, worst_d = 0.0;
    *worst = 0; *any_nan = false;
    for (uint32_t i = 0; i < n; ++i) {
        if (!(br[i] == br[i]) || !(bi[i] == bi[i]) ||
            !(ar[i] == ar[i]) || !(ai[i] == ai[i])) *any_nan = true;
        const double m = std::fabs((double)br[i]) + std::fabs((double)bi[i]);
        if (m > norm) norm = m;
    }
    if (norm <= 0.0) norm = 1.0;
    for (uint32_t i = 0; i < n; ++i) {
        const double d = std::fabs((double)ar[i] - (double)br[i]) +
                         std::fabs((double)ai[i] - (double)bi[i]);
        if (d / norm > worst_d) { worst_d = d / norm; *worst = i; }
    }
    return worst_d;
}

static void run_diag(uint32_t N, uint32_t m, const std::vector<float> &input,
                     uint32_t warmup, uint32_t bench_runs,
                     const std::string &variant_name, uint32_t radix)
{
    (void)m; (void)warmup; (void)bench_runs; (void)variant_name;
    // The same M Bluestein would use at this N, so the diagnosis is about the
    // configuration that actually fails rather than a convenient one.
    const uint32_t M = next_pow2(2u * N - 1u);
    InnerFFT f;
    std::string err;
    if (!build_inner(M, radix, g_MIN_T, f, &err)) {
        printf("[DIAG  ] build_inner failed: %s\n", err.c_str());
        return;
    }
    const uint32_t tile = ODDFFT_CLAMP_TILE(g_TILE_SIZE);
    printf("================================================================\n");
    printf("[DIAG  ] N=%u -> M=%u  inner radix=%u  tile=%u  minT=%u\n",
           N, M, radix, tile, g_MIN_T);
    printf("[DIAG  ] inner plan: %zu stages, first %u on the host\n",
           f.plan.stages.size(), f.plan.cpu_first);

    const size_t mb = (size_t)M * sizeof(float);
    float *hr, *hi;
    CHECK_ACL2(aclrtMallocHost((void **)&hr, mb));
    CHECK_ACL2(aclrtMallocHost((void **)&hi, mb));
    void *dr, *di, *er, *ei, *twDev;
    CHECK_ACL2(aclrtMalloc(&dr, mb, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL2(aclrtMalloc(&di, mb, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL2(aclrtMalloc(&er, mb, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL2(aclrtMalloc(&ei, mb, ACL_MEM_MALLOC_HUGE_FIRST));
    const size_t twb = (f.tw.empty() ? 1u : f.tw.size()) * sizeof(float);
    CHECK_ACL2(aclrtMalloc(&twDev, twb, ACL_MEM_MALLOC_HUGE_FIRST));
    aclrtStream stream;
    CHECK_ACL2(aclrtCreateStream(&stream));
    if (!f.tw.empty())
        CHECK_ACL2(aclrtMemcpy(twDev, twb, f.tw.data(), twb, ACL_MEMCPY_HOST_TO_DEVICE));
    const uint32_t bd = GetAivCoreNum(SOC_VERSION_STR);

    // Deliberately not symmetric and not real: a reversed permutation or a
    // swapped real/imaginary pair is invisible on data that is either.
    std::vector<float> ar(M), ai(M), br(M), bi(M);
    for (uint32_t i = 0; i < M; ++i) {
        ar[i] = (i < 2u * N) ? input[(2u * i) % input.size()] : 0.0f;
        ai[i] = (i < 2u * N) ? input[(2u * i + 1u) % input.size()] : 0.0f;
        br[i] = (float)std::cos(0.37 * (double)i + 0.11);
        bi[i] = (float)std::sin(0.53 * (double)i + 0.29);
    }
    int bad = 0;
    uint32_t w; bool nan;

    // ---- 1. arb_cmul ------------------------------------------------------
    {
        std::vector<float> gr(M), gi(M);
        for (uint32_t i = 0; i < M; ++i) {
            gr[i] = ar[i] * br[i] - ai[i] * bi[i];
            gi[i] = ar[i] * bi[i] + ai[i] * br[i];
        }
        CHECK_ACL2(aclrtMemcpy(dr, mb, ar.data(), mb, ACL_MEMCPY_HOST_TO_DEVICE));
        CHECK_ACL2(aclrtMemcpy(di, mb, ai.data(), mb, ACL_MEMCPY_HOST_TO_DEVICE));
        CHECK_ACL2(aclrtMemcpy(er, mb, br.data(), mb, ACL_MEMCPY_HOST_TO_DEVICE));
        CHECK_ACL2(aclrtMemcpy(ei, mb, bi.data(), mb, ACL_MEMCPY_HOST_TO_DEVICE));
        arb_cmul_do(bd, stream, dr, di, er, ei, M, tile, bd);
        CHECK_ACL2(aclrtSynchronizeStream(stream));
        CHECK_ACL2(aclrtMemcpy(hr, mb, dr, mb, ACL_MEMCPY_DEVICE_TO_HOST));
        CHECK_ACL2(aclrtMemcpy(hi, mb, di, mb, ACL_MEMCPY_DEVICE_TO_HOST));
        const double d = max_rel_diff(hr, hi, gr.data(), gi.data(), M, &w, &nan);
        printf("[DIAG 1] arb_cmul      max_rel=%.3e worst=%u nan=%d  %s\n",
               d, w, (int)nan, (d < 1e-5) ? "OK" : "<<< WRONG");
        if (d >= 1e-5) {
            printf("         at %u: got (%.6g, %.6g) want (%.6g, %.6g)\n",
                   w, hr[w], hi[w], gr[w], gi[w]);
            ++bad;
        }
    }

    // ---- 2. arb_cscale ----------------------------------------------------
    {
        const float sc = 0.25f;
        std::vector<float> gr(M), gi(M);
        for (uint32_t i = 0; i < M; ++i) { gr[i] = ar[i] * sc; gi[i] = -ai[i] * sc; }
        CHECK_ACL2(aclrtMemcpy(dr, mb, ar.data(), mb, ACL_MEMCPY_HOST_TO_DEVICE));
        CHECK_ACL2(aclrtMemcpy(di, mb, ai.data(), mb, ACL_MEMCPY_HOST_TO_DEVICE));
        arb_cscale_do(bd, stream, dr, di, M, sc, -1.0f, tile, bd);
        CHECK_ACL2(aclrtSynchronizeStream(stream));
        CHECK_ACL2(aclrtMemcpy(hr, mb, dr, mb, ACL_MEMCPY_DEVICE_TO_HOST));
        CHECK_ACL2(aclrtMemcpy(hi, mb, di, mb, ACL_MEMCPY_DEVICE_TO_HOST));
        const double d = max_rel_diff(hr, hi, gr.data(), gi.data(), M, &w, &nan);
        printf("[DIAG 2] arb_cscale    max_rel=%.3e worst=%u nan=%d  %s\n",
               d, w, (int)nan, (d < 1e-5) ? "OK" : "<<< WRONG");
        if (d >= 1e-5) {
            printf("         at %u: got (%.6g, %.6g) want (%.6g, %.6g)\n",
                   w, hr[w], hi[w], gr[w], gi[w]);
            ++bad;
        }
    }

    // ---- 3. the inner transform, STAGE BY STAGE ---------------------------
    // Both sides start from the same data: permuted, with the sub-floor
    // prefix already applied, exactly as inner_fft_device hands it over.
    {
        std::vector<float> zr(M), zi(M);
        apply_perm(f, ar.data(), ai.data(), zr.data(), zi.data());
        for (size_t s = 0; s < f.plan.stages.size(); ++s) {
            if (f.plan.stages[s].on_npu) break;
            host_stage_fast(f.plan, s, zr.data(), zi.data());
        }
        CHECK_ACL2(aclrtMemcpy(dr, mb, zr.data(), mb, ACL_MEMCPY_HOST_TO_DEVICE));
        CHECK_ACL2(aclrtMemcpy(di, mb, zi.data(), mb, ACL_MEMCPY_HOST_TO_DEVICE));

        for (size_t s = 0; s < f.plan.stages.size(); ++s) {
            const StageInfo &st = f.plan.stages[s];
            if (!st.on_npu) continue;
            host_stage_fast(f.plan, s, zr.data(), zi.data());   // the reference

            const OddKernelEntry *ke = nullptr;
            for (uint32_t i = 0; i < ODD_KERNEL_COUNT; ++i)
                if (ODD_KERNELS[i].radix == st.radix &&
                    std::string("direct") == ODD_KERNELS[i].butterfly) ke = &ODD_KERNELS[i];
            if (!ke) { printf("[DIAG 3] stage %zu radix %u: NO KERNEL\n", s, st.radix); ++bad; continue; }
            const uint32_t rowLen = ((st.t + 7u) / 8u) * 8u;
            const uint32_t nShift = ODDFFT_SHIFT_COUNT(st.t, tile);
            const uint32_t nTiles = st.nBlk * nShift;
            const uint32_t sbd = (nTiles < bd) ? nTiles : bd;
            ke->launch(sbd, stream, dr, di, twDev,
                       M, st.t, st.block, st.wOffset, rowLen, tile, sbd);
            CHECK_ACL2(aclrtSynchronizeStream(stream));
            CHECK_ACL2(aclrtMemcpy(hr, mb, dr, mb, ACL_MEMCPY_DEVICE_TO_HOST));
            CHECK_ACL2(aclrtMemcpy(hi, mb, di, mb, ACL_MEMCPY_DEVICE_TO_HOST));
            const double d = max_rel_diff(hr, hi, zr.data(), zi.data(), M, &w, &nan);
            printf("[DIAG 3] stage %-2zu r=%u t=%-6u block=%-6u nBlk=%-5u wOff=%-7u "
                   "rowLen=%-4u tiles=%-5u bd=%-3u max_rel=%.3e nan=%d %s\n",
                   s, st.radix, st.t, st.block, st.nBlk, st.wOffset, rowLen,
                   nTiles, sbd, d, (int)nan, (d < 1e-4) ? "OK" : "<<< WRONG");
            if (d >= 1e-4) {
                printf("         first divergence at %u: got (%.6g, %.6g) want (%.6g, %.6g)\n",
                       w, hr[w], hi[w], zr[w], zi[w]);
                ++bad;
                break;   // everything after the first wrong stage is noise
            }
        }
    }

    printf("[DIAG  ] %s\n", bad ? "DEFECT LOCALISED ABOVE" : "all three pieces agree with the host");
    printf("================================================================\n");

    aclrtFree(dr); aclrtFree(di); aclrtFree(er); aclrtFree(ei); aclrtFree(twDev);
    aclrtFreeHost(hr); aclrtFreeHost(hi);
    aclrtDestroyStream(stream);
    // No output file: main() reports NO_OUTPUT and the matrix records SKIP.
}

static void diag_r2(uint32_t N, uint32_t m, const std::vector<float> &in,
                    uint32_t wu, uint32_t br2, const std::string &vn)
{ run_diag(N, m, in, wu, br2, vn, 2); }
static void diag_r4(uint32_t N, uint32_t m, const std::vector<float> &in,
                    uint32_t wu, uint32_t br2, const std::string &vn)
{ run_diag(N, m, in, wu, br2, vn, 4); }
#endif  // !ASCENDC_CPU_DEBUG

// ===========================================================================
// Registration: 8 variants, three paired axes
//   construction  bluestein vs rader   (same N, same inner radix, same cores)
//   inner radix   r2 vs r4             (the power-of-two winner, re-measured
//                                       in a setting where M is not N)
//   cores         ps (1) vs psm (40)
// radix is recorded as 0 so main() accepts any N: for Bluestein that is the
// point, and for Rader the primality check lives in rader_prepare, which can
// say WHY a composite N was refused.
// ===========================================================================
static uint32_t blue_prewarm_r2(uint32_t N, uint32_t) { BluePrep p; std::string e; return blue_prepare(N, 2, g_MIN_T, p, &e) ? p.inner.plan.cpu_first : 0u; }
static uint32_t blue_prewarm_r4(uint32_t N, uint32_t) { BluePrep p; std::string e; return blue_prepare(N, 4, g_MIN_T, p, &e) ? p.inner.plan.cpu_first : 0u; }
static uint32_t rad_prewarm_r2(uint32_t N, uint32_t)  { RaderPrep p; std::string e; return rader_prepare(N, 2, g_MIN_T, p, &e) ? p.inner.plan.cpu_first : 0u; }
static uint32_t rad_prewarm_r4(uint32_t N, uint32_t)  { RaderPrep p; std::string e; return rader_prepare(N, 4, g_MIN_T, p, &e) ? p.inner.plan.cpu_first : 0u; }

#define ARB_THUNK(fn, body) \
    static void fn(uint32_t N, uint32_t m, const std::vector<float> &in, \
                   uint32_t wu, uint32_t br, const std::string &vn) { body; }
ARB_THUNK(blue_r2_ps,  run_blue(N, m, in, wu, br, vn, 2, false))
ARB_THUNK(blue_r2_psm, run_blue(N, m, in, wu, br, vn, 2, true))
ARB_THUNK(blue_r4_ps,  run_blue(N, m, in, wu, br, vn, 4, false))
ARB_THUNK(blue_r4_psm, run_blue(N, m, in, wu, br, vn, 4, true))
ARB_THUNK(rad_r2_ps,   run_rader(N, m, in, wu, br, vn, 2, false))
ARB_THUNK(rad_r2_psm,  run_rader(N, m, in, wu, br, vn, 2, true))
ARB_THUNK(rad_r4_ps,   run_rader(N, m, in, wu, br, vn, 4, false))
ARB_THUNK(rad_r4_psm,  run_rader(N, m, in, wu, br, vn, 4, true))
#undef ARB_THUNK

void register_arblen(Registry &reg)
{
    reg.add({"blu", "bluestein_r2_ps_xs",  0, "bluestein", "dit_r2", "perstage", "xs", "vector", 0, 0,  1, false, blue_r2_ps,  blue_prewarm_r2});
    reg.add({"blu", "bluestein_r2_psm_xs", 0, "bluestein", "dit_r2", "perstage", "xs", "vector", 0, 0, 40, false, blue_r2_psm, blue_prewarm_r2});
    reg.add({"blu", "bluestein_r4_ps_xs",  0, "bluestein", "dit_r4", "perstage", "xs", "vector", 0, 0,  1, false, blue_r4_ps,  blue_prewarm_r4});
    reg.add({"blu", "bluestein_r4_psm_xs", 0, "bluestein", "dit_r4", "perstage", "xs", "vector", 0, 0, 40, false, blue_r4_psm, blue_prewarm_r4});
    reg.add({"rad", "rader_r2_ps_xs",      0, "rader",     "dit_r2", "perstage", "xs", "vector", 0, 0,  1, false, rad_r2_ps,   rad_prewarm_r2});
    reg.add({"rad", "rader_r2_psm_xs",     0, "rader",     "dit_r2", "perstage", "xs", "vector", 0, 0, 40, false, rad_r2_psm,  rad_prewarm_r2});
    reg.add({"rad", "rader_r4_ps_xs",      0, "rader",     "dit_r4", "perstage", "xs", "vector", 0, 0,  1, false, rad_r4_ps,   rad_prewarm_r4});
    reg.add({"rad", "rader_r4_psm_xs",     0, "rader",     "dit_r4", "perstage", "xs", "vector", 0, 0, 40, false, rad_r4_psm,  rad_prewarm_r4});
#ifndef ASCENDC_CPU_DEBUG
    // Diagnostics, not benchmark cells. They produce no output file, so the
    // matrix records SKIP for them and they cannot contaminate a result.
    reg.add({"chk", "inner_r2_xs", 0, "diag", "dit_r2", "perstage", "xs", "vector", 0, 0, 40, false, diag_r2, nullptr});
    reg.add({"chk", "inner_r4_xs", 0, "diag", "dit_r4", "perstage", "xs", "vector", 0, 0, 40, false, diag_r4, nullptr});
#endif
}
