// host/impl_r8.cpp  (FFTBench)
//
// All radix-8 variants. The run-function bodies were carried over from
// live/Radix8/host/main.cpp WITH NO CHANGES — this is code already proven on
// hardware, and any edit here would be needless risk. Only the environment
// around them changed:
//   * the radix-specific part lives in namespace r8 (host/fft_host_r8.h);
//   * the run label (impl, axes, twiddle-table warmup) is carried by g_cell
//     (host/cell.h), so append_csv_row/print_bench_banner needed no new
//     arguments and the function bodies stayed untouched;
//   * g_TILE_SIZE is now shared, defined in host/main.cpp;
//   * at the bottom — register_r8(), which hands the variants to the shared
//     registry.
#include "kernel_decls.h"
#include "fft_host_r8.h"
#include "cube_tiling.h"
#include "cube1_common.h"
#include "bench_harness.h"
#include "registry.h"
#include "../common/data_utils.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>

#ifndef ASCENDC_CPU_DEBUG
#include "acl/acl.h"
#define CHECK_ACL2(x) do { \
    auto _rc = (x); \
    if (_rc != ACL_SUCCESS) fprintf(stderr, "[ERROR] ACL call failed: %s (rc=%d) at %s:%d\n", \
                                     #x, (int)_rc, __FILE__, __LINE__); \
} while (0)
#endif

using namespace r8;


// Must match this->tile in kernels/r8_dit_fused_xs.cpp — the tile width of
// the Vector kernel in elements. The host side needs it to work out the
// number of tiles in a stage and pick blockDim.
static constexpr uint32_t DIT_TILE_HOST = 64;

// The batch width of a Cube call (how many independent 8-point butterflies
// are computed in one matmul) — taken as exactly the same constant the device
// kernels see: there used to be a private literal 128 here, and the two could
// have diverged silently.
static constexpr uint32_t CUBE_BATCH_HOST = CUBE_BATCH;

#ifndef ASCENDC_CPU_DEBUG
// ---------------------------------------------------------------------------
// Profiling BY LAUNCH TYPE — switched on with RADIX8_PROFILE=1
//
// WHY. The sweep over CUBE_BATCH (run_matrix.sh -B) gave an unexpected
// result: the time of a group of tiles barely depends on the volume of work
// and is described well by the formula
//
//      t(group) ~ 40 us + 2.7 us * (tiles in the group)
//
// (B=256: 44.8, B=128: 49.1, B=64: 61.2, B=32: 83.8 us per group at 2/4/8/16
// tiles). Two conclusions follow from it at once: (1) the tiles inside a
// group do NOT run in parallel across the cores — otherwise the 2.7*t term
// would be flat; (2) at a large B everything is eaten by the constant 40 us
// per group, not by the arithmetic. But WHICH part of the group eats it is
// not visible: a group is three launches (repack on AIV, matmul on AIC,
// unpack on AIV) plus two round-trips through GM.
//
// Hence this mode: it synchronizes AFTER EVERY launch and accumulates the
// time by type. Two limitations that matter, and because of which the mode is
// switched on only by an environment variable:
//   * a sync after every launch breaks the pipelining and INFLATES the total
//     time — the numbers from this mode are NOT fit for the report;
//   * the only thing that means anything is the LAYOUT (which of the three
//     dominates) and how it changes between two values of CUBE_BATCH: the
//     difference tells which of the kernels the "per tile" term sits in.
struct LaunchProfile {
    double   us[3]    = {0.0, 0.0, 0.0};   // cube: repack / matmul / unpack
    uint64_t calls[3] = {0, 0, 0};         // dit:  slot 0 = stage
    uint64_t blocks   = 0;                 // sum of blockDim over all launches
};
static LaunchProfile g_prof;

// Zeroing the counters. Today main() dispatches exactly ONE variant per
// process, so a stale g_prof cannot leak between variants — but the counters
// are a file-scope global, and the moment two r8 variants run in one process
// the second one's "share" and "per launch" figures would silently include
// the first one's launches. One call at the top of each run function costs
// nothing and removes that trap.
static void profile_reset()
{
    g_prof = LaunchProfile{};
}

static bool profile_enabled()
{
    static const bool on = (getenv("RADIX8_PROFILE") != nullptr);
    return on;
}

// Run a launch; in profiling mode — wait for it to finish and record the time
// into the slot. Outside that mode — exactly the same call, with no overhead.
template <typename F>
static void profiled_launch(int slot, aclrtStream stream, uint32_t blockDim, F &&launch)
{
    if (!profile_enabled()) { launch(); return; }
    auto p0 = std::chrono::high_resolution_clock::now();
    launch();
    CHECK_ACL2(aclrtSynchronizeStream(stream));
    auto p1 = std::chrono::high_resolution_clock::now();
    g_prof.us[slot]    += std::chrono::duration<double, std::micro>(p1 - p0).count();
    g_prof.calls[slot] += 1;
    g_prof.blocks      += blockDim;
}

static void print_profile(const char *const names[3])
{
    if (!profile_enabled()) return;
    double total = g_prof.us[0] + g_prof.us[1] + g_prof.us[2];
    if (total <= 0.0) return;
    fprintf(stderr, "[PROFILE] WARNING: sync after every launch — the totals are inflated,\n");
    fprintf(stderr, "[PROFILE] only the layout and the per-launch average mean anything.\n");
    for (int i = 0; i < 3; ++i) {
        if (g_prof.calls[i] == 0) continue;
        fprintf(stderr, "[PROFILE] %-8s launches=%-5llu  total=%9.1f us  "
                        "per launch=%7.2f us  share=%5.1f%%\n",
                names[i], (unsigned long long)g_prof.calls[i], g_prof.us[i],
                g_prof.us[i] / (double)g_prof.calls[i], 100.0 * g_prof.us[i] / total);
    }
    uint64_t allCalls = g_prof.calls[0] + g_prof.calls[1] + g_prof.calls[2];
    fprintf(stderr, "[PROFILE] launches in total=%llu, blocks=%llu "
                    "(blockDim=%.1f on average), total=%.1f us\n",
            (unsigned long long)allCalls, (unsigned long long)g_prof.blocks,
            allCalls ? (double)g_prof.blocks / (double)allCalls : 0.0, total);
    fflush(stderr);
}
#endif

// A software fp32 -> fp16 (IEEE754 binary16) conversion for the host code
// (host/impl_r8.cpp is not device code, there is no `half` type here; W16 is
// uploaded to the device as raw uint16_t fp16 bits). A plain
// round-to-nearest-even implementation, enough for the one-off preparation of
// the constant 16x16 DFT weight matrix.
static uint16_t FloatToHalfBits(float f)
{
    uint32_t x; memcpy(&x, &f, sizeof(x));
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t  exp  = (int32_t)((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = x & 0x7FFFFFu;
    if (((x >> 23) & 0xFF) == 0xFF) { // inf/nan
        return (uint16_t)(sign | 0x7C00u | (mant ? 0x0200u : 0u));
    }
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign; // underflow -> 0
        mant |= 0x800000u;
        uint32_t shift = (uint32_t)(14 - exp);
        uint32_t halfMant = mant >> shift;
        if ((mant >> (shift - 1)) & 1u) halfMant++;
        return (uint16_t)(sign | halfMant);
    }
    if (exp >= 31) return (uint16_t)(sign | 0x7C00u); // overflow -> inf
    uint32_t halfMant = mant >> 13;
    if (mant & 0x1000u) {
        halfMant++;
        if (halfMant == 0x400u) { halfMant = 0; exp++; if (exp >= 31) return (uint16_t)(sign | 0x7C00u); }
    }
    return (uint16_t)(sign | (uint32_t)(exp << 10) | halfMant);
}

// The inverse of FloatToHalfBits: needed to compute the correction
// A_lo = M16 - float(A_hi) for the precision-recovery scheme (terms>=3, see
// kernels/r8_cube_fused_xs.cpp). Half-precision denormals are handled too —
// M16 never has any, but the function is a general one.
static float HalfBitsToFloat(uint16_t h)
{
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp  = (uint32_t)(h >> 10) & 0x1Fu;
    uint32_t mant = (uint32_t)(h & 0x3FFu);
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;                                  // +-0
        } else {                                          // half denormal
            exp = 127 - 15 + 1;
            while ((mant & 0x400u) == 0) { mant <<= 1; --exp; }
            mant &= 0x3FFu;
            bits = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 0x1Fu) {
        bits = sign | 0x7F800000u | (mant << 13);         // inf/nan
    } else {
        bits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float f; memcpy(&f, &bits, sizeof(f)); return f;
}

// ============================================================================
// DIT fused, X-Split  (dit_fs_xs, R8-DT-FS-XS)
// ============================================================================
static void run_dit_fs_xs(uint32_t N, uint32_t m3, const std::vector<float> &input,
                           uint32_t warmup, uint32_t bench_runs, const std::string &variant_name)
{
    std::vector<float> xr0(N), xi0(N);
    for (uint32_t i = 0; i < N; ++i) { xr0[i] = input[2*i]; xi0[i] = input[2*i+1]; }
    digit_reverse_permute8_split(xr0.data(), xi0.data(), N);

    std::vector<FusedStageInfo> stages;
    uint32_t w_total = 0, cpu_stages = 0;
    build_fused_plan8(m3, g_TILE_SIZE, stages, w_total, cpu_stages);
    std::vector<float> w(w_total);
    fill_fused_buffers8_packed(w.data(), stages);
    uint32_t numStages = (uint32_t)stages.size();

    std::vector<uint32_t> meta(numStages * 3);
    for (size_t i = 0; i < stages.size(); ++i) {
        meta[i*3+0] = stages[i].eighth; meta[i*3+1] = stages[i].aligned; meta[i*3+2] = stages[i].wOffset;
    }

    std::vector<float> output(2 * N);

#ifdef ASCENDC_CPU_DEBUG
    float *xr = (float *)AscendC::GmAlloc(N * sizeof(float));
    float *xi = (float *)AscendC::GmAlloc(N * sizeof(float));
    float *wDev = (float *)AscendC::GmAlloc(w_total * sizeof(float));
    uint32_t *metaDev = (uint32_t *)AscendC::GmAlloc(meta.size() * sizeof(uint32_t));
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    memcpy(wDev, w.data(), w_total * sizeof(float));
    memcpy(metaDev, meta.data(), meta.size() * sizeof(uint32_t));

    auto run_one = [&](bool verbose) -> RunResult {
        RunResult res{0, 0, 0, numStages, cpu_stages};
        memcpy(xr, xr0.data(), N * sizeof(float));
        memcpy(xi, xi0.data(), N * sizeof(float));

        if (cpu_stages > 0) {
            auto t0 = std::chrono::high_resolution_clock::now();
            for (uint32_t s = 1; s <= cpu_stages; ++s) host_butterfly8_stage_split(xr, xi, N, 1u << (3*(s-1)));
            auto t1 = std::chrono::high_resolution_clock::now();
            res.cpu_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
        }

        auto t0 = std::chrono::high_resolution_clock::now();
        if (numStages > 0) {
            // One call = one stage (see Process() in
            // kernels/r8_dit_fused_xs.cpp). In CPU-debug there is always just
            // one core, so blk_num = 1.
            for (uint32_t st = 0; st < numStages; ++st) {
                ICPU_RUN_KF(fft8_dit_fused_xs, 1,
                    reinterpret_cast<GM_ADDR>(xr), reinterpret_cast<GM_ADDR>(xi),
                    reinterpret_cast<GM_ADDR>(wDev), reinterpret_cast<GM_ADDR>(metaDev),
                    N, numStages, st, 1u);
            }
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        res.npu_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
        res.total_us = res.npu_us + res.cpu_us;
        return res;
    };
    BenchOutcome __out = run_benchmark(warmup, bench_runs, run_one);
    append_csv_row(variant_name, N, g_TILE_SIZE, __out);
    for (uint32_t i = 0; i < N; ++i) { output[2*i] = xr[i]; output[2*i+1] = xi[i]; }
    AscendC::GmFree(xr); AscendC::GmFree(xi); AscendC::GmFree(wDev); AscendC::GmFree(metaDev);
#else
    float *xrHost, *xiHost;
    CHECK_ACL2(aclrtMallocHost((void **)&xrHost, N * sizeof(float)));
    CHECK_ACL2(aclrtMallocHost((void **)&xiHost, N * sizeof(float)));
    void *xrDev, *xiDev, *wDev, *metaDev;
    CHECK_ACL2(aclrtMalloc(&xrDev, N * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL2(aclrtMalloc(&xiDev, N * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL2(aclrtMalloc(&wDev, w_total * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL2(aclrtMalloc(&metaDev, meta.size() * sizeof(uint32_t), ACL_MEM_MALLOC_HUGE_FIRST));
    aclrtStream stream; CHECK_ACL2(aclrtCreateStream(&stream));
    CHECK_ACL2(aclrtMemcpy(wDev, w_total * sizeof(float), w.data(), w_total * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL2(aclrtMemcpy(metaDev, meta.size() * sizeof(uint32_t), meta.data(), meta.size() * sizeof(uint32_t), ACL_MEMCPY_HOST_TO_DEVICE));

    const uint32_t aivCores = GetAivCoreNum(SOC_VERSION_STR);
    {
        fprintf(stderr, "[DIT8-PLAN] AIV cores=%u, stages on NPU=%u, launches=%u "
                        "(one per stage), tile=%u elements\n",
                aivCores, numStages, numStages, DIT_TILE_HOST);
        for (uint32_t st = 0; st < numStages; ++st) {
            const uint32_t eighth = stages[st].eighth;
            const uint32_t block  = eighth << 3;
            const uint32_t nTiles = (N / block) * ((eighth + DIT_TILE_HOST - 1) / DIT_TILE_HOST);
            const uint32_t bd     = (nTiles < aivCores) ? nTiles : aivCores;
            fprintf(stderr, "[DIT8-PLAN]   stage eighth=%-6u tiles=%-5u blockDim=%-3u "
                            "tiles per core=%u\n",
                    eighth, nTiles, bd, (nTiles + bd - 1) / bd);
        }
        fflush(stderr);
    }

    auto run_one = [&](bool verbose) -> RunResult {
        RunResult res{0, 0, 0, numStages, cpu_stages};
        memcpy(xrHost, xr0.data(), N * sizeof(float));
        memcpy(xiHost, xi0.data(), N * sizeof(float));
        if (cpu_stages > 0) {
            auto t0 = std::chrono::high_resolution_clock::now();
            for (uint32_t s = 1; s <= cpu_stages; ++s) host_butterfly8_stage_split(xrHost, xiHost, N, 1u << (3*(s-1)));
            auto t1 = std::chrono::high_resolution_clock::now();
            res.cpu_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
        }
        CHECK_ACL2(aclrtMemcpy(xrDev, N * sizeof(float), xrHost, N * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE));
        CHECK_ACL2(aclrtMemcpy(xiDev, N * sizeof(float), xiHost, N * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE));

        auto t0 = std::chrono::high_resolution_clock::now();
        // Stages go as launches of their own; inside a stage the tiles are
        // handed out across the AIV cores (see Process() in
        // kernels/r8_dit_fused_xs.cpp). One sync for the lot: the order
        // between stages is guaranteed by the stream.
        for (uint32_t st = 0; st < numStages; ++st) {
            const uint32_t eighth = stages[st].eighth;
            const uint32_t block  = eighth << 3;
            const uint32_t nTiles = (N / block) * ((eighth + DIT_TILE_HOST - 1) / DIT_TILE_HOST);
            const uint32_t bd     = (nTiles < aivCores) ? nTiles : aivCores;
            // Slot 0 is "stage": dit has one launch per stage, not three, so
            // the layout degenerates into a single line. It is needed for the
            // comparison: it shows what ONE Vector launch doing all the work
            // of a stage costs, against the three launches of a group on the
            // Cube path.
            profiled_launch(0, stream, bd, [&] {
                fft8_dit_fused_xs_do(bd, stream, xrDev, xiDev, wDev, metaDev,
                                     N, numStages, st, bd);
            });
        }
        if (numStages > 0) {
            CHECK_ACL2(aclrtSynchronizeStream(stream));
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        res.npu_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
        res.total_us = res.npu_us + res.cpu_us;
        return res;
    };
    profile_reset();
    BenchOutcome __out = run_benchmark(warmup, bench_runs, run_one);
    {
        static const char *const names[3] = {"stage", "-", "-"};
        print_profile(names);
    }
    append_csv_row(variant_name, N, g_TILE_SIZE, __out);
    CHECK_ACL2(aclrtMemcpy(xrHost, N * sizeof(float), xrDev, N * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK_ACL2(aclrtMemcpy(xiHost, N * sizeof(float), xiDev, N * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST));
    for (uint32_t i = 0; i < N; ++i) { output[2*i] = xrHost[i]; output[2*i+1] = xiHost[i]; }
    aclrtFree(xrDev); aclrtFree(xiDev); aclrtFree(wDev); aclrtFree(metaDev);
    aclrtFreeHost(xrHost); aclrtFreeHost(xiHost); aclrtDestroyStream(stream);
#endif
    WriteFile("./output/output_x.bin", output.data(), 2 * N * sizeof(float));
}

#ifndef ASCENDC_CPU_DEBUG
// ============================================================================
// cube1_fs_xs AND cube2_fs_xs — the real NPU path, ONE function for both.
// Repack/twiddle are the same in format as in dit_fs_xs (build_fused_plan8 +
// fill_fused_buffers8_packed, the same W buffer); combine is a Matmul through
// kernels/r8_cube_fused_xs.cpp (cube1) or kernels/r8_cube2_matmul.cpp (cube2).
// ============================================================================
// terms — the number of terms of the precision-recovery scheme (1, 2 or 3);
// it is spelled out in detail in kernels/r8_cube_fused_xs.cpp. The variants
// cube1_fs_xs / cube1x2_fs_xs / cube1x3_fs_xs differ in THIS ARGUMENT ONLY,
// so that the "precision against time" comparison lands in the CSV as
// separate rows.
//
// dual — THE CHOICE OF COMBINE SCHEME, and that is the second independent
// axis:
//   false (cube1): one 16x16 matmul with the block-real M16, the ± signs
//                  built into the matrix, unpack only sums the precision
//                  terms;
//   true  (cube2): TWO matmuls with A2 = [Re(M); Im(M)], the Re/Im combine
//                  step on Vector in unpack.
// Why one function and not two: the whole surrounding machinery (the stage
// plan, the cube twiddles, the tiling, the tile enumeration, the batching
// over k-blocks, the CPU cleanup, the timing) is code already proven on
// hardware, and a copy of it would diverge from the original at the very
// first edit. There are only four differences, and they are marked below as
// "cube1/cube2 DIFFERENCE": the weight matrix, the sizes of two scratch
// buffers and a pair of launches.
// Repack itself is SHARED and does not branch at all — see
// kernels/r8_cube2_matmul.cpp.
static void run_cube_fs_xs(uint32_t N, uint32_t m3, const std::vector<float> &input,
                            uint32_t warmup, uint32_t bench_runs, const std::string &variant_name,
                            uint32_t terms, bool dual)
{
    std::vector<FusedStageInfo> stages;
    uint32_t w_total = 0, cpu_stages = 0;
    build_fused_plan8(m3, g_TILE_SIZE, stages, w_total, cpu_stages);
    // FIX: build_fused_plan8() clamps min_npu_eighth from below at 8, while
    // the first radix-8 stage of ANY N=8^m3 (m3>=1) has eighth=1 (8^0) — it
    // is ALWAYS < 8, at any -t. So cpu_stages > 0 here is not "TILE_SIZE is
    // too large" (as the old error message claimed) but the normal case,
    // which simply had not been implemented in the cube1 path (even though
    // host_butterfly8_stage_split() is already there for dit_fs_xs below).
    // We implement exactly the same CPU cleanup here — stages s=1..cpu_stages
    // are computed on the CPU (host_butterfly8_stage_split), and the NPU/Cube
    // gets only the remaining stages[] (eighth >= TILE_SIZE), as before.
    uint32_t numStages = (uint32_t)stages.size();

    // The Vector twiddle buffer (fill_fused_buffers8_packed) is NOT used on
    // the cube path: the layout for batching over k-blocks is a different
    // one, see build_cube_twiddle_plan8 in host/fft_host_r8.h. The format of
    // the vector buffer is shared with dit_fs_xs, so we leave it alone — we
    // simply do not read it here.
    std::vector<CubeTwiddlePlan> cubePlan;
    uint32_t wc_total = 0;
    build_cube_twiddle_plan8(stages, CUBE_BATCH_HOST, cubePlan, wc_total);
    std::vector<float> wc(wc_total);
    fill_cube_twiddle8(wc.data(), stages, cubePlan);

    std::vector<uint32_t> meta(numStages * 3);
    for (size_t i = 0; i < stages.size(); ++i) {
        meta[i*3+0] = stages[i].eighth; meta[i*3+1] = stages[i].aligned; meta[i*3+2] = stages[i].wOffset;
    }
    // cube1/cube2 DIFFERENCE (1 of 4): the weight matrix.
    //   cube1 — M16 = [[Re(M), -Im(M)], [Im(M), Re(M)]], ± signs inside;
    //   cube2 — A2  = [[Re(M), 0],      [Im(M), 0]],     combine in unpack.
    // Both are 16x16, so EVERYTHING else — the tiling, the size of the W
    // buffer, the hi/lo split into half — is shared by them and does not
    // branch below. Why A2 is stored 16x16 with zeros instead of 16x8 is in
    // the header of kernels/r8_cube2_matmul.cpp.
    float WMAT[16][16];
    if (dual) build_dft8_matrix16_dual(WMAT);
    else      build_dft8_matrix16_blockreal(WMAT);

    std::vector<uint8_t> tilingBuf;
    // The N dimension of the matmul is exactly CUBE_BATCH (the third literal
    // 128, which once lived here separately from the other two; now there is
    // just the one constant).
    if (!BuildFixedWeightCubeTiling(SOC_VERSION_STR, 16, 16, CUBE_BATCH_HOST, tilingBuf)) {
        fprintf(stderr, "[ERROR] %s: BuildFixedWeightCubeTiling() failed "
                        "(GetTiling() != 0) — check the parameters in host/cube_tiling.h.\n",
                variant_name.c_str());
        return;
    }

    {
        auto *dbgT = reinterpret_cast<const int32_t *>(tilingBuf.data());
        fprintf(stderr, "[TILING-DEBUG] usedCoreNum=%d M=%d N=%d Ka=%d Kb=%d "
                "singleCoreM=%d singleCoreN=%d singleCoreK=%d "
                "baseM=%d baseN=%d baseK=%d depthA1=%d depthB1=%d "
                "stepM=%d stepN=%d\n",
                dbgT[0], dbgT[1], dbgT[2], dbgT[3], dbgT[4],
                dbgT[5], dbgT[6], dbgT[7],
                dbgT[8], dbgT[9], dbgT[10], dbgT[11], dbgT[12],
                dbgT[13], dbgT[14]);
        fflush(stderr);
    }

    std::vector<float> xr0(N), xi0(N);
    for (uint32_t i = 0; i < N; ++i) { xr0[i] = input[2*i]; xi0[i] = input[2*i+1]; }
    digit_reverse_permute8_split(xr0.data(), xi0.data(), N);
    std::vector<float> output(2 * N);

    float *xrHost, *xiHost;
    CHECK_ACL2(aclrtMallocHost((void **)&xrHost, N * sizeof(float)));
    CHECK_ACL2(aclrtMallocHost((void **)&xiHost, N * sizeof(float)));
    void *xrDev, *xiDev, *wcDev, *w16Dev, *tilingDev, *scratchXDev, *scratchYDev, *workspaceDev;
    CHECK_ACL2(aclrtMalloc(&xrDev, N * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL2(aclrtMalloc(&xiDev, N * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL2(aclrtMalloc(&wcDev, wc_total * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST));
    // W16: two 16x16 half blocks — A_hi and A_lo (A_lo needed at terms>=3).
    CHECK_ACL2(aclrtMalloc(&w16Dev, 2 * 16 * 16 * sizeof(uint16_t), ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL2(aclrtMalloc(&tilingDev, tilingBuf.size(), ACL_MEM_MALLOC_HUGE_FIRST));
    // X: two blocks (X_hi, X_lo); Y: three blocks (Y0, Y1, Y2) — see terms.
    // And all of it ONE SLOT PER TILE: one launch of a stage computes a group
    // of tiles, one per AIC block, and each has a slot of its own — otherwise
    // the blocks would overwrite each other's scratch. There are cubeMaxTiles
    // slots (the number of AIC cores). In memory that costs next to nothing:
    // 8 KB per slot for X and 24 KB for Y.
    const uint32_t cubeMaxTiles = GetAicCoreNum(SOC_VERSION_STR);
    // cube1/cube2 DIFFERENCE (2 and 3 of 4): the sizes of two scratch buffers.
    //
    // scratchX: in cube2 the B operand of the last matmul (Q1 = A_hi*Im_lo)
    // starts at row 24 of the slot and 16 rows are read, that is, up to row
    // 40 while the slot is 32 rows. For the LAST slot that is a read past the
    // end of the buffer — i.e. an aicore exception. We add 8 rows
    // (8*CUBE_BATCH half) of padding at the tail; they do not affect the
    // result, because columns 8..15 of the matrix A2 are zeros. The
    // walkthrough is in the header of kernels/r8_cube2_matmul.cpp.
    //
    // scratchY: cube1 has three blocks of terms (Y0,Y1,Y2), cube2 has SIX
    // (P0,Q0,P1,Q1,P2,Q2), two per term. In memory that costs next to nothing
    // (48 KB per slot instead of 24 at B=128), and in exchange unpack reads
    // exactly as much as the matmul wrote.
    const size_t xPadHalf  = dual ? (size_t)8 * CUBE_BATCH_HOST : 0;
    const size_t yBlocks   = dual ? 6 : 3;
    CHECK_ACL2(aclrtMalloc(&scratchXDev,
        ((size_t)cubeMaxTiles * 2 * 16 * CUBE_BATCH_HOST + xPadHalf) * sizeof(uint16_t),
        ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL2(aclrtMalloc(&scratchYDev,
        (size_t)cubeMaxTiles * yBlocks * 16 * CUBE_BATCH_HOST * sizeof(float),
        ACL_MEM_MALLOC_HUGE_FIRST));
    // The system workspace for Matmul. Mandatory, and it decides whether the
    // thing runs at all: the auto-wrapper of the MIX kernel (the cube library
    // is built with -DHAVE_WORKSPACE/-DHAVE_TILING, see
    // common/cmake/npu_lib.cmake) turns it into
    // SetSysWorkspaceForce + matmul::clearWorkspace() — the latter zero-fills
    // the message-exchange area between the AIC<->AIV halves of the matmul.
    // It was precisely the absence of that call (the defines were off) that
    // led to the eternal hardware deadlock on the first run of
    // fft8_cube1_matmul. The size is exactly as in the official
    // matmul_custom/main.cpp: GetLibApiWorkSpaceSize().
    uint64_t workspaceSize = GetCubeSystemWorkspaceSize(SOC_VERSION_STR);
    CHECK_ACL2(aclrtMalloc(&workspaceDev, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    aclrtStream stream; CHECK_ACL2(aclrtCreateStream(&stream));

    CHECK_ACL2(aclrtMemcpy(wcDev, wc_total * sizeof(float), wc.data(), wc_total * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE));
    // A = A_hi + A_lo, both half: A_hi is the ordinary rounding of M16 into
    // half, A_lo is what was lost in doing so. A constant, computed once.
    std::vector<uint16_t> M16Half(2 * 16 * 16);
    for (int ii = 0; ii < 16 * 16; ++ii) {
        const float    v  = (&WMAT[0][0])[ii];
        const uint16_t hi = FloatToHalfBits(v);
        M16Half[ii]            = hi;
        M16Half[16 * 16 + ii]  = FloatToHalfBits(v - HalfBitsToFloat(hi));
    }
    CHECK_ACL2(aclrtMemcpy(w16Dev, M16Half.size()*sizeof(uint16_t), M16Half.data(),
                           M16Half.size()*sizeof(uint16_t), ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL2(aclrtMemcpy(tilingDev, tilingBuf.size(), tilingBuf.data(), tilingBuf.size(), ACL_MEMCPY_HOST_TO_DEVICE));

    // !!! THE ZERO-FILL OF scratchX IS NOT HYGIENE, IT IS CORRECTNESS !!!
    // In cube2 the B operands of the matmuls are SLICES of the repack buffer
    // at offsets of 8 and 24 rows, and 16 rows are always read (K=16). So the
    // matmul inevitably takes in rows that repack did NOT write in this run:
    //   * at terms=1 Q0 reads rows 8..23, and 16..23 are X_lo, which is
    //     written only at terms>=2;
    //   * Q1 reads rows 24..39, that is, 8 rows beyond the slot (which is
    //     exactly what the host allocates the tail padding for, see above).
    // Columns 8..15 of the matrix A2 are zeros, so MATHEMATICALLY the
    // contribution of those rows is zero. But after aclrtMalloc there is
    // arbitrary garbage lying there, and if its bits add up to NaN or Inf
    // (for half that is merely an exponent of all ones — on random bytes the
    // probability is ~1/256 per element, that is, at least one per buffer
    // practically guaranteed), then 0 * NaN = NaN, and the NaN leaks into the
    // accumulator and on into the output. That would be a rare and very
    // unpleasant bug: it depends on what the previous task left in DDR, that
    // is, it would reproduce every other time.
    // One memcpy of zeros for the whole run (tens of kilobytes, outside the
    // timing) makes garbage impossible: the rows repack never touches are
    // zeros for good.
    // It is done for cube1 as well — it does not need it, but it costs
    // nothing, and the invariant "there is no garbage in scratchX" is better
    // kept shared.
    {
        const size_t xHalfCount = (size_t)cubeMaxTiles * 2 * 16 * CUBE_BATCH_HOST + xPadHalf;
        std::vector<uint16_t> zeros(xHalfCount, 0);
        CHECK_ACL2(aclrtMemcpy(scratchXDev, xHalfCount * sizeof(uint16_t),
                               zeros.data(), xHalfCount * sizeof(uint16_t),
                               ACL_MEMCPY_HOST_TO_DEVICE));
    }

    // How many Cube tiles the current plan gives and how many launches they
    // fit into — printed once, so that the report shows what the time runs up
    // against: the arithmetic or the overhead of launching the kernels.
    {
        uint32_t tiles = 0, groups = 0;
        for (const auto &st : stages) {
            uint32_t blk = st.eighth << 3, nb = N / blk, nt;
            if (st.eighth <= CUBE_BATCH_HOST) {
                uint32_t gMax = CUBE_BATCH_HOST / st.eighth;
                nt = nb / gMax;
                groups += (nt + cubeMaxTiles - 1) / cubeMaxTiles;
                if (nb % gMax) { ++nt; ++groups; }      // partial remainder — a group of its own
            } else {
                nt = nb * (st.eighth / CUBE_BATCH_HOST);
                groups += (nt + cubeMaxTiles - 1) / cubeMaxTiles;
            }
            tiles += nt;
        }
        // The total scratch footprint, printed because it — and not the batch
        // width, and not the choice of scheme — is what predicts whether this
        // configuration runs at a sane speed. Measured at N=262144, tile=8:
        //
        //    320 KB  cube1 B=64    16286 us
        //    561 KB  cube2 B=64    27902 us
        //    640 KB  cube1 B=128   11437 us     <- fastest
        //   1122 KB  cube2 B=128  272238 us     <- collapse
        //   1280 KB  cube1 B=256  486086 us     <- collapse
        //   2244 KB  cube2 B=256   >600000 us (timed out)
        //
        // Sorted by footprint the three smallest are healthy and the three
        // largest collapse, with the boundary between 640 KB and 1122 KB. The
        // scheme does not predict it; the byte count does. The penalty is paid
        // once per GROUP, so it only becomes visible when there are many of
        // them — at N=4096 (3 groups) a large batch is still mildly better,
        // at N=262144 (64 groups) it is 40x worse. The working hypothesis is
        // that the scratch has to stay resident between consecutive groups and
        // stops doing so above ~1 MB; see INSIGHTS.md section 4.1 for what is
        // and is not established about the mechanism.
        const size_t scratchBytes =
            (size_t)cubeMaxTiles * yBlocks * 16 * CUBE_BATCH_HOST * sizeof(float)
          + ((size_t)cubeMaxTiles * 2 * 16 * CUBE_BATCH_HOST + xPadHalf) * sizeof(uint16_t);
        fprintf(stderr, "[CUBE-PLAN] scheme=%s, AIC cores=%u, tiles=%u, groups=%u, "
                        "launches=%u (repack+matmul+unpack per group), batch=%u columns, "
                        "matmuls per tile=%u (terms=%u x %u matmul), scratch=%zu KB\n",
                dual ? "cube2 (2 matmuls)" : "cube1 (1 matmul)",
                cubeMaxTiles, tiles, groups, groups * 3, CUBE_BATCH_HOST,
                terms * (dual ? 2u : 1u), terms, dual ? 2u : 1u,
                scratchBytes / 1024);
        if (scratchBytes > 700u * 1024u && groups > 8u) {
            fprintf(stderr, "[CUBE-PLAN] WARNING: scratch %zu KB is above the measured "
                            "healthy range (<= 640 KB) and there are %u groups — expect a "
                            "10-40x slowdown. This is the footprint cliff, not the scheme: "
                            "use a smaller CUBE_BATCH.\n",
                    scratchBytes / 1024, groups);
        }
        fflush(stderr);
    }

    // The blockDim of the Cube kernels is NO LONGER derived from
    // tiling.usedCoreNum: by the reference's formula it used to come out as
    // one, and the whole pipeline ran on a single core. Now blockDim = the
    // number of tiles in the group, while the tiling describes exactly one
    // tile (16x16x128), which every block computes whole — see the comment
    // about the absence of a usedCoreNum guard in
    // kernels/r8_cube_fused_xs.cpp.
    //
    // One group of tiles = three launches (repack on AIV, matmul on AIC,
    // unpack on AIV). blockDim is the same for all three and equals the
    // number of tiles in the group, so the scratch slot (= blockIdx) of all
    // three points at one and the same tile.
    auto launchCubeGroup = [&](uint32_t tileBase, uint32_t kStride, uint32_t shiftStride,
                               uint32_t shiftsPerBlk, uint32_t eighth,
                               uint32_t gCount, uint32_t perBlock,
                               uint32_t wcOff, uint32_t wcRow, uint32_t cnt) {
        // With RADIX8_PROFILE=1 profiled_launch synchronizes after each of
        // the three calls and accumulates the time by type (see LaunchProfile
        // above); without the environment variable these are exactly the same
        // three calls as before.
        profiled_launch(0, stream, cnt, [&] {
            fft8_cube1_repack_do(cnt, stream, xrDev, xiDev, wcDev, scratchXDev,
                                 N, tileBase, kStride, shiftStride, shiftsPerBlk, eighth,
                                 gCount, perBlock, wcOff, wcRow, wc_total, terms);
        });
        // cube1/cube2 DIFFERENCE (4 of 4): the pair of launches. The repack
        // above is SHARED — it already writes [Re(t); Im(t)] into rows 0..15,
        // which suits both a single 16x16 matmul and two matmuls over slices
        // of those rows. The number of launches per group is the same (3):
        // the two matmuls of cube2 go inside ONE kernel, through one Cube
        // object. That matters for the honesty of the measurement —
        // otherwise what would be compared is not the schemes but the
        // overhead of an extra launch.
        profiled_launch(1, stream, cnt, [&] {
            if (dual) {
                ACLRT_LAUNCH_KERNEL(fft8_cube2_matmul)(cnt, stream,
                                     w16Dev, scratchXDev, scratchYDev, terms,
                                     workspaceDev, tilingDev);
            } else {
                ACLRT_LAUNCH_KERNEL(fft8_cube1_matmul)(cnt, stream,
                                     w16Dev, scratchXDev, scratchYDev, terms,
                                     workspaceDev, tilingDev);
            }
        });
        profiled_launch(2, stream, cnt, [&] {
            if (dual) {
                fft8_cube2_unpack_do(cnt, stream, xrDev, xiDev, scratchYDev,
                                     N, tileBase, kStride, shiftStride, shiftsPerBlk, eighth,
                                     gCount, perBlock, terms);
            } else {
                fft8_cube1_unpack_do(cnt, stream, xrDev, xiDev, scratchYDev,
                                     N, tileBase, kStride, shiftStride, shiftsPerBlk, eighth,
                                     gCount, perBlock, terms);
            }
        });
    };

    auto run_one = [&](bool verbose) -> RunResult {
        RunResult res{0, 0, 0, numStages, cpu_stages};
        memcpy(xrHost, xr0.data(), N * sizeof(float));
        memcpy(xiHost, xi0.data(), N * sizeof(float));

        if (cpu_stages > 0) {
            auto tc0 = std::chrono::high_resolution_clock::now();
            for (uint32_t s = 1; s <= cpu_stages; ++s) host_butterfly8_stage_split(xrHost, xiHost, N, 1u << (3*(s-1)));
            auto tc1 = std::chrono::high_resolution_clock::now();
            res.cpu_us = std::chrono::duration<double, std::micro>(tc1 - tc0).count();
        }

        CHECK_ACL2(aclrtMemcpy(xrDev, N * sizeof(float), xrHost, N * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE));
        CHECK_ACL2(aclrtMemcpy(xiDev, N * sizeof(float), xiHost, N * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE));

        // The loop over (stage s, k, shift) — it used to be on the device
        // inside a single "mix" AIC/AIV kernel (see the fix in
        // kernels/r8_cube_fused_xs.cpp about the AI Core hang); now it is on
        // the host, and every tile is three separate launches (repack on AIV,
        // matmul on AIC, unpack on AIV) on one stream. The order between the
        // launches is guaranteed by stream semantics — without any manual
        // cross-core synchronization inside a single kernel.
        // IMPORTANT about scratchX/scratchY: all three launches work on ONE
        // pair of scratch buffers, so the tiles CANNOT be pipelined — the
        // repack of the next tile would overwrite the scratchX that the
        // matmul of the current one is still reading. Correctness here rests
        // on the launches going down one stream, and a stream executing tasks
        // strictly in order: the unpack of tile i finishes before the repack
        // of tile i+1. So intermediate aclrtSynchronizeStream calls are not
        // needed (they were added only to pin down the hang, and slowed the
        // run noticeably) — one sync after the whole loop, before the timer
        // is stopped, is enough.
        auto t0 = std::chrono::high_resolution_clock::now();
        for (size_t si = 0; si < stages.size(); ++si) {
            const uint32_t eighth = stages[si].eighth;
            const uint32_t block  = eighth << 3;          // positions in one k-block
            const uint32_t nBlk   = N / block;            // k-blocks in the stage
            const uint32_t wcOff  = cubePlan[si].offset;
            const uint32_t wcRow  = cubePlan[si].rowLen;

            // The batch of a Cube call is CUBE_BATCH independent butterflies.
            // Inside one k-block there are only `eighth` of them, so at
            // eighth < CUBE_BATCH we top the batch up with NEIGHBOURING
            // k-BLOCKS (gMax blocks of eighth butterflies each), while at
            // eighth >= CUBE_BATCH we do the opposite and cut one block into
            // windows over shift.
            //
            // A stage's tiles go in GROUPS of cubeMaxTiles: in one launch
            // every block computes its own tile in its own scratch slot.
            // There used to be three launches per tile at blockDim=1, and at
            // ~8 us per launch practically the whole run time went into them.
            //
            // The tile-enumeration rule (the kernel reproduces the same one
            // from its own blockIdx, see Process() in
            // kernels/r8_cube1_repack.cpp):
            //   tileIdx -> k0     = (tileIdx / shiftsPerBlk) * kStride
            //              shift0 = (tileIdx % shiftsPerBlk) * shiftStride
            // Within one group all the tiles have THE SAME gCount and
            // perBlock — that is mandatory, because they are passed as
            // scalars for the whole launch. So the partial remainder over
            // k-blocks (gCount < gMax) goes off as a group of its own holding
            // a single tile.
            if (eighth <= CUBE_BATCH_HOST) {
                const uint32_t gMax    = CUBE_BATCH_HOST / eighth;
                const uint32_t fullT   = nBlk / gMax;      // tiles with a full gMax
                const uint32_t remBlk  = nBlk % gMax;      // remainder of k-blocks
                const uint32_t kStride = gMax * block;

                for (uint32_t t0 = 0; t0 < fullT; t0 += cubeMaxTiles) {
                    const uint32_t cnt = (fullT - t0 < cubeMaxTiles) ? (fullT - t0) : cubeMaxTiles;
                    launchCubeGroup(t0, kStride, 0u, 1u, eighth, gMax, eighth, wcOff, wcRow, cnt);
                }
                if (remBlk != 0) {
                    launchCubeGroup(fullT, kStride, 0u, 1u, eighth, remBlk, eighth, wcOff, wcRow, 1u);
                }
            } else {
                // eighth > CUBE_BATCH: eighth = 8^k at k >= 3, that is
                // 2^(3k) at 3k >= 9, and it is always divisible by
                // CUBE_BATCH = 2^7 — there are no partial windows over shift,
                // and perBlock is the same for every tile.
                const uint32_t chunks = eighth / CUBE_BATCH_HOST;
                const uint32_t nTiles = nBlk * chunks;
                for (uint32_t t0 = 0; t0 < nTiles; t0 += cubeMaxTiles) {
                    const uint32_t cnt = (nTiles - t0 < cubeMaxTiles) ? (nTiles - t0) : cubeMaxTiles;
                    launchCubeGroup(t0, block, CUBE_BATCH_HOST, chunks, eighth,
                                    1u, CUBE_BATCH_HOST, wcOff, wcRow, cnt);
                }
            }
        }
        CHECK_ACL2(aclrtSynchronizeStream(stream));
        auto t1 = std::chrono::high_resolution_clock::now();
        res.npu_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
        res.total_us = res.npu_us + res.cpu_us;
        return res;
    };
    profile_reset();
    BenchOutcome __out = run_benchmark(warmup, bench_runs, run_one);
    {
        static const char *const names[3] = {"repack", "matmul", "unpack"};
        print_profile(names);
    }
    append_csv_row(variant_name, N, g_TILE_SIZE, __out);
    CHECK_ACL2(aclrtMemcpy(xrHost, N * sizeof(float), xrDev, N * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK_ACL2(aclrtMemcpy(xiHost, N * sizeof(float), xiDev, N * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST));
    for (uint32_t i = 0; i < N; ++i) { output[2*i] = xrHost[i]; output[2*i+1] = xiHost[i]; }

    aclrtFree(xrDev); aclrtFree(xiDev); aclrtFree(wcDev); aclrtFree(w16Dev);
    aclrtFree(tilingDev); aclrtFree(scratchXDev); aclrtFree(scratchYDev); aclrtFree(workspaceDev);
    aclrtFreeHost(xrHost); aclrtFreeHost(xiHost); aclrtDestroyStream(stream);

    WriteFile("./output/output_x.bin", output.data(), 2 * N * sizeof(float));
}
#endif

// ============================================================================
// The Cube variants are deliberately not implemented in the CPU-debug build:
// tikicpulib does not simulate the Cube unit. An explicit error is better
// than a silently wrong result.
//
// The #ifdef is mandatory, not "just in case": ever since cube2 stopped being
// a stub, this function is called ONLY from the CPU-debug branch of
// run_cube_terms(). In the npu build it would be left unused, and the project
// is built with -Wall -Werror, so -Wunused-function would bring the build
// down. (Caught only by a full compile: -fsyntax-only does not notice it.)
// ============================================================================
#ifdef ASCENDC_CPU_DEBUG
static void run_cube_stub(const char *name, uint32_t N, uint32_t m3, const char *reason)
{
    std::vector<FusedStageInfo> stages;
    uint32_t w_total = 0, cpu_stages = 0;
    build_fused_plan8(m3, g_TILE_SIZE, stages, w_total, cpu_stages);

    fprintf(stderr, "[ERROR] variant=%s: %s (%u NPU stages, %u CPU stages computed correctly).\n",
            name, reason, (unsigned)stages.size(), cpu_stages);
}
#endif


// ===========================================================================
// Thin wrappers: run_cube_fs_xs has two extra parameters (terms and dual),
// while the registry stores functions of a single signature. Six variants =
// 2 schemes x 3 term counts; that is what gives the analyzer paired
// comparisons along two axes at once — "scheme terms" (at a fixed scheme) and
// "matmuls" (at a fixed number of terms). The npu/cpu-debug branch repeats
// what the main() of the original project used to do.
static void run_cube_terms(uint32_t N, uint32_t m3, const std::vector<float> &input,
                            uint32_t warmup, uint32_t bench_runs,
                            const std::string &variant_name, uint32_t terms, bool dual)
{
#ifndef ASCENDC_CPU_DEBUG
    run_cube_fs_xs(N, m3, input, warmup, bench_runs, variant_name, terms, dual);
#else
    (void)input; (void)warmup; (void)bench_runs; (void)terms; (void)dual;
    run_cube_stub(variant_name.c_str(), N, m3,
        "implemented only for -r npu; CPU-debug (tikicpulib) simulation of the Cube unit is not supported");
#endif
}

static void run_cube1_x1(uint32_t N, uint32_t m3, const std::vector<float> &i,
                          uint32_t w, uint32_t b, const std::string &v)
{ run_cube_terms(N, m3, i, w, b, v, 1u, false); }

static void run_cube1_x2(uint32_t N, uint32_t m3, const std::vector<float> &i,
                          uint32_t w, uint32_t b, const std::string &v)
{ run_cube_terms(N, m3, i, w, b, v, 2u, false); }

static void run_cube1_x3(uint32_t N, uint32_t m3, const std::vector<float> &i,
                          uint32_t w, uint32_t b, const std::string &v)
{ run_cube_terms(N, m3, i, w, b, v, 3u, false); }

static void run_cube2_x1(uint32_t N, uint32_t m3, const std::vector<float> &i,
                          uint32_t w, uint32_t b, const std::string &v)
{ run_cube_terms(N, m3, i, w, b, v, 1u, true); }

static void run_cube2_x2(uint32_t N, uint32_t m3, const std::vector<float> &i,
                          uint32_t w, uint32_t b, const std::string &v)
{ run_cube_terms(N, m3, i, w, b, v, 2u, true); }

static void run_cube2_x3(uint32_t N, uint32_t m3, const std::vector<float> &i,
                          uint32_t w, uint32_t b, const std::string &v)
{ run_cube_terms(N, m3, i, w, b, v, 3u, true); }

// ===========================================================================
// The mm field in the registry is the number of matmuls per tile per term of
// the precision scheme: 1 for cube1, 2 for cube2, 0 for the Vector variants
// (no matmuls at all). That is the "one matmul or two" axis itself, and for
// it to give a PAIRED comparison it has to be a CSV column of its own —
// otherwise cube1x3 and cube2x3 differ only in the variant name, and the
// analyzer will not pair them.
void register_r8(Registry &reg)
{
    reg.add({"r8", "dit_fs_xs", 8, "dit", "fused", "xs", "vector", 0, 0, 40, false, &run_dit_fs_xs, &r8::host_twiddle8_prewarm_N});
    reg.add({"r8", "cube1_fs_xs",   8, "cube", "fused", "xs", "cube", 1, 1, 20, true, &run_cube1_x1, &r8::host_twiddle8_prewarm_N});
    reg.add({"r8", "cube1x2_fs_xs", 8, "cube", "fused", "xs", "cube", 2, 1, 20, true, &run_cube1_x2, &r8::host_twiddle8_prewarm_N});
    reg.add({"r8", "cube1x3_fs_xs", 8, "cube", "fused", "xs", "cube", 3, 1, 20, true, &run_cube1_x3, &r8::host_twiddle8_prewarm_N});
    reg.add({"r8", "cube2_fs_xs",   8, "cube", "fused", "xs", "cube", 1, 2, 20, true, &run_cube2_x1, &r8::host_twiddle8_prewarm_N});
    reg.add({"r8", "cube2x2_fs_xs", 8, "cube", "fused", "xs", "cube", 2, 2, 20, true, &run_cube2_x2, &r8::host_twiddle8_prewarm_N});
    reg.add({"r8", "cube2x3_fs_xs", 8, "cube", "fused", "xs", "cube", 3, 2, 20, true, &run_cube2_x3, &r8::host_twiddle8_prewarm_N});
}
