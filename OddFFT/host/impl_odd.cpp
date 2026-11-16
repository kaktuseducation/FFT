// host/impl_odd.cpp  (OddFFT)
//
// ONE run-function for every odd-radix variant, and one loop that registers
// all of them from kernels/kernel_table.h.
//
// WHY THERE IS NO impl_r3.cpp / impl_r5.cpp / impl_r7.cpp
// The kernels are generated from one template (scripts/gen_kernels.py), so
// they all have the same launch signature and the same tiling. The host side
// therefore has nothing per-radix to say: the radix is a number in the stage
// plan. Three files would be three copies of this one function, and three
// copies drift -- which would quietly turn "these variants differ only in the
// butterfly" from a fact into a hope.
//
// Each table entry becomes TWO registered variants, differing in exactly one
// thing, the core count:
//     <algo>_ps_xs   blockDim = 1
//     <algo>_psm_xs  blockDim = min(tiles, AIV cores)
// Same body, so the `cores` axis is a comparison of core count and of nothing
// else.
#include "plan.h"
#include "butterflies.h"
#include "host_fft.h"
#include "host_twiddles.h"
#include "host_prefix_fast.h"
#include "frankenstein.h"
#include "bench_harness.h"
#include "registry.h"
#include "../kernels/oddfft_tiling.h"
#include "../common/data_utils.h"
#include "../kernels/kernel_limits.h"
#include "../kernels/kernel_table.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <deque>
#include <chrono>

#ifndef ASCENDC_CPU_DEBUG
#include "acl/acl.h"
#include "platform_cores.h"
#define CHECK_ACL2(x) do { \
    auto _rc = (x); \
    if (_rc != ACL_SUCCESS) fprintf(stderr, "[ERROR] ACL call failed: %s (rc=%d) at %s:%d\n", \
                                     #x, (int)_rc, __FILE__, __LINE__); \
} while (0)
#endif

using namespace oddfft;

// The registry hands a run-function no information about WHICH variant it is,
// so the dispatch has to be recoverable from the variant name. These are set
// by main() through the registry's prewarm/run pair; rather than thread an
// extra argument through RunFn (which would touch every project), the
// registration below creates one static thunk per table entry and each thunk
// knows its own entry index. g_cell carries everything else.
static uint32_t g_entry = 0;

static std::vector<uint32_t> radix_list(uint32_t N, uint32_t radix)
{
    std::vector<uint32_t> out;
    for (uint32_t v = N; v > 1; v /= radix) out.push_back(radix);
    return out;
}

// The twiddle-table warmup the harness runs BEFORE the clock starts, so the
// trigonometry is outside the timing on the host side exactly as it is on the
// device side. Returns the number of host stages, which the banner prints.
static uint32_t odd_prewarm(uint32_t N, uint32_t tile)
{
    const uint32_t radix = ODD_KERNELS[g_entry].radix;
    Plan p;
    std::string err;
    (void)tile;   // the plan depends on MIN_T, not on the tile width
    if (!build_plan(N, radix_list(N, radix), g_MIN_T, p, &err)) return 0;
    build_host_twiddles(p);
    return p.cpu_first;
}

static void run_odd(uint32_t N, uint32_t m, const std::vector<float> &input,
                    uint32_t warmup, uint32_t bench_runs,
                    const std::string &variant_name, bool multicore)
{
    (void)m;
    const OddKernelEntry &ent = ODD_KERNELS[g_entry];

    Plan p;
    std::string err;
    if (!build_plan(N, radix_list(N, ent.radix), g_MIN_T, p, &err)) {
        fprintf(stderr, "[ERROR ] build_plan failed: %s\n", err.c_str());
        return;   // no output_x.bin -> main() reports NO_OUTPUT, exit code 2
    }
    g_cell.cpu_first = p.cpu_first;

    // The EFFECTIVE tile, clamped by the same macro the kernel clamps with
    // (kernels/kernel_limits.h). This is what goes into the matrix: recording
    // the requested tile while the kernel used a smaller one would make the
    // tile axis compare cells that are not what the table says they are.
    const uint32_t tile = ODDFFT_CLAMP_TILE(g_TILE_SIZE);
    g_cell.tile = tile;

    std::vector<float> w(p.w_total ? p.w_total : 1u, 0.0f);
    fill_twiddles(w.data(), p);
    build_host_twiddles(p);

    // The digit-reverse permutation is part of the DIT algorithm, not part of
    // what is timed per run, so it is done once. NOTE it is a GATHER; see
    // host/plan.h for why that distinction is the whole of mixed radix.
    std::vector<float> xr0(N), xi0(N);
    for (uint32_t i = 0; i < N; ++i) { xr0[i] = input[2*i]; xi0[i] = input[2*i+1]; }
    digit_reverse_permute(xr0.data(), xi0.data(), p);

    std::vector<float> output(2 * (size_t)N);
    const uint32_t npuStages = (uint32_t)p.stages.size() - p.cpu_first;

#ifdef ASCENDC_CPU_DEBUG
    (void)multicore;   // there are no cores to hand tiles out to without a device
    // No device: the host does every stage. This exercises the plan, the
    // twiddles and the permutation; it does NOT exercise the kernels, and the
    // butterfly axis is meaningless here because every variant takes the same
    // host path. That is why the generated kernels have their own test
    // (tests/gen_butterfly_test.cpp), which runs the emitted bodies.
    auto run_one = [&](bool) -> RunResult {
        RunResult res{0, 0, 0, npuStages, p.cpu_first};
        std::vector<float> xr(xr0), xi(xi0);
        for (size_t s = 0; s < p.stages.size(); ++s) host_stage_tab(p, s, xr.data(), xi.data());
        for (uint32_t i = 0; i < N; ++i) { output[2*i] = xr[i]; output[2*i+1] = xi[i]; }
        return res;
    };
    BenchOutcome __out = run_benchmark(warmup, bench_runs, run_one);
    append_csv_row(variant_name, N, tile, __out);
#else
    float *xrHost, *xiHost;
    CHECK_ACL2(aclrtMallocHost((void **)&xrHost, N * sizeof(float)));
    CHECK_ACL2(aclrtMallocHost((void **)&xiHost, N * sizeof(float)));
    void *xrDev, *xiDev, *wDev;
    CHECK_ACL2(aclrtMalloc(&xrDev, N * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL2(aclrtMalloc(&xiDev, N * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL2(aclrtMalloc(&wDev, (size_t)w.size() * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST));
    aclrtStream stream;
    CHECK_ACL2(aclrtCreateStream(&stream));

    const uint32_t aivCores = multicore ? GetAivCoreNum(SOC_VERSION_STR) : 1u;
    CHECK_ACL2(aclrtMemcpy(wDev, (size_t)w.size() * sizeof(float),
                           w.data(), (size_t)w.size() * sizeof(float),
                           ACL_MEMCPY_HOST_TO_DEVICE));

    auto run_one = [&](bool) -> RunResult {
        RunResult res{0, 0, 0, npuStages, p.cpu_first};
        memcpy(xrHost, xr0.data(), N * sizeof(float));
        memcpy(xiHost, xi0.data(), N * sizeof(float));

        // --- the sub-floor prefix on the host, from the precomputed table ---
        if (p.cpu_first > 0) {
            auto c0 = std::chrono::high_resolution_clock::now();
            host_prefix_fast(p, xrHost, xiHost);
            auto c1 = std::chrono::high_resolution_clock::now();
            res.cpu_us = std::chrono::duration<double, std::micro>(c1 - c0).count();
        }
        CHECK_ACL2(aclrtMemcpy(xrDev, N * sizeof(float), xrHost, N * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE));
        CHECK_ACL2(aclrtMemcpy(xiDev, N * sizeof(float), xiHost, N * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE));

        // --- the rest on the device, one launch per stage. The inter-stage
        // barrier is free from the ordering of the launches on the stream.
        auto t0 = std::chrono::high_resolution_clock::now();
        for (const StageInfo &st : p.stages) {
            if (!st.on_npu) continue;
            const uint32_t rowLen = ((st.t + 7u) / 8u) * 8u;
            // The tile count MUST come from the same macro the kernel uses, or
            // the blocks would be handed the wrong tiles.
            const uint32_t nShift = ODDFFT_SHIFT_COUNT(st.t, tile);
            const uint32_t nTiles = st.nBlk * nShift;
            const uint32_t bd = multicore ? ((nTiles < aivCores) ? nTiles : aivCores) : 1u;
            // blk_num must equal blockDim: the kernel uses it as the stride
            // with which a block walks the tile list.
            ent.launch(bd, stream, xrDev, xiDev, wDev,
                       N, st.t, st.block, st.wOffset, rowLen, tile, bd);
            CHECK_ACL2(aclrtSynchronizeStream(stream));
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        res.npu_us   = std::chrono::duration<double, std::micro>(t1 - t0).count();
        res.total_us = res.npu_us + res.cpu_us;
        return res;
    };
    BenchOutcome __out = run_benchmark(warmup, bench_runs, run_one);
    append_csv_row(variant_name, N, tile, __out);

    CHECK_ACL2(aclrtMemcpy(xrHost, N * sizeof(float), xrDev, N * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK_ACL2(aclrtMemcpy(xiHost, N * sizeof(float), xiDev, N * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST));
    for (uint32_t i = 0; i < N; ++i) { output[2*i] = xrHost[i]; output[2*i+1] = xiHost[i]; }

    aclrtFree(xrDev); aclrtFree(xiDev); aclrtFree(wDev);
    aclrtFreeHost(xrHost); aclrtFreeHost(xiHost);
    aclrtDestroyStream(stream);
#endif
    WriteFile("./output/output_x.bin", output.data(), 2 * (size_t)N * sizeof(float));
}



// ===========================================================================
// THE COMBINED TRANSFORM ("Frankenstein")
// ===========================================================================
// A mixed-radix plan in which each stage runs on whichever generated kernel
// won its choice. See host/frankenstein.h for the three choices it makes and
// for the cost model, whose coefficients are NOT yet fitted to this hardware.
//
// Four variants, laid out so the two things worth measuring are paired axes:
//   mix/frank_hi_ps_xs    large-radix-first, 1 core
//   mix/frank_hi_psm_xs   large-radix-first, 40 cores
//   mix/frank_lo_ps_xs    small-radix-first, 1 core
//   mix/frank_lo_psm_xs   small-radix-first, 40 cores
// hi vs lo differs ONLY in stage order; ps vs psm only in core count.
// ===========================================================================

// Which generated kernel serves (radix, butterfly).
static const OddKernelEntry *find_kernel(uint32_t radix, const std::string &bf)
{
    for (uint32_t i = 0; i < ODD_KERNEL_COUNT; ++i) {
        if (ODD_KERNELS[i].radix == radix && bf == ODD_KERNELS[i].butterfly)
            return &ODD_KERNELS[i];
    }
    return nullptr;
}

static uint32_t frank_prewarm(uint32_t N, uint32_t tile)
{
    // The coefficients depend on the core count, so take the fit that matches
    // the variant being run. g_cell.cores is set by main() before the banner,
    // so the prewarm and the run agree on the plan; planning a 1-core variant
    // with the 40-core fit would compare a plan against a cost it was not
    // planned under.
    CostModel cm = cost_model_measured(g_cell.cores);
    FrankPlan fp;
    std::string err;
    (void)tile;   // the plan depends on MIN_T, not on the tile width
    if (!build_frankenstein(N, g_MIN_T, cm, fp, &err, true)) return 0;
    Plan p;
    if (!build_plan(N, fp.radices, g_MIN_T, p, &err)) return 0;
    build_host_twiddles(p);
    return p.cpu_first;
}

static void run_frank(uint32_t N, uint32_t m, const std::vector<float> &input,
                      uint32_t warmup, uint32_t bench_runs,
                      const std::string &variant_name,
                      bool descending, bool multicore)
{
    (void)m;
    // The coefficients depend on the core count, so take the fit that matches
    // the variant being run. g_cell.cores is set by main() before the banner,
    // so the prewarm and the run agree on the plan; planning a 1-core variant
    // with the 40-core fit would compare a plan against a cost it was not
    // planned under.
    CostModel cm = cost_model_measured(g_cell.cores);
    FrankPlan fp;
    std::string err;
    if (!build_frankenstein(N, g_MIN_T, cm, fp, &err, descending)) {
        fprintf(stderr, "[ERROR ] %s\n", err.c_str());
        return;
    }
    Plan p;
    if (!build_plan(N, fp.radices, g_MIN_T, p, &err)) {
        fprintf(stderr, "[ERROR ] build_plan failed: %s\n", err.c_str());
        return;
    }
    g_cell.cpu_first = p.cpu_first;
    print_frankenstein(fp, N, cm);

    // Every DEVICE stage must have a kernel. A host stage does not: it runs on
    // the host butterfly either way. Checked before anything is allocated, so
    // a missing kernel is a clear message rather than a null call.
    for (size_t sIdx = 0; sIdx < p.stages.size(); ++sIdx) {
        if (!p.stages[sIdx].on_npu) continue;
        if (!find_kernel(p.stages[sIdx].radix, fp.butterflies[sIdx])) {
            fprintf(stderr, "[ERROR ] no kernel for radix %u butterfly %s\n",
                    p.stages[sIdx].radix, fp.butterflies[sIdx].c_str());
            return;
        }
    }

    const uint32_t tile = ODDFFT_CLAMP_TILE(g_TILE_SIZE);
    g_cell.tile = tile;

    std::vector<float> w(p.w_total ? p.w_total : 1u, 0.0f);
    fill_twiddles(w.data(), p);
    build_host_twiddles(p);

    std::vector<float> xr0(N), xi0(N);
    for (uint32_t i = 0; i < N; ++i) { xr0[i] = input[2*i]; xi0[i] = input[2*i+1]; }
    digit_reverse_permute(xr0.data(), xi0.data(), p);

    std::vector<float> output(2 * (size_t)N);
    const uint32_t npuStages = (uint32_t)p.stages.size() - p.cpu_first;

#ifdef ASCENDC_CPU_DEBUG
    (void)multicore;
    auto run_one = [&](bool) -> RunResult {
        RunResult res{0, 0, 0, npuStages, p.cpu_first};
        std::vector<float> xr(xr0), xi(xi0);
        for (size_t sIdx = 0; sIdx < p.stages.size(); ++sIdx)
            host_stage_tab(p, sIdx, xr.data(), xi.data());
        for (uint32_t i = 0; i < N; ++i) { output[2*i] = xr[i]; output[2*i+1] = xi[i]; }
        return res;
    };
    BenchOutcome __out = run_benchmark(warmup, bench_runs, run_one);
    append_csv_row(variant_name, N, tile, __out);
#else
    float *xrHost, *xiHost;
    CHECK_ACL2(aclrtMallocHost((void **)&xrHost, N * sizeof(float)));
    CHECK_ACL2(aclrtMallocHost((void **)&xiHost, N * sizeof(float)));
    void *xrDev, *xiDev, *wDev;
    CHECK_ACL2(aclrtMalloc(&xrDev, N * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL2(aclrtMalloc(&xiDev, N * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL2(aclrtMalloc(&wDev, (size_t)w.size() * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST));
    aclrtStream stream;
    CHECK_ACL2(aclrtCreateStream(&stream));

    const uint32_t aivCores = multicore ? GetAivCoreNum(SOC_VERSION_STR) : 1u;
    CHECK_ACL2(aclrtMemcpy(wDev, (size_t)w.size() * sizeof(float),
                           w.data(), (size_t)w.size() * sizeof(float),
                           ACL_MEMCPY_HOST_TO_DEVICE));

    auto run_one = [&](bool) -> RunResult {
        RunResult res{0, 0, 0, npuStages, p.cpu_first};
        memcpy(xrHost, xr0.data(), N * sizeof(float));
        memcpy(xiHost, xi0.data(), N * sizeof(float));

        if (p.cpu_first > 0) {
            auto c0 = std::chrono::high_resolution_clock::now();
            host_prefix_fast(p, xrHost, xiHost);
            auto c1 = std::chrono::high_resolution_clock::now();
            res.cpu_us = std::chrono::duration<double, std::micro>(c1 - c0).count();
        }
        CHECK_ACL2(aclrtMemcpy(xrDev, N * sizeof(float), xrHost, N * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE));
        CHECK_ACL2(aclrtMemcpy(xiDev, N * sizeof(float), xiHost, N * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE));

        auto t0 = std::chrono::high_resolution_clock::now();
        for (size_t sIdx = 0; sIdx < p.stages.size(); ++sIdx) {
            const StageInfo &st = p.stages[sIdx];
            if (!st.on_npu) continue;
            const OddKernelEntry *ke = find_kernel(st.radix, fp.butterflies[sIdx]);
            const uint32_t rowLen = ((st.t + 7u) / 8u) * 8u;
            const uint32_t nShift = ODDFFT_SHIFT_COUNT(st.t, tile);
            const uint32_t nTiles = st.nBlk * nShift;
            const uint32_t bd = multicore ? ((nTiles < aivCores) ? nTiles : aivCores) : 1u;
            ke->launch(bd, stream, xrDev, xiDev, wDev,
                       N, st.t, st.block, st.wOffset, rowLen, tile, bd);
            CHECK_ACL2(aclrtSynchronizeStream(stream));
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        res.npu_us   = std::chrono::duration<double, std::micro>(t1 - t0).count();
        res.total_us = res.npu_us + res.cpu_us;
        return res;
    };
    BenchOutcome __out = run_benchmark(warmup, bench_runs, run_one);
    append_csv_row(variant_name, N, tile, __out);

    CHECK_ACL2(aclrtMemcpy(xrHost, N * sizeof(float), xrDev, N * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK_ACL2(aclrtMemcpy(xiHost, N * sizeof(float), xiDev, N * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST));
    for (uint32_t i = 0; i < N; ++i) { output[2*i] = xrHost[i]; output[2*i+1] = xiHost[i]; }

    aclrtFree(xrDev); aclrtFree(xiDev); aclrtFree(wDev);
    aclrtFreeHost(xrHost); aclrtFreeHost(xiHost);
    aclrtDestroyStream(stream);
#endif
    WriteFile("./output/output_x.bin", output.data(), 2 * (size_t)N * sizeof(float));
}

static void frank_hi_ps(uint32_t N, uint32_t m, const std::vector<float> &in,
                        uint32_t wu, uint32_t br, const std::string &vn)
{ run_frank(N, m, in, wu, br, vn, true, false); }
static void frank_hi_psm(uint32_t N, uint32_t m, const std::vector<float> &in,
                         uint32_t wu, uint32_t br, const std::string &vn)
{ run_frank(N, m, in, wu, br, vn, true, true); }
static void frank_lo_ps(uint32_t N, uint32_t m, const std::vector<float> &in,
                        uint32_t wu, uint32_t br, const std::string &vn)
{ run_frank(N, m, in, wu, br, vn, false, false); }
static void frank_lo_psm(uint32_t N, uint32_t m, const std::vector<float> &in,
                         uint32_t wu, uint32_t br, const std::string &vn)
{ run_frank(N, m, in, wu, br, vn, false, true); }

// ===========================================================================
// One thunk pair per table entry.
//
// RunFn carries no "which variant am I" argument, and adding one would touch
// every file that uses the registry. Instead each entry gets a thunk that sets
// g_entry before doing the work. A template parameter, not an array of
// std::function, so the thunks are ordinary function pointers.
// ===========================================================================
template <uint32_t E>
static void thunk_ps(uint32_t N, uint32_t m, const std::vector<float> &in,
                     uint32_t wu, uint32_t br, const std::string &vn)
{ g_entry = E; run_odd(N, m, in, wu, br, vn, false); }

template <uint32_t E>
static void thunk_psm(uint32_t N, uint32_t m, const std::vector<float> &in,
                      uint32_t wu, uint32_t br, const std::string &vn)
{ g_entry = E; run_odd(N, m, in, wu, br, vn, true); }

template <uint32_t E>
static uint32_t thunk_prewarm(uint32_t N, uint32_t tile)
{ g_entry = E; return odd_prewarm(N, tile); }

// The variant names are built once and kept alive for the process: VariantInfo
// holds const char*, and a std::string temporary would dangle.
//
// A DEQUE, not a vector. The registry stores c_str() of a string the moment it
// is pushed, and these names ("direct_ps_xs", 12 chars) are short enough to
// live in the small-string buffer INSIDE the string object -- so a vector
// reallocation would move that buffer and leave every pointer already handed
// to the registry dangling. The previous version avoided that with a
// reserve(2 * ODD_KERNEL_COUNT + 8) computed by hand, which was correct but
// held only as long as nobody added a kernel without re-deriving the count.
// A deque never invalidates references to its elements, so the guarantee no
// longer depends on arithmetic done at the other end of the file.
static std::deque<std::string> &name_pool()
{
    static std::deque<std::string> pool;
    return pool;
}

template <uint32_t E>
static void register_entry(Registry &reg)
{
    const OddKernelEntry &e = ODD_KERNELS[E];
    auto &pool = name_pool();
    pool.push_back(std::string(e.butterfly) + "_ps_xs");
    const char *n1 = pool.back().c_str();
    pool.push_back(std::string(e.butterfly) + "_psm_xs");
    const char *n2 = pool.back().c_str();
    static const char *impl[8] = {"r0", "r1", "r2", "r3", "r4", "r5", "r6", "r7"};

    reg.add({impl[e.radix], n1, e.radix, e.butterfly, e.scheme,
             "perstage", "xs", "vector", 0, 0, 1, false,
             &thunk_ps<E>, &thunk_prewarm<E>});
    reg.add({impl[e.radix], n2, e.radix, e.butterfly, e.scheme,
             "perstage", "xs", "vector", 0, 0, 40, false,
             &thunk_psm<E>, &thunk_prewarm<E>});
}

// Compile-time unroll over the table: one register_entry per entry, with E a
// constant so the thunks exist as distinct functions.
template <uint32_t E>
struct RegisterAll {
    static void go(Registry &reg)
    {
        RegisterAll<E - 1>::go(reg);
        register_entry<E - 1>(reg);
    }
};
template <>
struct RegisterAll<0> {
    static void go(Registry &) {}
};

void register_odd(Registry &reg)
{
    RegisterAll<ODD_KERNEL_COUNT>::go(reg);

    // The combined transform. impl "mix" and radix 0 mean "not a single
    // radix"; main() takes radix 0 as "any N the planner accepts" instead of
    // requiring a power.
    // THE STAGE ORDER GOES IN THE `scheme` COLUMN, as "dit_hi" / "dit_lo".
    //
    // It used to be "dit" for all four, with hi and lo distinguished only by
    // the variant NAME -- and scripts/analyze_matrix.py excludes the name from
    // every grouping key, deliberately, because for the single-radix families
    // the name only restates the butterfly and the core count. For mix it was
    // the one place the order lived, so the cores, tile and min_t axes were
    // silently grouping frank_hi together with frank_lo and reporting pairs
    // that differ in two axes under a heading promising they differ in one.
    // The verdicts did not flip -- checked against the collected matrix, every
    // winner stayed the same -- but half the groups were missing (36 instead
    // of 72 on the cores axis), and the stage-order axis itself could not be
    // reported at all because no column carried it.
    //
    // Putting it in `scheme` fixes both at once: the other axes now pair
    // correctly, and the scheme axis becomes the stage-order comparison. It
    // also matches what the arbitrary-length variants already do with their
    // inner radix ("dit_r2"/"dit_r4"). NOTE for cross-run comparison: rows
    // collected before this change carry "dit" in that column.
    reg.add({"mix", "frank_hi_ps_xs",  0, "mixed", "dit_hi", "perstage", "xs",
             "vector", 0, 0,  1, false, frank_hi_ps,  frank_prewarm});
    reg.add({"mix", "frank_hi_psm_xs", 0, "mixed", "dit_hi", "perstage", "xs",
             "vector", 0, 0, 40, false, frank_hi_psm, frank_prewarm});
    reg.add({"mix", "frank_lo_ps_xs",  0, "mixed", "dit_lo", "perstage", "xs",
             "vector", 0, 0,  1, false, frank_lo_ps,  frank_prewarm});
    reg.add({"mix", "frank_lo_psm_xs", 0, "mixed", "dit_lo", "perstage", "xs",
             "vector", 0, 0, 40, false, frank_lo_psm, frank_prewarm});
}
