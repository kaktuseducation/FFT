// host/impl_r4.cpp  (FFTBench)
//
// All radix-4 variants. The run-function bodies were carried over from
// r4/Radix4/host/main.cpp WITH NO CHANGES — this is code already proven on
// hardware, and any edit here would be needless risk. Only the environment
// around them changed:
//   * the radix-specific part lives in namespace r4 (host/fft_host_r4.h);
//   * the run label (impl, axes, twiddle-table warmup) is carried by g_cell
//     (host/cell.h), so append_csv_row/print_bench_banner needed no new
//     arguments and the function bodies stayed untouched;
//   * g_TILE_SIZE is now shared, defined in host/main.cpp;
//   * at the bottom — register_r4(), which hands the variants to the shared
//     registry.
#include "kernel_decls.h"
#include "fft_host_r4.h"
#include "bench_harness.h"
#include "registry.h"
#include "../common/data_utils.h"
#ifndef ASCENDC_CPU_DEBUG
// GetAivCoreNum() — the number of Vector cores on the platform; the multicore
// variant r4/dit_fsm_xs needs it as blockDim.
#include "cube_tiling.h"
#endif

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

using namespace r4;

// ===========================================================================
// Shared by ALL multicore variants of radix-4.
// Declared at the top of the file, instead of next to the first function that
// uses it: variants sitting earlier in the file have multicore versions too.
// ===========================================================================
// Must match this->tile in every radix-4 kernel — the tile width in
// elements. The host side needs it to work out the number of tiles in a
// stage and pick blockDim.
static constexpr uint32_t DIT4_TILE_HOST = 64;
// The same sentinel as FFT4_*_ALL_STAGES in the kernels: "walk all the stages
// yourself, on a single core" — the behaviour from before multicore existed.
static constexpr uint32_t R4_ALL_STAGES = 0xFFFFFFFFu;


// ============================================================================
// DIT per-stage, X-Split  (dit_ps_xs, R4-DT-PS-XS)
// ============================================================================
static void run_dit_ps_xs_impl(uint32_t N, uint32_t m2, const std::vector<float> &input,
                           uint32_t warmup, uint32_t bench_runs, const std::string &variant_name,
                               bool multicore)
{
    std::vector<float> xr0(N), xi0(N);
    for (uint32_t i = 0; i < N; ++i) { xr0[i] = input[2*i]; xi0[i] = input[2*i+1]; }
    digit_reverse_permute4_split(xr0.data(), xi0.data(), N);

    std::vector<FusedStageInfo> stages;
    uint32_t w_total = 0, cpu_stages = 0;
    build_fused_plan4(m2, g_TILE_SIZE, stages, w_total, cpu_stages);
    std::vector<float> w(w_total);
    fill_fused_buffers4_packed(w.data(), stages);

    std::vector<float> output(2 * N);

#ifdef ASCENDC_CPU_DEBUG
    float *xr = (float *)AscendC::GmAlloc(N * sizeof(float));
    float *xi = (float *)AscendC::GmAlloc(N * sizeof(float));
    float *wDev = (float *)AscendC::GmAlloc(w_total * sizeof(float));
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    memcpy(wDev, w.data(), w_total * sizeof(float));

    auto run_one = [&](bool verbose) -> RunResult {
        RunResult res{0, 0, 0, (uint32_t)stages.size(), cpu_stages};
        memcpy(xr, xr0.data(), N * sizeof(float));
        memcpy(xi, xi0.data(), N * sizeof(float));
        for (uint32_t s = 1; s <= cpu_stages; ++s) host_butterfly4_stage_split(xr, xi, N, 1u << (2*(s-1)));
        for (const auto &st : stages) {
            uint32_t block = st.quarter << 2;
            ICPU_RUN_KF(fft4_hybrid, 1, xr, xi, wDev + st.wOffset, N, st.quarter, block, st.aligned, 1u);
        }
        return res;
    };
    BenchOutcome __out = run_benchmark(warmup, bench_runs, run_one);
    append_csv_row(variant_name, N, g_TILE_SIZE, __out);
    for (uint32_t i = 0; i < N; ++i) { output[2*i] = xr[i]; output[2*i+1] = xi[i]; }
    AscendC::GmFree(xr); AscendC::GmFree(xi); AscendC::GmFree(wDev);
#else
    float *xrHost, *xiHost;
    CHECK_ACL2(aclrtMallocHost((void **)&xrHost, N * sizeof(float)));
    CHECK_ACL2(aclrtMallocHost((void **)&xiHost, N * sizeof(float)));
    void *xrDev, *xiDev, *wDev;
    CHECK_ACL2(aclrtMalloc(&xrDev, N * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL2(aclrtMalloc(&xiDev, N * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL2(aclrtMalloc(&wDev, w_total * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST));
    aclrtStream stream; CHECK_ACL2(aclrtCreateStream(&stream));

    // Multicore mode takes every AIV core; the single-core path is left as it
    // was, so its numbers stay comparable with the matrix already measured.
    const uint32_t aivCores = multicore ? GetAivCoreNum(SOC_VERSION_STR) : 1u;
    CHECK_ACL2(aclrtMemcpy(wDev, w_total * sizeof(float), w.data(), w_total * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE));

    auto run_one = [&](bool verbose) -> RunResult {
        RunResult res{0, 0, 0, (uint32_t)stages.size(), cpu_stages};
        memcpy(xrHost, xr0.data(), N * sizeof(float));
        memcpy(xiHost, xi0.data(), N * sizeof(float));

        if (cpu_stages > 0) {
            auto t0 = std::chrono::high_resolution_clock::now();
            for (uint32_t s = 1; s <= cpu_stages; ++s) host_butterfly4_stage_split(xrHost, xiHost, N, 1u << (2*(s-1)));
            auto t1 = std::chrono::high_resolution_clock::now();
            res.cpu_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
        }
        CHECK_ACL2(aclrtMemcpy(xrDev, N * sizeof(float), xrHost, N * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE));
        CHECK_ACL2(aclrtMemcpy(xiDev, N * sizeof(float), xiHost, N * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE));

        auto t0 = std::chrono::high_resolution_clock::now();
        for (const auto &st : stages) {
            uint32_t block = st.quarter << 2;
            void *wStage = (void *)((char *)wDev + (size_t)st.wOffset * sizeof(float));
            const uint32_t nTiles = (N / block) *
                ((st.quarter + DIT4_TILE_HOST - 1) / DIT4_TILE_HOST);
            const uint32_t bd = multicore
                ? ((nTiles < aivCores) ? nTiles : aivCores) : 1u;
            fft4_hybrid_do(bd, stream, xrDev, xiDev, wStage, N, st.quarter, block, st.aligned, bd);
            CHECK_ACL2(aclrtSynchronizeStream(stream));
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        res.npu_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
        res.total_us = res.npu_us + res.cpu_us;
        return res;
    };
    BenchOutcome __out = run_benchmark(warmup, bench_runs, run_one);
    append_csv_row(variant_name, N, g_TILE_SIZE, __out);
    CHECK_ACL2(aclrtMemcpy(xrHost, N * sizeof(float), xrDev, N * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK_ACL2(aclrtMemcpy(xiHost, N * sizeof(float), xiDev, N * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST));
    for (uint32_t i = 0; i < N; ++i) { output[2*i] = xrHost[i]; output[2*i+1] = xiHost[i]; }
    aclrtFree(xrDev); aclrtFree(xiDev); aclrtFree(wDev);
    aclrtFreeHost(xrHost); aclrtFreeHost(xiHost); aclrtDestroyStream(stream);
#endif
    WriteFile("./output/output_x.bin", output.data(), 2 * N * sizeof(float));
}

// ============================================================================
// DIT fused, X-Split  (dit_fs_xs / dit_fsm_xs, R4-DT-FS-XS)
// ============================================================================

// One body for two variants:
//   multicore=false -> r4/dit_fs_xs  (one launch for all stages, one core)
//   multicore=true  -> r4/dit_fsm_xs (one launch per stage, tiles across the
//                                     AIV cores)
// They differ in EXACTLY the number of cores occupied and in the way the
// stages are spread out; all the math is shared — otherwise a comparison
// along the "cores" axis would stop being a comparison of the number of
// cores.
// Two variants on top of one body: they differ only in the number of cores
// they occupy.
static void run_dit_ps_xs(uint32_t N, uint32_t m2, const std::vector<float> &input,
                   uint32_t warmup, uint32_t bench_runs, const std::string &variant_name)
{ run_dit_ps_xs_impl(N, m2, input, warmup, bench_runs, variant_name, false); }

static void run_dit_psm_xs(uint32_t N, uint32_t m2, const std::vector<float> &input,
                   uint32_t warmup, uint32_t bench_runs, const std::string &variant_name)
{ run_dit_ps_xs_impl(N, m2, input, warmup, bench_runs, variant_name, true); }

static void run_dit_fs_xs_impl(uint32_t N, uint32_t m2, const std::vector<float> &input,
                                uint32_t warmup, uint32_t bench_runs,
                                const std::string &variant_name, bool multicore)
{
    std::vector<float> xr0(N), xi0(N);
    for (uint32_t i = 0; i < N; ++i) { xr0[i] = input[2*i]; xi0[i] = input[2*i+1]; }
    digit_reverse_permute4_split(xr0.data(), xi0.data(), N);

    std::vector<FusedStageInfo> stages;
    uint32_t w_total = 0, cpu_stages = 0;
    build_fused_plan4(m2, g_TILE_SIZE, stages, w_total, cpu_stages);
    std::vector<float> w(w_total);
    fill_fused_buffers4_packed(w.data(), stages);
    uint32_t numStages = (uint32_t)stages.size();

    std::vector<uint32_t> meta(numStages * 3);
    for (size_t i = 0; i < stages.size(); ++i) {
        meta[i*3+0] = stages[i].quarter; meta[i*3+1] = stages[i].aligned; meta[i*3+2] = stages[i].wOffset;
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
        for (uint32_t s = 1; s <= cpu_stages; ++s) host_butterfly4_stage_split(xr, xi, N, 1u << (2*(s-1)));
        // CPU-debug: there is no multicore here, always the legacy path
        if (numStages > 0) ICPU_RUN_KF(fft4_dit_fused_xs, 1, xr, xi, wDev, metaDev,
                                        N, numStages, R4_ALL_STAGES, 1u);
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

    const uint32_t aivCores = multicore ? GetAivCoreNum(SOC_VERSION_STR) : 1u;
    if (multicore) {
        // The plan is printed once, so that the report shows how many cores
        // are really occupied at each stage: with a small quarter there can
        // be fewer tiles than cores, and then some of the cores sit idle.
        fprintf(stderr, "[DIT4-PLAN] AIV cores=%u, stages on NPU=%u, launches=%u "
                        "(one per stage), tile=%u elements\n",
                aivCores, numStages, numStages, DIT4_TILE_HOST);
        for (uint32_t st = 0; st < numStages; ++st) {
            const uint32_t quarter = stages[st].quarter;
            const uint32_t block   = quarter << 2;
            const uint32_t nTiles  = (N / block) *
                ((quarter + DIT4_TILE_HOST - 1) / DIT4_TILE_HOST);
            const uint32_t bd      = (nTiles < aivCores) ? nTiles : aivCores;
            fprintf(stderr, "[DIT4-PLAN]   stage quarter=%-6u tiles=%-6u blockDim=%-3u "
                            "tiles per core=%u\n",
                    quarter, nTiles, bd, (nTiles + bd - 1) / bd);
        }
        fflush(stderr);
    }

    auto run_one = [&](bool verbose) -> RunResult {
        RunResult res{0, 0, 0, numStages, cpu_stages};
        memcpy(xrHost, xr0.data(), N * sizeof(float));
        memcpy(xiHost, xi0.data(), N * sizeof(float));
        if (cpu_stages > 0) {
            auto t0 = std::chrono::high_resolution_clock::now();
            for (uint32_t s = 1; s <= cpu_stages; ++s) host_butterfly4_stage_split(xrHost, xiHost, N, 1u << (2*(s-1)));
            auto t1 = std::chrono::high_resolution_clock::now();
            res.cpu_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
        }
        CHECK_ACL2(aclrtMemcpy(xrDev, N * sizeof(float), xrHost, N * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE));
        CHECK_ACL2(aclrtMemcpy(xiDev, N * sizeof(float), xiHost, N * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE));

        auto t0 = std::chrono::high_resolution_clock::now();
        if (numStages > 0) {
            if (!multicore) {
                // As before: the kernel walks the stages itself, blockDim=1.
                fft4_dit_fused_xs_do(1, stream, xrDev, xiDev, wDev, metaDev,
                                     N, numStages, R4_ALL_STAGES, 1u);
            } else {
                // Stage = a launch of its own: the barrier between stages is
                // supplied by the stream itself. blockDim is the number of
                // tiles in the stage, but no more than the number of AIV
                // cores; the kernel hands the tiles out by blockIdx with
                // stride blockDim.
                for (uint32_t st = 0; st < numStages; ++st) {
                    const uint32_t quarter = stages[st].quarter;
                    const uint32_t block   = quarter << 2;
                    const uint32_t nTiles  = (N / block) *
                        ((quarter + DIT4_TILE_HOST - 1) / DIT4_TILE_HOST);
                    const uint32_t bd      = (nTiles < aivCores) ? nTiles : aivCores;
                    fft4_dit_fused_xs_do(bd, stream, xrDev, xiDev, wDev, metaDev,
                                         N, numStages, st, bd);
                }
            }
            CHECK_ACL2(aclrtSynchronizeStream(stream));
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        res.npu_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
        res.total_us = res.npu_us + res.cpu_us;
        return res;
    };
    BenchOutcome __out = run_benchmark(warmup, bench_runs, run_one);
    append_csv_row(variant_name, N, g_TILE_SIZE, __out);
    CHECK_ACL2(aclrtMemcpy(xrHost, N * sizeof(float), xrDev, N * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK_ACL2(aclrtMemcpy(xiHost, N * sizeof(float), xiDev, N * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST));
    for (uint32_t i = 0; i < N; ++i) { output[2*i] = xrHost[i]; output[2*i+1] = xiHost[i]; }
    aclrtFree(xrDev); aclrtFree(xiDev); aclrtFree(wDev); aclrtFree(metaDev);
    aclrtFreeHost(xrHost); aclrtFreeHost(xiHost); aclrtDestroyStream(stream);
#endif
    WriteFile("./output/output_x.bin", output.data(), 2 * N * sizeof(float));
}

// Two variants on top of one body. The registry stores functions of a single
// signature, so the multicore flag is passed through these thin wrappers.
static void run_dit_fs_xs(uint32_t N, uint32_t m2, const std::vector<float> &input,
                           uint32_t warmup, uint32_t bench_runs, const std::string &variant_name)
{ run_dit_fs_xs_impl(N, m2, input, warmup, bench_runs, variant_name, false); }

static void run_dit_fsm_xs(uint32_t N, uint32_t m2, const std::vector<float> &input,
                            uint32_t warmup, uint32_t bench_runs, const std::string &variant_name)
{ run_dit_fs_xs_impl(N, m2, input, warmup, bench_runs, variant_name, true); }

// ============================================================================
// Stockham per-stage, X-Split  (st_ps_xs, R4-ST-PS-XS) — NO digit-reversal,
// ping-pong between two buffers by stage parity (both on CPU and on NPU)
// ============================================================================
static void run_st_ps_xs_impl(uint32_t N, uint32_t m2, const std::vector<float> &input,
                          uint32_t warmup, uint32_t bench_runs, const std::string &variant_name,
                              bool multicore)
{
    std::vector<float> ar0(N), ai0(N);
    for (uint32_t i = 0; i < N; ++i) { ar0[i] = input[2*i]; ai0[i] = input[2*i+1]; }

    std::vector<FusedStageInfo> stages;
    uint32_t w_total = 0, cpu_stages = 0;
    build_fused_plan4(m2, g_TILE_SIZE, stages, w_total, cpu_stages);
    std::vector<float> w(w_total);
    fill_fused_buffers4_packed(w.data(), stages);

    std::vector<float> output(2 * N);

    auto run_one_common = [&](float *outR, float *outI) {
        std::vector<float> curR(ar0), curI(ai0), br(N), bi(N);
        float *srcR = curR.data(), *srcI = curI.data(), *dstR = br.data(), *dstI = bi.data();
        for (uint32_t s = 1; s <= cpu_stages; ++s) {
            host_stockham4_stage_split(srcR, srcI, dstR, dstI, N, 1u << (2*(s-1)));
            std::swap(srcR, dstR); std::swap(srcI, dstI);
        }
        memcpy(outR, srcR, N * sizeof(float));
        memcpy(outI, srcI, N * sizeof(float));
    };

#ifdef ASCENDC_CPU_DEBUG
    float *buf0R = (float *)AscendC::GmAlloc(N * sizeof(float));
    float *buf0I = (float *)AscendC::GmAlloc(N * sizeof(float));
    float *buf1R = (float *)AscendC::GmAlloc(N * sizeof(float));
    float *buf1I = (float *)AscendC::GmAlloc(N * sizeof(float));
    float *wDev = (float *)AscendC::GmAlloc(w_total * sizeof(float));
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    memcpy(wDev, w.data(), w_total * sizeof(float));

    bool lastFinalIsBuf1 = false;
    auto run_one = [&](bool verbose) -> RunResult {
        RunResult res{0, 0, 0, (uint32_t)stages.size(), cpu_stages};
        run_one_common(buf0R, buf0I);
        bool srcIsBuf1 = false;
        for (const auto &st : stages) {
            float *sR = srcIsBuf1 ? buf1R : buf0R, *sI = srcIsBuf1 ? buf1I : buf0I;
            float *dR = srcIsBuf1 ? buf0R : buf1R, *dI = srcIsBuf1 ? buf0I : buf1I;
            ICPU_RUN_KF(fft4_stockham_xs, 1, sR, sI, dR, dI, wDev + st.wOffset, N, st.quarter, st.aligned, 1u);
            srcIsBuf1 = !srcIsBuf1;
        }
        lastFinalIsBuf1 = srcIsBuf1;
        return res;
    };
    BenchOutcome __out = run_benchmark(warmup, bench_runs, run_one);
    append_csv_row(variant_name, N, g_TILE_SIZE, __out);
    float *finR = lastFinalIsBuf1 ? buf1R : buf0R;
    float *finI = lastFinalIsBuf1 ? buf1I : buf0I;
    for (uint32_t i = 0; i < N; ++i) { output[2*i] = finR[i]; output[2*i+1] = finI[i]; }
    AscendC::GmFree(buf0R); AscendC::GmFree(buf0I); AscendC::GmFree(buf1R); AscendC::GmFree(buf1I); AscendC::GmFree(wDev);
#else
    float *hostR, *hostI;
    CHECK_ACL2(aclrtMallocHost((void **)&hostR, N * sizeof(float)));
    CHECK_ACL2(aclrtMallocHost((void **)&hostI, N * sizeof(float)));
    void *buf0R, *buf0I, *buf1R, *buf1I, *wDev;
    CHECK_ACL2(aclrtMalloc(&buf0R, N * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL2(aclrtMalloc(&buf0I, N * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL2(aclrtMalloc(&buf1R, N * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL2(aclrtMalloc(&buf1I, N * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL2(aclrtMalloc(&wDev, w_total * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST));
    aclrtStream stream; CHECK_ACL2(aclrtCreateStream(&stream));

    // Multicore mode takes every AIV core; the single-core path is left as it
    // was, so its numbers stay comparable with the matrix already measured.
    const uint32_t aivCores = multicore ? GetAivCoreNum(SOC_VERSION_STR) : 1u;
    CHECK_ACL2(aclrtMemcpy(wDev, w_total * sizeof(float), w.data(), w_total * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE));

    bool lastFinalIsBuf1 = false;
    auto run_one = [&](bool verbose) -> RunResult {
        RunResult res{0, 0, 0, (uint32_t)stages.size(), cpu_stages};
        auto t0c = std::chrono::high_resolution_clock::now();
        run_one_common(hostR, hostI);
        auto t1c = std::chrono::high_resolution_clock::now();
        res.cpu_us = std::chrono::duration<double, std::micro>(t1c - t0c).count();

        CHECK_ACL2(aclrtMemcpy(buf0R, N * sizeof(float), hostR, N * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE));
        CHECK_ACL2(aclrtMemcpy(buf0I, N * sizeof(float), hostI, N * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE));

        auto t0 = std::chrono::high_resolution_clock::now();
        bool srcIsBuf1 = false;
        for (const auto &st : stages) {
            void *sR = srcIsBuf1 ? buf1R : buf0R, *sI = srcIsBuf1 ? buf1I : buf0I;
            void *dR = srcIsBuf1 ? buf0R : buf1R, *dI = srcIsBuf1 ? buf0I : buf1I;
            void *wStage = (void *)((char *)wDev + (size_t)st.wOffset * sizeof(float));
            const uint32_t nTiles = (N / (st.quarter << 2)) *
                ((st.quarter + DIT4_TILE_HOST - 1) / DIT4_TILE_HOST);
            const uint32_t bd = multicore
                ? ((nTiles < aivCores) ? nTiles : aivCores) : 1u;
            fft4_stockham_xs_do(bd, stream, sR, sI, dR, dI, wStage, N, st.quarter, st.aligned, bd);
            CHECK_ACL2(aclrtSynchronizeStream(stream));
            srcIsBuf1 = !srcIsBuf1;
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        res.npu_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
        res.total_us = res.npu_us + res.cpu_us;
        lastFinalIsBuf1 = srcIsBuf1;
        return res;
    };
    BenchOutcome __out = run_benchmark(warmup, bench_runs, run_one);
    append_csv_row(variant_name, N, g_TILE_SIZE, __out);
    void *finR = lastFinalIsBuf1 ? buf1R : buf0R;
    void *finI = lastFinalIsBuf1 ? buf1I : buf0I;
    CHECK_ACL2(aclrtMemcpy(hostR, N * sizeof(float), finR, N * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK_ACL2(aclrtMemcpy(hostI, N * sizeof(float), finI, N * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST));
    for (uint32_t i = 0; i < N; ++i) { output[2*i] = hostR[i]; output[2*i+1] = hostI[i]; }
    aclrtFree(buf0R); aclrtFree(buf0I); aclrtFree(buf1R); aclrtFree(buf1I); aclrtFree(wDev);
    aclrtFreeHost(hostR); aclrtFreeHost(hostI); aclrtDestroyStream(stream);
#endif
    WriteFile("./output/output_x.bin", output.data(), 2 * N * sizeof(float));
}

// ============================================================================
// Stockham fused, X-Split  (st_fs_xs, R4-ST-FS-XS)
// ============================================================================
// Two variants on top of one body: they differ only in the number of cores
// they occupy.
static void run_st_ps_xs(uint32_t N, uint32_t m2, const std::vector<float> &input,
                   uint32_t warmup, uint32_t bench_runs, const std::string &variant_name)
{ run_st_ps_xs_impl(N, m2, input, warmup, bench_runs, variant_name, false); }

static void run_st_psm_xs(uint32_t N, uint32_t m2, const std::vector<float> &input,
                   uint32_t warmup, uint32_t bench_runs, const std::string &variant_name)
{ run_st_ps_xs_impl(N, m2, input, warmup, bench_runs, variant_name, true); }

static void run_st_fs_xs_impl(uint32_t N, uint32_t m2, const std::vector<float> &input,
                          uint32_t warmup, uint32_t bench_runs, const std::string &variant_name,
                              bool multicore)
{
    std::vector<float> ar0(N), ai0(N);
    for (uint32_t i = 0; i < N; ++i) { ar0[i] = input[2*i]; ai0[i] = input[2*i+1]; }

    std::vector<FusedStageInfo> stages;
    uint32_t w_total = 0, cpu_stages = 0;
    build_fused_plan4(m2, g_TILE_SIZE, stages, w_total, cpu_stages);
    std::vector<float> w(w_total);
    fill_fused_buffers4_packed(w.data(), stages);
    uint32_t numStages = (uint32_t)stages.size();
    bool finalIsBuf1 = (numStages % 2 == 1);

    std::vector<uint32_t> meta(numStages * 3);
    for (size_t i = 0; i < stages.size(); ++i) {
        meta[i*3+0] = stages[i].quarter; meta[i*3+1] = stages[i].aligned; meta[i*3+2] = stages[i].wOffset;
    }

    auto run_one_common = [&](float *outR, float *outI) {
        std::vector<float> curR(ar0), curI(ai0), bR(N), bI(N);
        float *srcR = curR.data(), *srcI = curI.data(), *dstR = bR.data(), *dstI = bI.data();
        for (uint32_t s = 1; s <= cpu_stages; ++s) {
            host_stockham4_stage_split(srcR, srcI, dstR, dstI, N, 1u << (2*(s-1)));
            std::swap(srcR, dstR); std::swap(srcI, dstI);
        }
        memcpy(outR, srcR, N * sizeof(float));
        memcpy(outI, srcI, N * sizeof(float));
    };

    std::vector<float> output(2 * N);

#ifdef ASCENDC_CPU_DEBUG
    float *buf0R = (float *)AscendC::GmAlloc(N * sizeof(float));
    float *buf0I = (float *)AscendC::GmAlloc(N * sizeof(float));
    float *buf1R = (float *)AscendC::GmAlloc(N * sizeof(float));
    float *buf1I = (float *)AscendC::GmAlloc(N * sizeof(float));
    float *wDev = (float *)AscendC::GmAlloc(w_total * sizeof(float));
    uint32_t *metaDev = (uint32_t *)AscendC::GmAlloc(meta.size() * sizeof(uint32_t));
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    memcpy(wDev, w.data(), w_total * sizeof(float));
    memcpy(metaDev, meta.data(), meta.size() * sizeof(uint32_t));

    auto run_one = [&](bool verbose) -> RunResult {
        RunResult res{0, 0, 0, numStages, cpu_stages};
        run_one_common(buf0R, buf0I);
        if (numStages > 0)
            ICPU_RUN_KF(fft4_stockham_fused_xs, 1, buf0R, buf0I, buf1R, buf1I, wDev, metaDev, N, numStages, R4_ALL_STAGES, 1u);
        return res;
    };
    BenchOutcome __out = run_benchmark(warmup, bench_runs, run_one);
    append_csv_row(variant_name, N, g_TILE_SIZE, __out);
    float *finR = finalIsBuf1 ? buf1R : buf0R;
    float *finI = finalIsBuf1 ? buf1I : buf0I;
    for (uint32_t i = 0; i < N; ++i) { output[2*i] = finR[i]; output[2*i+1] = finI[i]; }
    AscendC::GmFree(buf0R); AscendC::GmFree(buf0I); AscendC::GmFree(buf1R); AscendC::GmFree(buf1I);
    AscendC::GmFree(wDev); AscendC::GmFree(metaDev);
#else
    float *hostR, *hostI;
    CHECK_ACL2(aclrtMallocHost((void **)&hostR, N * sizeof(float)));
    CHECK_ACL2(aclrtMallocHost((void **)&hostI, N * sizeof(float)));
    void *buf0R, *buf0I, *buf1R, *buf1I, *wDev, *metaDev;
    CHECK_ACL2(aclrtMalloc(&buf0R, N * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL2(aclrtMalloc(&buf0I, N * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL2(aclrtMalloc(&buf1R, N * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL2(aclrtMalloc(&buf1I, N * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL2(aclrtMalloc(&wDev, w_total * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL2(aclrtMalloc(&metaDev, meta.size() * sizeof(uint32_t), ACL_MEM_MALLOC_HUGE_FIRST));
    aclrtStream stream; CHECK_ACL2(aclrtCreateStream(&stream));

    // Multicore mode takes every AIV core; the single-core path is left as it
    // was, so its numbers stay comparable with the matrix already measured.
    const uint32_t aivCores = multicore ? GetAivCoreNum(SOC_VERSION_STR) : 1u;
    CHECK_ACL2(aclrtMemcpy(wDev, w_total * sizeof(float), w.data(), w_total * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL2(aclrtMemcpy(metaDev, meta.size() * sizeof(uint32_t), meta.data(), meta.size() * sizeof(uint32_t), ACL_MEMCPY_HOST_TO_DEVICE));

    auto run_one = [&](bool verbose) -> RunResult {
        RunResult res{0, 0, 0, numStages, cpu_stages};
        auto t0c = std::chrono::high_resolution_clock::now();
        run_one_common(hostR, hostI);
        auto t1c = std::chrono::high_resolution_clock::now();
        res.cpu_us = std::chrono::duration<double, std::micro>(t1c - t0c).count();

        CHECK_ACL2(aclrtMemcpy(buf0R, N * sizeof(float), hostR, N * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE));
        CHECK_ACL2(aclrtMemcpy(buf0I, N * sizeof(float), hostI, N * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE));

        auto t0 = std::chrono::high_resolution_clock::now();
        if (numStages > 0) {
            if (!multicore) {
                // As before: the kernel walks all stages itself, blockDim=1.
                fft4_stockham_fused_xs_do(1, stream, buf0R, buf0I, buf1R, buf1I,
                                          wDev, metaDev, N, numStages, R4_ALL_STAGES, 1u);
            } else {
                // Stage = a launch of its own. The barrier between stages is
                // supplied by the stream itself; a hand-rolled cross-core
                // barrier inside the kernel is a source of hangs we have
                // already stepped on.
                for (uint32_t stIdx = 0; stIdx < numStages; ++stIdx) {
                    const uint32_t sz     = stages[stIdx].quarter;
                    const uint32_t nTiles = (N / (sz << 2)) *
                        ((sz + DIT4_TILE_HOST - 1) / DIT4_TILE_HOST);
                    const uint32_t bd = (nTiles < aivCores) ? nTiles : aivCores;
                    fft4_stockham_fused_xs_do(bd, stream, buf0R, buf0I, buf1R, buf1I,
                                              wDev, metaDev, N, numStages, stIdx, bd);
                }
            }
            CHECK_ACL2(aclrtSynchronizeStream(stream));
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        res.npu_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
        res.total_us = res.npu_us + res.cpu_us;
        return res;
    };
    BenchOutcome __out = run_benchmark(warmup, bench_runs, run_one);
    append_csv_row(variant_name, N, g_TILE_SIZE, __out);
    void *finR = finalIsBuf1 ? buf1R : buf0R;
    void *finI = finalIsBuf1 ? buf1I : buf0I;
    CHECK_ACL2(aclrtMemcpy(hostR, N * sizeof(float), finR, N * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK_ACL2(aclrtMemcpy(hostI, N * sizeof(float), finI, N * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST));
    for (uint32_t i = 0; i < N; ++i) { output[2*i] = hostR[i]; output[2*i+1] = hostI[i]; }
    aclrtFree(buf0R); aclrtFree(buf0I); aclrtFree(buf1R); aclrtFree(buf1I); aclrtFree(wDev); aclrtFree(metaDev);
    aclrtFreeHost(hostR); aclrtFreeHost(hostI); aclrtDestroyStream(stream);
#endif
    WriteFile("./output/output_x.bin", output.data(), 2 * N * sizeof(float));
}


// ===========================================================================
// Two variants on top of one body: they differ only in the number of cores
// they occupy.
static void run_st_fs_xs(uint32_t N, uint32_t m2, const std::vector<float> &input,
                   uint32_t warmup, uint32_t bench_runs, const std::string &variant_name)
{ run_st_fs_xs_impl(N, m2, input, warmup, bench_runs, variant_name, false); }

static void run_st_fsm_xs(uint32_t N, uint32_t m2, const std::vector<float> &input,
                   uint32_t warmup, uint32_t bench_runs, const std::string &variant_name)
{ run_st_fs_xs_impl(N, m2, input, warmup, bench_runs, variant_name, true); }

void register_r4(Registry &reg)
{
    // Every variant comes as a pair: single-core (cores=1) and a multicore
    // twin with an 'm' in the name (cores=40). The pair shares the math and
    // the kernel — only blockDim and the spreading of stages across launches
    // differ.
    reg.add({"r4", "dit_ps_xs", 4, "dit", "perstage", "xs", "vector", 0, 0, 1, false, &run_dit_ps_xs, &r4::host_twiddle4_prewarm_N});
    reg.add({"r4", "dit_psm_xs", 4, "dit", "perstage", "xs", "vector", 0, 0, 40, true, &run_dit_psm_xs, &r4::host_twiddle4_prewarm_N});
    reg.add({"r4", "dit_fs_xs", 4, "dit", "fused", "xs", "vector", 0, 0, 1, false, &run_dit_fs_xs, &r4::host_twiddle4_prewarm_N});
    // The multicore twin of dit_fs_xs: the same math, one launch per stage,
    // the tiles handed out across 40 AIV cores. The pair (dit_fs_xs,
    // dit_fsm_xs) gives a paired comparison along exactly the "number of
    // cores" axis.
    reg.add({"r4", "dit_fsm_xs", 4, "dit", "fused", "xs", "vector", 0, 0, 40, true, &run_dit_fsm_xs, &r4::host_twiddle4_prewarm_N});
    reg.add({"r4", "st_ps_xs", 4, "stockham", "perstage", "xs", "vector", 0, 0, 1, false, &run_st_ps_xs, &r4::host_twiddle4_prewarm_N});
    reg.add({"r4", "st_psm_xs", 4, "stockham", "perstage", "xs", "vector", 0, 0, 40, true, &run_st_psm_xs, &r4::host_twiddle4_prewarm_N});
    reg.add({"r4", "st_fs_xs", 4, "stockham", "fused", "xs", "vector", 0, 0, 1, false, &run_st_fs_xs, &r4::host_twiddle4_prewarm_N});
    reg.add({"r4", "st_fsm_xs", 4, "stockham", "fused", "xs", "vector", 0, 0, 40, true, &run_st_fsm_xs, &r4::host_twiddle4_prewarm_N});
}
