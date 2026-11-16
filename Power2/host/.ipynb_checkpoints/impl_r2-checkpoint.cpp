// host/impl_r2.cpp  (FFTBench)
//
// Все radix-2 варианты. Тела run-функций перенесены из
// r2/Radix2/host/main.cpp БЕЗ ИЗМЕНЕНИЙ — это проверенный на железе код, любая
// правка здесь была бы лишним риском. Изменилось только окружение:
//   * radix-специфика живёт в namespace r2 (host/fft_host_r2.h);
//   * метку прогона (impl, оси, прогрев таблицы твиддлов) несёт g_cell
//     (host/cell.h), поэтому append_csv_row/print_bench_banner не потребовали
//     новых аргументов и тела функций остались нетронутыми;
//   * g_TILE_SIZE теперь общий, определён в host/main.cpp;
//   * внизу — register_r2(), отдающая варианты в общий реестр.
#include "kernel_decls.h"
#include "fft_host_r2.h"
#include "bench_harness.h"
#include "registry.h"
#include "../common/data_utils.h"
#ifndef ASCENDC_CPU_DEBUG
// GetAivCoreNum() — число Vector-ядер, нужно варианту r2/dit_fsm_xs как blockDim.
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

using namespace r2;


#ifdef ASCENDC_CPU_DEBUG
static bool g_lastFinalIsBuf1 = false;   // используется в run_st_ps_xs CPU-debug ветке
#endif

// ============================================================================
// DIT per-stage, X-Split  (dit_ps_xs, R2-DT-PS-XS)
// ============================================================================
static void run_dit_ps_xs(uint32_t N, uint32_t m, const std::vector<float> &input,
                           uint32_t warmup, uint32_t bench_runs, const std::string &variant_name)
{
    std::vector<float> xr0(N), xi0(N);
    for (uint32_t i = 0; i < N; ++i) { xr0[i] = input[2*i]; xi0[i] = input[2*i+1]; }
    bit_reverse_permute_split(xr0.data(), xi0.data(), N);

    std::vector<FusedStageInfo> stages;
    uint32_t w_total = 0, cpu_stages = 0;
    build_fused_plan(m, g_TILE_SIZE, stages, w_total, cpu_stages);
    std::vector<float> w(w_total);
    std::vector<uint32_t> meta_unused(stages.size() * 3);
    fill_fused_buffers_packed(w.data(), meta_unused.data(), stages);

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
        for (uint32_t s = 1; s <= cpu_stages; ++s) host_butterfly_stage_split(xr, xi, N, 1u << (s - 1));
        for (const auto &st : stages) {
            uint32_t block = st.half << 1;
            ICPU_RUN_KF(fft_hybrid, 1, xr, xi, wDev + st.wOffset, N, st.half, block, st.aligned);
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
    CHECK_ACL2(aclrtMemcpy(wDev, w_total * sizeof(float), w.data(), w_total * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE));

    auto run_one = [&](bool verbose) -> RunResult {
        RunResult res{0, 0, 0, (uint32_t)stages.size(), cpu_stages};
        memcpy(xrHost, xr0.data(), N * sizeof(float));
        memcpy(xiHost, xi0.data(), N * sizeof(float));

        if (cpu_stages > 0) {
            auto t0 = std::chrono::high_resolution_clock::now();
            for (uint32_t s = 1; s <= cpu_stages; ++s) host_butterfly_stage_split(xrHost, xiHost, N, 1u << (s - 1));
            auto t1 = std::chrono::high_resolution_clock::now();
            res.cpu_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
        }
        CHECK_ACL2(aclrtMemcpy(xrDev, N * sizeof(float), xrHost, N * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE));
        CHECK_ACL2(aclrtMemcpy(xiDev, N * sizeof(float), xiHost, N * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE));

        auto t0 = std::chrono::high_resolution_clock::now();
        for (const auto &st : stages) {
            uint32_t block = st.half << 1;
            void *wStage = (void *)((char *)wDev + (size_t)st.wOffset * sizeof(float));
            fft_hybrid_do(1, stream, xrDev, xiDev, wStage, N, st.half, block, st.aligned);
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
    aclrtFreeHost(xrHost); aclrtFreeHost(xiHost);
    aclrtDestroyStream(stream);
#endif
    WriteFile("./output/output_x.bin", output.data(), 2 * N * sizeof(float));
}

// ============================================================================
// DIT per-stage, X-Interleaved  (dit_ps_xi, R2-DT-PS-XI)
// ============================================================================
static void run_dit_ps_xi(uint32_t N, uint32_t m, const std::vector<float> &input,
                           uint32_t warmup, uint32_t bench_runs, const std::string &variant_name)
{
    std::vector<float> x0(input);
    bit_reverse_permute_interleaved(x0.data(), N);

    std::vector<FusedStageInfo> stages;
    uint32_t w_total = 0, cpu_stages = 0;
    build_fused_plan(m, g_TILE_SIZE, stages, w_total, cpu_stages);
    std::vector<float> w(w_total);
    std::vector<uint32_t> meta_unused(stages.size() * 3);
    fill_fused_buffers_packed(w.data(), meta_unused.data(), stages);

    std::vector<float> output(2 * N);

#ifdef ASCENDC_CPU_DEBUG
    float *x = (float *)AscendC::GmAlloc(2 * N * sizeof(float));
    float *wDev = (float *)AscendC::GmAlloc(w_total * sizeof(float));
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    memcpy(wDev, w.data(), w_total * sizeof(float));

    auto run_one = [&](bool verbose) -> RunResult {
        RunResult res{0, 0, 0, (uint32_t)stages.size(), cpu_stages};
        memcpy(x, x0.data(), 2 * N * sizeof(float));
        for (uint32_t s = 1; s <= cpu_stages; ++s) host_butterfly_stage_interleaved(x, N, 1u << (s - 1));
        for (const auto &st : stages) {
            uint32_t block = st.half << 1;
            ICPU_RUN_KF(fft_interleaved, 1, x, wDev + st.wOffset, N, st.half, block, st.aligned);
        }
        return res;
    };
    BenchOutcome __out = run_benchmark(warmup, bench_runs, run_one);
    append_csv_row(variant_name, N, g_TILE_SIZE, __out);
    memcpy(output.data(), x, 2 * N * sizeof(float));
    AscendC::GmFree(x); AscendC::GmFree(wDev);
#else
    float *xHost;
    CHECK_ACL2(aclrtMallocHost((void **)&xHost, 2 * N * sizeof(float)));
    void *xDev, *wDev;
    CHECK_ACL2(aclrtMalloc(&xDev, 2 * N * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL2(aclrtMalloc(&wDev, w_total * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST));
    aclrtStream stream; CHECK_ACL2(aclrtCreateStream(&stream));
    CHECK_ACL2(aclrtMemcpy(wDev, w_total * sizeof(float), w.data(), w_total * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE));

    auto run_one = [&](bool verbose) -> RunResult {
        RunResult res{0, 0, 0, (uint32_t)stages.size(), cpu_stages};
        memcpy(xHost, x0.data(), 2 * N * sizeof(float));
        if (cpu_stages > 0) {
            auto t0 = std::chrono::high_resolution_clock::now();
            for (uint32_t s = 1; s <= cpu_stages; ++s) host_butterfly_stage_interleaved(xHost, N, 1u << (s - 1));
            auto t1 = std::chrono::high_resolution_clock::now();
            res.cpu_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
        }
        CHECK_ACL2(aclrtMemcpy(xDev, 2 * N * sizeof(float), xHost, 2 * N * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE));

        auto t0 = std::chrono::high_resolution_clock::now();
        for (const auto &st : stages) {
            uint32_t block = st.half << 1;
            void *wStage = (void *)((char *)wDev + (size_t)st.wOffset * sizeof(float));
            fft_interleaved_do(1, stream, xDev, wStage, N, st.half, block, st.aligned);
            CHECK_ACL2(aclrtSynchronizeStream(stream));
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        res.npu_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
        res.total_us = res.npu_us + res.cpu_us;
        return res;
    };
    BenchOutcome __out = run_benchmark(warmup, bench_runs, run_one);
    append_csv_row(variant_name, N, g_TILE_SIZE, __out);
    CHECK_ACL2(aclrtMemcpy(xHost, 2 * N * sizeof(float), xDev, 2 * N * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST));
    memcpy(output.data(), xHost, 2 * N * sizeof(float));
    aclrtFree(xDev); aclrtFree(wDev); aclrtFreeHost(xHost); aclrtDestroyStream(stream);
#endif
    WriteFile("./output/output_x.bin", output.data(), 2 * N * sizeof(float));
}

// ============================================================================
// DIT fused, X-Split  (dit_fs_xs, R2-DT-FS-XS)
// ============================================================================
// Должно совпадать с this->tile в kernels/r2_dit_fused_xs.cpp.
static constexpr uint32_t DIT2_TILE_HOST = 64;
// Тот же сентинел, что FFT2_ALL_STAGES в кернеле.
static constexpr uint32_t R2_ALL_STAGES = 0xFFFFFFFFu;

// Одно тело на два варианта: multicore=false -> r2/dit_fs_xs (один launch,
// одно ядро), multicore=true -> r2/dit_fsm_xs (launch на стадию, тайлы по
// AIV-ядрам). Математика общая — иначе ось "ядер" перестала бы измерять ядра.
static void run_dit_fs_xs_impl(uint32_t N, uint32_t m, const std::vector<float> &input,
                                uint32_t warmup, uint32_t bench_runs,
                                const std::string &variant_name, bool multicore)
{
    std::vector<float> xr0(N), xi0(N);
    for (uint32_t i = 0; i < N; ++i) { xr0[i] = input[2*i]; xi0[i] = input[2*i+1]; }
    bit_reverse_permute_split(xr0.data(), xi0.data(), N);

    std::vector<FusedStageInfo> stages;
    uint32_t w_total = 0, cpu_stages = 0;
    build_fused_plan(m, g_TILE_SIZE, stages, w_total, cpu_stages);
    std::vector<float> w(w_total);
    std::vector<uint32_t> meta(stages.size() * 3);
    fill_fused_buffers_packed(w.data(), meta.data(), stages);
    uint32_t numStages = (uint32_t)stages.size();

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
        for (uint32_t s = 1; s <= cpu_stages; ++s) host_butterfly_stage_split(xr, xi, N, 1u << (s - 1));
        // CPU-debug: многоядерности нет, всегда legacy-путь
        if (numStages > 0) ICPU_RUN_KF(fft_dit_fused_xs, 1, xr, xi, wDev, metaDev,
                                        N, numStages, R2_ALL_STAGES, 1u);
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

    const uint32_t aivCores = multicore ? GetAivCoreNum(SOC_VERSION_STR) : 1u;
    if (multicore) {
        fprintf(stderr, "[DIT2-PLAN] AIV-ядер=%u, стадий на NPU=%u, launch'ей=%u "
                        "(по одному на стадию), тайл=%u элементов\n",
                aivCores, numStages, numStages, DIT2_TILE_HOST);
        for (uint32_t st = 0; st < numStages; ++st) {
            const uint32_t halfSize = stages[st].half;
            const uint32_t block    = halfSize << 1;
            const uint32_t nTiles   = (N / block) *
                ((halfSize + DIT2_TILE_HOST - 1) / DIT2_TILE_HOST);
            const uint32_t bd       = (nTiles < aivCores) ? nTiles : aivCores;
            fprintf(stderr, "[DIT2-PLAN]   стадия half=%-6u тайлов=%-6u blockDim=%-3u "
                            "тайлов на ядро=%u\n",
                    halfSize, nTiles, bd, (nTiles + bd - 1) / bd);
        }
        fflush(stderr);
    }
    CHECK_ACL2(aclrtMemcpy(wDev, w_total * sizeof(float), w.data(), w_total * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL2(aclrtMemcpy(metaDev, meta.size() * sizeof(uint32_t), meta.data(), meta.size() * sizeof(uint32_t), ACL_MEMCPY_HOST_TO_DEVICE));

    auto run_one = [&](bool verbose) -> RunResult {
        RunResult res{0, 0, 0, numStages, cpu_stages};
        memcpy(xrHost, xr0.data(), N * sizeof(float));
        memcpy(xiHost, xi0.data(), N * sizeof(float));
        if (cpu_stages > 0) {
            auto t0 = std::chrono::high_resolution_clock::now();
            for (uint32_t s = 1; s <= cpu_stages; ++s) host_butterfly_stage_split(xrHost, xiHost, N, 1u << (s - 1));
            auto t1 = std::chrono::high_resolution_clock::now();
            res.cpu_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
        }
        CHECK_ACL2(aclrtMemcpy(xrDev, N * sizeof(float), xrHost, N * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE));
        CHECK_ACL2(aclrtMemcpy(xiDev, N * sizeof(float), xiHost, N * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE));

        auto t0 = std::chrono::high_resolution_clock::now();
        if (numStages > 0) {
            if (!multicore) {
                fft_dit_fused_xs_do(1, stream, xrDev, xiDev, wDev, metaDev,
                                    N, numStages, R2_ALL_STAGES, 1u);
            } else {
                // Стадия = отдельный launch, барьер между ними даёт stream.
                for (uint32_t st = 0; st < numStages; ++st) {
                    const uint32_t halfSize = stages[st].half;
                    const uint32_t block    = halfSize << 1;
                    const uint32_t nTiles   = (N / block) *
                        ((halfSize + DIT2_TILE_HOST - 1) / DIT2_TILE_HOST);
                    const uint32_t bd       = (nTiles < aivCores) ? nTiles : aivCores;
                    fft_dit_fused_xs_do(bd, stream, xrDev, xiDev, wDev, metaDev,
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

// Два варианта поверх одного тела: реестр хранит функции одной сигнатуры.
static void run_dit_fs_xs(uint32_t N, uint32_t m, const std::vector<float> &input,
                           uint32_t warmup, uint32_t bench_runs, const std::string &variant_name)
{ run_dit_fs_xs_impl(N, m, input, warmup, bench_runs, variant_name, false); }

static void run_dit_fsm_xs(uint32_t N, uint32_t m, const std::vector<float> &input,
                            uint32_t warmup, uint32_t bench_runs, const std::string &variant_name)
{ run_dit_fs_xs_impl(N, m, input, warmup, bench_runs, variant_name, true); }

// ============================================================================
// DIT fused, X-Interleaved  (dit_fs_xi, R2-DT-FS-XI)
// ============================================================================
static void run_dit_fs_xi(uint32_t N, uint32_t m, const std::vector<float> &input,
                           uint32_t warmup, uint32_t bench_runs, const std::string &variant_name)
{
    std::vector<float> x0(input);
    bit_reverse_permute_interleaved(x0.data(), N);

    std::vector<FusedStageInfo> stages;
    uint32_t w_total = 0, cpu_stages = 0;
    build_fused_plan(m, g_TILE_SIZE, stages, w_total, cpu_stages);
    std::vector<float> w(w_total);
    std::vector<uint32_t> meta(stages.size() * 3);
    fill_fused_buffers_packed(w.data(), meta.data(), stages);
    uint32_t numStages = (uint32_t)stages.size();

    std::vector<float> output(2 * N);

#ifdef ASCENDC_CPU_DEBUG
    float *x = (float *)AscendC::GmAlloc(2 * N * sizeof(float));
    float *wDev = (float *)AscendC::GmAlloc(w_total * sizeof(float));
    uint32_t *metaDev = (uint32_t *)AscendC::GmAlloc(meta.size() * sizeof(uint32_t));
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    memcpy(wDev, w.data(), w_total * sizeof(float));
    memcpy(metaDev, meta.data(), meta.size() * sizeof(uint32_t));

    auto run_one = [&](bool verbose) -> RunResult {
        RunResult res{0, 0, 0, numStages, cpu_stages};
        memcpy(x, x0.data(), 2 * N * sizeof(float));
        for (uint32_t s = 1; s <= cpu_stages; ++s) host_butterfly_stage_interleaved(x, N, 1u << (s - 1));
        if (numStages > 0) ICPU_RUN_KF(fft_dit_fused_xi, 1, x, wDev, metaDev, N, numStages);
        return res;
    };
    BenchOutcome __out = run_benchmark(warmup, bench_runs, run_one);
    append_csv_row(variant_name, N, g_TILE_SIZE, __out);
    memcpy(output.data(), x, 2 * N * sizeof(float));
    AscendC::GmFree(x); AscendC::GmFree(wDev); AscendC::GmFree(metaDev);
#else
    float *xHost;
    CHECK_ACL2(aclrtMallocHost((void **)&xHost, 2 * N * sizeof(float)));
    void *xDev, *wDev, *metaDev;
    CHECK_ACL2(aclrtMalloc(&xDev, 2 * N * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL2(aclrtMalloc(&wDev, w_total * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL2(aclrtMalloc(&metaDev, meta.size() * sizeof(uint32_t), ACL_MEM_MALLOC_HUGE_FIRST));
    aclrtStream stream; CHECK_ACL2(aclrtCreateStream(&stream));
    CHECK_ACL2(aclrtMemcpy(wDev, w_total * sizeof(float), w.data(), w_total * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL2(aclrtMemcpy(metaDev, meta.size() * sizeof(uint32_t), meta.data(), meta.size() * sizeof(uint32_t), ACL_MEMCPY_HOST_TO_DEVICE));

    auto run_one = [&](bool verbose) -> RunResult {
        RunResult res{0, 0, 0, numStages, cpu_stages};
        memcpy(xHost, x0.data(), 2 * N * sizeof(float));
        if (cpu_stages > 0) {
            auto t0 = std::chrono::high_resolution_clock::now();
            for (uint32_t s = 1; s <= cpu_stages; ++s) host_butterfly_stage_interleaved(xHost, N, 1u << (s - 1));
            auto t1 = std::chrono::high_resolution_clock::now();
            res.cpu_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
        }
        CHECK_ACL2(aclrtMemcpy(xDev, 2 * N * sizeof(float), xHost, 2 * N * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE));

        auto t0 = std::chrono::high_resolution_clock::now();
        if (numStages > 0) {
            fft_dit_fused_xi_do(1, stream, xDev, wDev, metaDev, N, numStages);
            CHECK_ACL2(aclrtSynchronizeStream(stream));
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        res.npu_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
        res.total_us = res.npu_us + res.cpu_us;
        return res;
    };
    BenchOutcome __out = run_benchmark(warmup, bench_runs, run_one);
    append_csv_row(variant_name, N, g_TILE_SIZE, __out);
    CHECK_ACL2(aclrtMemcpy(xHost, 2 * N * sizeof(float), xDev, 2 * N * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST));
    memcpy(output.data(), xHost, 2 * N * sizeof(float));
    aclrtFree(xDev); aclrtFree(wDev); aclrtFree(metaDev); aclrtFreeHost(xHost); aclrtDestroyStream(stream);
#endif
    WriteFile("./output/output_x.bin", output.data(), 2 * N * sizeof(float));
}

// ============================================================================
// Stockham per-stage, X-Split  (st_ps_xs, R2-ST-PS-XS) — НЕТ bit-reversal,
// ping-pong между двумя буферами по чётности стадии (и на CPU, и на NPU)
// ============================================================================
static void run_st_ps_xs(uint32_t N, uint32_t m, const std::vector<float> &input,
                          uint32_t warmup, uint32_t bench_runs, const std::string &variant_name)
{
    std::vector<float> ar0(N), ai0(N);   // buffer "A" = вход
    for (uint32_t i = 0; i < N; ++i) { ar0[i] = input[2*i]; ai0[i] = input[2*i+1]; }

    std::vector<FusedStageInfo> stages;
    uint32_t w_total = 0, cpu_stages = 0;
    build_fused_plan(m, g_TILE_SIZE, stages, w_total, cpu_stages);
    std::vector<float> w(w_total);
    std::vector<uint32_t> meta_unused(stages.size() * 3);
    fill_fused_buffers_packed(w.data(), meta_unused.data(), stages);

    std::vector<float> output(2 * N);

    auto run_one_common = [&](float *outR, float *outI) {
        // CPU prefix, ping-pong в host-массивах
        std::vector<float> br(N), bi(N);
        float *srcR = ar0.data(), *srcI = ai0.data();
        // копия входа, т.к. ar0 не должен мутировать между прогонами бенчмарка
        std::vector<float> curR(ar0), curI(ai0);
        srcR = curR.data(); srcI = curI.data();
        float *dstR = br.data(), *dstI = bi.data();
        for (uint32_t s = 1; s <= cpu_stages; ++s) {
            host_stockham_stage_split(srcR, srcI, dstR, dstI, N, 1u << (s - 1));
            std::swap(srcR, dstR); std::swap(srcI, dstI);
        }
        // srcR/srcI теперь держат результат CPU-префикса
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

    auto run_one = [&](bool verbose) -> RunResult {
        RunResult res{0, 0, 0, (uint32_t)stages.size(), cpu_stages};
        run_one_common(buf0R, buf0I);
        bool srcIsBuf1 = false;
        for (const auto &st : stages) {
            float *sR = srcIsBuf1 ? buf1R : buf0R, *sI = srcIsBuf1 ? buf1I : buf0I;
            float *dR = srcIsBuf1 ? buf0R : buf1R, *dI = srcIsBuf1 ? buf0I : buf1I;
            ICPU_RUN_KF(fft_stockham_xs, 1, sR, sI, dR, dI, wDev + st.wOffset, N, st.half, st.aligned);
            srcIsBuf1 = !srcIsBuf1;
        }
        g_lastFinalIsBuf1 = srcIsBuf1;
        return res;
    };
    BenchOutcome __out = run_benchmark(warmup, bench_runs, run_one);
    append_csv_row(variant_name, N, g_TILE_SIZE, __out);
    float *finR = g_lastFinalIsBuf1 ? buf1R : buf0R;
    float *finI = g_lastFinalIsBuf1 ? buf1I : buf0I;
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
            fft_stockham_xs_do(1, stream, sR, sI, dR, dI, wStage, N, st.half, st.aligned);
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
// Stockham per-stage, X-Interleaved  (st_ps_xi, R2-ST-PS-XI)
// ============================================================================
static void run_st_ps_xi(uint32_t N, uint32_t m, const std::vector<float> &input,
                          uint32_t warmup, uint32_t bench_runs, const std::string &variant_name)
{
    std::vector<FusedStageInfo> stages;
    uint32_t w_total = 0, cpu_stages = 0;
    build_fused_plan(m, g_TILE_SIZE, stages, w_total, cpu_stages);
    std::vector<float> w(w_total);
    std::vector<uint32_t> meta_unused(stages.size() * 3);
    fill_fused_buffers_packed(w.data(), meta_unused.data(), stages);

    std::vector<float> output(2 * N);

    auto run_one_common = [&](float *out) {
        std::vector<float> cur(input);
        std::vector<float> buf1(2 * N);
        float *src = cur.data(), *dst = buf1.data();
        for (uint32_t s = 1; s <= cpu_stages; ++s) {
            host_stockham_stage_interleaved(src, dst, N, 1u << (s - 1));
            std::swap(src, dst);
        }
        memcpy(out, src, 2 * N * sizeof(float));
    };

#ifdef ASCENDC_CPU_DEBUG
    float *buf0 = (float *)AscendC::GmAlloc(2 * N * sizeof(float));
    float *buf1 = (float *)AscendC::GmAlloc(2 * N * sizeof(float));
    float *wDev = (float *)AscendC::GmAlloc(w_total * sizeof(float));
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    memcpy(wDev, w.data(), w_total * sizeof(float));

    bool lastFinalIsBuf1 = false;
    auto run_one = [&](bool verbose) -> RunResult {
        RunResult res{0, 0, 0, (uint32_t)stages.size(), cpu_stages};
        run_one_common(buf0);
        bool srcIsBuf1 = false;
        for (const auto &st : stages) {
            float *s = srcIsBuf1 ? buf1 : buf0, *d = srcIsBuf1 ? buf0 : buf1;
            ICPU_RUN_KF(fft_stockham_xi, 1, s, d, wDev + st.wOffset, N, st.half, st.aligned);
            srcIsBuf1 = !srcIsBuf1;
        }
        lastFinalIsBuf1 = srcIsBuf1;
        return res;
    };
    BenchOutcome __out = run_benchmark(warmup, bench_runs, run_one);
    append_csv_row(variant_name, N, g_TILE_SIZE, __out);
    float *fin = lastFinalIsBuf1 ? buf1 : buf0;
    memcpy(output.data(), fin, 2 * N * sizeof(float));
    AscendC::GmFree(buf0); AscendC::GmFree(buf1); AscendC::GmFree(wDev);
#else
    float *host;
    CHECK_ACL2(aclrtMallocHost((void **)&host, 2 * N * sizeof(float)));
    void *buf0, *buf1, *wDev;
    CHECK_ACL2(aclrtMalloc(&buf0, 2 * N * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL2(aclrtMalloc(&buf1, 2 * N * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL2(aclrtMalloc(&wDev, w_total * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST));
    aclrtStream stream; CHECK_ACL2(aclrtCreateStream(&stream));
    CHECK_ACL2(aclrtMemcpy(wDev, w_total * sizeof(float), w.data(), w_total * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE));

    bool lastFinalIsBuf1 = false;
    auto run_one = [&](bool verbose) -> RunResult {
        RunResult res{0, 0, 0, (uint32_t)stages.size(), cpu_stages};
        auto t0c = std::chrono::high_resolution_clock::now();
        run_one_common(host);
        auto t1c = std::chrono::high_resolution_clock::now();
        res.cpu_us = std::chrono::duration<double, std::micro>(t1c - t0c).count();

        CHECK_ACL2(aclrtMemcpy(buf0, 2 * N * sizeof(float), host, 2 * N * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE));

        auto t0 = std::chrono::high_resolution_clock::now();
        bool srcIsBuf1 = false;
        for (const auto &st : stages) {
            void *s = srcIsBuf1 ? buf1 : buf0, *d = srcIsBuf1 ? buf0 : buf1;
            void *wStage = (void *)((char *)wDev + (size_t)st.wOffset * sizeof(float));
            fft_stockham_xi_do(1, stream, s, d, wStage, N, st.half, st.aligned);
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
    void *fin = lastFinalIsBuf1 ? buf1 : buf0;
    CHECK_ACL2(aclrtMemcpy(host, 2 * N * sizeof(float), fin, 2 * N * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST));
    memcpy(output.data(), host, 2 * N * sizeof(float));
    aclrtFree(buf0); aclrtFree(buf1); aclrtFree(wDev); aclrtFreeHost(host); aclrtDestroyStream(stream);
#endif
    WriteFile("./output/output_x.bin", output.data(), 2 * N * sizeof(float));
}

// ============================================================================
// Stockham fused, X-Split  (st_fs_xs, R2-ST-FS-XS)
// ============================================================================
static void run_st_fs_xs(uint32_t N, uint32_t m, const std::vector<float> &input,
                          uint32_t warmup, uint32_t bench_runs, const std::string &variant_name)
{
    std::vector<float> ar0(N), ai0(N);
    for (uint32_t i = 0; i < N; ++i) { ar0[i] = input[2*i]; ai0[i] = input[2*i+1]; }

    std::vector<FusedStageInfo> stages;
    uint32_t w_total = 0, cpu_stages = 0;
    build_fused_plan(m, g_TILE_SIZE, stages, w_total, cpu_stages);
    std::vector<float> w(w_total);
    std::vector<uint32_t> meta(stages.size() * 3);
    fill_fused_buffers_packed(w.data(), meta.data(), stages);
    uint32_t numStages = (uint32_t)stages.size();
    bool finalIsBuf1 = (numStages % 2 == 1);   // см. вывод в kernels/stockham_fused_xs.cpp

    auto run_one_common = [&](float *outR, float *outI) {
        std::vector<float> curR(ar0), curI(ai0), bR(N), bI(N);
        float *srcR = curR.data(), *srcI = curI.data(), *dstR = bR.data(), *dstI = bI.data();
        for (uint32_t s = 1; s <= cpu_stages; ++s) {
            host_stockham_stage_split(srcR, srcI, dstR, dstI, N, 1u << (s - 1));
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
            ICPU_RUN_KF(fft_stockham_fused_xs, 1, buf0R, buf0I, buf1R, buf1I, wDev, metaDev, N, numStages);
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
            fft_stockham_fused_xs_do(1, stream, buf0R, buf0I, buf1R, buf1I, wDev, metaDev, N, numStages);
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

// ============================================================================
// Stockham fused, X-Interleaved  (st_fs_xi, R2-ST-FS-XI)
// ============================================================================
static void run_st_fs_xi(uint32_t N, uint32_t m, const std::vector<float> &input,
                          uint32_t warmup, uint32_t bench_runs, const std::string &variant_name)
{
    std::vector<FusedStageInfo> stages;
    uint32_t w_total = 0, cpu_stages = 0;
    build_fused_plan(m, g_TILE_SIZE, stages, w_total, cpu_stages);
    std::vector<float> w(w_total);
    std::vector<uint32_t> meta(stages.size() * 3);
    fill_fused_buffers_packed(w.data(), meta.data(), stages);
    uint32_t numStages = (uint32_t)stages.size();
    bool finalIsBuf1 = (numStages % 2 == 1);

    auto run_one_common = [&](float *out) {
        std::vector<float> cur(input), buf1(2 * N);
        float *src = cur.data(), *dst = buf1.data();
        for (uint32_t s = 1; s <= cpu_stages; ++s) {
            host_stockham_stage_interleaved(src, dst, N, 1u << (s - 1));
            std::swap(src, dst);
        }
        memcpy(out, src, 2 * N * sizeof(float));
    };

    std::vector<float> output(2 * N);

#ifdef ASCENDC_CPU_DEBUG
    float *buf0 = (float *)AscendC::GmAlloc(2 * N * sizeof(float));
    float *buf1 = (float *)AscendC::GmAlloc(2 * N * sizeof(float));
    float *wDev = (float *)AscendC::GmAlloc(w_total * sizeof(float));
    uint32_t *metaDev = (uint32_t *)AscendC::GmAlloc(meta.size() * sizeof(uint32_t));
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    memcpy(wDev, w.data(), w_total * sizeof(float));
    memcpy(metaDev, meta.data(), meta.size() * sizeof(uint32_t));

    auto run_one = [&](bool verbose) -> RunResult {
        RunResult res{0, 0, 0, numStages, cpu_stages};
        run_one_common(buf0);
        if (numStages > 0) ICPU_RUN_KF(fft_stockham_fused_xi, 1, buf0, buf1, wDev, metaDev, N, numStages);
        return res;
    };
    BenchOutcome __out = run_benchmark(warmup, bench_runs, run_one);
    append_csv_row(variant_name, N, g_TILE_SIZE, __out);
    float *fin = finalIsBuf1 ? buf1 : buf0;
    memcpy(output.data(), fin, 2 * N * sizeof(float));
    AscendC::GmFree(buf0); AscendC::GmFree(buf1); AscendC::GmFree(wDev); AscendC::GmFree(metaDev);
#else
    float *host;
    CHECK_ACL2(aclrtMallocHost((void **)&host, 2 * N * sizeof(float)));
    void *buf0, *buf1, *wDev, *metaDev;
    CHECK_ACL2(aclrtMalloc(&buf0, 2 * N * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL2(aclrtMalloc(&buf1, 2 * N * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL2(aclrtMalloc(&wDev, w_total * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL2(aclrtMalloc(&metaDev, meta.size() * sizeof(uint32_t), ACL_MEM_MALLOC_HUGE_FIRST));
    aclrtStream stream; CHECK_ACL2(aclrtCreateStream(&stream));
    CHECK_ACL2(aclrtMemcpy(wDev, w_total * sizeof(float), w.data(), w_total * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL2(aclrtMemcpy(metaDev, meta.size() * sizeof(uint32_t), meta.data(), meta.size() * sizeof(uint32_t), ACL_MEMCPY_HOST_TO_DEVICE));

    auto run_one = [&](bool verbose) -> RunResult {
        RunResult res{0, 0, 0, numStages, cpu_stages};
        auto t0c = std::chrono::high_resolution_clock::now();
        run_one_common(host);
        auto t1c = std::chrono::high_resolution_clock::now();
        res.cpu_us = std::chrono::duration<double, std::micro>(t1c - t0c).count();

        CHECK_ACL2(aclrtMemcpy(buf0, 2 * N * sizeof(float), host, 2 * N * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE));

        auto t0 = std::chrono::high_resolution_clock::now();
        if (numStages > 0) {
            fft_stockham_fused_xi_do(1, stream, buf0, buf1, wDev, metaDev, N, numStages);
            CHECK_ACL2(aclrtSynchronizeStream(stream));
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        res.npu_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
        res.total_us = res.npu_us + res.cpu_us;
        return res;
    };
    BenchOutcome __out = run_benchmark(warmup, bench_runs, run_one);
    append_csv_row(variant_name, N, g_TILE_SIZE, __out);
    void *fin = finalIsBuf1 ? buf1 : buf0;
    CHECK_ACL2(aclrtMemcpy(host, 2 * N * sizeof(float), fin, 2 * N * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST));
    memcpy(output.data(), host, 2 * N * sizeof(float));
    aclrtFree(buf0); aclrtFree(buf1); aclrtFree(wDev); aclrtFree(metaDev); aclrtFreeHost(host); aclrtDestroyStream(stream);
#endif
    WriteFile("./output/output_x.bin", output.data(), 2 * N * sizeof(float));
}


// ===========================================================================
void register_r2(Registry &reg)
{
    reg.add({"r2", "dit_ps_xs", 2, "dit", "perstage", "xs", "vector", 0, 1, false, &run_dit_ps_xs, &r2::host_twiddle2_prewarm_N});
    reg.add({"r2", "dit_ps_xi", 2, "dit", "perstage", "xi", "vector", 0, 1, false, &run_dit_ps_xi, &r2::host_twiddle2_prewarm_N});
    reg.add({"r2", "dit_fs_xs", 2, "dit", "fused", "xs", "vector", 0, 1, false, &run_dit_fs_xs, &r2::host_twiddle2_prewarm_N});
    // Многоядерный близнец: та же математика, launch на стадию, тайлы по
    // AIV-ядрам. Нужен, чтобы ось "радикс" сравнивала радиксы, а не число
    // занятых ядер: без него r2 оставался единственным одноядерным.
    reg.add({"r2", "dit_fsm_xs", 2, "dit", "fused", "xs", "vector", 0, 40, true, &run_dit_fsm_xs, &r2::host_twiddle2_prewarm_N});
    reg.add({"r2", "dit_fs_xi", 2, "dit", "fused", "xi", "vector", 0, 1, false, &run_dit_fs_xi, &r2::host_twiddle2_prewarm_N});
    reg.add({"r2", "st_ps_xs", 2, "stockham", "perstage", "xs", "vector", 0, 1, false, &run_st_ps_xs, &r2::host_twiddle2_prewarm_N});
    reg.add({"r2", "st_ps_xi", 2, "stockham", "perstage", "xi", "vector", 0, 1, false, &run_st_ps_xi, &r2::host_twiddle2_prewarm_N});
    reg.add({"r2", "st_fs_xs", 2, "stockham", "fused", "xs", "vector", 0, 1, false, &run_st_fs_xs, &r2::host_twiddle2_prewarm_N});
    reg.add({"r2", "st_fs_xi", 2, "stockham", "fused", "xi", "vector", 0, 1, false, &run_st_fs_xi, &r2::host_twiddle2_prewarm_N});
}