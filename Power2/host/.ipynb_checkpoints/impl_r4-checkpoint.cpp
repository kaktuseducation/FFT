// host/impl_r4.cpp  (FFTBench)
//
// Все radix-4 варианты. Тела run-функций перенесены из
// r4/Radix4/host/main.cpp БЕЗ ИЗМЕНЕНИЙ — это проверенный на железе код, любая
// правка здесь была бы лишним риском. Изменилось только окружение:
//   * radix-специфика живёт в namespace r4 (host/fft_host_r4.h);
//   * метку прогона (impl, оси, прогрев таблицы твиддлов) несёт g_cell
//     (host/cell.h), поэтому append_csv_row/print_bench_banner не потребовали
//     новых аргументов и тела функций остались нетронутыми;
//   * g_TILE_SIZE теперь общий, определён в host/main.cpp;
//   * внизу — register_r4(), отдающая варианты в общий реестр.
#include "kernel_decls.h"
#include "fft_host_r4.h"
#include "bench_harness.h"
#include "registry.h"
#include "../common/data_utils.h"
#ifndef ASCENDC_CPU_DEBUG
// GetAivCoreNum() — число Vector-ядер платформы, нужно многоядерному
// варианту r4/dit_fsm_xs как blockDim.
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


// ============================================================================
// DIT per-stage, X-Split  (dit_ps_xs, R4-DT-PS-XS)
// ============================================================================
static void run_dit_ps_xs(uint32_t N, uint32_t m2, const std::vector<float> &input,
                           uint32_t warmup, uint32_t bench_runs, const std::string &variant_name)
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
            ICPU_RUN_KF(fft4_hybrid, 1, xr, xi, wDev + st.wOffset, N, st.quarter, block, st.aligned);
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
            fft4_hybrid_do(1, stream, xrDev, xiDev, wStage, N, st.quarter, block, st.aligned);
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
// Должно совпадать с this->tile в kernels/r4_dit_fused_xs.cpp — ширина тайла в
// элементах. Нужна host-стороне, чтобы посчитать число тайлов стадии и выбрать
// blockDim.
static constexpr uint32_t DIT4_TILE_HOST = 64;
// Тот же сентинел, что FFT4_ALL_STAGES в кернеле: "пройди все стадии сам,
// одним ядром" — поведение до появления многоядерности.
static constexpr uint32_t R4_ALL_STAGES = 0xFFFFFFFFu;

// Одно тело на два варианта:
//   multicore=false -> r4/dit_fs_xs  (один launch на все стадии, одно ядро)
//   multicore=true  -> r4/dit_fsm_xs (launch на стадию, тайлы по AIV-ядрам)
// Отличаются РОВНО числом занятых ядер и способом разнесения стадий, вся
// математика общая — иначе сравнение по оси "ядер" перестало бы быть
// сравнением числа ядер.
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
        // CPU-debug: многоядерности нет, всегда legacy-путь
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
        // План печатается один раз, чтобы в отчёте было видно, сколько ядер
        // реально занято на каждой стадии: при маленьком quarter тайлов может
        // оказаться меньше, чем ядер, и тогда часть ядер простаивает.
        fprintf(stderr, "[DIT4-PLAN] AIV-ядер=%u, стадий на NPU=%u, launch'ей=%u "
                        "(по одному на стадию), тайл=%u элементов\n",
                aivCores, numStages, numStages, DIT4_TILE_HOST);
        for (uint32_t st = 0; st < numStages; ++st) {
            const uint32_t quarter = stages[st].quarter;
            const uint32_t block   = quarter << 2;
            const uint32_t nTiles  = (N / block) *
                ((quarter + DIT4_TILE_HOST - 1) / DIT4_TILE_HOST);
            const uint32_t bd      = (nTiles < aivCores) ? nTiles : aivCores;
            fprintf(stderr, "[DIT4-PLAN]   стадия quarter=%-6u тайлов=%-6u blockDim=%-3u "
                            "тайлов на ядро=%u\n",
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
                // Как было: кернел сам идёт по стадиям, blockDim=1.
                fft4_dit_fused_xs_do(1, stream, xrDev, xiDev, wDev, metaDev,
                                     N, numStages, R4_ALL_STAGES, 1u);
            } else {
                // Стадия = отдельный launch: барьер между стадиями даёт сам
                // stream. blockDim — число тайлов стадии, но не больше числа
                // AIV-ядер; кернел раздаёт тайлы по blockIdx шагом blockDim.
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

// Два варианта поверх одного тела. Реестр хранит функции одной сигнатуры,
// поэтому флаг multicore передаётся через эти тонкие обёртки.
static void run_dit_fs_xs(uint32_t N, uint32_t m2, const std::vector<float> &input,
                           uint32_t warmup, uint32_t bench_runs, const std::string &variant_name)
{ run_dit_fs_xs_impl(N, m2, input, warmup, bench_runs, variant_name, false); }

static void run_dit_fsm_xs(uint32_t N, uint32_t m2, const std::vector<float> &input,
                            uint32_t warmup, uint32_t bench_runs, const std::string &variant_name)
{ run_dit_fs_xs_impl(N, m2, input, warmup, bench_runs, variant_name, true); }

// ============================================================================
// Stockham per-stage, X-Split  (st_ps_xs, R4-ST-PS-XS) — НЕТ digit-reversal,
// ping-pong между двумя буферами по чётности стадии (и на CPU, и на NPU)
// ============================================================================
static void run_st_ps_xs(uint32_t N, uint32_t m2, const std::vector<float> &input,
                          uint32_t warmup, uint32_t bench_runs, const std::string &variant_name)
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
            ICPU_RUN_KF(fft4_stockham_xs, 1, sR, sI, dR, dI, wDev + st.wOffset, N, st.quarter, st.aligned);
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
            fft4_stockham_xs_do(1, stream, sR, sI, dR, dI, wStage, N, st.quarter, st.aligned);
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
static void run_st_fs_xs(uint32_t N, uint32_t m2, const std::vector<float> &input,
                          uint32_t warmup, uint32_t bench_runs, const std::string &variant_name)
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
            ICPU_RUN_KF(fft4_stockham_fused_xs, 1, buf0R, buf0I, buf1R, buf1I, wDev, metaDev, N, numStages);
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
            fft4_stockham_fused_xs_do(1, stream, buf0R, buf0I, buf1R, buf1I, wDev, metaDev, N, numStages);
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
void register_r4(Registry &reg)
{
    reg.add({"r4", "dit_ps_xs", 4, "dit", "perstage", "xs", "vector", 0, 1, false, &run_dit_ps_xs, &r4::host_twiddle4_prewarm_N});
    reg.add({"r4", "dit_fs_xs", 4, "dit", "fused", "xs", "vector", 0, 1, false, &run_dit_fs_xs, &r4::host_twiddle4_prewarm_N});
    // Многоядерный близнец dit_fs_xs: та же математика, launch на стадию,
    // тайлы раздаются по 40 AIV-ядрам. Пара (dit_fs_xs, dit_fsm_xs) даёт
    // парное сравнение ровно по оси "число ядер".
    reg.add({"r4", "dit_fsm_xs", 4, "dit", "fused", "xs", "vector", 0, 40, true, &run_dit_fsm_xs, &r4::host_twiddle4_prewarm_N});
    reg.add({"r4", "st_ps_xs", 4, "stockham", "perstage", "xs", "vector", 0, 1, false, &run_st_ps_xs, &r4::host_twiddle4_prewarm_N});
    reg.add({"r4", "st_fs_xs", 4, "stockham", "fused", "xs", "vector", 0, 1, false, &run_st_fs_xs, &r4::host_twiddle4_prewarm_N});
}