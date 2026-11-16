// host/bench_harness.h  (FFTBench)
//
// One output format for ALL the variants of all the radices. Every run prints
// exactly three kinds of line:
//
//   [BENCH ] impl=... N=... tile=... warmup=... runs=...      — the header
//   [RUN   ] i/n  npu_us=... cpu_us=... total_us=...          — a single run
//   [RESULT] impl=... N=... tile=... npu_us=... ...           — the total (machine-readable)
//
// The "NPU:/CPU:/Total: avg=... min=... max=..." lines are kept exactly as they
// are — the grep in the older scripts leans on them, and we did not break it.
//
// IMPORTANT: print_bench_banner() also builds the twiddle table for the CPU
// stages, that is, BEFORE the warmup and BEFORE the timing. That levels the
// methodology: inside the timed loop the NPU and the CPU stages alike only
// read ready-made twiddles.
//
// DIFFERENCES FROM THE VERSION IN THE THREE SOURCE PROJECTS (the file was
// identical in all of them):
//   * #include "fft_host_utils.h" is gone — the radix utilities now live in
//     three different namespaces (host/fft_host_r2.h / r4.h / r8.h), and all
//     that is needed from here is RunResult (host/cell.h);
//   * the macro FFT_IMPL_TAG was replaced by g_cell.impl, and the macro
//     HOST_TWIDDLE_PREWARM by g_cell.prewarm. Otherwise this header would have
//     to be compiled once per radix;
//   * append_csv_row() stows the total in g_lastOutcome so that main() can
//     write the matrix row out together with the result of the golden
//     verification (verification is only possible once the run-function has
//     written output_x.bin).
#pragma once
#include "cell.h"
#include <cstdio>
#include <string>
#include <vector>
#include <algorithm>
#include <functional>

struct BenchStats { double avg = 0, min = 0, max = 0; };

inline BenchStats compute_stats(const std::vector<double> &v)
{
    BenchStats s;
    if (v.empty()) return s;
    s.min = s.max = v[0];
    for (double x : v) { s.avg += x; s.min = std::min(s.min, x); s.max = std::max(s.max, x); }
    s.avg /= (double)v.size();
    return s;
}

struct BenchOutcome {
    BenchStats npu, cpu, total;
    uint32_t   npu_stages = 0;
    uint32_t   cpu_stages = 0;
};

// The last total computed — main() picks it up for the matrix row.
extern BenchOutcome g_lastOutcome;
extern bool         g_haveOutcome;

// ============================================================================
// One run header + the warmup of the CPU twiddle table.
// ============================================================================
inline void print_bench_banner(const std::string &variant, uint32_t N, uint32_t tile,
                                uint32_t warmup, uint32_t bench_runs)
{
    uint32_t tabs = g_cell.prewarm ? g_cell.prewarm(N, tile) : 0u;
    printf("================================================================\n");
    printf("[BENCH ] impl=%s/%s  N=%u  tile=%u  warmup=%u  runs=%u\n",
           g_cell.impl, variant.c_str(), N, tile, warmup, bench_runs);
    printf("[BENCH ] twiddles=precomputed (npu=table, cpu=table:%u stage(s)) "
           "— the trigonometry is outside the timing on both sides\n", tabs);
    printf("----------------------------------------------------------------\n");
}

// ============================================================================
// The CSV log for comparing the variants against each other and against NumPy.
// The results format has NOT changed relative to the three source projects —
// scripts/bench_numpy.py and scripts/analyze_matrix.py append to and read that
// very file. The rich table with all the axes is a separate file,
// ./output/matrix.csv, which main() writes (see write_matrix_row).
// ============================================================================
inline void ensure_csv_header(const std::string &csv_path)
{
    FILE *check = fopen(csv_path.c_str(), "r");
    if (check) { fclose(check); return; }
    FILE *f = fopen(csv_path.c_str(), "w");
    if (!f) { fprintf(stderr, "[ERROR ] could not create %s\n", csv_path.c_str()); return; }
    fprintf(f, "variant,N,tile_size,npu_avg_us,cpu_avg_us,total_avg_us,total_min_us,total_max_us\n");
    fclose(f);
}

inline void append_csv_row(const std::string &variant, uint32_t N, uint32_t tile_size,
                            const BenchOutcome &out, const std::string &csv_path = "./output/results.csv")
{
    g_lastOutcome = out;
    g_haveOutcome = true;

    ensure_csv_header(csv_path);
    FILE *f = fopen(csv_path.c_str(), "a");
    if (!f) { fprintf(stderr, "[ERROR ] could not open %s for append\n", csv_path.c_str()); return; }
    // The variant column gets the FULL name (r4/dit_fs_xs): in the merged
    // project the short name is no longer unique.
    fprintf(f, "%s/%s,%u,%u,%.2f,%.2f,%.2f,%.2f,%.2f\n",
            g_cell.impl, variant.c_str(), N, tile_size,
            out.npu.avg, out.cpu.avg, out.total.avg, out.total.min, out.total.max);
    fclose(f);

    printf("[RESULT] impl=%s/%s N=%u tile=%u npu_us=%.2f cpu_us=%.2f total_us=%.2f "
           "min_us=%.2f max_us=%.2f stages_npu=%u stages_cpu=%u\n",
           g_cell.impl, variant.c_str(), N, tile_size,
           out.npu.avg, out.cpu.avg, out.total.avg, out.total.min, out.total.max,
           out.npu_stages, out.cpu_stages);
    printf("================================================================\n");
}

inline BenchOutcome run_benchmark(uint32_t warmup, uint32_t bench_runs,
                                   const std::function<RunResult(bool verbose)> &run_one)
{
    // Even at warmup=0 we do one untimed run: caches, allocations and the
    // first launch must not land in the first timed iteration.
    uint32_t wu = (warmup > 0) ? warmup : 1;
    for (uint32_t i = 0; i < wu; ++i) {
        run_one(false);
        printf("[WARMUP] %u/%u done\n", i + 1, wu);
    }

    std::vector<double> npu_s, cpu_s, total_s;
    RunResult last{0, 0, 0, 0, 0};
    for (uint32_t r = 0; r < bench_runs; ++r) {
        RunResult res = run_one(true);
        last = res;
        npu_s.push_back(res.npu_us); cpu_s.push_back(res.cpu_us); total_s.push_back(res.total_us);
        printf("[RUN   ] %u/%u  npu_us=%10.2f  cpu_us=%10.2f  total_us=%10.2f\n",
               r + 1, bench_runs, res.npu_us, res.cpu_us, res.total_us);
    }

    BenchOutcome out;
    out.npu = compute_stats(npu_s);
    out.cpu = compute_stats(cpu_s);
    out.total = compute_stats(total_s);
    out.npu_stages = last.npu_stages;
    out.cpu_stages = last.cpu_stages;

    printf("----------------------------------------------------------------\n");
    printf("NPU:   avg=%.2f us  min=%.2f us  max=%.2f us\n", out.npu.avg, out.npu.min, out.npu.max);
    printf("CPU:   avg=%.2f us  min=%.2f us  max=%.2f us\n", out.cpu.avg, out.cpu.min, out.cpu.max);
    printf("Total: avg=%.2f us  min=%.2f us  max=%.2f us\n", out.total.avg, out.total.min, out.total.max);
    return out;
}
