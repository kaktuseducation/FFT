// host/main.cpp  (FFTBench — one project across the power-of-two radices)
//
// One binary for all the implementations: radix-2, radix-4, radix-8 (Vector
// and Cube), all the algorithms (DIT, Stockham), both granularities (fused /
// per-stage) and both layouts (xs / xi). For what exactly there is — see
// host/registry.h and `--list`.
//
// Usage:
//   ./ascendc_kernels_bbit <VARIANT> [TILE=8] [WARMUP=2] [RUNS=5]
//   ./ascendc_kernels_bbit --list
//
// VARIANT — the full name "r4/dit_fs_xs", or the short "st_fs_xi" when it is
// unambiguous (all three radices have a dit_fs_xs, so that one has to be
// written out in full).
//
// N is taken from the size of ./input/input_x.bin and must be a POWER OF THE
// RADIX of the chosen variant: radix-2 — any 2^k, radix-4 — only 4^k, radix-8
// — only 8^k. The N common to all three are 64, 4096, 262144 (that is,
// 2^(6k)): those are exactly the ones on which comparing the radices against
// each other means anything. An incompatible (variant, N) pair the binary
// rejects with code 1, and run_matrix.sh simply never queues such cells.
//
// Return codes (run_matrix.sh leans on them):
//   0 — the run happened, golden verification PASS
//   1 — an error (unknown variant, no input, N not a power of the radix)
//   2 — variant skipped: not implemented in this build (wrote no output_x.bin)
//   3 — the run happened, verification FAIL

#include "registry.h"
#include "bench_harness.h"
#include "verify.h"
#include "../common/data_utils.h"
#include "cube1_common.h"      // CUBE_BATCH — goes into the table as an axis

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#ifndef ASCENDC_CPU_DEBUG
#include "acl/acl.h"
#endif

// Definitions of the shared globals declared in cell.h / bench_harness.h.
Cell         g_cell;
uint32_t     g_TILE_SIZE   = 8;
BenchOutcome g_lastOutcome;
bool         g_haveOutcome = false;

// ============================================================================
// The rich table for analysis: one row per matrix cell, every axis an explicit
// column. Kept apart from ./output/results.csv, whose format stayed as it was
// for compatibility with scripts/bench_numpy.py and scripts/analyze_matrix.py.
// ============================================================================
static const char *MATRIX_CSV = "./output/matrix.csv";
static const char *MATRIX_HEADER =
    "impl,variant,radix,algo,granularity,layout,unit,terms,mm,N,tile,cube_batch,cores,"
    "npu_us,cpu_us,total_us,total_min_us,total_max_us,stages_npu,stages_cpu,"
    "status,mismatches,points,err_ratio,max_abs,max_rel\n";

static void write_matrix_row(uint32_t N, const VerifyStats &v, const char *status)
{
    FILE *check = fopen(MATRIX_CSV, "r");
    bool needHeader = (check == nullptr);
    if (check) fclose(check);
    FILE *f = fopen(MATRIX_CSV, "a");
    if (!f) { fprintf(stderr, "[ERROR ] could not open %s for append\n", MATRIX_CSV); return; }
    if (needHeader) fputs(MATRIX_HEADER, f);

    const BenchOutcome &o = g_lastOutcome;
    const bool haveTime = g_haveOutcome;
    fprintf(f, "%s,%s,%u,%s,%s,%s,%s,%u,%u,%u,%u,%u,%u,",
            g_cell.impl, g_cell.variant, g_cell.radix, g_cell.algo, g_cell.granularity,
            g_cell.layout, g_cell.unit, g_cell.terms, g_cell.mm, N, g_cell.tile,
            g_cell.cube_batch, g_cell.cores);
    if (haveTime) {
        fprintf(f, "%.2f,%.2f,%.2f,%.2f,%.2f,%u,%u,",
                o.npu.avg, o.cpu.avg, o.total.avg, o.total.min, o.total.max,
                o.npu_stages, o.cpu_stages);
    } else {
        fprintf(f, ",,,,,,,");
    }
    if (v.available) {
        fprintf(f, "%s,%llu,%llu,%.6f,%.6e,%.6e\n", status,
                (unsigned long long)v.mismatches, (unsigned long long)v.points,
                v.err_ratio, v.max_abs, v.max_rel);
    } else {
        fprintf(f, "%s,,,,,\n", status);
    }
    fclose(f);
}

static void print_list(const Registry &reg)
{
    // Machine-readable: run_matrix.sh uses this listing, so that the knowledge
    // about the radices and the variants' axes is not duplicated in bash.
    printf("impl,variant,radix,algo,granularity,layout,unit,terms,mm,cores\n");
    for (const auto &v : reg.all) {
        printf("%s,%s,%u,%s,%s,%s,%s,%u,%u,%u\n",
               v.impl, v.name, v.radix, v.algo, v.granularity, v.layout,
               v.unit, v.terms, v.mm, v.cores);
    }
}

// ============================================================================
int32_t main(int32_t argc, char *argv[])
{
    setvbuf(stdout, nullptr, _IOLBF, 0);
    Registry reg = build_registry();

    if (argc < 2) {
        fprintf(stderr, "Usage: %s <VARIANT> [TILE=8] [WARMUP=2] [RUNS=5]\n", argv[0]);
        fprintf(stderr, "       %s --list\n", argv[0]);
        fprintf(stderr, "VARIANT: the full impl/variant name, or an unambiguous short one.\n");
        fprintf(stderr, "Variant list: %s --list\n", argv[0]);
        return 1;
    }
    const std::string arg1 = argv[1];
    if (arg1 == "--list" || arg1 == "-l") { print_list(reg); return 0; }

    std::string err;
    const VariantInfo *vi = reg.find(arg1, &err);
    if (!vi) {
        fprintf(stderr, "[ERROR] %s\n", err.c_str());
        fprintf(stderr, "Variant list: %s --list\n", argv[0]);
        return 1;
    }

    g_TILE_SIZE         = (argc > 2) ? (uint32_t)atoi(argv[2]) : 8;
    uint32_t WARMUP     = (argc > 3) ? (uint32_t)atoi(argv[3]) : 2;
    uint32_t BENCH_RUNS = (argc > 4) ? (uint32_t)atoi(argv[4]) : 5;

    // ---- the input and the number of stages ----------------------------
    size_t file_size = 0;
    FILE *f = fopen("./input/input_x.bin", "rb");
    if (!f) {
        fprintf(stderr, "[ERROR] ./input/input_x.bin not found — run scripts/gen_data.py first\n");
        return 1;
    }
    fseek(f, 0, SEEK_END); file_size = (size_t)ftell(f); fclose(f);
    const uint32_t N = (uint32_t)(file_size / (2 * sizeof(float)));

    // N must be a power of the variant's radix; m — the stage count in that base.
    uint32_t m = 0, tmp = N;
    while (tmp > 1) {
        if (tmp % vi->radix != 0) {
            fprintf(stderr, "[ERROR] N=%u is not a power of %u — variant %s requires N=%u^k. "
                            "The N common to all the radices: 64, 4096, 262144.\n",
                    N, vi->radix, vi->full().c_str(), vi->radix);
            return 1;
        }
        tmp /= vi->radix;
        ++m;
    }

    std::vector<float> input(2 * (size_t)N);
    size_t got = 0;
    ReadFile("./input/input_x.bin", got, input.data(), 2 * (size_t)N * sizeof(float));

    // ---- fill in the cell description before the run -------------------
    g_cell.impl        = vi->impl;
    g_cell.variant     = vi->name;
    g_cell.radix       = vi->radix;
    g_cell.algo        = vi->algo;
    g_cell.granularity = vi->granularity;
    g_cell.layout      = vi->layout;
    g_cell.unit        = vi->unit;
    g_cell.terms       = vi->terms;
    g_cell.mm          = vi->mm;
    g_cell.cores       = vi->cores;
    g_cell.tile        = g_TILE_SIZE;
    g_cell.prewarm     = vi->prewarm;
    // The Cube batch width is a property of the BUILD (a compile-time
    // constant); for the Vector variants it means nothing and is written into
    // the table as a zero.
    g_cell.cube_batch  = (std::string(vi->unit) == "cube") ? (uint32_t)CUBE_BATCH : 0u;

#ifndef ASCENDC_CPU_DEBUG
    int32_t deviceId = 0;
    if (const char *envDev = getenv("ASCEND_DEVICE_ID")) deviceId = atoi(envDev);
    if (aclInit(nullptr) != ACL_SUCCESS)        fprintf(stderr, "[ERROR] aclInit failed\n");
    if (aclrtSetDevice(deviceId) != ACL_SUCCESS) fprintf(stderr, "[ERROR] aclrtSetDevice failed\n");
    // We do NOT create an explicit aclrtCreateContext: the official
    // matmul_custom uses the default context from aclrtSetDevice(), and a
    // context of our own initializes FFTS+ (the AIC/AIV scheduler for MIX
    // kernels) differently — at one time that was the prime suspect in the
    // deadlock of the Cube variant.
    printf("[INFO] ACL initialized, deviceId=%d\n", deviceId);
#endif

    // IMPORTANT: wipe the previous output_x.bin BEFORE the run. The file
    // lives on between invocations, and if a variant wrote nothing for
    // whatever reason (a TODO stub, a kernel error), the verification would
    // quietly compare the output of the PREVIOUS variant and could come back
    // PASS. The absence of the file after a run is a reliable sign of "the
    // variant did not do its job".
    remove("./output/output_x.bin");

    print_bench_banner(vi->name, N, g_TILE_SIZE, WARMUP, BENCH_RUNS);
    vi->run(N, m, input, WARMUP, BENCH_RUNS, std::string(vi->name));

#ifndef ASCENDC_CPU_DEBUG
    aclrtResetDevice(deviceId);
    aclFinalize();
#endif

    // ---- verification and the matrix row -------------------------------
    VerifyStats v = verify_against_golden();
    const char *status = !v.available ? "NO_OUTPUT" : (v.ok ? "PASS" : "FAIL");

    if (!v.available) {
        printf("[VERIFY] impl=%s/%s N=%u tile=%u status=NO_OUTPUT (%s) — the variant gave no result\n",
               g_cell.impl, g_cell.variant, N, g_TILE_SIZE, v.note);
    } else {
        printf("[VERIFY] impl=%s/%s N=%u tile=%u status=%s mismatches=%llu/%llu "
               "err_ratio=%.6f tol=%.1e max_abs=%.3e max_rel=%.3e\n",
               g_cell.impl, g_cell.variant, N, g_TILE_SIZE, status,
               (unsigned long long)v.mismatches, (unsigned long long)v.points,
               v.err_ratio, VERIFY_ERR_RATIO, v.max_abs, v.max_rel);
        printf(v.ok ? "test pass\n" : "[ERROR ] result mismatch\n");
    }
    write_matrix_row(N, v, status);

    if (!v.available) return 2;
    return v.ok ? 0 : 3;
}
