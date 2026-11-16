// host/main.cpp  (OddFFT — odd radices 3, 5, 7)
//
// One binary for every odd-radix implementation: each radix, each butterfly
// algorithm, each scheme, each granularity and layout. For what exists, see
// host/registry.h and `--list`.
//
// Usage:
//   ./oddfft <VARIANT> [TILE=64] [WARMUP=2] [RUNS=5] [MIN_T=8]
//   ./oddfft --list
//
// VARIANT — the full name "r3/sym_psm_xs", or the short "sym_psm_xs" when it
// is unambiguous.
//
// N is taken from the size of ./input/input_x.bin and must be a power of the
// chosen variant's radix.
//
// ===========================================================================
// THERE IS NO N COMMON TO RADIX 3, 5 AND 7
// ===========================================================================
// In the power-of-two project the radices shared N = 2^(6k) (64, 4096,
// 262144), so every cross-radix comparison was a paired one at identical N.
// Here no N above 1 is simultaneously a power of 3, of 5 and of 7, so that
// method is simply unavailable and pretending otherwise would be the
// methodological error this project is most at risk of.
//
// What replaces it: compare at the NEAREST sizes and normalise. The nearest
// triples are 2187 / 3125 / 2401 (within 1.43x of each other) and
// 19683 / 15625 / 16807 (within 1.26x). The normalised figure is
//
//     ns_per_butterfly_point = total_us * 1000 / (N * log2(N))
//
// written into the matrix as its own column so the analyzer never has to
// recompute it and every reader sees the same number. It is reported as a
// column and not as a verdict: a 1.26x spread in N is not nothing, and any
// conclusion drawn across it has to say so.
//
// Return codes (run_matrix.sh leans on them):
//   0 — the run happened, golden verification PASS
//   1 — an error (unknown variant, no input, N not a power of the radix)
//   2 — variant skipped or produced nothing
//   3 — the run happened, verification FAIL

#include "registry.h"
#include "bench_harness.h"
#include "verify.h"
#include "../common/data_utils.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#ifndef ASCENDC_CPU_DEBUG
#include "acl/acl.h"
#endif

Cell         g_cell;
uint32_t     g_TILE_SIZE   = 64;
uint32_t     g_MIN_T       = 8;
BenchOutcome g_lastOutcome;
bool         g_haveOutcome = false;

static const char *MATRIX_CSV = "./output/matrix.csv";
// cpu_first and ns_per_pt are columns the power-of-two project did not have.
// cpu_first, because how much of the transform the 32-byte floor pushes to the
// host differs per radix and is a result rather than a detail. ns_per_pt,
// because without a common N it is the only cross-radix figure there is.
static const char *MATRIX_HEADER =
    "impl,variant,radix,butterfly,scheme,granularity,layout,unit,terms,mm,"
    "N,tile,min_t,cube_batch,cores,cpu_first,"
    "npu_us,cpu_us,total_us,total_min_us,total_max_us,ns_per_pt,stages_npu,stages_cpu,"
    "status,mismatches,points,err_ratio,max_abs,max_rel\n";

static void write_matrix_row(uint32_t N, const VerifyStats &v, const char *status)
{
    FILE *check = fopen(MATRIX_CSV, "r");
    const bool needHeader = (check == nullptr);
    if (check) fclose(check);
    FILE *f = fopen(MATRIX_CSV, "a");
    if (!f) { fprintf(stderr, "[ERROR ] could not open %s for append\n", MATRIX_CSV); return; }
    if (needHeader) fputs(MATRIX_HEADER, f);

    const BenchOutcome &o = g_lastOutcome;
    fprintf(f, "%s,%s,%u,%s,%s,%s,%s,%s,%u,%u,%u,%u,%u,%u,%u,%u,",
            g_cell.impl, g_cell.variant, g_cell.radix, g_cell.butterfly, g_cell.scheme,
            g_cell.granularity, g_cell.layout, g_cell.unit, g_cell.terms, g_cell.mm,
            N, g_cell.tile, g_cell.min_t, g_cell.cube_batch, g_cell.cores,
            g_cell.cpu_first);
    if (g_haveOutcome) {
        const double work = (double)N * std::log2((double)N);
        const double nspp = (work > 0.0) ? (o.total.avg * 1000.0 / work) : 0.0;
        fprintf(f, "%.2f,%.2f,%.2f,%.2f,%.2f,%.4f,%u,%u,",
                o.npu.avg, o.cpu.avg, o.total.avg, o.total.min, o.total.max, nspp,
                o.npu_stages, o.cpu_stages);
    } else {
        fprintf(f, ",,,,,,,,");
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
    // Machine-readable: run_matrix.sh reads this, so the knowledge about the
    // variants' axes is not duplicated in bash.
    printf("impl,variant,radix,butterfly,scheme,granularity,layout,unit,terms,mm,cores\n");
    for (const auto &v : reg.all) {
        printf("%s,%s,%u,%s,%s,%s,%s,%s,%u,%u,%u\n",
               v.impl, v.name, v.radix, v.butterfly, v.scheme, v.granularity,
               v.layout, v.unit, v.terms, v.mm, v.cores);
    }
}

int32_t main(int32_t argc, char *argv[])
{
    setvbuf(stdout, nullptr, _IOLBF, 0);
    Registry reg = build_registry();

    if (argc < 2) {
        fprintf(stderr, "Usage: %s <VARIANT> [TILE=64] [WARMUP=2] [RUNS=5] [MIN_T=8]\n", argv[0]);
        fprintf(stderr, "       %s --list\n", argv[0]);
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

    g_TILE_SIZE              = (argc > 2) ? (uint32_t)atoi(argv[2]) : 64;
    const uint32_t WARMUP    = (argc > 3) ? (uint32_t)atoi(argv[3]) : 2;
    const uint32_t BENCH_RUNS= (argc > 4) ? (uint32_t)atoi(argv[4]) : 5;
    // MIN_T may go down to 1: DataCopyPad is byte-granular and was measured
    // correct below the 32-byte granule (see host/plan.h). The default stays
    // at 8 because a stage with a very short run may still be slower on the
    // device than on the host -- that is what the MIN_T axis measures.
    g_MIN_T                  = (argc > 5) ? (uint32_t)atoi(argv[5]) : 8;
    if (g_MIN_T < 1) g_MIN_T = 1;

    FILE *f = fopen("./input/input_x.bin", "rb");
    if (!f) {
        fprintf(stderr, "[ERROR] ./input/input_x.bin not found — run scripts/gen_data.py first\n");
        return 1;
    }
    fseek(f, 0, SEEK_END);
    const size_t file_size = (size_t)ftell(f);
    fclose(f);
    const uint32_t N = (uint32_t)(file_size / (2 * sizeof(float)));

    // radix 0 is the combined transform: N need not be a power of anything,
    // only factorisable over {2,3,4,5,7}, and the planner is what decides
    // that. Any other radix requires N = radix^k.
    uint32_t m = 0, tmp = (vi->radix == 0) ? 1u : N;
    while (tmp > 1) {
        if (tmp % vi->radix != 0) {
            fprintf(stderr, "[ERROR] N=%u is not a power of %u — variant %s requires N=%u^k.\n"
                            "        Unlike the power-of-two project there is NO N shared by "
                            "radix 3, 5 and 7; compare at the nearest sizes instead "
                            "(2187/3125/2401 or 19683/15625/16807) and read the ns_per_pt column.\n",
                    N, vi->radix, vi->full().c_str(), vi->radix);
            return 1;
        }
        tmp /= vi->radix;
        ++m;
    }

    std::vector<float> input(2 * (size_t)N);
    size_t got = 0;
    ReadFile("./input/input_x.bin", got, input.data(), 2 * (size_t)N * sizeof(float));

    g_cell.impl        = vi->impl;
    g_cell.variant     = vi->name;
    g_cell.radix       = vi->radix;
    g_cell.butterfly   = vi->butterfly;
    g_cell.scheme      = vi->scheme;
    g_cell.granularity = vi->granularity;
    g_cell.layout      = vi->layout;
    g_cell.unit        = vi->unit;
    g_cell.terms       = vi->terms;
    g_cell.mm          = vi->mm;
    g_cell.cores       = vi->cores;
    g_cell.tile        = g_TILE_SIZE;
    g_cell.min_t       = g_MIN_T;
    g_cell.cube_batch  = 0;
    g_cell.cpu_first   = 0;   // the run-function fills this from its plan
    g_cell.prewarm     = vi->prewarm;

#ifndef ASCENDC_CPU_DEBUG
    int32_t deviceId = 0;
    if (const char *envDev = getenv("ASCEND_DEVICE_ID")) deviceId = atoi(envDev);
    if (aclInit(nullptr) != ACL_SUCCESS)         fprintf(stderr, "[ERROR] aclInit failed\n");
    if (aclrtSetDevice(deviceId) != ACL_SUCCESS) fprintf(stderr, "[ERROR] aclrtSetDevice failed\n");
    printf("[INFO] ACL initialized, deviceId=%d\n", deviceId);
#endif

    // Wipe the previous output BEFORE the run. The file lives on between
    // invocations, and if a variant wrote nothing the verification would
    // quietly compare the PREVIOUS variant's output and could come back PASS.
    remove("./output/output_x.bin");

    print_bench_banner(vi->name, N, g_TILE_SIZE, WARMUP, BENCH_RUNS);
    vi->run(N, m, input, WARMUP, BENCH_RUNS, std::string(vi->name));

#ifndef ASCENDC_CPU_DEBUG
    aclrtResetDevice(deviceId);
    aclFinalize();
#endif

    const VerifyStats v = verify_against_golden();
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
        printf("[PLAN  ] stages_npu=%u stages_cpu=%u (the leading %u stage(s) have t below the "
               "32-byte DataCopy floor and run on the host)\n",
               g_lastOutcome.npu_stages, g_lastOutcome.cpu_stages, g_cell.cpu_first);
        printf(v.ok ? "test pass\n" : "[ERROR ] result mismatch\n");
    }
    write_matrix_row(N, v, status);

    if (!v.available) return 2;
    return v.ok ? 0 : 3;
}
