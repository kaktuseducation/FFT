// host/cell.h  (FFTBench)
//
// Two objects shared by all the radices.
//
// 1. RunResult — the result of one timed run. It used to be declared in the
//    fft_host_utils.h of EVERY one of the three projects, and in exactly the
//    same way each time. On the merge the radix utilities each moved off into
//    a namespace of their own (host/fft_host_r2.h / r4.h / r8.h), but
//    RunResult has to stay single: both bench_harness.h and every
//    impl_*.cpp see it.
//
// 2. g_cell — a description of the comparison-matrix "cell" being run right
//    now.
//    Why a global and not a parameter: the run-function bodies were carried
//    over from the three projects WITH NO CHANGES, and inside they call
//    append_csv_row() with the old argument set (variant, N, tile, outcome).
//    For the radix, the algorithm, the layout, the Cube batch width and the
//    other axes to land in the CSV as well, they have to reach the write
//    somehow without touching a single line of the proven code. main() sets
//    g_cell before the call, append_csv_row reads it.
//    There is always exactly one run at a time; there is no multithreading
//    here.
#pragma once
#include <cstdint>

struct RunResult {
    double   npu_us;
    double   cpu_us;
    double   total_us;
    uint32_t npu_stages;
    uint32_t cpu_stages;
};

// Every field but the timings is known BEFORE the run: part of it comes from
// the registry (host/registry.h), part from the command line and the build.
struct Cell {
    const char *impl        = "?";        // r2 / r4 / r8
    const char *variant     = "?";        // dit_fs_xs, cube1x3_fs_xs, ...
    uint32_t    radix       = 0;          // 2 / 4 / 8 — N must be radix^k
    const char *algo        = "?";        // dit / stockham / cube
    const char *granularity = "?";        // fused (one launch for all stages) / perstage
    const char *layout      = "?";        // xs (Re and Im apart) / xi (interleaved)
    const char *unit        = "vector";   // vector (AIV) / cube (AIC)
    uint32_t    terms       = 0;          // terms of the precision scheme (cube only)
    // mm — how many matmuls ONE term needs: 1 for cube1 (a single block-real
    // 16x16), 2 for cube2 (A2*Re and A2*Im separately), 0 for the Vector
    // variants. A column of its own rather than something derived from the
    // variant name: only this way does the analyzer get a PAIRED comparison
    // of "one matmul or two" with the remaining axes held fixed.
    uint32_t    mm          = 0;
    uint32_t    cube_batch  = 0;          // the build's CUBE_BATCH (cube only)
    uint32_t    cores       = 1;          // the DECLARED blockDim policy, see below
    uint32_t    tile        = 0;          // the min_npu_* threshold (= g_TILE_SIZE)

    // Warmup of the CPU twiddle table of the radix in question. This used to
    // be the macro HOST_TWIDDLE_PREWARM, one per project; now it is a pointer
    // the registry sets, otherwise bench_harness.h would have to be compiled
    // three times over.
    uint32_t (*prewarm)(uint32_t N, uint32_t tile) = nullptr;
};

// IMPORTANT about cores: this is NOT measured core occupancy, it is the
// blockDim policy the registry DECLARES for the variant. Every r2 and r4
// variant now exists as a single-core/multicore pair — blockDim=1 in all of
// one twin's launches, tiles handed out across the 40 AIV cores in the
// other's — so for those the field is exactly what tells the twins apart.
// r8/dit hands its tiles out across the 40 AIV cores, r8/cube across the 20
// AIC. The field is here precisely so that when the radices are compared this
// difference is visible in the table instead of getting lost.
extern Cell     g_cell;
extern uint32_t g_TILE_SIZE;
