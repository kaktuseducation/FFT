// host/cell.h  (OddFFT)
//
// Two objects shared by all the radices: RunResult (the result of one timed
// run) and g_cell (a description of the comparison-matrix cell being run).
//
// WHAT CHANGED RELATIVE TO THE POWER-OF-TWO PROJECT'S cell.h
// There, one field called `algo` carried "dit" or "stockham" and, for the
// radix-8 Cube variants, "cube". That worked because the butterfly was never
// a free choice: for radix 2 and 4 there is only one sensible butterfly and
// nobody writes a second one.
//
// Here the butterfly IS the main axis. Each of 3, 5 and 7 has four of them
// (direct / sym / winograd / rader) with operation counts that differ by up to
// 3x, and each of those can be run under either scheme (DIT or Stockham).
// Folding both into one column would make every butterfly-vs-butterfly
// comparison unpaired, which is exactly the methodological error the
// power-of-two project's §2.1 was written to prevent. So there are two
// columns: `butterfly` and `scheme`.
#pragma once
#include <cstdint>

struct RunResult {
    double   npu_us;
    double   cpu_us;
    double   total_us;
    uint32_t npu_stages;
    uint32_t cpu_stages;
};

struct Cell {
    const char *impl        = "?";        // r3 / r5 / r7
    const char *variant     = "?";        // sym_psm_xs, winograd_fsm_xs, ...
    uint32_t    radix       = 0;          // 3 / 5 / 7 — N must be radix^k
    // The butterfly algorithm. This is the axis the project exists to measure.
    //   direct   — the r x r DFT matrix applied as written
    //   sym      — conjugate-pair symmetry, the FFTW-codelet form
    //   winograd — Winograd's short-DFT algorithm
    //   rader    — Rader's prime-length algorithm via a cyclic convolution
    //   cube     — the butterfly as a matmul on the AIC cores
    const char *butterfly   = "?";
    // DIT (in place, digit-reversed input) or Stockham (out of place, no
    // permutation). Measured again from scratch here: the power-of-two project
    // found DIT the winner, but it won partly because every stride it needed
    // was a multiple of 8, which no odd radix can offer.
    const char *scheme      = "?";
    const char *granularity = "?";        // fused (one launch for all stages) / perstage
    const char *layout       = "?";       // xs (Re and Im apart) / xi (interleaved)
    const char *unit        = "vector";   // vector (AIV) / cube (AIC)
    uint32_t    terms       = 0;          // precision-scheme terms (cube only)
    uint32_t    mm          = 0;          // matmuls per tile per term (cube only)
    uint32_t    cube_batch  = 0;          // the build's CUBE_BATCH (cube only)
    uint32_t    cores       = 1;          // the DECLARED blockDim policy
    uint32_t    tile        = 0;          // the j-run length a tile covers
    // The smallest operand distance t a stage may have and still run on the
    // device. NOT the same thing as the tile width, though an earlier version
    // of this project conflated them -- it passed the tile as the threshold,
    // so asking for tile=64 silently pushed every stage with t < 64 onto the
    // host. At N=2187 that was four stages of seven and 78% of the runtime.
    // The hardware floor is 8 (the 32-byte DataCopy granule); anything above
    // that is a deliberate choice, and making it an axis is how the cost of
    // the host prefix gets measured instead of assumed.
    uint32_t    min_t       = 0;
    // How many LEADING stages ran on the host because their operand distance t
    // was below the 32-byte DataCopy floor. In the power-of-two project this
    // was a detail; here it is a result in its own right, because the floor
    // removes a very different fraction of the work depending on the radix
    // (t = 1, 3, 9 for radix 3 against t = 1, 7 for radix 7).
    uint32_t    cpu_first   = 0;

    uint32_t (*prewarm)(uint32_t N, uint32_t tile) = nullptr;
};

extern Cell     g_cell;
extern uint32_t g_TILE_SIZE;
extern uint32_t g_MIN_T;
