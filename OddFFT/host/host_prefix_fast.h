// host/host_prefix_fast.h  (OddFFT)
//
// The sub-floor prefix, specialised.
//
// WHY THIS FILE EXISTS
// The first measured matrix made the shape of the problem unmistakable: the
// stages that run on the host take 80-93% of the total time, while the device
// stages they feed take the rest. Every odd-radix variant lost to NumPy, and
// it lost on the host, not on the device.
//
// The generic host_stage_tab in host_twiddles.h is written for clarity and
// pays for it three times over:
//
//   1. The radix is a RUNTIME value, so the butterfly's loops cannot unroll
//      and its operands live in arrays rather than registers.
//   2. A stage with t == 1 has only j == 0, and W^0 = 1 -- that stage needs NO
//      twiddle multiplies at all. The generic code multiplies anyway: for
//      radix 7 that is 24 wasted real multiplies and 12 wasted adds per
//      butterfly, on the stage that has the most butterflies.
//   3. The twiddle row offset (2*(m-1)+ri)*t + j is recomputed for every
//      (m, j) instead of being hoisted per j.
//
// This header fixes all three by templating on the radix, so the butterfly
// unrolls into registers, and by splitting the t == 1 case out.
//
// IT IS NOT A DIFFERENT ALGORITHM. It computes the same conjugate-pair
// butterfly as butterfly_sym, against which it is checked element for element
// in tests/host_selftest.cpp. The arithmetic is identical; only the code
// generated for it differs.
//
// ===========================================================================
// AND WHY IT IS PARALLEL
// ===========================================================================
// The host of this board is a Kunpeng-920 with 192 hardware threads, and this
// prefix was running on exactly one of them. The matrix says what that cost:
// at N = 100800 the two sub-floor stages (t = 1 and t = 7) take 8370 us on the
// host, against 401.67 us for the six device stages that follow. 94% of the
// transform, single-threaded, next to 191 idle cores.
//
// The outer loop over blocks is embarrassingly parallel: block k0 touches only
// the indices k0 + j + m*t for j < t and m < R, so two different blocks share
// no element. Nothing is reduced across blocks and no element is read by one
// block and written by another, which is why this parallelises without a
// single atomic and WITHOUT CHANGING THE RESULT: the arithmetic inside a block
// is untouched and blocks do not interact, so the output is bit-identical to
// the serial version. tests/host_selftest.cpp checks exactly that rather than
// taking it on trust -- a parallel pass that is merely "close enough" would be
// a different algorithm, and the project's whole verification story rests on
// the host path being the reference the device path is compared against.
//
// The sub-floor stages, which are the ones that matter here, always have many
// blocks: nBlk = N/(R*t) and t is small by construction. The large-t stages
// have few blocks but a long j loop, so those parallelise over j instead. The
// switch between the two is a runtime `if` on the pragma, not two code paths.
//
// Thread count comes from OMP_NUM_THREADS. It is deliberately NOT fixed here:
// this pass is memory-bound, so the useful number is well below 192 and has to
// be measured on the machine rather than guessed.
#pragma once
#include "plan.h"
#include "butterflies.h"
#include "host_twiddles.h"
#include <cstdint>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace oddfft {

// One conjugate-pair butterfly, radix fixed at compile time so every loop
// below unrolls and the operands stay in registers.
template <uint32_t R>
inline void bf_sym_fixed(float *br, float *bi)
{
    constexpr uint32_t H = (R - 1u) / 2u;
    const RootTable &T = roots();

    float ar[4], ai[4], dr[4], di[4];
    for (uint32_t k = 1; k <= H; ++k) {
        ar[k] = br[k] + br[R - k];  ai[k] = bi[k] + bi[R - k];
        dr[k] = br[k] - br[R - k];  di[k] = bi[k] - bi[R - k];
    }
    const float x0r = br[0], x0i = bi[0];
    float o0r = x0r, o0i = x0i;
    for (uint32_t k = 1; k <= H; ++k) { o0r += ar[k]; o0i += ai[k]; }

    float outr[8], outi[8];
    outr[0] = o0r; outi[0] = o0i;
    for (uint32_t p = 1; p <= H; ++p) {
        double ur = x0r, ui = x0i, vr = 0.0, vi = 0.0;
        for (uint32_t k = 1; k <= H; ++k) {
            const double cc = T.c[R][(p * k) % R];
            const double ss = T.s[R][(p * k) % R];
            ur += cc * (double)ar[k];  ui += cc * (double)ai[k];
            vr += ss * (double)dr[k];  vi += ss * (double)di[k];
        }
        outr[p]     = (float)(ur + vi);  outi[p]     = (float)(ui - vr);
        outr[R - p] = (float)(ur - vi);  outi[R - p] = (float)(ui + vr);
    }
    for (uint32_t m = 0; m < R; ++m) { br[m] = outr[m]; bi[m] = outi[m]; }
}

// Even radices have a self-conjugate middle term the conjugate-pair form has
// no place for, so they take the naive route, exactly as butterfly_host does.
template <uint32_t R>
inline void bf_fixed(float *br, float *bi)
{
    if (R % 2u == 0u) butterfly_direct(R, br, bi, 1);
    else              bf_sym_fixed<R>(br, bi);
}

// One host stage with the radix fixed at compile time.
template <uint32_t R>
inline void host_stage_fixed(const Plan &p, size_t s, float *xr, float *xi)
{
    const StageInfo &st = p.stages[s];
    const HostTwiddles &tw = host_twiddle_cache();
    const float *base = tw.w.data() + tw.offset[s];
    const uint32_t t = st.t, block = st.block, N = p.N;
    const uint32_t nBlk = (block > 0u) ? (N / block) : 1u;
    // Parallelise over blocks when there are enough of them to go round, and
    // over j otherwise. 8 is a floor, not a tuned number: below it the loop is
    // too short for the fork to pay for itself whatever the thread count.
    const bool parBlocks = (nBlk >= 8u);
#ifndef _OPENMP
    // Without OpenMP the pragmas vanish and this would be an unused variable,
    // which the build turns into an error (-Wall -Werror). The serial build
    // has to stay buildable: tests/run_tests.sh is the check a reviewer runs,
    // and it must not depend on the compiler having OpenMP.
    (void)parBlocks;
#endif

    if (t == 1u) {
        // Every twiddle on this stage is W^0 = 1, so there are none to apply.
        // This is also the stage with the most butterflies, which is why the
        // generic code's wasted multiplies here cost the most -- and why it is
        // the stage the parallel pass buys the most on.
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(parBlocks)
#endif
        for (uint32_t b = 0; b < nBlk; ++b) {
            const uint32_t k0 = b * block;
            float br[8], bi[8];
            for (uint32_t m = 0; m < R; ++m) { br[m] = xr[k0 + m]; bi[m] = xi[k0 + m]; }
            bf_fixed<R>(br, bi);
            for (uint32_t m = 0; m < R; ++m) { xr[k0 + m] = br[m]; xi[k0 + m] = bi[m]; }
        }
        return;
    }

    // Row pointers hoisted out of the j loop.
    const float *wr[8], *wi[8];
    for (uint32_t m = 1; m < R; ++m) {
        wr[m] = base + (2u * (m - 1u) + 0u) * t;
        wi[m] = base + (2u * (m - 1u) + 1u) * t;
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(parBlocks)
#endif
    for (uint32_t b = 0; b < nBlk; ++b) {
        const uint32_t k0 = b * block;
        // j == 0 carries W^0 = 1 on every operand: no multiplies.
        {
            float br[8], bi[8];
            for (uint32_t m = 0; m < R; ++m) {
                const uint32_t idx = k0 + m * t;
                br[m] = xr[idx]; bi[m] = xi[idx];
            }
            bf_fixed<R>(br, bi);
            for (uint32_t m = 0; m < R; ++m) {
                const uint32_t idx = k0 + m * t;
                xr[idx] = br[m]; xi[idx] = bi[m];
            }
        }
        // When there are too few blocks to spread, spread the j loop instead.
        // Only one of the two pragmas is ever active, so this never nests.
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(!parBlocks && t >= 64u)
#endif
        for (uint32_t j = 1; j < t; ++j) {
            float br[8], bi[8];
            br[0] = xr[k0 + j]; bi[0] = xi[k0 + j];
            for (uint32_t m = 1; m < R; ++m) {
                const uint32_t idx = k0 + j + m * t;
                const float c = wr[m][j], s2 = wi[m][j];
                const float vr = xr[idx], vi = xi[idx];
                br[m] = c * vr - s2 * vi;
                bi[m] = c * vi + s2 * vr;
            }
            bf_fixed<R>(br, bi);
            for (uint32_t m = 0; m < R; ++m) {
                const uint32_t idx = k0 + j + m * t;
                xr[idx] = br[m]; xi[idx] = bi[m];
            }
        }
    }
}

// Dispatch on the stage's radix. Any radix without a specialisation falls back
// to the generic path, so adding a radix cannot silently lose correctness.
inline void host_stage_fast(const Plan &p, size_t s, float *xr, float *xi)
{
    switch (p.stages[s].radix) {
        case 2: host_stage_fixed<2>(p, s, xr, xi); return;
        case 3: host_stage_fixed<3>(p, s, xr, xi); return;
        case 4: host_stage_fixed<4>(p, s, xr, xi); return;
        case 5: host_stage_fixed<5>(p, s, xr, xi); return;
        case 7: host_stage_fixed<7>(p, s, xr, xi); return;
        default: host_stage_tab(p, s, xr, xi); return;
    }
}

inline uint32_t host_prefix_fast(const Plan &p, float *xr, float *xi)
{
    uint32_t done = 0;
    for (size_t s = 0; s < p.stages.size(); ++s) {
        if (p.stages[s].on_npu) break;
        host_stage_fast(p, s, xr, xi);
        ++done;
    }
    return done;
}

} // namespace oddfft
