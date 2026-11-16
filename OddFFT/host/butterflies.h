// host/butterflies.h — OddFFT
//
// The host-side radix-3/5/7 butterfly, used for two things and only two:
//   1. the sub-floor stages, whose rows are shorter than the 32-byte DataCopy
//      granularity and therefore cannot run on the NPU at all (see plan.h);
//   2. a cross-check inside the self-test.
//
// !!! WHY THERE IS ONE IMPLEMENTATION HERE AND FOUR IN THE KERNELS !!!
// The four butterfly algorithms (direct, sym, winograd, cube) are a KERNEL-side
// axis. Every variant of the project shares this same host code for its
// sub-floor prefix, deliberately: if each variant brought its own CPU butterfly
// the algorithm axis would be measuring the host code as much as the device
// code, and the comparison would be worthless. One host implementation keeps
// the CPU contribution identical across every cell in the matrix, so the axis
// isolates what it is supposed to isolate.
//
// The one implemented here is `sym` — conjugate-pair symmetry, the FFTW-codelet
// form — because the measured operation counts make it the best of the four at
// r=5 and r=7 and within 12% of the best at r=3:
//
//      r   direct   sym   winograd   rader      (multiplies + adds)
//      3       36    18       16        40
//      5      136    52       54       144
//      7      300   102      148       312
//
// Those are measured, not quoted: the counter is in odd_butterflies.py, which
// also checks every one of the twelve against numpy.fft.
//
// The correctness oracle for the project as a whole is NOT this file — it is
// the golden data from numpy (scripts/gen_data.py). This code only has to be
// right, not independent.
#pragma once
#include <cmath>
#include <cstdint>

namespace oddfft {

// Precomputed cos/sin of 2*pi*k/r for the radices we support. Built once.
struct RootTable {
    // c[r][k], s[r][k] for r in {2..7}, k < r. Indexed by radix directly, so
    // slots 0 and 1 are unused — a few floats of waste for index arithmetic
    // that cannot go wrong. Radix 2 and 4 are in here because a MIXED plan may
    // contain power-of-two stages (see host/plan.h).
    double c[8][8];
    double s[8][8];
    RootTable()
    {
        for (uint32_t r = 2; r <= 7; ++r) {
            for (uint32_t k = 0; k < 8; ++k) {
                const double a = 6.28318530717958647692 * (double)(k % (r ? r : 1)) / (double)r;
                c[r][k] = std::cos(a);
                s[r][k] = std::sin(a);
            }
        }
    }
};
inline const RootTable &roots()
{
    static const RootTable t;
    return t;
}

// The r-point DFT of (xr[m*stride], xi[m*stride]) for m = 0..r-1, in place,
// with W = exp(-2*pi*i/r).
//
// Conjugate-pair symmetry: the pairs (k, r-k) are summed and differenced once
// each, and then each output pair (p, r-p) is formed from the same cos and sin
// accumulations with one sign flip. That is what takes the multiply count from
// r^2-ish down to roughly r^2/2.
inline void butterfly_sym(uint32_t r, float *xr, float *xi, uint32_t stride)
{
    const uint32_t h = (r - 1u) / 2u;
    const RootTable &R = roots();

    float ar[4], ai[4], br[4], bi[4];        // h <= 3 for r <= 7
    for (uint32_t k = 1; k <= h; ++k) {
        const uint32_t a = k * stride, b = (r - k) * stride;
        ar[k] = xr[a] + xr[b];  ai[k] = xi[a] + xi[b];
        br[k] = xr[a] - xr[b];  bi[k] = xi[a] - xi[b];
    }

    const float x0r = xr[0], x0i = xi[0];
    float outr[8], outi[8];

    outr[0] = x0r; outi[0] = x0i;
    for (uint32_t k = 1; k <= h; ++k) { outr[0] += ar[k]; outi[0] += ai[k]; }

    for (uint32_t p = 1; p <= h; ++p) {
        double ur = x0r, ui = x0i, vr = 0.0, vi = 0.0;
        for (uint32_t k = 1; k <= h; ++k) {
            const double cc = R.c[r][(p * k) % r];
            const double ss = R.s[r][(p * k) % r];
            ur += cc * (double)ar[k];  ui += cc * (double)ai[k];
            vr += ss * (double)br[k];  vi += ss * (double)bi[k];
        }
        // W = exp(-2*pi*i/r), so y_p = u - i*v and y_{r-p} = u + i*v.
        outr[p]     = (float)(ur + vi);  outi[p]     = (float)(ui - vr);
        outr[r - p] = (float)(ur - vi);  outi[r - p] = (float)(ui + vr);
    }

    for (uint32_t m = 0; m < r; ++m) {
        xr[m * stride] = outr[m];
        xi[m * stride] = outi[m];
    }
}

// The naive r-point DFT. Present only so the self-test can check `sym` against
// something that shares none of its structure; never used in a measured path.
inline void butterfly_direct(uint32_t r, float *xr, float *xi, uint32_t stride)
{
    const RootTable &R = roots();
    float outr[8], outi[8];
    for (uint32_t p = 0; p < r; ++p) {
        double sr = 0.0, si = 0.0;
        for (uint32_t k = 0; k < r; ++k) {
            const double cc =  R.c[r][(p * k) % r];
            const double ss = -R.s[r][(p * k) % r];       // exp(-2*pi*i*pk/r)
            const double vr = xr[k * stride], vi = xi[k * stride];
            sr += cc * vr - ss * vi;
            si += cc * vi + ss * vr;
        }
        outr[p] = (float)sr; outi[p] = (float)si;
    }
    for (uint32_t m = 0; m < r; ++m) { xr[m * stride] = outr[m]; xi[m * stride] = outi[m]; }
}

// The butterfly the HOST uses, for any radix a plan may contain.
//
// `sym` covers the odd radices, which is what this project is about. It does
// NOT cover the even ones: its conjugate-pair decomposition has no place for
// the self-conjugate middle term k = r/2 that an even radix has, so for r = 4
// output 2 would never be written and for r = 2 nothing would be.
//
// Rather than grow `sym` a second code path for a case it was not designed
// for, the even radices go through the naive DFT. That is not a compromise on
// a measured result: the host butterfly is only ever used for the sub-floor
// prefix (a handful of stages with t < 8) and for the self-test. The butterfly
// ALGORITHMS being compared live in the kernels, where each has a file of its
// own -- a host-side shortcut here cannot contaminate that axis.
inline void butterfly_host(uint32_t r, float *xr, float *xi, uint32_t stride)
{
    if (r % 2u == 0u) butterfly_direct(r, xr, xi, stride);
    else              butterfly_sym(r, xr, xi, stride);
}

} // namespace oddfft
