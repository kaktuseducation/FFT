// host/host_fft.h — OddFFT
//
// One stage of the transform, on the host. Used for the sub-floor prefix of
// every variant (plan.h explains why that prefix exists and why it is
// contiguous), and by the self-test to run a whole transform on the host and
// check it against a reference DFT.
//
// The stage form is the one modelled and verified before any of this was
// written: stage s has radix r, t = prod(r_0..r_{s-1}), block = t*r, and the
// butterfly at (k, j) acts on x[k*block + j + m*t] for m = 0..r-1 with the
// operand m premultiplied by W_block^(m*j).
#pragma once
#include "butterflies.h"
#include "plan.h"
#include <cmath>
#include <cstdint>

namespace oddfft {

// One stage in place over the whole array.
inline void host_stage(const StageInfo &st, float *xr, float *xi, uint32_t N)
{
    const uint32_t r = st.radix, t = st.t, block = st.block;
    const double ang = -6.28318530717958647692 / (double)block;
    float br[8], bi[8];
    for (uint32_t k0 = 0; k0 < N; k0 += block) {
        for (uint32_t j = 0; j < t; ++j) {
            // Gather the r operands and apply the twiddles. m=0 has W^0=1 and
            // is never multiplied — the same saving the kernels make.
            for (uint32_t m = 0; m < r; ++m) {
                const uint32_t idx = k0 + j + m * t;
                if (m == 0) { br[m] = xr[idx]; bi[m] = xi[idx]; continue; }
                const double a  = ang * (double)m * (double)j;
                const double wc = std::cos(a), ws = std::sin(a);
                const double vr = xr[idx], vi = xi[idx];
                br[m] = (float)(wc * vr - ws * vi);
                bi[m] = (float)(wc * vi + ws * vr);
            }
            butterfly_host(r, br, bi, 1);
            for (uint32_t m = 0; m < r; ++m) {
                const uint32_t idx = k0 + j + m * t;
                xr[idx] = br[m];
                xi[idx] = bi[m];
            }
        }
    }
}

// The whole transform on the host: permutation, then every stage. The
// benchmark only ever runs the sub-floor prefix this way; this entry point is
// for the self-test.
inline void host_fft(const Plan &p, float *xr, float *xi)
{
    digit_reverse_permute(xr, xi, p);
    for (const StageInfo &st : p.stages) host_stage(st, xr, xi, p.N);
}

// Just the sub-floor prefix — what the benchmark actually calls before handing
// over to the device. Returns how many stages were done.
inline uint32_t host_prefix(const Plan &p, float *xr, float *xi)
{
    uint32_t done = 0;
    for (const StageInfo &st : p.stages) {
        if (st.on_npu) break;
        host_stage(st, xr, xi, p.N);
        ++done;
    }
    return done;
}

} // namespace oddfft
