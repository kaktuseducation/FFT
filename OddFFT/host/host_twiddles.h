// host/host_twiddles.h  (OddFFT)
//
// A PRECOMPUTED twiddle table for the stages that run on the host.
//
// WHY THIS FILE EXISTS
// host_fft.h's host_stage() computes cos/sin inside the j loop. That is fine
// for a self-test but not for a benchmark: it would put trigonometry inside
// the timed CPU region while the NPU stages read a table built before the
// clock started. The power-of-two project set the rule explicitly — "the
// trigonometry is outside the timing on both sides" — and a comparison of
// radices is exactly where breaking it would do damage, because the number of
// sub-floor host stages is itself one of the things being compared (three for
// radix 3, two for radix 5 and 7).
//
// The table is built once per (N, radix) and cached, so repeated runs in one
// process (warmup + timed runs) pay for it once, outside run_benchmark.
#pragma once
#include "plan.h"
#include "butterflies.h"
#include <cmath>
#include <cstdint>
#include <vector>

namespace oddfft {

// Row layout per host stage, mirroring the device table in plan.h but WITHOUT
// the 8-float padding: nothing here is DataCopy'd, so padding would be waste.
// Row (2*(m-1) + ri) holds W_block^(m*j) for j = 0..t-1.
struct HostTwiddles {
    uint32_t              N = 0;
    // The WHOLE radix list, not just the first radix. An earlier version kept
    // only radices[0] and treated (N, radices[0]) as the cache key, which is
    // not a key at all: at N = 32 the plans [4,4,2] and [4,2,4] share it, and
    // so do [2,2,2,2,2] and [2,4,4]. The second plan to ask then silently got
    // the first plan's table and produced a wrong transform with no error --
    // errors of order 1 relative to the spectrum peak, not a loss of
    // precision. It never fired in the benchmark because one process runs one
    // variant with one plan; it fired immediately in a test that sweeps many.
    std::vector<uint32_t> radices;
    std::vector<uint32_t> offset;   // per stage index, into `w`
    std::vector<float>    w;
    bool built = false;
};

inline HostTwiddles &host_twiddle_cache()
{
    static HostTwiddles tw;
    return tw;
}

inline void build_host_twiddles(const Plan &p)
{
    HostTwiddles &tw = host_twiddle_cache();
    if (tw.built && tw.N == p.N && tw.radices == p.radices) return;

    tw = HostTwiddles{};
    tw.N = p.N;
    tw.radices = p.radices;
    // EVERY stage gets a row set, not just the sub-floor ones. The benchmark
    // only ever calls host_prefix_tab, which stops at the first NPU stage, but
    // the CPU-debug path runs the WHOLE transform on the host and would
    // otherwise index past the end of `w` for the device stages — an
    // out-of-bounds read that showed up as inf in the first self-test rather
    // than as a crash. The extra rows cost about 2N floats in total.
    uint32_t total = 0;
    for (const StageInfo &st : p.stages) {
        tw.offset.push_back(total);
        total += 2u * (st.radix - 1u) * st.t;
    }
    tw.w.assign(total ? total : 1u, 0.0f);

    for (size_t s = 0; s < p.stages.size(); ++s) {
        const StageInfo &st = p.stages[s];
        float *base = tw.w.data() + tw.offset[s];
        const double ang = -6.28318530717958647692 / (double)st.block;
        for (uint32_t m = 1; m < st.radix; ++m) {
            float *re = base + (2u * (m - 1u) + 0u) * st.t;
            float *im = base + (2u * (m - 1u) + 1u) * st.t;
            for (uint32_t j = 0; j < st.t; ++j) {
                const double a = ang * (double)m * (double)j;
                re[j] = (float)std::cos(a);
                im[j] = (float)std::sin(a);
            }
        }
    }
    tw.built = true;
}

// One host stage using the table. Same arithmetic as host_stage(), with the
// cos/sin calls replaced by two loads.
inline void host_stage_tab(const Plan &p, size_t s, float *xr, float *xi)
{
    const StageInfo &st = p.stages[s];
    const HostTwiddles &tw = host_twiddle_cache();
    const float *base = tw.w.data() + tw.offset[s];
    const uint32_t r = st.radix, t = st.t, block = st.block;
    float br[8], bi[8];
    for (uint32_t k0 = 0; k0 < p.N; k0 += block) {
        for (uint32_t j = 0; j < t; ++j) {
            br[0] = xr[k0 + j];
            bi[0] = xi[k0 + j];
            for (uint32_t m = 1; m < r; ++m) {
                const uint32_t idx = k0 + j + m * t;
                const float wc = base[(2u * (m - 1u) + 0u) * t + j];
                const float ws = base[(2u * (m - 1u) + 1u) * t + j];
                const float vr = xr[idx], vi = xi[idx];
                br[m] = wc * vr - ws * vi;
                bi[m] = wc * vi + ws * vr;
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

// The sub-floor prefix, from the table. Returns how many stages it did.
inline uint32_t host_prefix_tab(const Plan &p, float *xr, float *xi)
{
    uint32_t done = 0;
    for (size_t s = 0; s < p.stages.size(); ++s) {
        if (p.stages[s].on_npu) break;
        host_stage_tab(p, s, xr, xi);
        ++done;
    }
    return done;
}

} // namespace oddfft
