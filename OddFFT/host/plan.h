// host/plan.h — OddFFT
//
// The stage plan, the twiddle tables and the digit-reverse permutation, all
// written against an ARBITRARY LIST OF RADICES rather than a single radix.
//
// WHY A LIST AND NOT A RADIX
// A pure power is the special case whose list is [r, r, ..., r]. Keeping the
// list general from the start costs nothing now and is what makes mixed radix
// (N = 3^a * 5^b * 7^c) a host-side change later rather than a kernel rewrite.
// The kernels never see the list: each launch is told one stage's (radix, t,
// block) and nothing else.
//
// !!! VERIFIED FOR UNIFORM *AND* MIXED LISTS !!!
// The plan below -- stage s has radix r_s, t_s = prod(r_0..r_{s-1}),
// block = t*r, butterfly (k,j) over x[k*block + j + m*t] with twiddle
// W_block^(m*j) -- is exact for every radix list tested, uniform or not,
// over radices {2, 3, 4, 5, 7}. See scratchpad derivation notes reproduced in
// docs/MIXED_RADIX.md.
//
// HOW THAT CHANGED, AND WHY THE EARLIER VERDICT WAS WRONG
// An earlier model of this plan reported mixed lists as broken: [3,5], [5,3],
// [3,5,7], [7,5,3] and [3,3,5] all came out with errors of order the signal
// itself, and build_plan() refused them. That verdict blamed the digit-reverse
// FORMULA. It was wrong: the formula was right and the DIRECTION it was
// applied in was not.
//
// The permutation needed is a GATHER:
//     z[i] = x[rev(i)]
// and what was being done was a SCATTER:
//     z[rev(i)] = x[i]
//
// For a UNIFORM list digit reversal is an involution (reversing the digit
// string twice returns it), so the gather and the scatter are the same map and
// the bug is invisible. For a non-uniform list it is not an involution,
// because the digit ranges differ, and the two maps diverge.
//
// This is worth recording as a method lesson, not just a fix: every test that
// passed had involution rate 1.00, and that included the only non-uniform list
// that passed, [3,5,3] -- which is a PALINDROME. A test suite of uniform lists
// plus one accidental palindrome cannot see this bug at all. The way it was
// finally found was not more algebra but a derivation: apply the stage plan to
// each basis vector, match the resulting columns against the DFT matrix and
// read the permutation off, then compare the two readings.
#pragma once
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace oddfft {

// One stage of the transform. `t` is this project's analogue of the
// half/quarter/eighth of the power-of-two project: the distance in elements
// between the r operands of a butterfly.
struct StageInfo {
    uint32_t radix;     // r_s
    uint32_t t;         // prod(r_0 .. r_{s-1}); 1 for the first stage
    uint32_t block;     // t * radix — the span of one k-block
    uint32_t nBlk;      // N / block
    uint32_t wOffset;   // where this stage's twiddles start in the W buffer
    bool     on_npu;    // false => computed on the host (see the floor below)
};

struct Plan {
    uint32_t              N = 0;
    std::vector<uint32_t> radices;
    std::vector<StageInfo> stages;
    uint32_t w_total   = 0;   // floats in the twiddle buffer
    uint32_t cpu_first = 0;   // how many leading stages run on the host
};

// The smallest operand distance t a stage may have and still be fed by the
// device.
//
// THIS WAS 8 AND IT WAS WRONG.
// Plain DataCopy does move whole 32-byte granules, so a run shorter than 8
// floats transfers nothing through it. That is a real constraint, and it was
// carried over from the power-of-two project as a floor of 8 on every stage.
//
// But these kernels do not use plain DataCopy. They use DataCopyPad, which is
// byte-granular by construction -- and nobody had tested it below the granule.
// AlignProbe's `padodd` mode now has: lengths 1, 2, 3, 5 and 7 at every
// unaligned base tested, 96 cases, all correct, with no write past the end.
//
// So the floor for THIS data path is 1, not 8. That matters more than it
// sounds: with a floor of 8 the leading stages of every odd radix (t = 1 and
// t = 3 for radix 3, t = 1 and t = 5 for radix 5, t = 1 and t = 7 for radix 7)
// were stranded on the host, and the first measured matrix put 80-93% of the
// total runtime there.
//
// POSSIBLE IS NOT PROFITABLE. A stage with t = 1 moves one float per run, and
// a transfer costs roughly the same whatever its length. Whether a tiny stage
// is better on the device or on the host is therefore an empirical question,
// not a structural one -- which is why the threshold is a command-line
// argument and a matrix axis (MIN_T) rather than a constant, and why this
// value is only its lower bound.
static constexpr uint32_t DATACOPY_FLOOR = 1;

// N must be the product of the list. For a pure power pass {r, r, ..., r}.
inline bool build_plan(uint32_t N, const std::vector<uint32_t> &radices,
                       uint32_t tile_threshold, Plan &out, std::string *err)
{
    out = Plan{};
    if (radices.empty()) { if (err) *err = "empty radix list"; return false; }

    // {2, 3, 4, 5, 7}: the odd radices this project exists for, plus 2 and 4 so
    // a mixed plan can reuse the power-of-two kernels for the stages where they
    // win. The plan is verified for every order and every mixture of these.
    // NOTE: having a radix in the plan is not the same as having a kernel for
    // it -- StageInfo::radix is what a launch dispatches on, and a stage whose
    // radix has no kernel registered has to run on the host.
    for (uint32_t r : radices) {
        if (r != 2 && r != 3 && r != 4 && r != 5 && r != 7) {
            if (err) *err = "radix " + std::to_string(r) +
                            " is not one of 2, 3, 4, 5, 7";
            return false;
        }
    }
    uint64_t prod = 1;
    for (uint32_t r : radices) prod *= r;
    if (prod != (uint64_t)N) {
        if (err) *err = "N=" + std::to_string(N) + " is not the product of the radix list";
        return false;
    }

    out.N = N;
    out.radices = radices;

    uint32_t t = 1;
    uint32_t w = 0;
    bool still_cpu = true;
    const uint32_t need = (tile_threshold > DATACOPY_FLOOR) ? tile_threshold : DATACOPY_FLOOR;
    for (uint32_t r : radices) {
        StageInfo st{};
        st.radix = r;
        st.t     = t;
        st.block = t * r;
        st.nBlk  = N / st.block;
        // A stage goes to the NPU once its row clears both the hardware floor
        // and the user's threshold. `still_cpu` keeps the CPU prefix
        // CONTIGUOUS: a plan that bounced between host and device mid-transform
        // would need a copy each way per stage and is never what we want.
        if (still_cpu && t < need) {
            st.on_npu = false;
            ++out.cpu_first;
        } else {
            still_cpu = false;
            st.on_npu = true;
        }
        st.wOffset = w;
        if (st.on_npu) {
            // (r-1) non-trivial multipliers per j, Re and Im on separate rows,
            // each row padded to a multiple of 8 so that the row's own
            // DataCopy starts on a 32-byte boundary. Note what is and is not
            // padded: the twiddle ROW is, because we allocate it; the operand
            // bases inside the signal are not, and need not be — the alignment
            // probe showed unaligned DataCopy works on this device, which is
            // the whole reason this project is shaped the way it is.
            w += 2u * (r - 1u) * (((t + 7u) / 8u) * 8u);
        }
        out.stages.push_back(st);
        t = st.block;
    }
    out.w_total = w;
    return true;
}

// Twiddles for the NPU stages, laid out so a kernel reads one contiguous run
// per (m, Re/Im): row (2*(m-1) + ri) of a stage holds W_block^(m*j) for
// j = 0..t-1, padded to a multiple of 8.
inline void fill_twiddles(float *w, const Plan &p)
{
    for (const StageInfo &st : p.stages) {
        if (!st.on_npu) continue;
        const uint32_t rowLen = ((st.t + 7u) / 8u) * 8u;
        float *base = w + st.wOffset;
        const double ang = -6.28318530717958647692 / (double)st.block;
        for (uint32_t m = 1; m < st.radix; ++m) {
            float *re = base + (2u * (m - 1u) + 0u) * rowLen;
            float *im = base + (2u * (m - 1u) + 1u) * rowLen;
            for (uint32_t j = 0; j < rowLen; ++j) {
                if (j < st.t) {
                    const double a = ang * (double)m * (double)j;
                    re[j] = (float)std::cos(a);
                    im[j] = (float)std::sin(a);
                } else {
                    // Padding. Zero rather than left uninitialised: the kernel
                    // reads the padded length, and the power-of-two project
                    // learned the hard way that 0 * NaN is NaN and a fresh
                    // allocation is full of NaN.
                    re[j] = 0.0f;
                    im[j] = 0.0f;
                }
            }
        }
    }
}

// Mixed-radix digit reversal: index i decomposes as i = sum_s d_s * t_s with
// t_s = prod(r_0..r_{s-1}); the reversed index puts digit d_s at the weight
// prod(r_{s+1}..r_{L-1}). Verified to be a bijection and verified correct for
// uniform lists AND for mixed ones, provided it is applied as a gather --
// see the header comment, which is where the direction is explained.
inline uint32_t digit_reverse_index(uint32_t i, const std::vector<uint32_t> &radices)
{
    const size_t L = radices.size();
    uint32_t j = 0, v = i;
    for (size_t s = 0; s < L; ++s) {
        const uint32_t d = v % radices[s];
        v /= radices[s];
        uint32_t weight = 1;
        for (size_t u = s + 1; u < L; ++u) weight *= radices[u];
        j += d * weight;
    }
    return j;
}

// A GATHER, z[i] = x[rev(i)], and NOT a scatter. The difference is invisible
// for a uniform radix list, where rev is an involution, and is the whole
// difference between right and wrong for a mixed one. See the header.
inline void digit_reverse_permute(float *xr, float *xi, const Plan &p)
{
    std::vector<float> ar(p.N), ai(p.N);
    for (uint32_t i = 0; i < p.N; ++i) {
        const uint32_t j = digit_reverse_index(i, p.radices);
        ar[i] = xr[j];
        ai[i] = xi[j];
    }
    for (uint32_t i = 0; i < p.N; ++i) { xr[i] = ar[i]; xi[i] = ai[i]; }
}

} // namespace oddfft
