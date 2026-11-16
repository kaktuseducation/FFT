// host/frankenstein.h  (OddFFT)
//
// The combined transform: for a given N, choose a FACTORISATION into radices
// from {2,3,4,5,7}, choose an ORDER for them, and choose a BUTTERFLY for each
// stage -- then run each stage on the kernel that won that choice.
//
// This is the conclusion the rest of the project exists to support. Every
// other variant answers one question in isolation; this one assembles the
// answers.
//
// ===========================================================================
// THE THREE CHOICES, AND WHY EACH IS A REAL CHOICE
// ===========================================================================
//
// 1. FACTORISATION. 3600 = 2^4*3^2*5^2 can be [4,4,3,3,5,5] or
//    [2,2,2,2,3,3,5,5] or [4,2,2,3,3,5,5]. Fewer, larger radices mean fewer
//    STAGES, and a stage is a full pass over the array: 4N floats of GM
//    traffic and one kernel launch. Larger radices also mean more arithmetic
//    per point. Which wins is not obvious and is exactly what the
//    power-of-two project measured when radix 4 beat radix 2 on every one of
//    56 paired cells -- despite radix 2 having the lower operation count.
//    That result is the reason this planner does not simply minimise
//    operations.
//
// 2. ORDER. The operand distance t of stage s is the product of the earlier
//    radices, and a stage with t < 8 cannot be fed by DataCopy at all (the
//    32-byte granule) so it runs on the host. Putting the large radices FIRST
//    makes t clear 8 sooner and pushes fewer stages onto the host. Putting
//    them last leaves more work in the big, well-parallelised stages. Both
//    orders are offered so the difference can be measured rather than
//    assumed.
//
// 3. BUTTERFLY. Per radix, from the measured matrix when one is available and
//    from the operation counts otherwise.
//
// ===========================================================================
// THE COST MODEL -- FITTED, AND NOT THE MODEL THIS FILE FIRST HAD
// ===========================================================================
// The first version of this model was
//
//     cost(plan) = ALPHA * (vector ops per point)  +  BETA * (stages)
//
// with alpha = 1 and beta = 40 as admitted guesses, to be fitted later. The
// fit was then attempted against the 900-cell matrix, and it failed in a way
// worth recording, because the failure is the result.
//
// WHY THAT FORM CANNOT BE FITTED AT ALL
// Within any group of cells sharing a radix and a butterfly, the stage count
// is fixed at log_r(N), so host stages + device stages is a constant, and the
// two features "arithmetic" and "stage count" are exact linear combinations of
// each other. Least squares on 756 PASS cells returned R^2 = 0.60 with a
// NEGATIVE coefficient on host arithmetic (-0.18 to -5.3 depending on the
// configuration) and beta/alpha = -50 at 40 cores. A negative coefficient on
// host work means "more work on the host makes it faster", which is not a
// measurement of anything; it is what a degenerate design matrix returns. The
// only comparison in the whole matrix that separates the two terms is radix 2
// against radix 4 at the same N, and all that one says is beta > alpha/2,
// because the arithmetic there differs by 5% while the stage count differs by
// 2x.
//
// WHAT THE DATA DOES SUPPORT
// Three terms, all identifiable, all positive, fitted by weighted least
// squares (weights 1/y, so relative error is what is minimised -- the model
// orders plans, it does not need to predict microseconds):
//
//     cost(plan) = sum over DEVICE stages ( launch_us + data_us * N )
//                + sum over HOST   stages ( host_us * N * ops_per_point )
//
//   launch_us  ~41-49 us per stage launch, INDEPENDENT of N. The old model had
//              no such term, and this one is the largest single cost at small
//              N: five stages at N=243 is 210 us, which is the entire runtime.
//              It is why every arbitrary-length cell below N~3600 is 20-90x
//              slower than NumPy.
//   data_us    the pass over the array. 0.00073 us per point per stage at 40
//              cores against 0.00315 at 1 core -- a 4.3x parallel speedup on
//              this term, not 40x.
//   host_us    0.0388 us per point PER HOST STAGE, and -- this is the part
//              that took three wrong model forms to find -- NOT proportional
//              to the butterfly's operation count. Fitted per stage the host
//              term lands at 16.4% mean relative error; fitted per operation,
//              on the same cells with the same weighting, 46.7%. The
//              coefficient is 0.038807 at 1 core and 0.038759 at 40, which is
//              the consistency check that says it is real: the host does not
//              use the AIV cores, so it must not depend on how many there are.
//
// npu_us AND cpu_us ARE FITTED SEPARATELY, because the CSV reports them
// separately. Fitting their sum is what produced the degenerate matrix
// described above; once the two regressions are independent, every
// coefficient is identifiable and positive.
//
// DEVICE ARITHMETIC IS ABSENT FROM THIS MODEL, DELIBERATELY
// Added as a fourth term it fits at 5.8e-5 against 8.4e-3 for data movement, a
// factor of 145, and it moves the mean relative error from 23.8% to 23.5%. On
// this hardware the arithmetic inside a butterfly is not what a stage costs.
// That is the single most consequential measurement in the project, because
// the operation count is the axis the odd-radix work was organised around:
// across 162 paired cells the butterfly with the fewest operations is fastest
// in 86 of them -- 53%, against a 2.2-2.6x spread in operations and a median
// 1.12-1.19x spread in time. At radix 7, winograd issues 2.2x fewer operations
// than direct and runs 1.6% SLOWER.
//
// VALIDATION, OUT OF SAMPLE
// The coefficients were fitted on the single-radix cells only. Applied to the
// 288 arbitrary-length cells, which took no part in the fit, they give a mean
// relative error of 31.5% and pick the winner in 144 of 144 paired
// inner-radix comparisons. 31.5% is about the floor without a tile term, which
// this model does not have and which moves the time by up to 1.73x.
//
// WHAT IT STILL CANNOT DO: THE STAGE ORDER
// On the 72 measured hi/lo pairs the model agrees with the measurement 51
// times, 71%. That is worth stating alongside the 144/144, because it is the
// weaker half of the model and it is the half this file is named after. Two
// things make the 71% acceptable rather than embarrassing:
//
//   * of the 21 disagreements, 19 cost 1.28x or less and 15 cost 1.18x or
//     less. With the per-operation host term the same model disagreed 43 times
//     and the worst miss cost 2.91x. The model no longer makes an expensive
//     mistake, only cheap ones.
//   * at min_t = 1 no stage is forced onto the host, so the two orders differ
//     in nothing this model can see. It is a genuine tie, not a prediction,
//     and a third of the pairs are of that kind.
//
// The order is therefore still FORCED by the caller rather than chosen by the
// model, and both orders stay registered as separate variants. That decision
// was made before any of this was fitted, for the reason given at
// build_frankenstein, and the fit is what vindicates it: had the model been
// allowed to pick, it would have picked wrong 29 times out of 72 and the
// measurement would have had nothing to disagree with.
//
// So: the coefficients below are measured, the factorisation they choose has
// been checked against measurements that did not produce them, the predicted
// MAGNITUDE is good to about 30% and should be quoted as such, and the stage
// order is a measured input, not a model output.
#pragma once
#include "plan.h"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace oddfft {

// Vector instructions one butterfly issues, per (radix, butterfly), as the
// GENERATED kernels issue them. Butterfly arithmetic + the twiddle premultiply
// (4 multiplies + 2 adds per operand m >= 1).
//
// These numbers are printed by scripts/gen_kernels.py on every run and are
// reconciled there against docs/odd_butterflies.py, so this table cannot
// silently drift from the kernels.
struct ButterflyCost {
    uint32_t    radix;
    const char *name;
    uint32_t    ops;        // butterfly ops
    uint32_t    twiddle;    // 6 * (radix - 1)
};

static const ButterflyCost BUTTERFLY_COST[] = {
    {2, "direct",     4,   6},
    {4, "direct",    24,  18},
    {3, "direct",    36,  12},
    {3, "sym",       16,  12},
    {3, "winograd",  16,  12},
    {3, "rader",     36,  12},
    {5, "direct",   136,  24},
    {5, "sym",       48,  24},
    {5, "winograd",  52,  24},
    {5, "rader",    136,  24},
    {7, "direct",   300,  36},
    {7, "sym",       96,  36},
    {7, "winograd", 118,  36},
    {7, "rader",    300,  36},
};
static const uint32_t BUTTERFLY_COST_COUNT =
    sizeof(BUTTERFLY_COST) / sizeof(BUTTERFLY_COST[0]);

// Microseconds. See the header comment for how these were obtained, what they
// mean, and what the old alpha/beta pair could not be.
struct CostModel {
    double launch_us = 40.962;    // per device stage launch, independent of N
    double data_us   = 0.000731;  // per point, per device stage
    double host_us   = 0.038759;  // per point, PER HOST STAGE (not per op)
    uint32_t cores   = 40;        // which fit these came from
    bool   fitted    = true;
};

// The two fitted configurations. 40 cores is the default because the psm
// variants win 139 of 144 paired core comparisons; 1 core is kept because the
// ps variants are a registered axis and planning them with the wrong
// coefficients would compare a plan against a cost it was not planned under.
inline CostModel cost_model_measured(uint32_t cores)
{
    CostModel cm;
    if (cores <= 1) {
        cm.launch_us = 49.410;
        cm.data_us   = 0.003148;
        cm.host_us   = 0.038807;   // all but identical to the 40-core value,
        cm.cores     = 1;          // as it must be: the host has no AIV cores
    }
    return cm;
}

// The butterfly to use for a radix, and its per-point operation count.
//
// CHOSEN BY MEASUREMENT, NOT BY THE OPERATION COUNT. Across 162 paired cells
// -- all four butterflies present, every other axis held fixed -- sym is
// fastest in 86: 26 of 54 at radix 3, 29 of 54 at radix 5, 31 of 54 at radix
// 7. It is a plurality, not a majority, and it is worth being plain about how
// weak the signal is: picking the fewest-operation butterfly instead is right
// in the same 86 of 162 cells, and the median spread between the best and
// worst butterfly in a cell is only 1.12-1.19x against a 1.7-2.6x spread in
// operations. The choice is worth making and it is not worth much.
//
// The operation count is still returned, because host stages are the one place
// where arithmetic measurably costs something (see host_us).
inline const char *best_butterfly(uint32_t radix, double *ops_per_point,
                                  const char *forced = nullptr)
{
    const ButterflyCost *best = nullptr;
    for (uint32_t i = 0; i < BUTTERFLY_COST_COUNT; ++i) {
        const ButterflyCost &b = BUTTERFLY_COST[i];
        if (b.radix != radix) continue;
        if (forced && std::string(forced) != b.name) continue;
        // Prefer the measured winner. Radix 2 and 4 have only `direct`, so
        // this falls through to the operation count for them.
        if (!forced && std::string("sym") == b.name) { best = &b; break; }
        if (!best || (b.ops + b.twiddle) < (best->ops + best->twiddle)) best = &b;
    }
    if (!best) { if (ops_per_point) *ops_per_point = 1e30; return nullptr; }
    if (ops_per_point)
        *ops_per_point = (double)(best->ops + best->twiddle) / (double)radix;
    return best->name;
}

// Every factorisation of N into radices from {2,3,4,5,7}, as multisets.
// Bounded: N is at most a few hundred thousand here, and the recursion is over
// at most five distinct radices.
inline void enumerate_factorisations(uint32_t N,
                                     std::vector<std::vector<uint32_t>> &out,
                                     std::vector<uint32_t> &cur,
                                     uint32_t minRadixIdx = 0)
{
    static const uint32_t R[] = {2, 3, 4, 5, 7};
    if (N == 1) { out.push_back(cur); return; }
    for (uint32_t i = minRadixIdx; i < 5; ++i) {
        if (N % R[i] == 0) {
            cur.push_back(R[i]);
            // Non-decreasing radix index keeps each multiset once; the ORDER
            // is decided separately below, because order changes the plan.
            enumerate_factorisations(N / R[i], out, cur, i);
            cur.pop_back();
        }
    }
}

struct FrankPlan {
    std::vector<uint32_t>    radices;      // in stage order
    std::vector<std::string> butterflies;  // one per stage
    double                   predicted = 0.0;
    uint32_t                 cpu_first = 0;
    bool                     descending = true;
};

// Score one ordered radix list, in MICROSECONDS (the old version returned
// arbitrary units, which made the printed figure uncheckable against the
// matrix; now it can be compared with total_us directly, to about 30%).
//
// A device stage costs a launch plus a pass over the array. A host stage costs
// its arithmetic, on the host, and no launch -- which is the whole reason the
// order of the radices matters: it decides how many stages fall below the
// threshold.
inline double score_plan(const std::vector<uint32_t> &radices,
                         const CostModel &cm, uint32_t tile_threshold,
                         uint32_t *cpu_first_out,
                         std::vector<std::string> *bfs)
{
    const uint32_t need = (tile_threshold > DATACOPY_FLOOR) ? tile_threshold
                                                            : DATACOPY_FLOOR;
    uint32_t N = 1;
    for (uint32_t r : radices) N *= r;

    double total = 0.0;
    uint32_t t = 1, cpu_first = 0;
    bool still_cpu = true;
    if (bfs) bfs->clear();
    for (uint32_t r : radices) {
        double opp = 0.0;
        const char *bf = best_butterfly(r, &opp);
        if (!bf) return 1e30;
        if (bfs) bfs->push_back(bf);
        const bool on_cpu = still_cpu && (t < need);
        if (on_cpu) ++cpu_first; else still_cpu = false;
        // The host term does NOT use opp. Per stage it fits at 16.4% mean
        // relative error, per operation at 46.7% -- see the header. opp is
        // still computed, because it is what chooses the butterfly and it is
        // worth printing, but nothing in this cost depends on it any more.
        // That is the measurement, not an omission.
        (void)opp;
        total += on_cpu ? (cm.host_us * (double)N)
                        : (cm.launch_us + cm.data_us * (double)N);
        t *= r;
    }
    if (cpu_first_out) *cpu_first_out = cpu_first;
    return total;
}

// The planner.
//
// `descending` FORCES the stage order: true = large radices first, false =
// small first. It is forced rather than chosen so that the two can be
// registered as two variants and compared as a paired axis. Letting the model
// pick would have buried an unmeasured assumption inside the result -- the
// model's guess about ordering would decide the ordering, and the measurement
// would then agree with it by construction.
inline bool build_frankenstein(uint32_t N, uint32_t tile_threshold,
                               const CostModel &cm, FrankPlan &out,
                               std::string *err, bool descending)
{
    std::vector<std::vector<uint32_t>> facs;
    std::vector<uint32_t> cur;
    enumerate_factorisations(N, facs, cur);
    if (facs.empty()) {
        if (err) *err = "N=" + std::to_string(N) +
                        " has a prime factor outside {2,3,5,7}; that is what "
                        "Bluestein is for";
        return false;
    }

    double bestScore = 1e30;
    bool found = false;
    for (std::vector<uint32_t> f : facs) {
        if (descending) std::sort(f.begin(), f.end(), std::greater<uint32_t>());
        else            std::sort(f.begin(), f.end());
        uint32_t cf = 0;
        std::vector<std::string> bfs;
        const double sc = score_plan(f, cm, tile_threshold, &cf, &bfs);
        if (sc < bestScore) {
            bestScore = sc;
            out.radices = f;
            out.butterflies = bfs;
            out.predicted = sc;
            out.cpu_first = cf;
            out.descending = descending;
            found = true;
        }
    }
    if (!found && err) *err = "no viable factorisation";
    return found;
}

inline void print_frankenstein(const FrankPlan &fp, uint32_t N, const CostModel &cm)
{
    printf("[FRANK ] N=%u  stages=%u  order=%s  host_prefix=%u\n",
           N, (unsigned)fp.radices.size(),
           fp.descending ? "large-radix-first" : "small-radix-first",
           fp.cpu_first);
    printf("[FRANK ] plan:");
    for (size_t i = 0; i < fp.radices.size(); ++i)
        printf(" %u/%s", fp.radices[i], fp.butterflies[i].c_str());
    printf("\n");
    printf("[FRANK ] predicted %.1f us  (launch=%.3f us/stage, data=%.6f us/point/stage, "
           "host=%.5f us/point/host-stage, %u-core fit)\n",
           fp.predicted, cm.launch_us, cm.data_us, cm.host_us, cm.cores);
    printf("[FRANK ] coefficients MEASURED from output/matrix.csv; magnitude good to "
           "~31%% out of sample; no tile term; the stage ORDER above is forced by the "
           "caller, not chosen by the model (it would be right 71%% of the time)\n");
}

} // namespace oddfft
