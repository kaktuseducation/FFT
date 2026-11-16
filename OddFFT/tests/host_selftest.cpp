// tests/host_selftest.cpp  (OddFFT)
//
// Checks the HOST side against a naive O(N^2) DFT computed in double. No
// numpy and no device: this is the test a reviewer can run anywhere.
//
// What it covers:
//   1. butterfly_sym vs butterfly_direct for r = 3, 5, 7 — two independent
//      pieces of arithmetic for the same transform.
//   2. host_fft (permutation + all stages) vs the naive DFT, for every
//      (radix, N) the project supports.
//   3. host_prefix_tab (the TABLE-driven prefix, which is what the benchmark
//      actually times) against host_prefix (the on-the-fly one). These must
//      agree: if they ever drifted, the timed CPU path would be computing
//      something other than what the self-test blesses.
//   4. digit_reverse_index is a bijection.
//   5. MIXED radix lists against the naive DFT -- the case an earlier version
//      of this file asserted was unsupported. See the test's own comment for
//      why that verdict was wrong and what hid it.
//   6. a radix outside {2,3,4,5,7} is still refused.
#include "plan.h"
#include "butterflies.h"
#include "host_fft.h"
#include "host_twiddles.h"
#include "host_prefix_fast.h"
#include <cstring>

#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

using namespace oddfft;

static int failures = 0;

static std::string vec_str(const std::vector<uint32_t> &v)
{
    std::string out = "[";
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) out += ",";
        out += std::to_string(v[i]);
    }
    return out + "]";
}

static void report(const char *what, double rel, double tol)
{
    const bool ok = (rel < tol);
    if (!ok) ++failures;
    printf("%-52s rel=%.3e  %s\n", what, rel, ok ? "OK" : "<<<< FAIL");
}

// Naive DFT in double — the reference.
static void naive_dft(const std::vector<float> &xr, const std::vector<float> &xi,
                      std::vector<double> &yr, std::vector<double> &yi)
{
    const size_t N = xr.size();
    yr.assign(N, 0.0);
    yi.assign(N, 0.0);
    for (size_t k = 0; k < N; ++k) {
        double sr = 0.0, si = 0.0;
        for (size_t j = 0; j < N; ++j) {
            const double a = -6.28318530717958647692 * (double)k * (double)j / (double)N;
            const double c = std::cos(a), s = std::sin(a);
            sr += (double)xr[j] * c - (double)xi[j] * s;
            si += (double)xr[j] * s + (double)xi[j] * c;
        }
        yr[k] = sr;
        yi[k] = si;
    }
}

static double rel_err(const std::vector<float> &ar, const std::vector<float> &ai,
                      const std::vector<double> &br, const std::vector<double> &bi)
{
    double maxe = 0.0, scale = 0.0;
    for (size_t i = 0; i < ar.size(); ++i) {
        maxe  = std::max(maxe, std::abs((double)ar[i] - br[i]));
        maxe  = std::max(maxe, std::abs((double)ai[i] - bi[i]));
        scale = std::max(scale, std::abs(br[i]));
        scale = std::max(scale, std::abs(bi[i]));
    }
    return (scale > 0.0) ? maxe / scale : maxe;
}

int main()
{
    std::mt19937 rng(2024);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // ---- 1. sym vs direct ------------------------------------------------
    printf("--- 1. butterfly sym vs direct -------------------------------------\n");
    for (uint32_t r : {3u, 5u, 7u}) {
        float ar[8], ai[8], br[8], bi[8];
        for (uint32_t i = 0; i < r; ++i) {
            ar[i] = br[i] = dist(rng);
            ai[i] = bi[i] = dist(rng);
        }
        butterfly_sym(r, ar, ai, 1);
        butterfly_direct(r, br, bi, 1);
        double maxe = 0.0, scale = 0.0;
        for (uint32_t i = 0; i < r; ++i) {
            maxe  = std::max(maxe, (double)std::abs(ar[i] - br[i]));
            maxe  = std::max(maxe, (double)std::abs(ai[i] - bi[i]));
            scale = std::max(scale, (double)std::abs(br[i]));
            scale = std::max(scale, (double)std::abs(bi[i]));
        }
        char buf[64];
        snprintf(buf, sizeof(buf), "  r=%u  sym vs direct", r);
        report(buf, scale > 0 ? maxe / scale : maxe, 1e-5);
    }

    // ---- 2. host_fft vs naive DFT ----------------------------------------
    printf("--- 2. host_fft vs naive O(N^2) DFT in double ----------------------\n");
    struct Case { uint32_t radix, N; };
    const Case cases[] = {
        {3, 3}, {3, 9}, {3, 27}, {3, 81}, {3, 243},
        {5, 5}, {5, 25}, {5, 125}, {5, 625},
        {7, 7}, {7, 49}, {7, 343},
    };
    for (const Case &c : cases) {
        std::vector<uint32_t> radices;
        for (uint32_t v = c.N; v > 1; v /= c.radix) radices.push_back(c.radix);
        Plan p;
        std::string err;
        if (!build_plan(c.N, radices, 8, p, &err)) {
            printf("  radix=%u N=%-5u build_plan FAILED: %s\n", c.radix, c.N, err.c_str());
            ++failures;
            continue;
        }
        std::vector<float> xr(c.N), xi(c.N);
        for (uint32_t i = 0; i < c.N; ++i) { xr[i] = dist(rng); xi[i] = dist(rng); }
        std::vector<double> gr, gi;
        naive_dft(xr, xi, gr, gi);
        std::vector<float> yr(xr), yi(xi);
        host_fft(p, yr.data(), yi.data());
        char buf[64];
        snprintf(buf, sizeof(buf), "  radix=%u N=%-6u host_fft", c.radix, c.N);
        report(buf, rel_err(yr, yi, gr, gi), 1e-5);
    }

    // ---- 3. the table-driven prefix vs the on-the-fly one ---------------
    printf("--- 3. host_prefix_tab vs host_prefix (the timed path vs the blessed one)\n");
    for (const Case &c : cases) {
        std::vector<uint32_t> radices;
        for (uint32_t v = c.N; v > 1; v /= c.radix) radices.push_back(c.radix);
        Plan p;
        std::string err;
        if (!build_plan(c.N, radices, 64, p, &err)) { ++failures; continue; }
        build_host_twiddles(p);
        std::vector<float> ar(c.N), ai(c.N);
        for (uint32_t i = 0; i < c.N; ++i) { ar[i] = dist(rng); ai[i] = dist(rng); }
        std::vector<float> br(ar), bi(ai);
        const uint32_t n1 = host_prefix(p, ar.data(), ai.data());
        const uint32_t n2 = host_prefix_tab(p, br.data(), bi.data());
        double maxe = 0.0, scale = 0.0;
        for (uint32_t i = 0; i < c.N; ++i) {
            maxe  = std::max(maxe, (double)std::abs(ar[i] - br[i]));
            maxe  = std::max(maxe, (double)std::abs(ai[i] - bi[i]));
            scale = std::max(scale, (double)std::abs(ar[i]));
            scale = std::max(scale, (double)std::abs(ai[i]));
        }
        char buf[80];
        snprintf(buf, sizeof(buf), "  radix=%u N=%-6u prefix (%u stage(s))", c.radix, c.N, n1);
        if (n1 != n2) { printf("  stage-count mismatch %u vs %u\n", n1, n2); ++failures; }
        report(buf, scale > 0 ? maxe / scale : maxe, 1e-6);
    }

    // ---- 4. digit_reverse_index is a bijection --------------------------
    printf("--- 4. digit_reverse_index is a bijection --------------------------\n");
    for (const Case &c : cases) {
        std::vector<uint32_t> radices;
        for (uint32_t v = c.N; v > 1; v /= c.radix) radices.push_back(c.radix);
        std::vector<int> seen(c.N, 0);
        bool ok = true;
        for (uint32_t i = 0; i < c.N; ++i) {
            const uint32_t j = digit_reverse_index(i, radices);
            if (j >= c.N || seen[j]) { ok = false; break; }
            seen[j] = 1;
        }
        if (!ok) ++failures;
        printf("  radix=%u N=%-6u %s\n", c.radix, c.N, ok ? "OK" : "<<<< FAIL");
    }

    // ---- 4b. the SPECIALISED prefix against the generic one ------------
    // host_prefix_fast is what the benchmark actually times. It is a code
    // transformation, not a different algorithm, so it must agree with the
    // generic path EXACTLY -- not to a tolerance. A tolerance here would hide
    // precisely the kind of divergence the specialisation could introduce.
    printf("--- 4b. host_prefix_fast vs host_prefix_tab (must be exact) --------\n");
    for (const Case &c : cases) {
        std::vector<uint32_t> radices;
        for (uint32_t v = c.N; v > 1; v /= c.radix) radices.push_back(c.radix);
        Plan p2;
        std::string e2;
        if (!build_plan(c.N, radices, 8, p2, &e2)) { ++failures; continue; }
        build_host_twiddles(p2);
        std::vector<float> ar(c.N), ai(c.N);
        for (uint32_t i = 0; i < c.N; ++i) { ar[i] = dist(rng); ai[i] = dist(rng); }
        std::vector<float> br(ar), bi(ai);
        const uint32_t n1 = host_prefix_tab(p2, ar.data(), ai.data());
        const uint32_t n2 = host_prefix_fast(p2, br.data(), bi.data());
        bool exact = (n1 == n2);
        for (uint32_t i = 0; i < c.N && exact; ++i)
            if (ar[i] != br[i] || ai[i] != bi[i]) exact = false;
        if (!exact) ++failures;
        printf("  radix=%u N=%-6u %u stage(s)  %s\n", c.radix, c.N, n1,
               exact ? "bit-identical" : "<<<< FAIL: differs");
    }

    // ---- 5. MIXED radix lists vs the naive DFT -------------------------
    // This is what used to be a test that build_plan() REFUSED a mixed list.
    // The refusal was based on a wrong diagnosis: the digit-reverse formula was
    // correct and the direction it was applied in was not (a scatter where a
    // gather was needed). For a uniform list digit reversal is an involution,
    // so the two are the same map -- which is why every uniform test passed and
    // hid the bug, and why the one non-uniform list that also passed, [3,5,3],
    // passed for being a PALINDROME.
    //
    // So the list below is chosen to be adversarial about exactly that: mixed
    // orders, repeated radices, power-of-two radices mixed with odd ones, and
    // deliberately NO palindromes except one kept as a control.
    printf("--- 5. mixed radix lists vs naive DFT ------------------------------\n");
    {
        const std::vector<std::vector<uint32_t>> mixed = {
            {3, 5}, {5, 3}, {3, 7}, {7, 3}, {5, 7}, {7, 5},
            {3, 5, 7}, {7, 5, 3}, {3, 3, 5}, {5, 3, 3},
            {3, 5, 3},                       // palindrome: the control
            {2, 3}, {3, 2}, {4, 3}, {3, 4}, {2, 2, 3}, {2, 3, 5},
            {4, 4, 3}, {3, 3, 4}, {2, 3, 3, 5}, {4, 9 / 3, 3},
        };
        for (const auto &radices : mixed) {
            uint32_t N = 1;
            for (uint32_t r : radices) N *= r;
            Plan p2;
            std::string e2;
            if (!build_plan(N, radices, 8, p2, &e2)) {
                printf("  %-18s build_plan FAILED: %s\n",
                       vec_str(radices).c_str(), e2.c_str());
                ++failures;
                continue;
            }
            std::vector<float> xr(N), xi(N);
            for (uint32_t i = 0; i < N; ++i) { xr[i] = dist(rng); xi[i] = dist(rng); }
            std::vector<double> gr, gi;
            naive_dft(xr, xi, gr, gi);
            std::vector<float> yr(xr), yi(xi);
            host_fft(p2, yr.data(), yi.data());
            char buf[96];
            snprintf(buf, sizeof(buf), "  %-18s N=%-5u", vec_str(radices).c_str(), N);
            report(buf, rel_err(yr, yi, gr, gi), 1e-5);
        }
    }

    // ---- 6. a radix outside the supported set is still refused ----------
    printf("--- 6. an unsupported radix is refused -----------------------------\n");
    {
        Plan p2;
        std::string e2;
        if (build_plan(33, {3, 11}, 8, p2, &e2)) {
            printf("  <<<< FAIL: radix 11 was ACCEPTED\n");
            ++failures;
        } else {
            printf("  refused, as it must be: %s\n", e2.c_str());
        }
    }

    // ---- 7. two DIFFERENT plans with the same N must not share a twiddle
    // table ------------------------------------------------------------------
    // The host twiddle table is cached. The cache key used to be
    // (N, radices[0]), which is not a key: at N = 32 the plans [4,4,2] and
    // [4,2,4] agree on both, and so do [2,2,2,2,2] and [2,4,4]. The second
    // plan to ask then got the first plan's table and returned a transform
    // wrong by order 1 relative to the spectrum peak -- no warning, no loss of
    // precision, just a different answer.
    //
    // It never fired in the benchmark because one process runs one variant
    // with one plan, and it never fired in this test file because every case
    // here had a distinct N. It fired the moment a sweep asked for several
    // plans of the same length, which is what Bluestein's inner transform
    // does. The pairs below are chosen so each shares (N, radices[0]) with the
    // one before it, and each is run SECOND at least once.
    printf("--- 7. two plans of the same N do not share a twiddle table ---------\n");
    {
        const std::vector<std::vector<uint32_t>> fam[] = {
            { {4,4,2}, {4,2,4}, {2,4,4}, {2,2,2,2,2} },   // N = 32
            { {3,3,5,5}, {3,5,3,5}, {5,5,3,3}, {5,3,5,3} },   // N = 225
            { {2,2,7}, {2,7,2}, {7,2,2} },   // N = 28
        };
        std::mt19937 rng2(7);
        std::uniform_real_distribution<float> d2(-1.0f, 1.0f);
        for (const auto &fm : fam) {
            // Both orders, so whichever plan is asked second is also asked
            // first in the other pass: a stale table is only visible to the
            // second asker.
            for (int pass = 0; pass < 2; ++pass) {
                for (size_t k = 0; k < fm.size(); ++k) {
                    const std::vector<uint32_t> &rs = fm[pass ? (fm.size() - 1 - k) : k];
                    uint32_t N = 1;
                    for (uint32_t r : rs) N *= r;
                    Plan p3;
                    std::string e3;
                    if (!build_plan(N, rs, 1, p3, &e3)) {
                        printf("  <<<< FAIL: build_plan(%u, %s): %s\n",
                               N, vec_str(rs).c_str(), e3.c_str());
                        ++failures;
                        continue;
                    }
                    std::vector<float> xr(N), xi(N);
                    for (uint32_t i = 0; i < N; ++i) { xr[i] = d2(rng2); xi[i] = d2(rng2); }
                    std::vector<double> gr, gi;
                    naive_dft(xr, xi, gr, gi);
                    // THROUGH THE TABLE, not through host_fft(). host_fft()
                    // calls host_stage(), which computes its own cos/sin and
                    // never touches the cache -- so an earlier version of this
                    // very test passed with the broken cache key still in
                    // place. The negative control is what caught that: a check
                    // that does not exercise the cached path cannot certify
                    // the cache. Both table readers are run, because
                    // host_stage_fast falls back to host_stage_tab for radices
                    // it has no specialisation for.
                    build_host_twiddles(p3);
                    for (int which = 0; which < 2; ++which) {
                        std::vector<float> yr(xr), yi(xi);
                        digit_reverse_permute(yr.data(), yi.data(), p3);
                        for (size_t sIdx = 0; sIdx < p3.stages.size(); ++sIdx) {
                            if (which == 0) host_stage_fast(p3, sIdx, yr.data(), yi.data());
                            else            host_stage_tab(p3, sIdx, yr.data(), yi.data());
                        }
                        char buf[96];
                        snprintf(buf, sizeof(buf), "  pass%d %-7s %-16s N=%-5u",
                                 pass, which ? "tab" : "fast", vec_str(rs).c_str(), N);
                        report(buf, rel_err(yr, yi, gr, gi), 1e-5);
                    }
                }
            }
        }
    }

    // ---- 8. the parallel host prefix must be BIT-IDENTICAL to the serial one
    // ------------------------------------------------------------------------
    // host_prefix_fast.h parallelises the block loop. Blocks touch disjoint
    // elements, nothing is reduced across them, and the arithmetic inside a
    // block is unchanged -- so the result must match the serial run exactly,
    // not approximately. "Close enough" would mean the parallel pass is a
    // different algorithm, and the whole verification story rests on the host
    // path being the reference the device path is checked against.
    //
    // Compared as RAW BITS, not with a tolerance: a tolerance would pass a
    // reordering that happens to be small on this input and wrong on another.
    printf("--- 8. the parallel host prefix is bit-identical to the serial one --\n");
#ifdef _OPENMP
    {
        std::mt19937 rng3(99);
        std::uniform_real_distribution<float> d3(-1.0f, 1.0f);
        const int maxT = omp_get_max_threads();
        printf("  OpenMP reports %d thread(s) available\n", maxT);
        struct Case { uint32_t N; std::vector<uint32_t> r; };
        const Case cases[] = {
            {2187,  {3,3,3,3,3,3,3}},
            {3125,  {5,5,5,5,5}},
            {2401,  {7,7,7,7}},
            {4096,  {4,4,4,4,4,4}},
            {30240, {7,5,4,4,3,3,3,2}},
            {30240, {2,3,3,3,4,4,5,7}},
        };
        for (const Case &c : cases) {
            Plan p4;
            std::string e4;
            if (!build_plan(c.N, c.r, 1, p4, &e4)) {
                printf("  <<<< FAIL: build_plan(%u): %s\n", c.N, e4.c_str());
                ++failures;
                continue;
            }
            build_host_twiddles(p4);
            std::vector<float> x0r(c.N), x0i(c.N);
            for (uint32_t i = 0; i < c.N; ++i) { x0r[i] = d3(rng3); x0i[i] = d3(rng3); }

            // EVERY stage, not just the sub-floor ones: build_plan with a
            // threshold of 1 puts them all on the device, but host_stage_fast
            // is called directly here so the large-t stages -- the ones that
            // parallelise over j instead of over blocks -- are covered too.
            std::vector<float> sr(x0r), si(x0i), pr(x0r), pi(x0i);
            omp_set_num_threads(1);
            for (size_t sIdx = 0; sIdx < p4.stages.size(); ++sIdx)
                host_stage_fast(p4, sIdx, sr.data(), si.data());
            omp_set_num_threads(maxT);
            for (size_t sIdx = 0; sIdx < p4.stages.size(); ++sIdx)
                host_stage_fast(p4, sIdx, pr.data(), pi.data());

            size_t diff = 0;
            for (uint32_t i = 0; i < c.N; ++i) {
                if (std::memcmp(&sr[i], &pr[i], sizeof(float)) != 0) ++diff;
                if (std::memcmp(&si[i], &pi[i], sizeof(float)) != 0) ++diff;
            }
            printf("  %-18s N=%-6u stages=%zu  %s (%zu of %u words differ)\n",
                   vec_str(c.r).c_str(), c.N, p4.stages.size(),
                   diff ? "<<<< FAIL" : "identical", diff, 2u * c.N);
            if (diff) ++failures;
        }
        omp_set_num_threads(maxT);
    }
#else
    printf("  skipped: this binary was built without OpenMP, so there is no\n");
    printf("  parallel path to compare against. Build the tests with -fopenmp\n");
    printf("  to check it (tests/run_tests.sh does when the compiler has it).\n");
#endif

    printf("\n%s  (failures=%d)\n", failures ? "SOME CHECKS FAILED" : "ALL HOST CHECKS PASS",
           failures);
    return failures ? 1 : 0;
}
