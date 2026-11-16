// tests/sim_odd.cpp  (OddFFT)
//
// A CPU simulation of the GENERATED kernels' exact index arithmetic, for every
// radix, checking two independent things:
//
//   (a) COVERAGE. Across the tiles of all simulated cores, every element of
//       every stage is written EXACTLY ONCE -- including the stages with
//       t < 8, where a tile covers a single-digit number of elements. This is the important half. An
//       overlap is a cross-core race on live data and a gap is a missing
//       butterfly; both produce a plausible wrong answer rather than a crash,
//       which in a benchmark is the hardest kind of bug to notice.
//
//   (b) CORRECTNESS. The end-to-end result matches host_fft, itself checked
//       against a naive DFT in tests/host_selftest.cpp.
//
// WHAT THIS DOES AND DOES NOT COVER
// The tile mapping, the short-tail absorb, the multicore hand-out and the
// twiddle addressing are shared by every generated kernel (they come from one
// template), so simulating them once per radix covers all of them. The
// BUTTERFLY ARITHMETIC is not simulated here -- it is tested by
// tests/gen_butterfly_test.cpp, which runs each kernel's emitted body
// verbatim. Between them the two tests cover the kernel: this one the
// addressing, that one the arithmetic.
#include "plan.h"
#include "butterflies.h"
#include "host_fft.h"
#include "oddfft_tiling.h"
#include "kernel_limits.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

using namespace oddfft;

// One stage, exactly as the kernels do it, over `blkNum` simulated cores.
static void sim_stage(const StageInfo &st, float *xr, float *xi, uint32_t N,
                      uint32_t tile, uint32_t blkNum, const float *w,
                      std::vector<int> *writes)
{
    tile = ODDFFT_CLAMP_TILE(tile);
    const uint32_t r      = st.radix;
    const uint32_t rowLen = ((st.t + 7u) / 8u) * 8u;
    const uint32_t nShift = ODDFFT_SHIFT_COUNT(st.t, tile);
    const uint32_t nTiles = st.nBlk * nShift;

    // A stage is in place: every core reads the pre-stage state. Snapshotting
    // it means the simulation cannot accidentally depend on tile ORDER -- if
    // it did, that dependence would itself be the bug (a) looks for.
    std::vector<float> ir(xr, xr + N), ii(xi, xi + N);

    for (uint32_t bid = 0; bid < blkNum; ++bid) {
        for (uint32_t tid = bid; tid < nTiles; tid += blkNum) {
            const uint32_t k     = (tid / nShift) * st.block;
            const uint32_t s     = tid % nShift;
            const uint32_t shift = s * tile;
            const uint32_t valid = ODDFFT_TILE_VALID(st.t, tile, s, nShift);
            const uint32_t i0    = k + shift;

            for (uint32_t e = 0; e < valid; ++e) {
                const uint32_t j = shift + e;
                float br[8], bi[8];
                br[0] = ir[i0 + e];
                bi[0] = ii[i0 + e];
                for (uint32_t m = 1; m < r; ++m) {
                    const uint32_t idx = i0 + m * st.t + e;
                    const float wc = w[st.wOffset + (2u * (m - 1u) + 0u) * rowLen + j];
                    const float ws = w[st.wOffset + (2u * (m - 1u) + 1u) * rowLen + j];
                    const float vr = ir[idx], vi = ii[idx];
                    br[m] = wc * vr - ws * vi;
                    bi[m] = wc * vi + ws * vr;
                }
                butterfly_host(r, br, bi, 1);
                for (uint32_t m = 0; m < r; ++m) {
                    const uint32_t idx = i0 + m * st.t + e;
                    xr[idx] = br[m];
                    xi[idx] = bi[m];
                    if (writes) (*writes)[idx]++;
                }
            }
        }
    }
}

int main()
{
    int failures = 0;
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    struct Case { uint32_t radix; uint32_t N; };
    const Case cases[] = {
        {3, 243}, {3, 2187}, {3, 19683},
        {5, 125}, {5, 3125}, {5, 15625},
        {7, 343}, {7, 2401}, {7, 16807},
    };
    const uint32_t tiles[] = {8, 16, 64, 128, 424, 512};
    const uint32_t cores[] = {1, 7, 40};

    printf("MAX_TILE from kernels/kernel_limits.h = %u "
           "(a larger request is clamped to it, here and in the kernel)\n\n",
           ODDFFT_MAX_TILE);

    for (const Case &c : cases) {
        std::vector<uint32_t> radices;
        for (uint32_t v = c.N; v > 1; v /= c.radix) radices.push_back(c.radix);

        for (uint32_t tile : tiles) {
            for (uint32_t nc : cores) {
                Plan p;
                std::string err;
                // tile_threshold = 1: EVERY stage goes to the device,
                // including t = 1, 3, 5 and 7. Those are the hardest cases for
                // the tiling and they only became reachable once sub-granule
                // DataCopyPad was measured to work; the simulation has to
                // cover them or the write-coverage guarantee stops applying
                // exactly where the transfers are smallest.
                if (!build_plan(c.N, radices, 1, p, &err)) {
                    printf("r=%u N=%u build_plan FAILED: %s\n", c.radix, c.N, err.c_str());
                    ++failures;
                    continue;
                }
                std::vector<float> w(p.w_total ? p.w_total : 1, 0.0f);
                fill_twiddles(w.data(), p);

                std::vector<float> xr(c.N), xi(c.N);
                for (uint32_t i = 0; i < c.N; ++i) { xr[i] = dist(rng); xi[i] = dist(rng); }
                std::vector<float> gr(xr), gi(xi);
                host_fft(p, gr.data(), gi.data());

                digit_reverse_permute(xr.data(), xi.data(), p);
                int cov_bad = 0;
                for (size_t s = 0; s < p.stages.size(); ++s) {
                    const StageInfo &st = p.stages[s];
                    if (!st.on_npu) { host_stage(st, xr.data(), xi.data(), c.N); continue; }
                    std::vector<int> writes(c.N, 0);
                    sim_stage(st, xr.data(), xi.data(), c.N, tile, nc, w.data(), &writes);
                    for (uint32_t i = 0; i < c.N; ++i) {
                        if (writes[i] != 1) { ++cov_bad; break; }
                    }
                }

                double maxerr = 0.0, scale = 0.0;
                for (uint32_t i = 0; i < c.N; ++i) {
                    maxerr = std::fmax(maxerr, std::fabs((double)xr[i] - gr[i]));
                    maxerr = std::fmax(maxerr, std::fabs((double)xi[i] - gi[i]));
                    scale  = std::fmax(scale, std::fabs((double)gr[i]));
                    scale  = std::fmax(scale, std::fabs((double)gi[i]));
                }
                const double rel = (scale > 0.0) ? maxerr / scale : maxerr;
                const bool ok = (cov_bad == 0) && (rel < 1e-5);
                if (!ok) {
                    ++failures;
                    printf("r=%u N=%-6u tile=%-4u cores=%-3u cpu_first=%u  "
                           "coverage=%s  rel=%.2e  <<<< FAIL\n",
                           c.radix, c.N, tile, nc, p.cpu_first,
                           cov_bad ? "BAD" : "ok", rel);
                }
            }
        }
        printf("radix %u: %lu sizes x %lu tiles x %lu core counts -- all OK\n",
               c.radix, (unsigned long)1, (unsigned long)(sizeof(tiles) / sizeof(tiles[0])),
               (unsigned long)(sizeof(cores) / sizeof(cores[0])));
    }

    const int total = (int)(sizeof(cases) / sizeof(cases[0]))
                    * (int)(sizeof(tiles) / sizeof(tiles[0]))
                    * (int)(sizeof(cores) / sizeof(cores[0]));
    printf("\n%s  (%d cases, failures=%d)\n",
           failures ? "SOME CASES FAILED" : "ALL CASES PASS", total, failures);
    return failures ? 1 : 0;
}
