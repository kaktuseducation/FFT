// tests/arb_selftest.cpp  (OddFFT)
//
// Does Bluestein and Rader compute the DFT?
//
// This test links host/impl_arblen.cpp ITSELF, in its CPU-debug configuration,
// and drives it through the registry exactly as main() does. It does not
// re-derive the algorithm in the test and compare two copies of the same
// reasoning: a copy would agree with the original about any index algebra both
// got wrong. The reference is a naive O(N^2) DFT in double precision, which
// shares nothing with the code under test.
//
// WHAT IT IS ACTUALLY CHECKING, beyond "the numbers match"
//   * Bluestein at composite and at prime N, including N where next_pow2(2N-1)
//     has an odd log2 -- that is the case where pow2_radices() has to put a
//     single radix-2 stage first, and where a radix-4 plan is not all radix 4.
//   * Rader's index algebra: the gather g^q, the scatter g^-p, the fold
//     c[p] + c[p+L], and the DC term. Three of those four are permutations,
//     and a permutation bug is invisible on an input that is symmetric -- so
//     the input here is deliberately asymmetric and complex.
//   * That Rader REFUSES a composite N instead of returning a wrong answer.
#include "../host/registry.h"
#include "../host/bench_harness.h"
#include "../host/arb_common.h"

#include <fstream>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

Cell         g_cell;
uint32_t     g_TILE_SIZE   = 64;
uint32_t     g_MIN_T       = 8;
BenchOutcome g_lastOutcome;
bool         g_haveOutcome = false;

void register_arblen(Registry &reg);

static int g_fail = 0;

static std::vector<float> make_input(uint32_t N)
{
    // Asymmetric and complex: a real or palindromic input would hide a
    // reversed permutation, which is the failure this test exists to catch.
    std::vector<float> in(2 * (size_t)N);
    for (uint32_t i = 0; i < N; ++i) {
        in[2*i]   = (float)std::sin(0.7 * (double)i + 0.3) + 0.01f * (float)i;
        in[2*i+1] = (float)std::cos(1.3 * (double)i + 1.1) - 0.02f * (float)(i % 7);
    }
    return in;
}

static void naive_dft(const std::vector<float> &in, uint32_t N, std::vector<double> &out)
{
    out.assign(2 * (size_t)N, 0.0);
    for (uint32_t k = 0; k < N; ++k) {
        double sr = 0.0, si = 0.0;
        for (uint32_t n = 0; n < N; ++n) {
            const double a = -2.0 * 3.14159265358979323846 * (double)k * (double)n / (double)N;
            const double c = std::cos(a), s = std::sin(a);
            sr += (double)in[2*n] * c - (double)in[2*n+1] * s;
            si += (double)in[2*n] * s + (double)in[2*n+1] * c;
        }
        out[2*k] = sr; out[2*k+1] = si;
    }
}

static void check(const Registry &reg, const std::string &vname, uint32_t N)
{
    std::string err;
    const VariantInfo *vi = reg.find(vname, &err);
    if (!vi) { printf("  [FAIL] %-24s N=%-6u  %s\n", vname.c_str(), N, err.c_str()); ++g_fail; return; }

    std::remove("./output/output_x.bin");
    const std::vector<float> in = make_input(N);
    g_cell.impl = vi->impl; g_cell.variant = vi->name; g_cell.prewarm = vi->prewarm;
    vi->run(N, 0, in, 0, 1, vi->name);

    std::vector<float> got(2 * (size_t)N, 0.0f);
    std::ifstream fin("./output/output_x.bin", std::ios::binary);
    const size_t want = got.size() * sizeof(float);
    if (!fin) { printf("  [FAIL] %-24s N=%-6u  no output file\n", vname.c_str(), N); ++g_fail; return; }
    fin.read(reinterpret_cast<char *>(got.data()), (std::streamsize)want);
    const size_t sz = (size_t)fin.gcount();
    if (sz != want) {
        printf("  [FAIL] %-24s N=%-6u  short output: %zu of %zu bytes\n",
               vname.c_str(), N, sz, want);
        ++g_fail; return;
    }

    std::vector<double> ref;
    naive_dft(in, N, ref);
    double maxrel = 0.0, norm = 0.0;
    for (uint32_t k = 0; k < N; ++k)
        norm = std::max(norm, std::max(std::fabs(ref[2*k]), std::fabs(ref[2*k+1])));
    if (norm <= 0.0) norm = 1.0;
    uint32_t worst = 0;
    for (uint32_t k = 0; k < 2 * N; ++k) {
        const double d = std::fabs((double)got[k] - ref[k]) / norm;
        if (d > maxrel) { maxrel = d; worst = k; }
    }
    // Single precision through two length-M transforms plus three elementwise
    // multiplies; 2e-4 relative to the spectrum's peak is the tolerance the
    // existing tests use for the same arithmetic depth.
    const bool ok = (maxrel < 2e-4);
    printf("  [%s] %-24s N=%-6u  max_rel=%.3e (worst k=%u)\n",
           ok ? " OK " : "FAIL", vname.c_str(), N, maxrel, worst / 2);
    if (!ok) ++g_fail;
}

int main()
{
    Registry reg;
    register_arblen(reg);
    printf("=== arbitrary-length self-test: %zu variants registered ===\n", reg.all.size());

    // --- the structural facts the constructions rest on ---
    printf("\n[1] number theory\n");
    struct { uint32_t n; bool prime; } pt[] = {
        {2,true},{3,true},{4,false},{9,false},{13,true},{17,true},{25,false},
        {31,true},{61,true},{97,true},{100,false},{1009,true},{1024,false}
    };
    for (auto &c : pt) {
        const bool g = oddfft::is_prime(c.n);
        if (g != c.prime) { printf("  [FAIL] is_prime(%u)=%d\n", c.n, (int)g); ++g_fail; }
    }
    for (uint32_t n : {3u,5u,7u,13u,17u,31u,61u,97u,1009u}) {
        const uint32_t g = oddfft::primitive_root(n);
        // g is a primitive root iff g^k mod n runs through 1..n-1 exactly once.
        std::vector<int> seen(n, 0);
        uint32_t bad = 0;
        for (uint32_t k = 0; k < n - 1u; ++k) {
            const uint32_t v = (uint32_t)oddfft::mod_pow(g, k, n);
            if (v == 0 || seen[v]) ++bad;
            seen[v] = 1;
        }
        if (bad) { printf("  [FAIL] primitive_root(%u)=%u is not primitive\n", n, g); ++g_fail; }
    }
    for (uint32_t m = 1; m <= (1u<<20); m <<= 1) {
        if (oddfft::next_pow2(m) != m) { printf("  [FAIL] next_pow2(%u)\n", m); ++g_fail; }
        // From m = 4 up: next_pow2(1) is 1, not 2, so m = 2 is not a case
        // where next_pow2(m-1) == m. That was a wrong expectation in this
        // test, not a defect in the function.
        if (m > 2 && oddfft::next_pow2(m - 1u) != m) { printf("  [FAIL] next_pow2(%u)\n", m-1u); ++g_fail; }
    }
    // pow2_radices must multiply back to M, and a radix-4 request on an odd
    // exponent must put the single radix-2 stage FIRST.
    for (uint32_t e = 1; e <= 18; ++e) {
        const uint32_t M = 1u << e;
        for (uint32_t r : {2u, 4u}) {
            auto rs = oddfft::pow2_radices(M, r);
            uint32_t prod = 1; for (uint32_t x : rs) prod *= x;
            if (prod != M) { printf("  [FAIL] pow2_radices(%u,%u) product %u\n", M, r, prod); ++g_fail; }
            if (r == 4 && (e & 1u) && rs.front() != 2u) {
                printf("  [FAIL] pow2_radices(%u,4) starts with %u, not 2\n", M, rs.front()); ++g_fail;
            }
        }
    }
    printf("  done\n");

    // --- Rader must refuse a composite N rather than answer it wrongly ---
    // Driven through the shipped run-function, not through a predicate written
    // for the test: what matters is that the REAL path leaves no output file,
    // because that is how main() turns a refusal into exit code 2 rather than
    // into a confident wrong answer verified against nothing.
    printf("\n[2] Rader refuses composite N\n");
    {
        std::string err;
        const VariantInfo *vi = reg.find("rad/rader_r2_ps_xs", &err);
        if (!vi) { printf("  [FAIL] %s\n", err.c_str()); ++g_fail; }
        else for (uint32_t N : {4u, 9u, 100u, 1024u}) {
            std::remove("./output/output_x.bin");
            const std::vector<float> in = make_input(N);
            g_cell.impl = vi->impl; g_cell.variant = vi->name; g_cell.prewarm = nullptr;
            vi->run(N, 0, in, 0, 1, vi->name);
            FILE *f = fopen("./output/output_x.bin", "rb");
            if (f) { fclose(f); printf("  [FAIL] Rader produced output for composite N=%u\n", N); ++g_fail; }
        }
    }
    printf("  done\n");

    // --- Bluestein: any N ---
    printf("\n[3] Bluestein against a naive DFT\n");
    // 12 and 100: composite. 13, 17, 61: prime (Bluestein does not care).
    // 24 -> M=64 (log2 even); 12 -> M=32 (log2 ODD, so a radix-4 plan needs
    // its leading radix-2 stage); 100 -> M=256; 63 -> M=128 (odd log2 again).
    for (uint32_t N : {7u, 12u, 13u, 17u, 24u, 61u, 63u, 100u, 128u, 210u}) {
        check(reg, "blu/bluestein_r2_ps_xs", N);
        check(reg, "blu/bluestein_r4_ps_xs", N);
    }

    // --- Rader: prime N only ---
    printf("\n[4] Rader against a naive DFT\n");
    // L = N-1 and M = next_pow2(2L-1): 7->L=6,M=16; 13->L=12,M=32 (odd log2);
    // 17->L=16,M=32; 31->L=30,M=64; 61->L=60,M=128 (odd log2); 97->L=96,M=256.
    for (uint32_t N : {3u, 5u, 7u, 13u, 17u, 31u, 61u, 97u, 211u}) {
        check(reg, "rad/rader_r2_ps_xs", N);
        check(reg, "rad/rader_r4_ps_xs", N);
    }

    printf("\n================================\n");
    printf("%s  (%d failure(s))\n", g_fail ? "FAILED" : "ALL PASS", g_fail);
    return g_fail ? 1 : 0;
}
