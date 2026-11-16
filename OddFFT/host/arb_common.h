// host/arb_common.h  (OddFFT)
//
// The pieces both arbitrary-length constructions share: a power-of-two inner
// transform built out of this project's own plan and kernels, and the number
// theory Rader needs.
//
// WHY THE INNER TRANSFORM IS REBUILT HERE RATHER THAN REUSED AS WAS
// The earlier implementation of Bluestein and Rader carried its own inner FFT,
// and that inner FFT shuttled data to the host constantly. Per call at
// M = 4096 it performed, inside the timed region:
//
//   * 4 PCIe transfers for the bit-reversal permutation;
//   * a full host round trip -- 4 more transfers -- for EVERY stage whose
//     half-distance fell below the tile size, which at M = 4096 is six of
//     twelve stages, so 24 transfers;
//   * aclrtMalloc + aclrtFree of a twiddle buffer on EVERY device stage;
//   * 4 further transfers to conjugate and scale for the inverse.
//
// Bluestein calls it twice, so a single timed iteration performed on the order
// of sixty PCIe transfers and a dozen device allocation pairs. The published
// latencies for these algorithms therefore measure data movement between host
// and device, not the device.
//
// This version keeps everything on the device except one permutation: the
// twiddles for every stage are built once into a single buffer before the
// clock starts (exactly as the odd-radix path does), the conjugate and scale
// of the inverse are kernels, and no allocation happens inside a timed run.
//
// WHAT REMAINS ON THE HOST, AND WHY
// The digit-reverse permutation. A decimation-in-time transform needs its
// input permuted, and for Bluestein the data being permuted is produced fresh
// each call, so it cannot be hoisted out of the loop the way the odd-radix
// path hoists it. Doing it on the device would need either a scattered
// single-element transfer per point or a permutation kernel; doing it on the
// host costs four transfers per inner transform. Four is small against the
// sixty it replaces, so it stays, and the structural fix is recorded as future
// work: a decimation-in-FREQUENCY forward pass takes natural-order input and
// produces digit-reversed output, which the inverse decimation-in-time pass
// then consumes directly -- no permutation anywhere.
#pragma once
#include "plan.h"
#include "host_twiddles.h"
#include <cstdio>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "../kernels/kernel_table.h"
#include <cstdint>
#include <string>
#include <vector>

namespace oddfft {

inline uint32_t next_pow2(uint32_t v)
{
    uint32_t p = 1;
    while (p < v) p <<= 1;
    return p;
}

inline uint32_t log2_exact(uint32_t v)
{
    uint32_t k = 0;
    while ((1u << k) < v) ++k;
    return k;
}

inline bool is_prime(uint32_t n)
{
    if (n < 2) return false;
    for (uint32_t d = 2; d * d <= n; ++d) if (n % d == 0) return false;
    return true;
}

inline uint64_t mod_pow(uint64_t b, uint64_t e, uint64_t m)
{
    uint64_t r = 1;
    b %= m;
    while (e) {
        if (e & 1) r = (r * b) % m;
        b = (b * b) % m;
        e >>= 1;
    }
    return r;
}

// The smallest primitive root of a prime p.
inline uint32_t primitive_root(uint32_t p)
{
    std::vector<uint32_t> fac;
    uint32_t phi = p - 1, t = phi;
    for (uint32_t d = 2; d * d <= t; ++d) {
        if (t % d == 0) { fac.push_back(d); while (t % d == 0) t /= d; }
    }
    if (t > 1) fac.push_back(t);
    for (uint32_t g = 2; g < p; ++g) {
        bool ok = true;
        for (uint32_t f : fac) {
            if (mod_pow(g, phi / f, p) == 1) { ok = false; break; }
        }
        if (ok) return g;
    }
    return 0;
}

// A radix list for a power-of-two M: radix 4 throughout, with one radix-2
// stage FIRST when the exponent is odd.
//
// Radix 4 because the power-of-two matrix settled that question -- radix 4
// beat radix 2 on 46 of 48 paired comparisons, despite issuing more operations
// per point, because it halves the number of passes over the array. The
// leftover radix-2 stage goes first, where its operand distance is 1 and it is
// cheapest.
inline std::vector<uint32_t> pow2_radices(uint32_t M, uint32_t radix)
{
    std::vector<uint32_t> r;
    uint32_t m = log2_exact(M);
    if (radix == 2) {
        r.assign(m, 2u);
        return r;
    }
    if (m & 1u) r.push_back(2u);
    for (uint32_t i = 0; i < m / 2u; ++i) r.push_back(4u);
    return r;
}

// Everything a power-of-two inner transform needs, built once.
struct InnerFFT {
    Plan                  plan;
    std::vector<float>    tw;        // twiddles, host side
    std::vector<uint32_t> perm;      // perm[i] = digit_reverse_index(i)
    uint32_t              M = 0;
    uint32_t              radix = 4;
};

// Binary dump of a result buffer.
//
// WHY NOT common/data_utils.h's WriteFile(), which does the same thing: that
// header includes "acl/acl.h" unconditionally, so anything that touches it
// cannot be compiled without the CANN include path. tests/run_tests.sh is
// meant to run anywhere, with no device and no toolkit -- that is the whole
// point of having it -- and the arbitrary-length test links the shipped
// implementation rather than a copy of it. One of those two properties had to
// give, and this is the cheaper one: an eight-line write against a vendor
// header that pulls in the driver.
//
// Same semantics as WriteFile: truncate-or-create, and a short write is a
// failure rather than a partial file that verifies against nothing.
inline bool write_bin(const char *path, const void *buf, size_t bytes)
{
    if (buf == nullptr) { fprintf(stderr, "[ERROR ] write_bin: null buffer\n"); return false; }
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "[ERROR ] write_bin: cannot open %s\n", path); return false; }
    const size_t wrote = fwrite(buf, 1, bytes, f);
    const bool ok = (fclose(f) == 0) && (wrote == bytes);
    if (!ok) fprintf(stderr, "[ERROR ] write_bin: %s wrote %zu of %zu bytes\n", path, wrote, bytes);
    return ok;
}

inline bool build_inner(uint32_t M, uint32_t radix, uint32_t min_t,
                        InnerFFT &out, std::string *err)
{
    out = InnerFFT{};
    out.M = M;
    out.radix = radix;
    if (!build_plan(M, pow2_radices(M, radix), min_t, out.plan, err)) return false;
    out.tw.assign(out.plan.w_total ? out.plan.w_total : 1u, 0.0f);
    fill_twiddles(out.tw.data(), out.plan);
    out.perm.resize(M);
    for (uint32_t i = 0; i < M; ++i)
        out.perm[i] = digit_reverse_index(i, out.plan.radices);
    // The HOST twiddle table too, not only the device one. host_stage_fast()
    // reads from that table, and both the filter spectrum built here before
    // the clock starts and every sub-floor stage inside a run go through it.
    // Leaving it unbuilt does not fail loudly: the table would simply be
    // empty, and the filter spectrum -- hence every transform using it --
    // would come out as quiet garbage.
    build_host_twiddles(out.plan);
    return true;
}

// Apply the permutation as a GATHER: z[i] = x[perm[i]]. The direction matters
// and is not symmetric for a mixed radix list -- see host/plan.h.
// The digit-reverse permutation, as a GATHER. See host/plan.h for why the
// direction is the whole of mixed radix.
//
// PARALLEL, and the easiest parallel loop in the project: there is no
// arithmetic here at all, the destination index is the loop counter so two
// iterations never write the same element, and the source is read-only. The
// result is therefore bit-identical for any thread count by construction, not
// by argument.
//
// It is worth parallelising because it is what is left. With every stage of
// Bluestein's inner transform on the device and `stages_cpu` at zero, this
// gather still accounted for 1737.78 us of a 7105.96 us call at N = 10007 --
// 24.5%, and rising with N. It is the one host step the port could not remove,
// and until a decimation-in-frequency forward pass removes the permutation
// outright, spreading it is the available remedy.
//
// WHETHER IT ACTUALLY HELPS IS NOT YET MEASURED ON THE TARGET, and the one
// measurement in hand points the other way: on the two-core machine this was
// developed on, M = 32768 took 61.84 us serial and 80.41 us on two threads --
// a 1.30x LOSS. Two float arrays of 128 KB fit in one core's cache there, so
// the gather is cache-latency-bound and the fork is pure overhead.
//
// The reason to ship it anyway, labelled as a prediction: the same gather
// takes about 869 us on the Kunpeng host, 14x longer per element, which puts
// it in DRAM-latency territory rather than cache. A latency-bound gather gains
// from more threads because each thread adds outstanding misses, which is the
// opposite regime from the one measured above. The next arbitrary-length
// matrix run settles it; if cpu_us does not fall, raise this threshold above
// the largest M in use and the loop is serial again.
//
// The threshold mirrors host_prefix_fast.h: below a few thousand elements the
// fork costs more than the copy saves.
inline void apply_perm(const InnerFFT &f, const float *xr, const float *xi,
                       float *zr, float *zi)
{
    const uint32_t M = f.M;
    const uint32_t *perm = f.perm.data();
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(M >= 4096u)
#endif
    for (uint32_t i = 0; i < M; ++i) {
        const uint32_t j = perm[i];
        zr[i] = xr[j];
        zi[i] = xi[j];
    }
}

} // namespace oddfft
