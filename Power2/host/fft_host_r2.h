// host/fft_host_r2.h  (radix2)
//
// CHANGE RELATIVE TO THE PREVIOUS VERSION
// ----------------------------------------
// The CPU stages (host_butterfly_stage_* / host_stockham_stage_*) used to call
// std::cos/std::sin in double INSIDE the timed loop, for every butterfly, on
// every run. The NPU stages meanwhile took their twiddles out of the
// precomputed buffer W. The timing was asymmetric: the NPU part measured only
// the arithmetic, the CPU part the arithmetic plus the trigonometry, and that
// is why cpu_us came out inflated.
//
// Now the CPU stages have a twiddle table of their own as well: g_host_tw2,
// keyed by half. It is built ONCE before the timing (host_twiddle2_prewarm_N,
// called from print_bench_banner() in bench_harness.h). Inside the timed loop
// there is only the read from the table and the butterfly arithmetic, exactly
// as on the NPU.
//
// The half=1 stage (j=0, w=1) goes down a branch with no complex
// multiplication at all.

#pragma once

#include <cstdint>
#include <cstring>
#include <cmath>
#include <map>
#include <vector>

#include "cell.h"

// All the radix-2-specific code lives in a namespace of its own: the three
// projects had the name FusedStageInfo and a couple of helpers in common, and
// we carry the function bodies over WITH NO edits, so as not to break proven
// code.
namespace r2 {

// The implementation tag for uniform output lines ([RESULT]/[VERIFY]).

// ============================================================================
// DIT: bit-reversal (Stockham does NOT use it)
// ============================================================================
static void bit_reverse_permute_split(float *xr, float *xi, uint32_t N)
{
    uint32_t m = 0, tmp = N;
    while (tmp > 1) { tmp >>= 1; ++m; }
    std::vector<float> cr(xr, xr + N), ci(xi, xi + N);
    for (uint32_t i = 0; i < N; ++i) {
        uint32_t rev = 0, ii = i;
        for (uint32_t j = 0; j < m; ++j) { rev = (rev << 1) | (ii & 1); ii >>= 1; }
        xr[rev] = cr[i]; xi[rev] = ci[i];
    }
}

static void bit_reverse_permute_interleaved(float *x, uint32_t N)
{
    uint32_t m = 0, tmp = N;
    while (tmp > 1) { tmp >>= 1; ++m; }
    std::vector<float> copy(x, x + 2 * N);
    for (uint32_t i = 0; i < N; ++i) {
        uint32_t rev = 0, ii = i;
        for (uint32_t j = 0; j < m; ++j) { rev = (rev << 1) | (ii & 1); ii >>= 1; }
        x[2 * rev] = copy[2 * i]; x[2 * rev + 1] = copy[2 * i + 1];
    }
}

// ============================================================================
// The twiddle table for the CPU stages (the same status as the NPU's W
// buffer: computed before the timing, not part of it). The key is half,
// because base = -2pi/(2*half) and j runs over [0, half).
// ============================================================================
struct HostTwiddle2 {
    std::vector<float> wr, wi;
};

static std::map<uint32_t, HostTwiddle2> g_host_tw2;

static const HostTwiddle2 &host_twiddle2(uint32_t half)
{
    std::map<uint32_t, HostTwiddle2>::iterator it = g_host_tw2.find(half);
    if (it != g_host_tw2.end()) return it->second;

    HostTwiddle2 t;
    t.wr.resize(half); t.wi.resize(half);
    double base = -6.28318530717958647692 / (double)(half << 1);
    for (uint32_t j = 0; j < half; ++j) {
        double ang = base * j;
        t.wr[j] = (float)std::cos(ang);
        t.wi[j] = (float)std::sin(ang);
    }
    it = g_host_tw2.insert(std::make_pair(half, t)).first;
    return it->second;
}

// Warmup of the table for ALL the CPU stages of the given configuration
// (N, min_npu_half).
static uint32_t host_twiddle2_prewarm_N(uint32_t N, uint32_t min_npu_half)
{
    if (min_npu_half < 8) min_npu_half = 8;
    uint32_t built = 0;
    for (uint32_t half = 1; half <= N / 2; half <<= 1) {
        if (half >= min_npu_half) break;
        host_twiddle2(half);
        ++built;
    }
    return built;
}

#define HOST_TWIDDLE_PREWARM(N, TILE) host_twiddle2_prewarm_N((N), (TILE))

// ============================================================================
// DIT: CPU-fallback butterfly stage (in-place, for small half)
// ============================================================================
static void host_butterfly_stage_split(float *xr, float *xi, uint32_t N, uint32_t half)
{
    uint32_t block = half << 1;
    const HostTwiddle2 &t = host_twiddle2(half);
    const float *wrT = t.wr.data(), *wiT = t.wi.data();

    for (uint32_t k = 0; k < N; k += block) {
        for (uint32_t j = 0; j < half; ++j) {
            uint32_t idxA = k + j, idxB = idxA + half;
            float ar = xr[idxA], br = xr[idxB], ai = xi[idxA], bi = xi[idxB];
            float tr, ti;
            if (half == 1) { tr = br; ti = bi; }          // w = 1
            else {
                float wr = wrT[j], wi = wiT[j];
                tr = br * wr - bi * wi; ti = br * wi + bi * wr;
            }
            xr[idxA] = ar + tr; xr[idxB] = ar - tr;
            xi[idxA] = ai + ti; xi[idxB] = ai - ti;
        }
    }
}

static void host_butterfly_stage_interleaved(float *x, uint32_t N, uint32_t half)
{
    uint32_t block = half << 1;
    const HostTwiddle2 &t = host_twiddle2(half);
    const float *wrT = t.wr.data(), *wiT = t.wi.data();

    for (uint32_t k = 0; k < N; k += block) {
        for (uint32_t j = 0; j < half; ++j) {
            uint32_t idxA = k + j, idxB = idxA + half;
            float ar = x[2*idxA], ai = x[2*idxA+1], br = x[2*idxB], bi = x[2*idxB+1];
            float tr, ti;
            if (half == 1) { tr = br; ti = bi; }
            else {
                float wr = wrT[j], wi = wiT[j];
                tr = br * wr - bi * wi; ti = br * wi + bi * wr;
            }
            x[2*idxA] = ar + tr; x[2*idxB] = ar - tr;
            x[2*idxA+1] = ai + ti; x[2*idxB+1] = ai - ti;
        }
    }
}

// ============================================================================
// Stockham: CPU-fallback butterfly stage (OUT-OF-PLACE, a fixed N/2 split on
// the read — see the derivation of the formula in
// kernels/r2_stockham_perstage_xs.cpp)
// ============================================================================
static void host_stockham_stage_split(const float *xr_in, const float *xi_in,
                                       float *xr_out, float *xi_out,
                                       uint32_t N, uint32_t half)
{
    uint32_t L = half << 1;
    uint32_t r = N / L;
    const HostTwiddle2 &t = host_twiddle2(half);
    const float *wrT = t.wr.data(), *wiT = t.wi.data();

    for (uint32_t k = 0; k < r; ++k) {
        for (uint32_t j = 0; j < half; ++j) {
            uint32_t idxIn = k * half + j, idxIn2 = idxIn + N / 2;
            uint32_t idxOut1 = k * L + j, idxOut2 = idxOut1 + half;
            float ar = xr_in[idxIn], ai = xi_in[idxIn];
            float br = xr_in[idxIn2], bi = xi_in[idxIn2];
            float wbr, wbi;
            if (half == 1) { wbr = br; wbi = bi; }
            else {
                float wr = wrT[j], wi = wiT[j];
                wbr = br * wr - bi * wi; wbi = br * wi + bi * wr;
            }
            xr_out[idxOut1] = ar + wbr; xi_out[idxOut1] = ai + wbi;
            xr_out[idxOut2] = ar - wbr; xi_out[idxOut2] = ai - wbi;
        }
    }
}

static void host_stockham_stage_interleaved(const float *x_in, float *x_out,
                                             uint32_t N, uint32_t half)
{
    uint32_t L = half << 1;
    uint32_t r = N / L;
    const HostTwiddle2 &t = host_twiddle2(half);
    const float *wrT = t.wr.data(), *wiT = t.wi.data();

    for (uint32_t k = 0; k < r; ++k) {
        for (uint32_t j = 0; j < half; ++j) {
            uint32_t idxIn = k * half + j, idxIn2 = idxIn + N / 2;
            uint32_t idxOut1 = k * L + j, idxOut2 = idxOut1 + half;
            float ar = x_in[2*idxIn], ai = x_in[2*idxIn+1];
            float br = x_in[2*idxIn2], bi = x_in[2*idxIn2+1];
            float wbr, wbi;
            if (half == 1) { wbr = br; wbi = bi; }
            else {
                float wr = wrT[j], wi = wiT[j];
                wbr = br * wr - bi * wi; wbi = br * wi + bi * wr;
            }
            x_out[2*idxOut1] = ar + wbr; x_out[2*idxOut1+1] = ai + wbi;
            x_out[2*idxOut2] = ar - wbr; x_out[2*idxOut2+1] = ai - wbi;
        }
    }
}

// ============================================================================
// Twiddle-factor precompute for the NPU — Packed layout
// ============================================================================
static void compute_twiddle_packed(float *w, uint32_t half, uint32_t aligned)
{
    float *wr = w, *wi = w + aligned;
    std::memset(w, 0, aligned * 2 * sizeof(float));
    uint32_t block = half << 1;
    double base = -6.28318530717958647692 / block;
    for (uint32_t j = 0; j < half; ++j) {
        double ang = base * j;
        wr[j] = (float)std::cos(ang);
        wi[j] = (float)std::sin(ang);
    }
}

// ============================================================================
// The result of a run
// ============================================================================
// No default member initializers — GCC 7.3.0 (in this environment) does not

// ============================================================================
struct FusedStageInfo {
    uint32_t half;
    uint32_t aligned;
    uint32_t wOffset;
};

// THRESHOLD 8, not 4 — see the analysis of bug 1 (the round8 padding only
// overflows when valid<8). Keep half<8 on the CPU always, the same way for DIT
// and for Stockham.
static void build_fused_plan(uint32_t m, uint32_t min_npu_half,
                              std::vector<FusedStageInfo> &stages,
                              uint32_t &w_total, uint32_t &cpu_stages)
{
    stages.clear();
    w_total = 0;
    cpu_stages = 0;
    if (min_npu_half < 8) min_npu_half = 8;

    for (uint32_t s = 1; s <= m; ++s) {
        uint32_t half = 1u << (s - 1);
        if (half < min_npu_half) { ++cpu_stages; continue; }
        FusedStageInfo info;
        info.half = half;
        info.aligned = ((half + 63) / 64) * 64;
        info.wOffset = w_total;
        w_total += info.aligned * 2;
        stages.push_back(info);
    }
}

static void fill_fused_buffers_packed(float *w, uint32_t *meta,
                                       const std::vector<FusedStageInfo> &stages)
{
    for (size_t i = 0; i < stages.size(); ++i) {
        const FusedStageInfo &st = stages[i];
        compute_twiddle_packed(w + st.wOffset, st.half, st.aligned);
        meta[i * 3 + 0] = st.half;
        meta[i * 3 + 1] = st.aligned;
        meta[i * 3 + 2] = st.wOffset;
    }
}
} // namespace r2
