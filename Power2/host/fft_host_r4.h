// host/fft_host_r4.h  (radix4)
//
// CHANGE RELATIVE TO THE PREVIOUS VERSION
// ----------------------------------------
// The CPU stages (host_butterfly4_stage_split / host_stockham4_stage_split)
// used to call std::cos/std::sin in double INSIDE the timed loop — 6 calls per
// butterfly, on every run. The NPU stages meanwhile took their twiddles out of
// the precomputed buffer W. The timing was asymmetric: the NPU part measured
// only the arithmetic, the CPU part the arithmetic plus the trigonometry. For
// N=4096 that is ~12000 sin/cos calls per run (~220 us), and that is what
// inflated cpu_us.
//
// Now the CPU stages have a twiddle table of their own as well: g_host_tw4,
// keyed by quarter. It is built ONCE before the timing
// (host_twiddle4_prewarm_N, called from print_bench_banner() in
// bench_harness.h), and inside the timed loop all that is left is the read
// from the table and the butterfly arithmetic — exactly as on the NPU.
//
// Separately: at the quarter=1 stage every twiddle equals 1 — that stage now
// goes down a branch with no complex multiplications at all (THE FORMULA IS
// THE SAME, it is just that w=1).
//
// The butterfly formula (DIT, radix4), for the group (x0,x1,x2,x3) at offsets
// (k+j, k+j+quarter, k+j+2*quarter, k+j+3*quarter) inside the block block=4*quarter:
//   w1=exp(-2pi*i*j/block), w2=w1^2, w3=w1^3   (twiddles on x1,x2,x3; x0 untwiddled)
//   x1'=x1*w1, x2'=x2*w2, x3'=x3*w3
//   y0 = x0 + x1' + x2' + x3'
//   y1 = x0 - i*x1' - x2' + i*x3'
//   y2 = x0 - x1' + x2' - x3'
//   y3 = x0 + i*x1' - x2' - i*x3'
//
// The Stockham version (out-of-place) differs only in the FIXED split on the
// read: x0..x3 are read with a stride of N/4 from the positions
// (0, N/4, N/2, 3N/4). The twiddle formula is identical to DIT — which is why
// the table is SHARED between them.
//
// IMPORTANT: N MUST be a power of FOUR (4, 16, 64, 256, 1024, 4096, ...).

#pragma once

#include <cstdint>
#include <cstring>
#include <cmath>
#include <map>
#include <vector>

#include "cell.h"

// All the radix-4-specific code lives in a namespace of its own: the three
// projects had the name FusedStageInfo and a couple of helpers in common, and
// we carry the function bodies over WITH NO edits, so as not to break proven
// code.
namespace r4 {

// The implementation tag for uniform output lines ([RESULT]/[VERIFY]).

// ============================================================================
// Digit-reverse permutation in base 4 (the radix2 bit-reversal's counterpart)
// ============================================================================
static void digit_reverse_permute4_split(float *xr, float *xi, uint32_t N)
{
    uint32_t m2 = 0, tmp = N;
    while (tmp > 1) { tmp >>= 2; ++m2; }   // the number of base-4 digits, N=4^m2

    std::vector<float> cr(xr, xr + N), ci(xi, xi + N);
    for (uint32_t i = 0; i < N; ++i) {
        uint32_t rev = 0, ii = i;
        for (uint32_t d = 0; d < m2; ++d) {
            rev = (rev << 2) | (ii & 3);
            ii >>= 2;
        }
        xr[rev] = cr[i];
        xi[rev] = ci[i];
    }
}

// ============================================================================
// The twiddle table for the CPU stages (the same status as the NPU's W
// buffer: computed before the timing, not part of it). The key is quarter,
// because base = -2pi/(4*quarter) and j runs over [0, quarter): everything is
// determined by quarter. The table is small: the CPU stages are the ones with
// quarter < TILE_SIZE.
// ============================================================================
struct HostTwiddle4 {
    std::vector<float> w1r, w1i, w2r, w2i, w3r, w3i;
};

static std::map<uint32_t, HostTwiddle4> g_host_tw4;

static const HostTwiddle4 &host_twiddle4(uint32_t quarter)
{
    std::map<uint32_t, HostTwiddle4>::iterator it = g_host_tw4.find(quarter);
    if (it != g_host_tw4.end()) return it->second;

    HostTwiddle4 t;
    t.w1r.resize(quarter); t.w1i.resize(quarter);
    t.w2r.resize(quarter); t.w2i.resize(quarter);
    t.w3r.resize(quarter); t.w3i.resize(quarter);

    double base = -6.28318530717958647692 / (double)(quarter << 2);
    for (uint32_t j = 0; j < quarter; ++j) {
        double a1 = base * j, a2 = base * 2 * j, a3 = base * 3 * j;
        t.w1r[j] = (float)std::cos(a1); t.w1i[j] = (float)std::sin(a1);
        t.w2r[j] = (float)std::cos(a2); t.w2i[j] = (float)std::sin(a2);
        t.w3r[j] = (float)std::cos(a3); t.w3i[j] = (float)std::sin(a3);
    }
    it = g_host_tw4.insert(std::make_pair(quarter, t)).first;
    return it->second;
}

// Warmup of the table for ALL the CPU stages of the given configuration
// (N, min_npu_quarter). Called once before warmup/bench — see
// print_bench_banner().
static uint32_t host_twiddle4_prewarm_N(uint32_t N, uint32_t min_npu_quarter)
{
    if (min_npu_quarter < 8) min_npu_quarter = 8;
    uint32_t built = 0;
    for (uint32_t quarter = 1; quarter <= N / 4; quarter <<= 2) {
        if (quarter >= min_npu_quarter) break;
        host_twiddle4(quarter);
        ++built;
    }
    return built;
}

#define HOST_TWIDDLE_PREWARM(N, TILE) host_twiddle4_prewarm_N((N), (TILE))

// ============================================================================
// DIT: CPU-fallback radix4 butterfly stage (in-place, for small quarter)
// The twiddles are read from the table — no sin/cos is called inside.
// ============================================================================
static void host_butterfly4_stage_split(float *xr, float *xi, uint32_t N, uint32_t quarter)
{
    uint32_t block = quarter << 2;
    const HostTwiddle4 &t = host_twiddle4(quarter);
    const float *w1r = t.w1r.data(), *w1i = t.w1i.data();
    const float *w2r = t.w2r.data(), *w2i = t.w2i.data();
    const float *w3r = t.w3r.data(), *w3i = t.w3i.data();

    for (uint32_t k = 0; k < N; k += block) {
        for (uint32_t j = 0; j < quarter; ++j) {
            uint32_t i0 = k + j, i1 = i0 + quarter, i2 = i1 + quarter, i3 = i2 + quarter;

            float x0r = xr[i0], x0i = xi[i0];
            float x1r = xr[i1], x1i = xi[i1];
            float x2r = xr[i2], x2i = xi[i2];
            float x3r = xr[i3], x3i = xi[i3];

            float t1r, t1i, t2r, t2i, t3r, t3i;
            if (quarter == 1) {           // j==0: w1=w2=w3=1, no multiplies needed
                t1r = x1r; t1i = x1i;
                t2r = x2r; t2i = x2i;
                t3r = x3r; t3i = x3i;
            } else {
                float a1r = w1r[j], a1i = w1i[j];
                float a2r = w2r[j], a2i = w2i[j];
                float a3r = w3r[j], a3i = w3i[j];
                t1r = x1r * a1r - x1i * a1i; t1i = x1r * a1i + x1i * a1r;
                t2r = x2r * a2r - x2i * a2i; t2i = x2r * a2i + x2i * a2r;
                t3r = x3r * a3r - x3i * a3i; t3i = x3r * a3i + x3i * a3r;
            }

            xr[i0] = x0r + t1r + t2r + t3r;
            xi[i0] = x0i + t1i + t2i + t3i;
            xr[i1] = x0r + t1i - t2r - t3i;
            xi[i1] = x0i - t1r - t2i + t3r;
            xr[i2] = x0r - t1r + t2r - t3r;
            xi[i2] = x0i - t1i + t2i - t3i;
            xr[i3] = x0r - t1i - t2r + t3i;
            xi[i3] = x0i + t1r - t2i - t3r;
        }
    }
}

// ============================================================================
// Stockham: CPU-fallback radix4 butterfly stage (OUT-OF-PLACE, a fixed
// N/4-split on the read). The same twiddle table.
// ============================================================================
static void host_stockham4_stage_split(const float *xr_in, const float *xi_in,
                                        float *xr_out, float *xi_out,
                                        uint32_t N, uint32_t quarter)
{
    uint32_t L = quarter << 2;
    uint32_t r = N / L;
    const HostTwiddle4 &t = host_twiddle4(quarter);
    const float *w1r = t.w1r.data(), *w1i = t.w1i.data();
    const float *w2r = t.w2r.data(), *w2i = t.w2i.data();
    const float *w3r = t.w3r.data(), *w3i = t.w3i.data();

    for (uint32_t k = 0; k < r; ++k) {
        for (uint32_t j = 0; j < quarter; ++j) {
            uint32_t i0 = k * quarter + j;
            uint32_t i1 = i0 + N / 4;
            uint32_t i2 = i0 + N / 2;
            uint32_t i3 = i0 + 3 * N / 4;
            uint32_t o0 = k * L + j;
            uint32_t o1 = o0 + quarter, o2 = o1 + quarter, o3 = o2 + quarter;

            float x0r = xr_in[i0], x0i = xi_in[i0];
            float x1r = xr_in[i1], x1i = xi_in[i1];
            float x2r = xr_in[i2], x2i = xi_in[i2];
            float x3r = xr_in[i3], x3i = xi_in[i3];

            float t1r, t1i, t2r, t2i, t3r, t3i;
            if (quarter == 1) {
                t1r = x1r; t1i = x1i;
                t2r = x2r; t2i = x2i;
                t3r = x3r; t3i = x3i;
            } else {
                float a1r = w1r[j], a1i = w1i[j];
                float a2r = w2r[j], a2i = w2i[j];
                float a3r = w3r[j], a3i = w3i[j];
                t1r = x1r * a1r - x1i * a1i; t1i = x1r * a1i + x1i * a1r;
                t2r = x2r * a2r - x2i * a2i; t2i = x2r * a2i + x2i * a2r;
                t3r = x3r * a3r - x3i * a3i; t3i = x3r * a3i + x3i * a3r;
            }

            xr_out[o0] = x0r + t1r + t2r + t3r;
            xi_out[o0] = x0i + t1i + t2i + t3i;
            xr_out[o1] = x0r + t1i - t2r - t3i;
            xi_out[o1] = x0i - t1r - t2i + t3r;
            xr_out[o2] = x0r - t1r + t2r - t3r;
            xi_out[o2] = x0i - t1i + t2i - t3i;
            xr_out[o3] = x0r - t1i - t2r + t3i;
            xi_out[o3] = x0i + t1r - t2i - t3r;
        }
    }
}

// ============================================================================
// Twiddle-factor precompute for the NPU — Packed layout, 3 factors (w1,w2,w3)
// x (re,im) one after another, each block `aligned` long:
//   w[0*aligned .. 1*aligned) = w1r    w[1*aligned .. 2*aligned) = w1i
//   w[2*aligned .. 3*aligned) = w2r    w[3*aligned .. 4*aligned) = w2i
//   w[4*aligned .. 5*aligned) = w3r    w[5*aligned .. 6*aligned) = w3i
// ============================================================================
static void compute_twiddle3_packed(float *w, uint32_t quarter, uint32_t aligned)
{
    std::memset(w, 0, aligned * 6 * sizeof(float));
    float *w1r = w, *w1i = w + aligned;
    float *w2r = w + 2 * aligned, *w2i = w + 3 * aligned;
    float *w3r = w + 4 * aligned, *w3i = w + 5 * aligned;

    uint32_t block = quarter << 2;
    double base = -6.28318530717958647692 / block;
    for (uint32_t j = 0; j < quarter; ++j) {
        double a1 = base * j, a2 = base * 2 * j, a3 = base * 3 * j;
        w1r[j] = (float)std::cos(a1); w1i[j] = (float)std::sin(a1);
        w2r[j] = (float)std::cos(a2); w2i[j] = (float)std::sin(a2);
        w3r[j] = (float)std::cos(a3); w3i[j] = (float)std::sin(a3);
    }
}


// ============================================================================
// Fused single-launch plan. THRESHOLD 8 (min_npu_quarter) — the same meaning
// as in radix2 (the round8 padding only overflows when valid<8).
// ============================================================================
struct FusedStageInfo {
    uint32_t quarter;
    uint32_t aligned;
    uint32_t wOffset;
};

static void build_fused_plan4(uint32_t m2, uint32_t min_npu_quarter,
                               std::vector<FusedStageInfo> &stages,
                               uint32_t &w_total, uint32_t &cpu_stages)
{
    stages.clear();
    w_total = 0;
    cpu_stages = 0;
    if (min_npu_quarter < 8) min_npu_quarter = 8;

    for (uint32_t s = 1; s <= m2; ++s) {
        uint32_t quarter = 1u << (2 * (s - 1));   // 4^(s-1)
        if (quarter < min_npu_quarter) { ++cpu_stages; continue; }
        FusedStageInfo info;
        info.quarter = quarter;
        info.aligned = ((quarter + 63) / 64) * 64;
        info.wOffset = w_total;
        w_total += info.aligned * 6;
        stages.push_back(info);
    }
}

static void fill_fused_buffers4_packed(float *w, const std::vector<FusedStageInfo> &stages)
{
    for (const auto &st : stages)
        compute_twiddle3_packed(w + st.wOffset, st.quarter, st.aligned);
}
} // namespace r4
