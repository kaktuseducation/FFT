// host/fft_host_r8.h  (radix8)
//
// IMPORTANT: N MUST be a power of EIGHT (8, 64, 512, 4096, 32768, 262144, ...).
// Out of the list of N this project normally uses for Radix2/Radix4
// (256, 1024, 4096, 16384, 65536, 262144), the ONLY powers of 8 are
// 4096 (=8^4) and 262144 (=8^6). On any other N host/main.cpp exits with an
// explicit error — mixed-radix (radix8 + cleanup stages) is NOT implemented in
// this version; that is a deliberate limitation, not a forgotten case.
//
// ============================================================================
// The mathematics of the radix-8 DIT butterfly
// ============================================================================
// For the group (x0..x7) at the positions (k+j, k+j+eighth, ..., k+j+7*eighth)
// inside the block block = 8*eighth, with the base twiddle
// W_block = exp(-2*pi*i/block):
//
//   t0 = x0                       (untwiddled)
//   t_m = x_m * W_block^(m*j),    m = 1..7      (7 UNIQUE twiddles per group)
//
// Then the "combine" — an honest 8-point DFT of the vector (t0..t7) by the
// fixed eighth roots of unity W8^k = exp(-2*pi*i*k/8), k=0..7:
//
//   y_p = sum_{k=0}^{7} t_k * W8^(p*k mod 8),   p = 0..7
//
// W8^k takes only the values {0, +-1, +-c, +-i, +-i*c} with c = 1/sqrt(2) —
// that is, the combine step has NO dependence on the j/k of the current stage:
// it is ONE and the same fixed 8x8 matrix for absolutely every group on every
// stage. That fact is what the Cube variant uses (see build_dft8_matrices()
// below) — ONLY the combine step goes there, while the twiddle multiplication
// t_m = x_m*W_block^(mj) stays on the Vector engine (it is honestly
// data-dependent on j and on the stage, and there is no folding it into the
// Cube in one go without a separate matrix for every j).
//
// The combine implementation comes out of the classical (and proven)
// Cooley-Tukey factorization radix-8 = radix-4 (over the even t0,t2,t4,t6 and
// the odd t1,t3,t5,t7) + a final radix-2 combine step with the twiddles
// W8^0..W8^3:
//   A = DFT4(t0,t2,t4,t6)              -> A0..A3
//   B = DFT4(t1,t3,t5,t7)              -> B0..B3
//   B'_p = B_p * W8^p   (p=0..3; W8^0=1 — trivial, no multiplication needed)
//   Y_p     = A_p + B'_p        (p=0..3)
//   Y_{p+4} = A_p - B'_p        (p=0..3)
// DFT4(a,b,c,d) is THE SAME formula already validated in
// kernels/r4_dit_fused_xs.cpp (the base case, with no extra twiddles):
//   Y0 = a+b+c+d
//   Y1 = a -i*b -c +i*d   =>  Re= Re(a)+Im(b)-Re(c)-Im(d),  Im= Im(a)-Re(b)-Im(c)+Re(d)
//   Y2 = a-b+c-d
//   Y3 = a +i*b -c -i*d   =>  Re= Re(a)-Im(b)-Re(c)+Im(d),  Im= Im(a)+Re(b)-Im(c)-Re(d)
//
// This is ONE fused function (one Compute() on the Vector engine, with no
// intermediate kernel launches and no round-trips through GM between the
// "sub-radix2 stages") — that is exactly what was meant by "not three nested
// radix-2 stages": the whole 8-point butterfly is one atomic computation in
// registers/UB, not three separate kernel launches. The internal factorization
// is used only so as not to invent, and hand-check, an unstable O(64) formula
// — the Cooley-Tukey factorization for the even/odd split of the DFT is
// mathematically exact (a textbook result), not an approximation.

#pragma once

#include <cstdint>
#include <cstring>
#include <cmath>
#include <map>
#include <vector>

#include "cell.h"

// All the radix-8-specific code lives in a namespace of its own: the three
// projects had the name FusedStageInfo and a couple of helpers in common, and
// we carry the function bodies over WITH NO edits, so as not to break proven
// code.
namespace r8 {


static const double R8_C = 0.70710678118654752440;  // 1/sqrt(2)

// ============================================================================
// Digit-reverse permutation in base 8
// ============================================================================
static void digit_reverse_permute8_split(float *xr, float *xi, uint32_t N)
{
    uint32_t m3 = 0, tmp = N;
    while (tmp > 1) { tmp /= 8; ++m3; }   // N = 8^m3

    std::vector<float> cr(xr, xr + N), ci(xi, xi + N);
    for (uint32_t i = 0; i < N; ++i) {
        uint32_t rev = 0, ii = i;
        for (uint32_t d = 0; d < m3; ++d) {
            rev = rev * 8 + (ii % 8);
            ii /= 8;
        }
        xr[rev] = cr[i];
        xi[rev] = ci[i];
    }
}

// ============================================================================
// The table of twiddles W_block^(m*j), m=1..7, for the CPU stages (the same
// role g_host_tw4 has in Radix4 — built once before the timing, reads from the
// table inside the timed loop, no sin/cos getting in there).
// ============================================================================
struct HostTwiddle8 {
    std::vector<float> wr[7], wi[7];   // wr[m-1][j], wi[m-1][j], m=1..7
};

static std::map<uint32_t, HostTwiddle8> g_host_tw8;

static const HostTwiddle8 &host_twiddle8(uint32_t eighth)
{
    auto it = g_host_tw8.find(eighth);
    if (it != g_host_tw8.end()) return it->second;

    HostTwiddle8 t;
    for (int m = 0; m < 7; ++m) { t.wr[m].resize(eighth); t.wi[m].resize(eighth); }

    double base = -6.28318530717958647692 / (double)(eighth * 8);
    for (uint32_t j = 0; j < eighth; ++j) {
        for (int m = 1; m <= 7; ++m) {
            double a = base * m * j;
            t.wr[m-1][j] = (float)std::cos(a);
            t.wi[m-1][j] = (float)std::sin(a);
        }
    }
    it = g_host_tw8.insert(std::make_pair(eighth, t)).first;
    return it->second;
}

static uint32_t host_twiddle8_prewarm_N(uint32_t N, uint32_t min_npu_eighth)
{
    if (min_npu_eighth < 8) min_npu_eighth = 8;
    uint32_t built = 0;
    for (uint32_t eighth = 1; eighth <= N / 8; eighth *= 8) {
        if (eighth >= min_npu_eighth) break;
        host_twiddle8(eighth);
        ++built;
    }
    return built;
}
#define HOST_TWIDDLE_PREWARM(N, TILE) host_twiddle8_prewarm_N((N), (TILE))

// ---- the 4-point DFT (base case, no extra twiddles) — a shared subroutine
static inline void dft4_base(float ar,float ai,float br,float bi,
                              float cr,float ci,float dr,float di,
                              float &y0r,float &y0i,float &y1r,float &y1i,
                              float &y2r,float &y2i,float &y3r,float &y3i)
{
    y0r = ar+br+cr+dr;               y0i = ai+bi+ci+di;
    y1r = ar+bi-cr-di;               y1i = ai-br-ci+dr;
    y2r = ar-br+cr-dr;               y2i = ai-bi+ci-di;
    y3r = ar-bi-cr+di;               y3i = ai+br-ci-dr;
}

// ---- the 8-point combine (the fixed W8 matrix, does NOT depend on j/stage)
static inline void dft8_combine(const float tr[8], const float ti[8], float yr[8], float yi[8])
{
    float A0r,A0i,A1r,A1i,A2r,A2i,A3r,A3i;
    float B0r,B0i,B1r,B1i,B2r,B2i,B3r,B3i;
    dft4_base(tr[0],ti[0],tr[2],ti[2],tr[4],ti[4],tr[6],ti[6], A0r,A0i,A1r,A1i,A2r,A2i,A3r,A3i);
    dft4_base(tr[1],ti[1],tr[3],ti[3],tr[5],ti[5],tr[7],ti[7], B0r,B0i,B1r,B1i,B2r,B2i,B3r,B3i);

    // B'_p = B_p * W8^p ; W8^0=1, W8^1=c-ic, W8^2=-i, W8^3=-c-ic
    float Bp0r=B0r, Bp0i=B0i;
    float Bp1r = R8_C*B1r + R8_C*B1i, Bp1i = -R8_C*B1r + R8_C*B1i; // (c-ic)*(B1r+iB1i)
    float Bp2r = B2i,  Bp2i = -B2r;                                 // -i*B2
    float Bp3r = -R8_C*B3r + R8_C*B3i, Bp3i = -R8_C*B3r - R8_C*B3i; // (-c-ic)*B3

    yr[0]=A0r+Bp0r; yi[0]=A0i+Bp0i; yr[4]=A0r-Bp0r; yi[4]=A0i-Bp0i;
    yr[1]=A1r+Bp1r; yi[1]=A1i+Bp1i; yr[5]=A1r-Bp1r; yi[5]=A1i-Bp1i;
    yr[2]=A2r+Bp2r; yi[2]=A2i+Bp2i; yr[6]=A2r-Bp2r; yi[6]=A2i-Bp2i;
    yr[3]=A3r+Bp3r; yi[3]=A3i+Bp3i; yr[7]=A3r-Bp3r; yi[7]=A3i-Bp3i;
}

// ============================================================================
// CPU-fallback radix8 butterfly stage (for quarter/eighth < TILE_SIZE)
// ============================================================================
static void host_butterfly8_stage_split(float *xr, float *xi, uint32_t N, uint32_t eighth)
{
    uint32_t block = eighth << 3;
    const HostTwiddle8 &t = host_twiddle8(eighth);

    for (uint32_t k = 0; k < N; k += block) {
        for (uint32_t j = 0; j < eighth; ++j) {
            uint32_t idx[8];
            for (int m = 0; m < 8; ++m) idx[m] = k + j + m * eighth;

            float tr[8], ti[8];
            tr[0] = xr[idx[0]]; ti[0] = xi[idx[0]];
            if (eighth == 1) {
                for (int m = 1; m < 8; ++m) { tr[m] = xr[idx[m]]; ti[m] = xi[idx[m]]; }
            } else {
                for (int m = 1; m < 8; ++m) {
                    float wr = t.wr[m-1][j], wi = t.wi[m-1][j];
                    float xr_m = xr[idx[m]], xi_m = xi[idx[m]];
                    tr[m] = xr_m*wr - xi_m*wi;
                    ti[m] = xr_m*wi + xi_m*wr;
                }
            }
            float yr[8], yi[8];
            dft8_combine(tr, ti, yr, yi);
            for (int m = 0; m < 8; ++m) { xr[idx[m]] = yr[m]; xi[idx[m]] = yi[m]; }
        }
    }
}

// ============================================================================
// Twiddle precompute for the NPU (the Vector variant dit_fs_xs) — Packed
// layout, 7 factors (w1..w7), re/im one after another, a block `aligned` long:
//   w[(2m-2)*aligned .. (2m-1)*aligned) = w_m_re
//   w[(2m-1)*aligned .. (2m  )*aligned) = w_m_im     (m=1..7)
// 14*aligned floats per stage in all.
// ============================================================================
static void compute_twiddle7_packed(float *w, uint32_t eighth, uint32_t aligned)
{
    std::memset(w, 0, aligned * 14 * sizeof(float));
    uint32_t block = eighth << 3;
    double base = -6.28318530717958647692 / block;
    for (uint32_t j = 0; j < eighth; ++j) {
        for (int m = 1; m <= 7; ++m) {
            double a = base * m * j;
            w[(2*m-2)*aligned + j] = (float)std::cos(a);
            w[(2*m-1)*aligned + j] = (float)std::sin(a);
        }
    }
}


struct FusedStageInfo {
    uint32_t eighth;
    uint32_t aligned;
    uint32_t wOffset;
};

// min_npu_eighth (the TILE_SIZE argument) — the threshold below which a stage
// is computed on the CPU (as in Radix2/Radix4). The minimum is clamped at 8
// for the same reasons (the rounding of alignedValid up to 8).
static void build_fused_plan8(uint32_t m3, uint32_t min_npu_eighth,
                               std::vector<FusedStageInfo> &stages,
                               uint32_t &w_total, uint32_t &cpu_stages)
{
    stages.clear();
    w_total = 0;
    cpu_stages = 0;
    if (min_npu_eighth < 8) min_npu_eighth = 8;

    for (uint32_t s = 1; s <= m3; ++s) {
        uint32_t eighth = 1u << (3 * (s - 1));   // 8^(s-1)
        if (eighth < min_npu_eighth) { ++cpu_stages; continue; }
        FusedStageInfo info;
        info.eighth  = eighth;
        info.aligned = ((eighth + 63) / 64) * 64;
        info.wOffset = w_total;
        w_total += info.aligned * 14;
        stages.push_back(info);
    }
}

static void fill_fused_buffers8_packed(float *w, const std::vector<FusedStageInfo> &stages)
{
    for (const auto &st : stages)
        compute_twiddle7_packed(w + st.wOffset, st.eighth, st.aligned);
}

// ============================================================================
// Twiddles for the Cube path: the same mathematics, but a layout for BATCHING
// OVER k-BLOCKS.
//
// Why a separate buffer. A Cube call processes CUBE_BATCH (=128) independent
// 8-point butterflies at a time. Inside one k-block there are only `eighth`
// butterflies, that is, the eighth=8 stage has 8 columns out of 128 occupied
// (6%), and at N=4096 such a stage yields 64 separate tiles — the run time
// goes entirely into the launch overhead and not into the arithmetic. So the
// batch columns are gathered from SEVERAL consecutive k-blocks: column
// b = jBlock*eighth + shift.
//
// The twiddle W_block^(m*shift) depends ONLY on shift and not on k (see
// compute_twiddle7_packed above: the index j is precisely the shift). Which
// means that under batching over k the same `eighth` values have to repeat
// across the columns G = CUBE_BATCH/eighth times. We replicate them here, on
// the host, once per run — then all the kernel has left is one contiguous row
// read, without G small DataCopy calls for each of the 14 rows.
//
// The layout (per stage): 14 rows of rowLen float, row (2*(m-1)) — cos,
// (2*(m-1)+1) — sin, m=1..7; the value in column c equals the twiddle for
// shift = c % eighth. rowLen = max(eighth, CUBE_BATCH): at eighth < CUBE_BATCH
// the row is `eighth` values tiled out to 128 columns; at eighth >=
// CUBE_BATCH there is no tiling (c % eighth == c), the row holds all eighth
// values one after another, and the kernel reads a window out of it at the
// offset shift0.
// The buffer is separate from the vector path's (dit_fs_xs) — we leave that
// one's layout alone.
// ============================================================================
struct CubeTwiddlePlan {
    uint32_t rowLen;   // the length of one row in float
    uint32_t offset;   // the stage's offset in the shared buffer, in float
};

static void build_cube_twiddle_plan8(const std::vector<FusedStageInfo> &stages,
                                      uint32_t cube_batch,
                                      std::vector<CubeTwiddlePlan> &plan,
                                      uint32_t &wc_total)
{
    plan.clear();
    wc_total = 0;
    for (const auto &st : stages) {
        CubeTwiddlePlan p;
        p.rowLen = (st.eighth < cube_batch) ? cube_batch : st.eighth;
        p.offset = wc_total;
        wc_total += p.rowLen * 14;
        plan.push_back(p);
    }
}

static void fill_cube_twiddle8(float *wc, const std::vector<FusedStageInfo> &stages,
                                const std::vector<CubeTwiddlePlan> &plan)
{
    for (size_t s = 0; s < stages.size(); ++s) {
        const uint32_t eighth = stages[s].eighth;
        const uint32_t rowLen = plan[s].rowLen;
        float *w = wc + plan[s].offset;
        const uint32_t block = eighth << 3;
        const double   base  = -6.28318530717958647692 / block;
        for (uint32_t c = 0; c < rowLen; ++c) {
            const uint32_t j = c % eighth;   // batch column -> shift
            for (int m = 1; m <= 7; ++m) {
                const double a = base * (double)m * (double)j;
                w[(2*m-2)*rowLen + c] = (float)std::cos(a);
                w[(2*m-1)*rowLen + c] = (float)std::sin(a);
            }
        }
    }
}

// ============================================================================
// The Cube variants: fixed DFT8 matrices (independent of j/stage). They are
// built ONCE for the whole run (not per stage!) — that is the key difference
// from the Vector twiddles above, which do depend on eighth. The twiddle
// multiplication t_m = x_m * W_block^(mj) for the Cube variants is done
// separately — see the kernel kernels/r8_cube_fused_xs.cpp and the comment
// about the repack/prep step there.
//
// cube1: the block-real 16x16 form:
//   [Re(y)]   [ Re(M)  -Im(M) ] [Re(t)]
//   [Im(y)] = [ Im(M)   Re(M) ] [Im(t)]
// where M is the complex DFT8 matrix (M[p][k] = W8^(p*k mod 8)).
//
// cube2: the same M, but as TWO matmuls with a single "vertical" matrix
// A2 = [Re(M); Im(M)] (see build_dft8_matrix16_dual below):
//   P = A2*Re(t) = [Re(M)*Re(t); Im(M)*Re(t)]
//   Q = A2*Im(t) = [Re(M)*Im(t); Im(M)*Im(t)]
//   Re(y) = P[0..7]  - Q[8..15]
//   Im(y) = P[8..15] + Q[0..7]      — the combine step on Vector, in unpack.
// ============================================================================
static void build_dft8_matrix(float MR[8][8], float MI[8][8])
{
    for (int p = 0; p < 8; ++p) {
        for (int k = 0; k < 8; ++k) {
            int m = (p * k) % 8;
            double a = -6.28318530717958647692 * m / 8.0;
            MR[p][k] = (float)std::cos(a);
            MI[p][k] = (float)std::sin(a);
        }
    }
}

// Row-major 16x16, for cube1 (single matmul). Layout: rows/columns 0..7 — the
// Re block, 8..15 — the Im block, as in the formula above.
static void build_dft8_matrix16_blockreal(float M16[16][16])
{
    float MR[8][8], MI[8][8];
    build_dft8_matrix(MR, MI);
    for (int r = 0; r < 16; ++r)
        for (int c = 0; c < 16; ++c)
            M16[r][c] = 0.0f;
    for (int r = 0; r < 8; ++r) {
        for (int c = 0; c < 8; ++c) {
            M16[r][c]       =  MR[r][c];   // Re(M)
            M16[r][c+8]     = -MI[r][c];   // -Im(M)
            M16[r+8][c]     =  MI[r][c];   //  Im(M)
            M16[r+8][c+8]   =  MR[r][c];   //  Re(M)
        }
    }
}

// Row-major 16x16 for cube2 (two matmuls). Logically this is 16x8 —
// A2 = [Re(M); Im(M)] — but it is STORED as 16x16 with ZERO columns 8..15.
//
// The zeros here are not "just in case", they carry three functions at once
// (the detailed account is in the header of kernels/r8_cube2_matmul.cpp):
//   1. a row of A becomes 16 half = 32 bytes, that is, it falls on the
//      granularity of DataCopy (at 16 bytes the pipeline quietly produced
//      zeros);
//   2. the tiling stays the same (16, 16, B) as cube1's — the only shape
//      tested on hardware, so there is no new tiling to debug;
//   3. the B operand of the matmul is rows 0..15 (for Re) or 8..23 (for Im)
//      of ONE AND THE SAME repack buffer, that is, the second half of K
//      always holds the "wrong" data. The zero columns of A guarantee their
//      contribution is exactly zero, which is why repack is reused by cube1
//      and cube2 without a single edit.
// The price is twice as many multiplications as the mathematics demands. But
// an "honest" K=8 would have paid exactly as much: the Cube engine's fractal
// for fp16 is 16 wide along K.
static void build_dft8_matrix16_dual(float A2[16][16])
{
    float MR[8][8], MI[8][8];
    build_dft8_matrix(MR, MI);
    for (int r = 0; r < 16; ++r)
        for (int c = 0; c < 16; ++c)
            A2[r][c] = 0.0f;
    for (int r = 0; r < 8; ++r) {
        for (int c = 0; c < 8; ++c) {
            A2[r][c]     = MR[r][c];       // the top 8 rows — Re(M)
            A2[r+8][c]   = MI[r][c];       // the bottom 8 rows — Im(M)
        }
    }
}
} // namespace r8
