// kernels/r8_dit_fused_xs.cpp
// R8-DT-FS-XS  (radix8 DIT, fused, X=Split, W=Packed-7)
//
// A direct analogue of kernels/r4_dit_fused_xs.cpp, but the group is 8
// points, there are 7 twiddles per group (w1..w7), and the combine step is a
// real 8-point DFT (not three nested radix-2 stages): for the detailed
// derivation of the formula and the justification of the factorization see
// host/fft_host_r8.h (dft8_combine).
// As in the radix4 version: one input and one output TQue, everything lives
// in one buffer and is carved up by TILE_F float — otherwise the hardware
// event IDs run out.

#include "kernel_operator.h"
#include <cstdint>

class KernelFFT8FusedXS {
public:
    __aicore__ inline KernelFFT8FusedXS() {}

    __aicore__ inline void Init(GM_ADDR XR, GM_ADDR XI, GM_ADDR W, GM_ADDR META,
                                 uint32_t full_size, uint32_t num_stages, uint32_t blk_num)
    {
        this->n         = full_size;
        this->tile      = 64;
        this->numStages = num_stages;
        this->blkNum    = blk_num > 0 ? blk_num : 1;

        uint32_t bufS = this->tile * sizeof(float);

        xrGm.SetGlobalBuffer((__gm__ float *)XR, this->n);
        xiGm.SetGlobalBuffer((__gm__ float *)XI, this->n);
        metaGm.SetGlobalBuffer((__gm__ uint32_t *)META, this->numStages * 3);

        // FIX: the size of the W tensor is the REAL one (out of meta), not an
        // n*14 estimate. On the host, W gets exactly w_total =
        // sum(aligned*14) over the NPU stages (build_fused_plan8), which at
        // N=4096 comes to 8960 float against 57344 by the old estimate.
        // Declaring a GM tensor larger than the actual allocation is
        // dangerous: that is exactly what caught an aicore exception in the
        // cube1 branch, when DataCopy ran past the end of the real buffer.
        // Here the reads did land inside the bounds, but the margin rested on
        // the arithmetic and not on the declaration — now it rests on the
        // declaration.
        uint32_t wTotal = 0;
        for (uint32_t s = 0; s < this->numStages; ++s) {
            uint32_t aligned = metaGm.GetValue(s * 3 + 1);
            uint32_t wOffset = metaGm.GetValue(s * 3 + 2);
            uint32_t end     = wOffset + aligned * 14;
            if (end > wTotal) wTotal = end;
        }
        wGm.SetGlobalBuffer((__gm__ float *)W, wTotal > 0 ? wTotal : 1);

        pipe.InitBuffer(inQ,  2, NIN_T  * bufS);
        pipe.InitBuffer(outQ, 2, NOUT_T * bufS);
        pipe.InitBuffer(calcBuf, NCALC_T * bufS);
    }

    // FIX (multicore): one launch = ONE stage, not the whole FFT.
    // The loop over stages used to be here, inside the kernel, and everything
    // was computed on a single AIV core (blockDim=1) — out of 40 available.
    // That loop cannot be parallelized inside the kernel: stage s+1 reads
    // what stage s wrote, that is, between stages a barrier ACROSS ALL CORES
    // is needed, and inter-core synchronization on Ascend is the most fragile
    // part there is. So the stages are spread over separate launches on one
    // stream: the order between them is guaranteed by stream semantics, for
    // free and reliably.
    // Within a stage the tiles are handed out to the cores with a stride of
    // blkNum. This is correct without any synchronization whatsoever: tiles
    // of one stage write to non-overlapping GM positions (a tile's positions
    // are k + shift + m*eighth, and different (k, shift) do not overlap), so
    // two different cores never touch the same cell.
    __aicore__ inline void Process(uint32_t stage)
    {
        uint32_t eighth  = metaGm.GetValue(stage * 3 + 0);
        uint32_t aligned = metaGm.GetValue(stage * 3 + 1);
        uint32_t wOffset = metaGm.GetValue(stage * 3 + 2);
        uint32_t block   = eighth << 3;

        uint32_t shiftsPerBlk = (eighth + this->tile - 1) / this->tile;
        uint32_t nTiles       = (this->n / block) * shiftsPerBlk;
        uint32_t bid          = (uint32_t)AscendC::GetBlockIdx();

        for (uint32_t tid = bid; tid < nTiles; tid += this->blkNum) {
            uint32_t k     = (tid / shiftsPerBlk) * block;
            uint32_t shift = (tid % shiftsPerBlk) * this->tile;
            ProcessTile(eighth, aligned, wOffset, k, shift);
        }
    }

private:
    __aicore__ inline void ProcessTile(uint32_t eighth, uint32_t aligned, uint32_t wOffset,
                                        uint32_t k, uint32_t shift)
    {
        uint32_t i[8];
        for (int m = 0; m < 8; ++m) i[m] = k + shift + (uint32_t)m * eighth;
        uint32_t valid = (eighth - shift < this->tile) ? (eighth - shift) : this->tile;

        this->curValid     = valid;
        this->alignedValid = ((valid + 7) / 8) * 8;   // 8 float = 32 bytes = one DataCopy block

        CopyIn(i, shift, aligned, wOffset);
        Compute();
        CopyOut(i);
    }

    __aicore__ inline void CopyIn(uint32_t i[8], uint32_t shift, uint32_t aligned, uint32_t wOffset)
    {
        auto inT = inQ.AllocTensor<float>();
        // x0..x7 (re,im) -> slots 0..15
        for (int m = 0; m < 8; ++m) {
            AscendC::DataCopy(inT[(2*m)   * TILE_F], xrGm[i[m]], this->alignedValid);
            AscendC::DataCopy(inT[(2*m+1) * TILE_F], xiGm[i[m]], this->alignedValid);
        }
        // w1..w7 (re,im) -> slots 16..29
        for (int m = 1; m <= 7; ++m) {
            uint32_t base = wOffset + (2*m-2) * aligned + shift;
            AscendC::DataCopy(inT[(16 + 2*(m-1))   * TILE_F], wGm[base],           this->alignedValid);
            AscendC::DataCopy(inT[(16 + 2*(m-1)+1) * TILE_F], wGm[base + aligned], this->alignedValid);
        }
        inQ.EnQue(inT);
    }

    __aicore__ inline void Compute()
    {
        auto inT = inQ.DeQue<float>();
        auto x = [&](int m, int ri) { return inT[(2*m+ri) * TILE_F]; };       // ri: 0=re,1=im
        auto w = [&](int m, int ri) { return inT[(16 + 2*(m-1)+ri) * TILE_F]; };

        // calc buffer (NCALC_T=36): t1..t7 (re,im) -> [0..13]; A0..A3 (re,im)
        // -> [14..21]; B0..B3 (re,im) -> [22..29]; u,v -> [30..31];
        // scr0..scr3 (scratch for assembling the DFT4) -> [32..35].
        auto calcT = calcBuf.Get<float>();
        auto t = [&](int m, int ri) { return calcT[(2*(m-1)+ri) * TILE_F]; };
        auto A = [&](int p, int ri) { return calcT[(14 + 2*p + ri) * TILE_F]; };
        auto B = [&](int p, int ri) { return calcT[(22 + 2*p + ri) * TILE_F]; };
        auto u = calcT[30 * TILE_F];
        auto v = calcT[31 * TILE_F];
        auto scr0 = calcT[32 * TILE_F];
        auto scr1 = calcT[33 * TILE_F];
        auto scr2 = calcT[34 * TILE_F];
        auto scr3 = calcT[35 * TILE_F];

        // FIX: the length of the vector operations is alignedValid (exactly
        // what CopyIn loaded), not a round-up to 64.
        // It used to be len = computeAligned = 64 here for any valid <= 64.
        // On the stage eighth=8 (and that is 64 tiles out of 80 at N=4096) 8
        // elements were loaded but 64 were computed, that is, 56 lanes of
        // every row were taken out of UNINITIALIZED UB — the contents
        // depended on whatever had run on the core earlier. This did not
        // affect the result (the operations are element-wise, and CopyOut
        // writes only alignedValid lanes), but it meant eight times more
        // vector work than needed on the dominant stage, plus
        // nondeterministic data in the registers. The tail zeroing of the
        // twiddles through Duplicate became unnecessary as well: the lanes
        // past alignedValid are no longer read.
        uint32_t len = this->alignedValid;

        // t0 = x0 (no twiddle); t1..t7 = x_m * w_m
        for (int m = 1; m <= 7; ++m) {
            AscendC::Mul(u, x(m,0), w(m,0), len); AscendC::Mul(v, x(m,1), w(m,1), len);
            AscendC::Sub(t(m,0), u, v, len);
            AscendC::Mul(u, x(m,0), w(m,1), len); AscendC::Mul(v, x(m,1), w(m,0), len);
            AscendC::Add(t(m,1), u, v, len);
        }
        AscendC::PipeBarrier<PIPE_V>();

        // -- DFT4(t0,t2,t4,t6) -> A0..A3 ; DFT4(t1,t3,t5,t7) -> B0..B3 --
        // The base DFT4 formula (see dft4_base in host/fft_host_r8.h),
        // W4 = W8^2 = -i:
        //   Y0 = a+b+c+d
        //   Y1 = a - i*b - c + i*d :  Re = ar+bi-cr-di ; Im = ai-br-ci+dr
        //   Y2 = a - b + c - d     :  Re = ar-br+cr-dr ; Im = ai-bi+ci-di
        //   Y3 = a + i*b - c - i*d :  Re = ar-bi-cr+di ; Im = ai+br-ci-dr
        //
        // !!! WHY THIS IS COMPUTED HERE, NOT IN A SEPARATE DFT4 FUNCTION !!!
        // The previous version called an __aicore__ function DFT4() whose
        // operands were passed as an array of LocalTensor objects
        // (T in[4][2]) — a trick to get around the
        // --cce-aicore-input-parameter-size limit — and it accumulated the
        // result IN PLACE: Sub(out[1][0], out[1][0], cr, len).
        // That gave a consistently wrong result for exactly p=1,2,3: at
        // N=4096 only the pair y0/y4 stayed correct, that is, 2 outputs out
        // of 8, and every NPU stage multiplied the fraction of correct values
        // by 1/4 (checked by bisection over -t: 0 stages -> PASS,
        // 1 -> 1024/4096 correct, 2 -> 256, 3 -> 64). Tellingly, p=0 was
        // CORRECT — it was the only one computed through the scratch register
        // u rather than by in-place accumulation.
        // The formulas themselves are right: reproducing the whole pipeline in
        // host arithmetic gives a 2e-13 discrepancy against numpy.fft.
        // Therefore: no separate function and no in-place accumulation —
        // every output is assembled in u and written to its destination once.
        // Four INDEPENDENT scratch registers (scr0..scr3), round-robin: each
        // output is assembled by a chain of three dependent operations, and if
        // all 16 outputs are driven through one register the chains glue
        // together into a single sequential one — the vector pipeline has
        // nothing to overlap. That is exactly what slowed dit down from ~640
        // to 2602 us. With four registers there are four independent chains in
        // flight at once.
        // The operations are unrolled explicitly, with no helper
        // function/lambda: a lambda would have five LocalTensor parameters by
        // value, and there is no guarantee the cce compiler would inline it.
        // A = DFT4(t0, t2, t4, t6)
        AscendC::Add(scr0, x(0,0), t(2,0), len);
        AscendC::Add(scr0, scr0, t(4,0), len);
        AscendC::Add(A(0,0), scr0, t(6,0), len);   // ar+br+cr+dr
        AscendC::Add(scr1, x(0,1), t(2,1), len);
        AscendC::Add(scr1, scr1, t(4,1), len);
        AscendC::Add(A(0,1), scr1, t(6,1), len);   // ai+bi+ci+di
        AscendC::Add(scr2, x(0,0), t(2,1), len);
        AscendC::Sub(scr2, scr2, t(4,0), len);
        AscendC::Sub(A(1,0), scr2, t(6,1), len);   // ar+bi-cr-di
        AscendC::Sub(scr3, x(0,1), t(2,0), len);
        AscendC::Sub(scr3, scr3, t(4,1), len);
        AscendC::Add(A(1,1), scr3, t(6,0), len);   // ai-br-ci+dr
        AscendC::Sub(scr0, x(0,0), t(2,0), len);
        AscendC::Add(scr0, scr0, t(4,0), len);
        AscendC::Sub(A(2,0), scr0, t(6,0), len);   // ar-br+cr-dr
        AscendC::Sub(scr1, x(0,1), t(2,1), len);
        AscendC::Add(scr1, scr1, t(4,1), len);
        AscendC::Sub(A(2,1), scr1, t(6,1), len);   // ai-bi+ci-di
        AscendC::Sub(scr2, x(0,0), t(2,1), len);
        AscendC::Sub(scr2, scr2, t(4,0), len);
        AscendC::Add(A(3,0), scr2, t(6,1), len);   // ar-bi-cr+di
        AscendC::Add(scr3, x(0,1), t(2,0), len);
        AscendC::Sub(scr3, scr3, t(4,1), len);
        AscendC::Sub(A(3,1), scr3, t(6,0), len);   // ai+br-ci-dr

        // B = DFT4(t1, t3, t5, t7)
        AscendC::Add(scr0, t(1,0), t(3,0), len);
        AscendC::Add(scr0, scr0, t(5,0), len);
        AscendC::Add(B(0,0), scr0, t(7,0), len);   // ar+br+cr+dr
        AscendC::Add(scr1, t(1,1), t(3,1), len);
        AscendC::Add(scr1, scr1, t(5,1), len);
        AscendC::Add(B(0,1), scr1, t(7,1), len);   // ai+bi+ci+di
        AscendC::Add(scr2, t(1,0), t(3,1), len);
        AscendC::Sub(scr2, scr2, t(5,0), len);
        AscendC::Sub(B(1,0), scr2, t(7,1), len);   // ar+bi-cr-di
        AscendC::Sub(scr3, t(1,1), t(3,0), len);
        AscendC::Sub(scr3, scr3, t(5,1), len);
        AscendC::Add(B(1,1), scr3, t(7,0), len);   // ai-br-ci+dr
        AscendC::Sub(scr0, t(1,0), t(3,0), len);
        AscendC::Add(scr0, scr0, t(5,0), len);
        AscendC::Sub(B(2,0), scr0, t(7,0), len);   // ar-br+cr-dr
        AscendC::Sub(scr1, t(1,1), t(3,1), len);
        AscendC::Add(scr1, scr1, t(5,1), len);
        AscendC::Sub(B(2,1), scr1, t(7,1), len);   // ai-bi+ci-di
        AscendC::Sub(scr2, t(1,0), t(3,1), len);
        AscendC::Sub(scr2, scr2, t(5,0), len);
        AscendC::Add(B(3,0), scr2, t(7,1), len);   // ar-bi-cr+di
        AscendC::Add(scr3, t(1,1), t(3,0), len);
        AscendC::Sub(scr3, scr3, t(5,1), len);
        AscendC::Sub(B(3,1), scr3, t(7,0), len);   // ai+br-ci-dr
        AscendC::PipeBarrier<PIPE_V>();

        // B'_p = B_p * W8^p :  W8^0=1 (B'0=B0) ; W8^1=c-ic ; W8^2=-i ; W8^3=-c-ic
        // Bp1..Bp3 go back into t1..t3 (the t slots are no longer needed in
        // their original form).
        auto Bp1r=t(1,0); auto Bp1i=t(1,1);
        auto Bp2r=t(2,0); auto Bp2i=t(2,1);
        auto Bp3r=t(3,0); auto Bp3i=t(3,1);
        const float C = 0.70710678118654752440f;

        // Bp1 = (c - i c) * B1 = (c*B1r + c*B1i,  -c*B1r + c*B1i)
        AscendC::Muls(u, B(1,0), C, len); AscendC::Muls(v, B(1,1), C, len);
        AscendC::Add(Bp1r, u, v, len);
        AscendC::Muls(u, B(1,0), -C, len);
        AscendC::Add(Bp1i, u, v, len);

        // Bp2 = -i * B2 = (B2i, -B2r)
        AscendC::Adds(Bp2r, B(2,1), 0.0f, len);
        AscendC::Muls(Bp2i, B(2,0), -1.0f, len);

        // Bp3 = (-c - i c) * B3 = (-c*B3r + c*B3i,  -c*B3r - c*B3i)
        AscendC::Muls(u, B(3,0), -C, len); AscendC::Muls(v, B(3,1), C, len);
        AscendC::Add(Bp3r, u, v, len);
        AscendC::Muls(v, B(3,1), -C, len);
        AscendC::Add(Bp3i, u, v, len);
        AscendC::PipeBarrier<PIPE_V>();

        // Y_p = A_p + Bp_p ; Y_{p+4} = A_p - Bp_p   (p=0..3; Bp0 = B0)
        // The result is written straight into outT — the only buffer that
        // actually leaves through outQ.EnQue().
        auto outT = outQ.AllocTensor<float>();
        auto y = [&](int m, int ri) { return outT[(2*m+ri) * TILE_F]; };

        AscendC::Add(y(0,0), A(0,0), B(0,0), len); AscendC::Add(y(0,1), A(0,1), B(0,1), len);
        AscendC::Sub(y(4,0), A(0,0), B(0,0), len); AscendC::Sub(y(4,1), A(0,1), B(0,1), len);

        AscendC::Add(y(1,0), A(1,0), Bp1r, len);   AscendC::Add(y(1,1), A(1,1), Bp1i, len);
        AscendC::Sub(y(5,0), A(1,0), Bp1r, len);   AscendC::Sub(y(5,1), A(1,1), Bp1i, len);

        AscendC::Add(y(2,0), A(2,0), Bp2r, len);   AscendC::Add(y(2,1), A(2,1), Bp2i, len);
        AscendC::Sub(y(6,0), A(2,0), Bp2r, len);   AscendC::Sub(y(6,1), A(2,1), Bp2i, len);

        AscendC::Add(y(3,0), A(3,0), Bp3r, len);   AscendC::Add(y(3,1), A(3,1), Bp3i, len);
        AscendC::Sub(y(7,0), A(3,0), Bp3r, len);   AscendC::Sub(y(7,1), A(3,1), Bp3i, len);
        AscendC::PipeBarrier<PIPE_V>();

        outQ.EnQue(outT);
        inQ.FreeTensor(inT);
    }

    __aicore__ inline void CopyOut(uint32_t i[8])
    {
        auto outT = outQ.DeQue<float>();
        if (this->alignedValid > this->curValid) {
            for (uint32_t p = this->curValid; p < this->alignedValid; p++) {
                for (int m = 0; m < 8; ++m) {
                    outT[(2*m)   * TILE_F].SetValue(p, xrGm.GetValue(i[m] + p));
                    outT[(2*m+1) * TILE_F].SetValue(p, xiGm.GetValue(i[m] + p));
                }
            }
        }
        for (int m = 0; m < 8; ++m) {
            AscendC::DataCopy(xrGm[i[m]], outT[(2*m)   * TILE_F], this->alignedValid);
            AscendC::DataCopy(xiGm[i[m]], outT[(2*m+1) * TILE_F], this->alignedValid);
        }
        outQ.FreeTensor(outT);
    }

private:
    AscendC::TPipe pipe;
    AscendC::GlobalTensor<float>    xrGm, xiGm, wGm;
    AscendC::GlobalTensor<uint32_t> metaGm;

    static constexpr uint32_t TILE_F  = 64;
    static constexpr uint32_t NIN_T   = 30;   // x0..x7 (re,im)=16 + w1..w7 (re,im)=14
    static constexpr uint32_t NOUT_T  = 16;   // y0..y7 (re,im) — the final result
    static constexpr uint32_t NCALC_T = 36;   // t1..t7(14) + A0..3(8) + B0..3(8) + u,v(2) + scr0..3(4)

    AscendC::TQue<AscendC::TPosition::VECIN,  2> inQ;
    AscendC::TQue<AscendC::TPosition::VECOUT, 2> outQ;
    AscendC::TBuf<AscendC::TPosition::VECCALC>   calcBuf;

    uint32_t n, tile, numStages, blkNum;
    uint32_t curValid, alignedValid;
};

extern "C" __global__ __aicore__ void fft8_dit_fused_xs(
    GM_ADDR XR, GM_ADDR XI, GM_ADDR W, GM_ADDR META,
    uint32_t full_size, uint32_t num_stages, uint32_t stage, uint32_t blk_num)
{
    KernelFFT8FusedXS op;
    op.Init(XR, XI, W, META, full_size, num_stages, blk_num);
    op.Process(stage);
}

#ifndef ASCENDC_CPU_DEBUG
void fft8_dit_fused_xs_do(uint32_t blockDim, void *stream,
                           void *xr, void *xi, void *w, void *meta,
                           uint32_t full_size, uint32_t num_stages,
                           uint32_t stage, uint32_t blk_num)
{
    fft8_dit_fused_xs<<<blockDim, nullptr, stream>>>(xr, xi, w, meta,
                                                      full_size, num_stages, stage, blk_num);
}
#endif