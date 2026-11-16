// kernels/r2_stockham_fused_xs.cpp
// R2-ST-FS-XS  (Stockham, fused, X=Split, W=Packed)
//
// All NPU stages in one kernel launch. The difference from
// r2_dit_fused_xs.cpp — here a ping-pong between TWO pairs of GM buffers is
// needed (buf0: XR0/XI0, buf1: XR1/XI1), because Stockham is out-of-place:
// stage s reads from one buffer and writes into the other, and the next stage
// s+1 swaps them.
//
// IMPORTANT about where the result finally lands: if the host computed some of
// the small stages on the CPU beforehand (with its own ping-pong there), then
// to work out whether the final answer sits in XR0/XI0 or in XR1/XI1 AFTER
// this launch you need the overall parity (cpu_stages + npu_stages), not just
// num_stages of this call. Only the internal parity by num_stages (how many
// stages THIS launch actually ran) is computed here — the host must add it to
// the parity of its own CPU prefix and decide where to read the D2H from.
// See the TODO in host/impl_r2.cpp when we get to it.
//
// The same PipeBarrier<PIPE_ALL> candidate between stages as in
// r2_dit_fused_*.cpp — here it matters even more: in Stockham the buffer that
// stage s+1 is going to READ from is precisely the buffer stage s has JUST
// written — if the synchronization between stages is not enough, the race will
// not hit tile "tails" but any address at all from the previous stage.
//
// As in r2_stockham_perstage_xs.cpp: there is NO tail-restore trick for
// halfSize<8 — the host must guarantee halfSize>=8 for every stage, which
// already goes through build_fused_plan_split_w/build_fused_plan (the
// min_npu_half>=8 threshold).

#include "kernel_operator.h"
#include <cstdint>

class KernelFFTStockhamFusedXS {
public:
    __aicore__ inline KernelFFTStockhamFusedXS() {}

    __aicore__ inline void Init(GM_ADDR XR0, GM_ADDR XI0, GM_ADDR XR1, GM_ADDR XI1,
                                 GM_ADDR W, GM_ADDR META,
                                 uint32_t full_size, uint32_t num_stages, uint32_t blk_num)
    {
        this->n         = full_size;
        this->tile      = 64;
        this->numStages = num_stages;
        // Tile-distribution stride across blocks in multicore mode. In
        // single-core (legacy) mode it is not used at all.
        this->blkNum    = blk_num > 0 ? blk_num : 1;

        uint32_t bufSizeSingle = this->tile * sizeof(float);

        xr0Gm.SetGlobalBuffer((__gm__ float *)XR0, this->n);
        xi0Gm.SetGlobalBuffer((__gm__ float *)XI0, this->n);
        xr1Gm.SetGlobalBuffer((__gm__ float *)XR1, this->n);
        xi1Gm.SetGlobalBuffer((__gm__ float *)XI1, this->n);
        wGm.SetGlobalBuffer((__gm__ float *)W, this->n * 2);       // upper bound, see r2_dit_fused_xs.cpp
        metaGm.SetGlobalBuffer((__gm__ uint32_t *)META, this->numStages * 3);

        pipe.InitBuffer(arIn, 2, bufSizeSingle);
        pipe.InitBuffer(aiIn, 2, bufSizeSingle);
        pipe.InitBuffer(brIn, 2, bufSizeSingle);
        pipe.InitBuffer(biIn, 2, bufSizeSingle);
        pipe.InitBuffer(wrIn, 2, bufSizeSingle);
        pipe.InitBuffer(wiIn, 2, bufSizeSingle);

        pipe.InitBuffer(out1ROut, 2, bufSizeSingle);
        pipe.InitBuffer(out1IOut, 2, bufSizeSingle);
        pipe.InitBuffer(out2ROut, 2, bufSizeSingle);
        pipe.InitBuffer(out2IOut, 2, bufSizeSingle);

        pipe.InitBuffer(t1Buf, bufSizeSingle);
        pipe.InitBuffer(t2Buf, bufSizeSingle);
        pipe.InitBuffer(t3Buf, bufSizeSingle);
        pipe.InitBuffer(t4Buf, bufSizeSingle);
    }

    // ========================================================================
    // TWO OPERATING MODES — exactly as in kernels/r2_dit_fused_xs.cpp and
    // kernels/r4_dit_fused_xs.cpp, see the detailed rationale there.
    //
    // Process() — as it was: all stages inside ONE launch, one core (variant
    // r2/stk_fs_xs). Kept verbatim so that this variant's numbers stay
    // bit-for-bit the same and stay comparable with the matrix already taken.
    //
    // ProcessOneStage(stage) — multicore mode: ONE stage per launch, the
    // stage's tiles are handed out across blocks (cores) with stride blkNum
    // (variant r2/stk_fsm_xs). Splitting the stages into separate launches is
    // mandatory here, and even more sharply so than in DIT: in Stockham, the
    // buffer stage s+1 is going to READ from is exactly the buffer stage s has
    // JUST written across the WHOLE array, so a barrier visible to all cores
    // is needed between stages. A hand-rolled cross-core barrier inside the
    // kernel is a known source of hangs, whereas ordering between launches on
    // one stream comes for free from the semantics of the stream itself.
    //
    // Handing tiles out across cores is correct: tile (k, shift) writes its
    // own two segments idxOut1/idxOut2 of length alignedValid into the
    // destination buffer, segments of different tiles do not overlap (k-blocks
    // of L run back to back, the shifts cut [0, halfSize) into tile-sized
    // pieces), and it reads from the OTHER buffer, which nobody writes to on
    // this stage. There is no tail-restore trick here at all (the host keeps
    // halfSize>=8), so there are no writes into anyone else's data by
    // construction.
    //
    // IMPORTANT about the src/dst choice in multicore mode: the ping-pong is
    // determined by the PARITY OF THE STAGE NUMBER, not by the call order, so
    // ProcessOneStage(stage) computes srcIsBuf1 = (stage % 2 == 1) — exactly
    // what Process() would have computed for s == stage. The host, when it
    // slices up the launches, must pass absolute stage numbers
    // (0..numStages-1), or the ping-pong goes off.
    // ========================================================================
    __aicore__ inline void Process()
    {
        for (uint32_t s = 0; s < this->numStages; ++s) {
            uint32_t halfSize    = metaGm.GetValue(s * 3 + 0);
            uint32_t aligned = metaGm.GetValue(s * 3 + 1);
            uint32_t wOffset = metaGm.GetValue(s * 3 + 2);

            bool srcIsBuf1 = (s % 2 == 1);   // s=0: src=buf0,dst=buf1; s=1: src=buf1,dst=buf0; ...

            AscendC::GlobalTensor<float> &srcR = srcIsBuf1 ? xr1Gm : xr0Gm;
            AscendC::GlobalTensor<float> &srcI = srcIsBuf1 ? xi1Gm : xi0Gm;
            AscendC::GlobalTensor<float> &dstR = srcIsBuf1 ? xr0Gm : xr1Gm;
            AscendC::GlobalTensor<float> &dstI = srcIsBuf1 ? xi0Gm : xi1Gm;

            ProcessStage(srcR, srcI, dstR, dstI, halfSize, aligned, wOffset);

            // --- CANDIDATE FIX for Bug 2 (see the file header) ---
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }

    __aicore__ inline void ProcessOneStage(uint32_t stage)
    {
        uint32_t halfSize = metaGm.GetValue(stage * 3 + 0);
        uint32_t aligned  = metaGm.GetValue(stage * 3 + 1);
        uint32_t wOffset  = metaGm.GetValue(stage * 3 + 2);

        // The same ping-pong by stage parity as in Process() — see above.
        bool srcIsBuf1 = (stage % 2 == 1);

        AscendC::GlobalTensor<float> &srcR = srcIsBuf1 ? xr1Gm : xr0Gm;
        AscendC::GlobalTensor<float> &srcI = srcIsBuf1 ? xi1Gm : xi0Gm;
        AscendC::GlobalTensor<float> &dstR = srcIsBuf1 ? xr0Gm : xr1Gm;
        AscendC::GlobalTensor<float> &dstI = srcIsBuf1 ? xi0Gm : xi1Gm;

        uint32_t L = halfSize << 1;
        uint32_t r = this->n / L;

        // The stage's tiles are numbered by one running tid: the outer loop of
        // ProcessStage went over k = 0..r-1 (here k is the block INDEX, not an
        // offset), the inner one over shift with stride tile (shiftsPerBlk
        // shifts in total).
        uint32_t shiftsPerBlk = (halfSize + this->tile - 1) / this->tile;
        uint32_t nTiles       = r * shiftsPerBlk;
        uint32_t bid          = (uint32_t)AscendC::GetBlockIdx();

        for (uint32_t tid = bid; tid < nTiles; tid += this->blkNum) {
            uint32_t k     = tid / shiftsPerBlk;
            uint32_t shift = (tid % shiftsPerBlk) * this->tile;
            ProcessTile(srcR, srcI, dstR, dstI, halfSize, aligned, wOffset, L, k, shift);
        }
    }

private:
    __aicore__ inline void ProcessStage(AscendC::GlobalTensor<float> &srcR,
                                         AscendC::GlobalTensor<float> &srcI,
                                         AscendC::GlobalTensor<float> &dstR,
                                         AscendC::GlobalTensor<float> &dstI,
                                         uint32_t halfSize, uint32_t aligned, uint32_t wOffset)
    {
        uint32_t L = halfSize << 1;
        uint32_t r = this->n / L;

        for (uint32_t k = 0; k < r; ++k) {
            for (uint32_t shift = 0; shift < halfSize; shift += this->tile) {
                ProcessTile(srcR, srcI, dstR, dstI, halfSize, aligned, wOffset, L, k, shift);
            }
        }
    }

    // The body of what used to be the inner loop of ProcessStage, lifted out
    // as is — shared by both modes, so the math is certain to be the same.
    // L is passed as a parameter because in the original it was computed in
    // ProcessStage and the body relies on it (idxOut1 = k * L + shift).
    __aicore__ inline void ProcessTile(AscendC::GlobalTensor<float> &srcR,
                                        AscendC::GlobalTensor<float> &srcI,
                                        AscendC::GlobalTensor<float> &dstR,
                                        AscendC::GlobalTensor<float> &dstI,
                                        uint32_t halfSize, uint32_t aligned, uint32_t wOffset,
                                        uint32_t L, uint32_t k, uint32_t shift)
    {
        uint32_t valid = (halfSize - shift < this->tile) ? (halfSize - shift) : this->tile;

        this->curValid       = valid;
        this->alignedValid   = ((valid + 7)  / 8)  * 8;
        this->computeAligned = ((valid + 63) / 64) * 64;

        uint32_t idxIn   = k * halfSize + shift;
        uint32_t idxIn2  = idxIn + this->n / 2;
        uint32_t idxOut1 = k * L + shift;
        uint32_t idxOut2 = idxOut1 + halfSize;

        CopyIn(srcR, srcI, idxIn, idxIn2, shift, aligned, wOffset);
        Compute();
        CopyOut(dstR, dstI, idxOut1, idxOut2);
    }

    __aicore__ inline void CopyIn(AscendC::GlobalTensor<float> &srcR,
                                   AscendC::GlobalTensor<float> &srcI,
                                   uint32_t idxIn, uint32_t idxIn2, uint32_t shift,
                                   uint32_t aligned, uint32_t wOffset)
    {
        AscendC::LocalTensor<float> ar = arIn.AllocTensor<float>();
        AscendC::LocalTensor<float> ai = aiIn.AllocTensor<float>();
        AscendC::LocalTensor<float> br = brIn.AllocTensor<float>();
        AscendC::LocalTensor<float> bi = biIn.AllocTensor<float>();

        AscendC::DataCopy(ar, srcR[idxIn],  this->alignedValid);
        AscendC::DataCopy(ai, srcI[idxIn],  this->alignedValid);
        AscendC::DataCopy(br, srcR[idxIn2], this->alignedValid);
        AscendC::DataCopy(bi, srcI[idxIn2], this->alignedValid);

        AscendC::LocalTensor<float> wr = wrIn.AllocTensor<float>();
        AscendC::LocalTensor<float> wi = wiIn.AllocTensor<float>();
        AscendC::DataCopy(wr, wGm[wOffset + shift],           this->alignedValid);
        AscendC::DataCopy(wi, wGm[wOffset + aligned + shift], this->alignedValid);

        AscendC::PipeBarrier<PIPE_MTE2>();

        arIn.EnQue(ar); aiIn.EnQue(ai);
        brIn.EnQue(br); biIn.EnQue(bi);
        wrIn.EnQue(wr); wiIn.EnQue(wi);
    }

    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<float> ar = arIn.DeQue<float>();
        AscendC::LocalTensor<float> ai = aiIn.DeQue<float>();
        AscendC::LocalTensor<float> br = brIn.DeQue<float>();
        AscendC::LocalTensor<float> bi = biIn.DeQue<float>();
        AscendC::LocalTensor<float> wr = wrIn.DeQue<float>();
        AscendC::LocalTensor<float> wi = wiIn.DeQue<float>();

        AscendC::LocalTensor<float> t1 = t1Buf.Get<float>();
        AscendC::LocalTensor<float> t2 = t2Buf.Get<float>();
        AscendC::LocalTensor<float> t3 = t3Buf.Get<float>();
        AscendC::LocalTensor<float> t4 = t4Buf.Get<float>();

        if (this->computeAligned > this->alignedValid) {
            uint32_t tail = this->computeAligned - this->alignedValid;
            AscendC::Duplicate(wr[this->alignedValid], (float)0.0f, tail);
            AscendC::Duplicate(wi[this->alignedValid], (float)0.0f, tail);
            AscendC::PipeBarrier<PIPE_V>();
        }

        uint32_t len = this->computeAligned;

        AscendC::Mul(t1, br, wr, len);
        AscendC::Mul(t2, bi, wi, len);
        AscendC::Mul(t3, br, wi, len);
        AscendC::Mul(t4, bi, wr, len);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Sub(wr, t1, t2, len);   // wbr
        AscendC::Add(wi, t3, t4, len);   // wbi
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::LocalTensor<float> o1r = out1ROut.AllocTensor<float>();
        AscendC::LocalTensor<float> o1i = out1IOut.AllocTensor<float>();
        AscendC::LocalTensor<float> o2r = out2ROut.AllocTensor<float>();
        AscendC::LocalTensor<float> o2i = out2IOut.AllocTensor<float>();

        AscendC::Add(o1r, ar, wr, len);
        AscendC::Add(o1i, ai, wi, len);
        AscendC::Sub(o2r, ar, wr, len);
        AscendC::Sub(o2i, ai, wi, len);
        AscendC::PipeBarrier<PIPE_V>();

        out1ROut.EnQue(o1r); out1IOut.EnQue(o1i);
        out2ROut.EnQue(o2r); out2IOut.EnQue(o2i);

        arIn.FreeTensor(ar); aiIn.FreeTensor(ai);
        brIn.FreeTensor(br); biIn.FreeTensor(bi);
        wrIn.FreeTensor(wr); wiIn.FreeTensor(wi);
    }

    __aicore__ inline void CopyOut(AscendC::GlobalTensor<float> &dstR,
                                    AscendC::GlobalTensor<float> &dstI,
                                    uint32_t idxOut1, uint32_t idxOut2)
    {
        AscendC::LocalTensor<float> o1r = out1ROut.DeQue<float>();
        AscendC::LocalTensor<float> o1i = out1IOut.DeQue<float>();
        AscendC::LocalTensor<float> o2r = out2ROut.DeQue<float>();
        AscendC::LocalTensor<float> o2i = out2IOut.DeQue<float>();

        // NO tail-restore trick — see the warning in the file header.
        AscendC::DataCopy(dstR[idxOut1], o1r, this->alignedValid);
        AscendC::DataCopy(dstI[idxOut1], o1i, this->alignedValid);
        AscendC::DataCopy(dstR[idxOut2], o2r, this->alignedValid);
        AscendC::DataCopy(dstI[idxOut2], o2i, this->alignedValid);
        AscendC::PipeBarrier<PIPE_MTE3>();

        out1ROut.FreeTensor(o1r); out1IOut.FreeTensor(o1i);
        out2ROut.FreeTensor(o2r); out2IOut.FreeTensor(o2i);
    }

private:
    AscendC::TPipe pipe;
    AscendC::GlobalTensor<float>    xr0Gm, xi0Gm, xr1Gm, xi1Gm, wGm;
    AscendC::GlobalTensor<uint32_t> metaGm;

    AscendC::TQue<AscendC::TPosition::VECIN,  2> arIn, aiIn, brIn, biIn, wrIn, wiIn;
    AscendC::TQue<AscendC::TPosition::VECOUT, 2> out1ROut, out1IOut, out2ROut, out2IOut;
    AscendC::TBuf<AscendC::TPosition::VECCALC> t1Buf, t2Buf, t3Buf, t4Buf;

    uint32_t n, tile, numStages;
    uint32_t blkNum;                 // tile stride across blocks (multicore mode)
    uint32_t curValid, alignedValid, computeAligned;
};

// stage == FFT_STK_XS_ALL_STAGES — single-core legacy mode (variant
// r2/stk_fs_xs); any other value means multicore: exactly this stage is
// computed and its tiles are handed out across blocks (variant r2/stk_fsm_xs).
// One kernel with two modes rather than a copy of the file, so that the math
// (CopyIn/Compute/CopyOut) of both variants is literally one and the same —
// otherwise "one core versus forty" stops being a comparison of core count.
#define FFT_STK_XS_ALL_STAGES 0xFFFFFFFFu

extern "C" __global__ __aicore__ void fft_stockham_fused_xs(
    GM_ADDR XR0, GM_ADDR XI0, GM_ADDR XR1, GM_ADDR XI1, GM_ADDR W, GM_ADDR META,
    uint32_t full_size, uint32_t num_stages, uint32_t stage, uint32_t blk_num)
{
    KernelFFTStockhamFusedXS op;
    op.Init(XR0, XI0, XR1, XI1, W, META, full_size, num_stages, blk_num);
    if (stage == FFT_STK_XS_ALL_STAGES) {
        op.Process();
    } else {
        op.ProcessOneStage(stage);
    }
}

#ifndef ASCENDC_CPU_DEBUG
void fft_stockham_fused_xs_do(uint32_t blockDim, void *stream,
                               void *xr0, void *xi0, void *xr1, void *xi1,
                               void *w, void *meta,
                               uint32_t full_size, uint32_t num_stages,
                               uint32_t stage, uint32_t blk_num)
{
    fft_stockham_fused_xs<<<blockDim, nullptr, stream>>>(xr0, xi0, xr1, xi1, w, meta,
        full_size, num_stages, stage, blk_num);
}
#endif