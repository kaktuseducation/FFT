// kernels/r2_stockham_fused_xi.cpp
// R2-ST-FS-XI  (Stockham, fused, X=Interleaved, W=Packed)
//
// The last of the 8 radix-2 files. A combination of what is already explained:
//  - ping-pong between two GM buffers by stage parity, the meta buffer, the
//    PipeBarrier<PIPE_ALL> candidate between stages — see
//    r2_stockham_fused_xs.cpp;
//  - the scalar de-interleave/interleave of re/im — see r2_dit_fused_xi.cpp,
//    r2_stockham_perstage_xi.cpp.
//
// Every warning from r2_stockham_fused_xs.cpp applies unchanged: there is NO
// tail-restore trick (the host keeps halfSize>=8), and the final buffer parity
// after this launch depends on num_stages of THIS call plus the parity of the
// CPU prefix, which the host computed separately before the launch.

#include "kernel_operator.h"
#include <cstdint>

class KernelFFTStockhamFusedXI {
public:
    __aicore__ inline KernelFFTStockhamFusedXI() {}

    __aicore__ inline void Init(GM_ADDR X0, GM_ADDR X1, GM_ADDR W, GM_ADDR META,
                                 uint32_t full_size, uint32_t num_stages, uint32_t blk_num)
    {
        this->n         = full_size;
        this->tile      = 64;
        this->numStages = num_stages;
        // Tile-distribution stride across blocks in multicore mode. In
        // single-core (legacy) mode it is not used at all.
        this->blkNum    = blk_num > 0 ? blk_num : 1;

        uint32_t bufSizeSingle = this->tile * sizeof(float);
        uint32_t bufSizePair   = 2 * this->tile * sizeof(float);

        x0Gm.SetGlobalBuffer((__gm__ float *)X0, 2 * this->n);
        x1Gm.SetGlobalBuffer((__gm__ float *)X1, 2 * this->n);
        wGm.SetGlobalBuffer((__gm__ float *)W, this->n * 2);
        metaGm.SetGlobalBuffer((__gm__ uint32_t *)META, this->numStages * 3);

        pipe.InitBuffer(aIn, 2, bufSizePair);
        pipe.InitBuffer(bIn, 2, bufSizePair);
        pipe.InitBuffer(wrIn, 2, bufSizeSingle);
        pipe.InitBuffer(wiIn, 2, bufSizeSingle);

        pipe.InitBuffer(out1Out, 2, bufSizePair);
        pipe.InitBuffer(out2Out, 2, bufSizePair);

        pipe.InitBuffer(arBuf, bufSizeSingle);
        pipe.InitBuffer(aiBuf, bufSizeSingle);
        pipe.InitBuffer(brBuf, bufSizeSingle);
        pipe.InitBuffer(biBuf, bufSizeSingle);
        pipe.InitBuffer(t1Buf, bufSizeSingle);
        pipe.InitBuffer(t2Buf, bufSizeSingle);
        pipe.InitBuffer(t3Buf, bufSizeSingle);
        pipe.InitBuffer(t4Buf, bufSizeSingle);
    }

    // ========================================================================
    // TWO OPERATING MODES — exactly as in kernels/r2_dit_fused_xs.cpp,
    // kernels/r4_dit_fused_xs.cpp and kernels/r2_stockham_fused_xs.cpp, see
    // the detailed rationale there.
    //
    // Process() — as it was: all stages inside ONE launch, one core (variant
    // r2/stk_fs_xi). Kept verbatim so that this variant's numbers stay
    // bit-for-bit the same and stay comparable with the matrix already taken.
    //
    // ProcessOneStage(stage) — multicore mode: ONE stage per launch, the
    // stage's tiles are handed out across blocks (cores) with stride blkNum
    // (variant r2/stk_fsm_xi). Splitting the stages into separate launches is
    // mandatory here: in Stockham, the buffer stage s+1 is going to READ from
    // is exactly the buffer stage s has JUST written across the WHOLE array,
    // so a barrier visible to all cores is needed between stages. A
    // hand-rolled cross-core barrier inside the kernel is a known source of
    // hangs, whereas ordering between launches on one stream comes for free
    // from the semantics of the stream itself.
    //
    // Handing tiles out across cores is correct: tile (k, shift) writes its
    // own two segments idxOut1/idxOut2 (2*alignedValid floats each,
    // interleaved) into the destination buffer, segments of different tiles do
    // not overlap, and it reads from the OTHER buffer, which nobody writes to
    // on this stage. There is no tail-restore trick here at all (the host
    // keeps halfSize>=8), so there are no writes into anyone else's data by
    // construction.
    //
    // IMPORTANT about the src/dst choice in multicore mode: the ping-pong is
    // determined by the PARITY OF THE STAGE NUMBER, not by the call order, so
    // ProcessOneStage(stage) computes srcIsBuf1 = (stage % 2 == 1) — exactly
    // what Process() would have computed for s == stage. The host must pass
    // absolute stage numbers (0..numStages-1), or the ping-pong goes off.
    // ========================================================================
    __aicore__ inline void Process()
    {
        for (uint32_t s = 0; s < this->numStages; ++s) {
            uint32_t halfSize    = metaGm.GetValue(s * 3 + 0);
            uint32_t aligned = metaGm.GetValue(s * 3 + 1);
            uint32_t wOffset = metaGm.GetValue(s * 3 + 2);

            bool srcIsBuf1 = (s % 2 == 1);
            AscendC::GlobalTensor<float> &src = srcIsBuf1 ? x1Gm : x0Gm;
            AscendC::GlobalTensor<float> &dst = srcIsBuf1 ? x0Gm : x1Gm;

            ProcessStage(src, dst, halfSize, aligned, wOffset);

            // --- CANDIDATE FIX for Bug 2 (see r2_stockham_fused_xs.cpp) ---
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
        AscendC::GlobalTensor<float> &src = srcIsBuf1 ? x1Gm : x0Gm;
        AscendC::GlobalTensor<float> &dst = srcIsBuf1 ? x0Gm : x1Gm;

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
            ProcessTile(src, dst, halfSize, aligned, wOffset, L, k, shift);
        }
    }

private:
    __aicore__ inline void ProcessStage(AscendC::GlobalTensor<float> &src,
                                         AscendC::GlobalTensor<float> &dst,
                                         uint32_t halfSize, uint32_t aligned, uint32_t wOffset)
    {
        uint32_t L = halfSize << 1;
        uint32_t r = this->n / L;

        for (uint32_t k = 0; k < r; ++k) {
            for (uint32_t shift = 0; shift < halfSize; shift += this->tile) {
                ProcessTile(src, dst, halfSize, aligned, wOffset, L, k, shift);
            }
        }
    }

    // The body of what used to be the inner loop of ProcessStage, lifted out
    // as is — shared by both modes, so the math is certain to be the same.
    // L is passed as a parameter because in the original it was computed in
    // ProcessStage and the body relies on it (idxOut1 = k * L + shift).
    __aicore__ inline void ProcessTile(AscendC::GlobalTensor<float> &src,
                                        AscendC::GlobalTensor<float> &dst,
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

        CopyIn(src, idxIn, idxIn2, shift, aligned, wOffset);
        Compute();
        CopyOut(dst, idxOut1, idxOut2);
    }

    __aicore__ inline void CopyIn(AscendC::GlobalTensor<float> &src,
                                   uint32_t idxIn, uint32_t idxIn2, uint32_t shift,
                                   uint32_t aligned, uint32_t wOffset)
    {
        AscendC::LocalTensor<float> a = aIn.AllocTensor<float>();
        AscendC::LocalTensor<float> b = bIn.AllocTensor<float>();

        AscendC::DataCopy(a, src[2 * idxIn],  2 * this->alignedValid);
        AscendC::DataCopy(b, src[2 * idxIn2], 2 * this->alignedValid);

        AscendC::LocalTensor<float> wr = wrIn.AllocTensor<float>();
        AscendC::LocalTensor<float> wi = wiIn.AllocTensor<float>();
        AscendC::DataCopy(wr, wGm[wOffset + shift],           this->alignedValid);
        AscendC::DataCopy(wi, wGm[wOffset + aligned + shift], this->alignedValid);

        AscendC::PipeBarrier<PIPE_MTE2>();

        aIn.EnQue(a); bIn.EnQue(b);
        wrIn.EnQue(wr); wiIn.EnQue(wi);
    }

    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<float> a  = aIn.DeQue<float>();
        AscendC::LocalTensor<float> b  = bIn.DeQue<float>();
        AscendC::LocalTensor<float> wr = wrIn.DeQue<float>();
        AscendC::LocalTensor<float> wi = wiIn.DeQue<float>();

        AscendC::LocalTensor<float> ar = arBuf.Get<float>();
        AscendC::LocalTensor<float> ai = aiBuf.Get<float>();
        AscendC::LocalTensor<float> br = brBuf.Get<float>();
        AscendC::LocalTensor<float> bi = biBuf.Get<float>();

        for (uint32_t i = 0; i < this->alignedValid; ++i) {
            ar.SetValue(i, a.GetValue(2 * i));
            ai.SetValue(i, a.GetValue(2 * i + 1));
            br.SetValue(i, b.GetValue(2 * i));
            bi.SetValue(i, b.GetValue(2 * i + 1));
        }

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

        AscendC::LocalTensor<float> out1 = out1Out.AllocTensor<float>();
        AscendC::LocalTensor<float> out2 = out2Out.AllocTensor<float>();

        AscendC::Add(t1, ar, wr, len);   // out1r -> t1 (reuse)
        AscendC::Add(t2, ai, wi, len);   // out1i -> t2 (reuse)
        AscendC::Sub(br, ar, wr, len);   // out2r -> br (reuse)
        AscendC::Sub(bi, ai, wi, len);   // out2i -> bi (reuse)
        AscendC::PipeBarrier<PIPE_V>();

        for (uint32_t i = 0; i < this->alignedValid; ++i) {
            out1.SetValue(2 * i,     t1.GetValue(i));
            out1.SetValue(2 * i + 1, t2.GetValue(i));
            out2.SetValue(2 * i,     br.GetValue(i));
            out2.SetValue(2 * i + 1, bi.GetValue(i));
        }

        out1Out.EnQue(out1); out2Out.EnQue(out2);
        aIn.FreeTensor(a); bIn.FreeTensor(b);
        wrIn.FreeTensor(wr); wiIn.FreeTensor(wi);
    }

    __aicore__ inline void CopyOut(AscendC::GlobalTensor<float> &dst,
                                    uint32_t idxOut1, uint32_t idxOut2)
    {
        AscendC::LocalTensor<float> out1 = out1Out.DeQue<float>();
        AscendC::LocalTensor<float> out2 = out2Out.DeQue<float>();

        // NO tail-restore trick — the host keeps halfSize>=8 on the NPU path.
        AscendC::DataCopy(dst[2 * idxOut1], out1, 2 * this->alignedValid);
        AscendC::DataCopy(dst[2 * idxOut2], out2, 2 * this->alignedValid);
        AscendC::PipeBarrier<PIPE_MTE3>();

        out1Out.FreeTensor(out1); out2Out.FreeTensor(out2);
    }

private:
    AscendC::TPipe pipe;
    AscendC::GlobalTensor<float>    x0Gm, x1Gm, wGm;
    AscendC::GlobalTensor<uint32_t> metaGm;

    AscendC::TQue<AscendC::TPosition::VECIN,  2> aIn, bIn, wrIn, wiIn;
    AscendC::TQue<AscendC::TPosition::VECOUT, 2> out1Out, out2Out;
    AscendC::TBuf<AscendC::TPosition::VECCALC> arBuf, aiBuf, brBuf, biBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> t1Buf, t2Buf, t3Buf, t4Buf;

    uint32_t n, tile, numStages;
    uint32_t blkNum;                 // tile stride across blocks (multicore mode)
    uint32_t curValid, alignedValid, computeAligned;
};

// stage == FFT_STK_XI_ALL_STAGES — single-core legacy mode (variant
// r2/stk_fs_xi); any other value means multicore: exactly this stage is
// computed and its tiles are handed out across blocks (variant r2/stk_fsm_xi).
// One kernel with two modes rather than a copy of the file, so that the math
// (CopyIn/Compute/CopyOut) of both variants is literally one and the same —
// otherwise "one core versus forty" stops being a comparison of core count.
#define FFT_STK_XI_ALL_STAGES 0xFFFFFFFFu

extern "C" __global__ __aicore__ void fft_stockham_fused_xi(
    GM_ADDR X0, GM_ADDR X1, GM_ADDR W, GM_ADDR META,
    uint32_t full_size, uint32_t num_stages, uint32_t stage, uint32_t blk_num)
{
    KernelFFTStockhamFusedXI op;
    op.Init(X0, X1, W, META, full_size, num_stages, blk_num);
    if (stage == FFT_STK_XI_ALL_STAGES) {
        op.Process();
    } else {
        op.ProcessOneStage(stage);
    }
}

#ifndef ASCENDC_CPU_DEBUG
void fft_stockham_fused_xi_do(uint32_t blockDim, void *stream,
                               void *x0, void *x1, void *w, void *meta,
                               uint32_t full_size, uint32_t num_stages,
                               uint32_t stage, uint32_t blk_num)
{
    fft_stockham_fused_xi<<<blockDim, nullptr, stream>>>(x0, x1, w, meta,
        full_size, num_stages, stage, blk_num);
}
#endif