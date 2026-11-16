// kernels/r2_dit_fused_xi.cpp
// R2-DT-FS-XI  (DIT, fused, X=Interleaved, W=Packed)
//
// A combination of two mechanisms that have already been explained:
//  - the stage structure and the meta buffer [halfSize,aligned,wOffset] x
//    numStages — as in r2_dit_fused_xs.cpp (one kernel launch for the whole
//    FFT);
//  - de-interleave/interleave of re/im by a scalar loop around the butterfly —
//    as in r2_dit_perstage_xi.cpp (the price of the interleaved layout,
//    expected to be slower than per-stage XI, and slower AGAIN because of the
//    extra scalar loop on each of the log2(N) stages run back to back with no
//    host synchronization between them).
//
// The same PipeBarrier<PIPE_ALL> candidate between stages as in
// r2_dit_fused_xs.cpp — see the comment there about Bug 2; it carries over here
// unchanged in meaning.

#include "kernel_operator.h"
#include <cstdint>

class KernelFFTFusedXI {
public:
    __aicore__ inline KernelFFTFusedXI() {}

    __aicore__ inline void Init(GM_ADDR X, GM_ADDR W, GM_ADDR META,
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

        xGm.SetGlobalBuffer((__gm__ float *)X, 2 * this->n);
        wGm.SetGlobalBuffer((__gm__ float *)W, this->n * 2);   // upper bound, see r2_dit_fused_xs.cpp
        metaGm.SetGlobalBuffer((__gm__ uint32_t *)META, this->numStages * 3);

        pipe.InitBuffer(xAIn,  2, bufSizePair);
        pipe.InitBuffer(xBIn,  2, bufSizePair);
        pipe.InitBuffer(xAOut, 2, bufSizePair);
        pipe.InitBuffer(xBOut, 2, bufSizePair);
        pipe.InitBuffer(wrIn, 2, bufSizeSingle);
        pipe.InitBuffer(wiIn, 2, bufSizeSingle);

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
    // TWO OPERATING MODES — exactly as in kernels/r2_dit_fused_xs.cpp and
    // kernels/r4_dit_fused_xs.cpp, see the detailed rationale there.
    //
    // Process() — as it was: all stages inside ONE launch, one core (variant
    // r2/dit_fs_xi). Kept verbatim so that this variant's numbers stay
    // bit-for-bit the same and stay comparable with the matrix already taken.
    //
    // ProcessOneStage(stage) — multicore mode: ONE stage per launch, the
    // stage's tiles are handed out across blocks (cores) with stride blkNum
    // (variant r2/dit_fsm_xi). Splitting the stages into separate launches is
    // mandatory here: stage s+1 reads what stage s wrote across the WHOLE
    // array, so a barrier visible to all cores is needed between them. A
    // hand-rolled cross-core barrier inside the kernel is a known source of
    // hangs, whereas ordering between launches on one stream comes for free
    // from the semantics of the stream itself.
    //
    // Handing tiles out across cores is correct: tile (k, shift) reads and
    // writes its own two segments idxA/idxB of length valid (here — 2*valid
    // floats each, since interleaved), and segments of different tiles do not
    // overlap. The tail trick in CopyOut (when alignedValid > curValid) never
    // fires at present, because halfSize on the NPU is a power of two no
    // smaller than 8, and such numbers are multiples of eight; otherwise it
    // would write out SOMEONE ELSE'S elements and would turn into a race
    // between cores.
    // ========================================================================
    __aicore__ inline void Process()
    {
        for (uint32_t s = 0; s < this->numStages; ++s) {
            uint32_t halfSize    = metaGm.GetValue(s * 3 + 0);
            uint32_t aligned = metaGm.GetValue(s * 3 + 1);
            uint32_t wOffset = metaGm.GetValue(s * 3 + 2);
            uint32_t block   = halfSize << 1;

            ProcessStage(halfSize, aligned, wOffset, block);

            // --- CANDIDATE FIX for Bug 2 (see r2_dit_fused_xs.cpp) ---
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }

    __aicore__ inline void ProcessOneStage(uint32_t stage)
    {
        uint32_t halfSize = metaGm.GetValue(stage * 3 + 0);
        uint32_t aligned  = metaGm.GetValue(stage * 3 + 1);
        uint32_t wOffset  = metaGm.GetValue(stage * 3 + 2);
        uint32_t block    = halfSize << 1;

        // The stage's tiles are numbered by one running tid: the outer loop of
        // Process() went over k with stride block (n/block blocks in total),
        // the inner one over shift with stride tile (shiftsPerBlk shifts).
        uint32_t shiftsPerBlk = (halfSize + this->tile - 1) / this->tile;
        uint32_t nTiles       = (this->n / block) * shiftsPerBlk;
        uint32_t bid          = (uint32_t)AscendC::GetBlockIdx();

        for (uint32_t tid = bid; tid < nTiles; tid += this->blkNum) {
            uint32_t k     = (tid / shiftsPerBlk) * block;
            uint32_t shift = (tid % shiftsPerBlk) * this->tile;
            ProcessTile(halfSize, aligned, wOffset, k, shift);
        }
    }

private:
    __aicore__ inline void ProcessStage(uint32_t halfSize, uint32_t aligned,
                                         uint32_t wOffset, uint32_t block)
    {
        for (uint32_t k = 0; k < this->n; k += block) {
            for (uint32_t shift = 0; shift < halfSize; shift += this->tile) {
                ProcessTile(halfSize, aligned, wOffset, k, shift);
            }
        }
    }

    // The body of what used to be the inner loop of ProcessStage, lifted out
    // as is — shared by both modes, so the math is certain to be the same.
    __aicore__ inline void ProcessTile(uint32_t halfSize, uint32_t aligned,
                                        uint32_t wOffset, uint32_t k, uint32_t shift)
    {
        uint32_t idxA  = k + shift;
        uint32_t idxB  = k + shift + halfSize;
        uint32_t valid = (halfSize - shift < this->tile) ? (halfSize - shift) : this->tile;

        this->curValid       = valid;
        this->alignedValid   = ((valid + 7)  / 8)  * 8;
        this->computeAligned = ((valid + 63) / 64) * 64;

        CopyIn(idxA, idxB, shift, aligned, wOffset);
        Compute();
        CopyOut(idxA, idxB);
    }

    __aicore__ inline void CopyIn(uint32_t idxA, uint32_t idxB, uint32_t shift,
                                   uint32_t aligned, uint32_t wOffset)
    {
        AscendC::LocalTensor<float> a = xAIn.AllocTensor<float>();
        AscendC::LocalTensor<float> b = xBIn.AllocTensor<float>();

        AscendC::DataCopy(a, xGm[2 * idxA], 2 * this->alignedValid);
        AscendC::DataCopy(b, xGm[2 * idxB], 2 * this->alignedValid);

        AscendC::LocalTensor<float> wr = wrIn.AllocTensor<float>();
        AscendC::LocalTensor<float> wi = wiIn.AllocTensor<float>();
        AscendC::DataCopy(wr, wGm[wOffset + shift],           this->alignedValid);
        AscendC::DataCopy(wi, wGm[wOffset + aligned + shift], this->alignedValid);

        AscendC::PipeBarrier<PIPE_MTE2>();

        xAIn.EnQue(a); xBIn.EnQue(b);
        wrIn.EnQue(wr); wiIn.EnQue(wi);
    }

    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<float> a  = xAIn.DeQue<float>();
        AscendC::LocalTensor<float> b  = xBIn.DeQue<float>();
        AscendC::LocalTensor<float> wr = wrIn.DeQue<float>();
        AscendC::LocalTensor<float> wi = wiIn.DeQue<float>();

        AscendC::LocalTensor<float> ar = arBuf.Get<float>();
        AscendC::LocalTensor<float> ai = aiBuf.Get<float>();
        AscendC::LocalTensor<float> br = brBuf.Get<float>();
        AscendC::LocalTensor<float> bi = biBuf.Get<float>();

        // De-interleave — the price of the interleaved layout, see
        // r2_dit_perstage_xi.cpp
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

        AscendC::Sub(wr, t1, t2, len);   // tr
        AscendC::Add(wi, t3, t4, len);   // ti
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::LocalTensor<float> outA = xAOut.AllocTensor<float>();
        AscendC::LocalTensor<float> outB = xBOut.AllocTensor<float>();

        // outAr/outAi -> t1/t2 (reuse), outBr/outBi -> br/bi (reuse) —
        // compute BOTH from the original ar/ai before any overwrite (the same
        // ordering I fixed in r2_dit_perstage_xi.cpp — not making the same
        // mistake over again here)
        AscendC::Add(t1, ar, wr, len);   // outAr
        AscendC::Add(t2, ai, wi, len);   // outAi
        AscendC::Sub(br, ar, wr, len);   // outBr
        AscendC::Sub(bi, ai, wi, len);   // outBi
        AscendC::PipeBarrier<PIPE_V>();

        for (uint32_t i = 0; i < this->alignedValid; ++i) {
            outA.SetValue(2 * i,     t1.GetValue(i));
            outA.SetValue(2 * i + 1, t2.GetValue(i));
            outB.SetValue(2 * i,     br.GetValue(i));
            outB.SetValue(2 * i + 1, bi.GetValue(i));
        }

        xAOut.EnQue(outA);
        xBOut.EnQue(outB);
        xAIn.FreeTensor(a); xBIn.FreeTensor(b);
        wrIn.FreeTensor(wr); wiIn.FreeTensor(wi);
    }

    __aicore__ inline void CopyOut(uint32_t idxA, uint32_t idxB)
    {
        AscendC::LocalTensor<float> outA = xAOut.DeQue<float>();
        AscendC::LocalTensor<float> outB = xBOut.DeQue<float>();

        if (this->alignedValid > this->curValid) {
            for (uint32_t i = this->curValid; i < this->alignedValid; i++) {
                outA.SetValue(2 * i,     xGm.GetValue(2 * idxA + 2 * i));
                outA.SetValue(2 * i + 1, xGm.GetValue(2 * idxA + 2 * i + 1));
                outB.SetValue(2 * i,     xGm.GetValue(2 * idxB + 2 * i));
                outB.SetValue(2 * i + 1, xGm.GetValue(2 * idxB + 2 * i + 1));
            }
        }

        AscendC::DataCopy(xGm[2 * idxA], outA, 2 * this->alignedValid);
        AscendC::DataCopy(xGm[2 * idxB], outB, 2 * this->alignedValid);
        AscendC::PipeBarrier<PIPE_MTE3>();

        xAOut.FreeTensor(outA); xBOut.FreeTensor(outB);
    }

private:
    AscendC::TPipe pipe;
    AscendC::GlobalTensor<float>    xGm, wGm;
    AscendC::GlobalTensor<uint32_t> metaGm;

    AscendC::TQue<AscendC::TPosition::VECIN,  2> xAIn, xBIn;
    AscendC::TQue<AscendC::TPosition::VECOUT, 2> xAOut, xBOut;
    AscendC::TQue<AscendC::TPosition::VECIN,  2> wrIn, wiIn;
    AscendC::TBuf<AscendC::TPosition::VECCALC> arBuf, aiBuf, brBuf, biBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> t1Buf, t2Buf, t3Buf, t4Buf;

    uint32_t n, tile, numStages;
    uint32_t blkNum;                 // tile stride across blocks (multicore mode)
    uint32_t curValid, alignedValid, computeAligned;
};

// stage == FFT_DIT_XI_ALL_STAGES — single-core legacy mode (variant
// r2/dit_fs_xi); any other value means multicore: exactly this stage is
// computed and its tiles are handed out across blocks (variant r2/dit_fsm_xi).
// One kernel with two modes rather than a copy of the file, so that the math
// (CopyIn/Compute/CopyOut) of both variants is literally one and the same —
// otherwise "one core versus forty" stops being a comparison of core count.
#define FFT_DIT_XI_ALL_STAGES 0xFFFFFFFFu

extern "C" __global__ __aicore__ void fft_dit_fused_xi(
    GM_ADDR X, GM_ADDR W, GM_ADDR META,
    uint32_t full_size, uint32_t num_stages, uint32_t stage, uint32_t blk_num)
{
    KernelFFTFusedXI op;
    op.Init(X, W, META, full_size, num_stages, blk_num);
    if (stage == FFT_DIT_XI_ALL_STAGES) {
        op.Process();
    } else {
        op.ProcessOneStage(stage);
    }
}

#ifndef ASCENDC_CPU_DEBUG
void fft_dit_fused_xi_do(uint32_t blockDim, void *stream,
                          void *x, void *w, void *meta,
                          uint32_t full_size, uint32_t num_stages,
                          uint32_t stage, uint32_t blk_num)
{
    fft_dit_fused_xi<<<blockDim, nullptr, stream>>>(x, w, meta,
        full_size, num_stages, stage, blk_num);
}
#endif