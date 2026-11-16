// kernels/r2_dit_perstage_xi.cpp
// R2-DT-PS-XI  (DIT, per-stage, X=Interleaved, W=Packed — the only W)
//
// The difference from r2_dit_perstage_xs.cpp: a single X buffer (interleaved
// re/im) instead of xr/xi. The GM tile is then read by ONE contiguous
// DataCopy (2*valid floats in a row — that is exactly the physical advantage
// of the interleaved layout on reads/writes), but the butterfly is computed
// on separate re/im, so the de-interleave/interleave is done by an explicit
// scalar loop AFTER CopyIn and BEFORE CopyOut. That is certainly slower than
// vectorized strided access — the comparison is meant to show precisely this
// (see the discussion in insights.md: the Ascend vector unit penalizes
// non-scalable scalar/strided access). Do not treat the scalar loop as a bug
// — it is a deliberate choice in favor of guaranteed correctness over
// guessing DataCopyParams strides without access to the hardware.

#include "kernel_operator.h"
#include <cstdint>

class KernelFFTInterleaved {
public:
    __aicore__ inline KernelFFTInterleaved() {}

    __aicore__ inline void Init(GM_ADDR X, GM_ADDR W,
                                 uint32_t full_size, uint32_t half_size, uint32_t aligned,
                                 uint32_t blk_num)
    {
        this->n       = full_size;
        this->h       = half_size;
        this->tile    = 64;
        this->aligned = aligned;
        // Tile-distribution stride across blocks (cores).
        this->blkNum  = blk_num > 0 ? blk_num : 1;

        uint32_t bufSizeSingle = this->tile * sizeof(float);
        uint32_t bufSizePair   = 2 * this->tile * sizeof(float);

        // X interleaved: a full buffer of 2*n floats (re,im alternating)
        xGm.SetGlobalBuffer((__gm__ float *)X, 2 * this->n);
        // W layout: [wr0..wr_{aligned-1} | wi0..wi_{aligned-1}] — as in XS
        wGm.SetGlobalBuffer((__gm__ float *)W, aligned * 2);

        // The "A" tile is read/written as one contiguous block of 2*tile float
        pipe.InitBuffer(xAIn,  2, bufSizePair);
        pipe.InitBuffer(xBIn,  2, bufSizePair);
        pipe.InitBuffer(xAOut, 2, bufSizePair);
        pipe.InitBuffer(xBOut, 2, bufSizePair);

        pipe.InitBuffer(wrIn, 2, bufSizeSingle);
        pipe.InitBuffer(wiIn, 2, bufSizeSingle);

        // De-interleaved re/im working buffers — TBuf here, since they are
        // used only inside a single Compute() and never overlap with the
        // neighboring tile along the pipeline (unlike xAIn/xBIn/wIn).
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
    // Tiles of ONE stage are handed out across blocks (cores): a block takes
    // tiles starting from its own blockIdx and onward with stride blkNum. For
    // that, the double loop over (k, shift) is flattened into a single linear
    // tile numbering — otherwise only the outer loop could be handed out
    // across cores, and with a small number of k-blocks (late stages:
    // block_size close to n) only a handful of cores would have any work.
    //
    // Why the distribution is correct: tile (k, shift) reads and writes
    // exactly its own two segments idxA/idxB of length valid (here — 2*valid
    // floats each, but the same non-overlapping segments), and segments of
    // different tiles do not overlap (the shifts cut [0, h) into tile-sized
    // pieces, the k-blocks run back to back). There are no inter-tile
    // dependencies within a stage, and the barrier between stages is given by
    // the launch order on the stream itself.
    //
    // NOTE for the future: the tail trick in CopyOut (the curValid..
    // alignedValid loop that reads GM) writes SOMEONE ELSE'S elements back.
    // At present it never fires, because h on the NPU path is a power of two
    // (or of four) no smaller than 8, that is, a multiple of eight, and
    // alignedValid always equals curValid. If h ever becomes smaller than 8,
    // this trick turns into a race between cores: by that point the neighbor
    // may already have written its own segment, and its result gets clobbered
    // by the old value.
    // ========================================================================
    __aicore__ inline void Process(uint32_t block_size)
    {
        uint32_t shiftsPerBlk = (this->h + this->tile - 1) / this->tile;
        uint32_t nTiles       = (this->n / block_size) * shiftsPerBlk;
        uint32_t bid          = (uint32_t)AscendC::GetBlockIdx();

        for (uint32_t tid = bid; tid < nTiles; tid += this->blkNum) {
            uint32_t k     = (tid / shiftsPerBlk) * block_size;
            uint32_t shift = (tid % shiftsPerBlk) * this->tile;
            ProcessTile(k, shift);
        }
    }

private:
    // The body of what used to be the inner loop of Process, lifted out as is,
    // so the tile math stays certain to match the single-core version.
    __aicore__ inline void ProcessTile(uint32_t k, uint32_t shift)
    {
        uint32_t idxA  = k + shift;
        uint32_t idxB  = k + shift + this->h;
        uint32_t valid = (this->h - shift < this->tile)
                         ? (this->h - shift) : this->tile;

        this->curValid       = valid;
        this->alignedValid   = ((valid + 7)  / 8)  * 8;
        this->computeAligned = ((valid + 63) / 64) * 64;

        CopyIn(idxA, idxB, shift);
        Compute();
        CopyOut(idxA, idxB);
    }

    __aicore__ inline void CopyIn(uint32_t idxA, uint32_t idxB, uint32_t shift)
    {
        AscendC::LocalTensor<float> a = xAIn.AllocTensor<float>();
        AscendC::LocalTensor<float> b = xBIn.AllocTensor<float>();

        // Contiguous block: alignedValid COMPLEX elements = 2*alignedValid float
        AscendC::DataCopy(a, xGm[2 * idxA], 2 * this->alignedValid);
        AscendC::DataCopy(b, xGm[2 * idxB], 2 * this->alignedValid);

        AscendC::LocalTensor<float> wr = wrIn.AllocTensor<float>();
        AscendC::LocalTensor<float> wi = wiIn.AllocTensor<float>();
        AscendC::DataCopy(wr, wGm[shift],                 this->alignedValid);
        AscendC::DataCopy(wi, wGm[this->aligned + shift], this->alignedValid);

        AscendC::PipeBarrier<PIPE_MTE2>();

        xAIn.EnQue(a); xBIn.EnQue(b);
        wrIn.EnQue(wr); wiIn.EnQue(wi);
    }

    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<float> a = xAIn.DeQue<float>();
        AscendC::LocalTensor<float> b = xBIn.DeQue<float>();
        AscendC::LocalTensor<float> wr = wrIn.DeQue<float>();
        AscendC::LocalTensor<float> wi = wiIn.DeQue<float>();

        AscendC::LocalTensor<float> ar = arBuf.Get<float>();
        AscendC::LocalTensor<float> ai = aiBuf.Get<float>();
        AscendC::LocalTensor<float> br = brBuf.Get<float>();
        AscendC::LocalTensor<float> bi = biBuf.Get<float>();

        // De-interleave: a[2*i]->ar[i], a[2*i+1]->ai[i] (and the same for b).
        // THE PRICE of the interleaved layout versus XS is this scalar loop.
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

        // Butterfly: tr = br*wr - bi*wi,  ti = br*wi + bi*wr — identical to XS
        AscendC::Mul(t1, br, wr, len);
        AscendC::Mul(t2, bi, wi, len);
        AscendC::Mul(t3, br, wi, len);
        AscendC::Mul(t4, bi, wr, len);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Sub(wr, t1, t2, len);   // tr (reuse)
        AscendC::Add(wi, t3, t4, len);   // ti (reuse)
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::LocalTensor<float> outA = xAOut.AllocTensor<float>();
        AscendC::LocalTensor<float> outB = xBOut.AllocTensor<float>();

        // wr/wi now hold tr/ti. ar/ai/br/bi (the original A/B) are still
        // untouched — compute BOTH outA and outB from them before overwriting
        // anything, putting the result into t1/t2 (outA) and into br/bi (outB,
        // safe to reuse — the original br/bi are no longer needed).
        AscendC::Add(t1, ar, wr, len);   // outAr = ar + tr
        AscendC::Add(t2, ai, wi, len);   // outAi = ai + ti
        AscendC::Sub(br, ar, wr, len);   // outBr = ar - tr   (reuse br)
        AscendC::Sub(bi, ai, wi, len);   // outBi = ai - ti   (reuse bi)
        AscendC::PipeBarrier<PIPE_V>();

        // Interleave back into outA (from t1/t2) and outB (from br/bi)
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
    AscendC::GlobalTensor<float> xGm, wGm;

    AscendC::TQue<AscendC::TPosition::VECIN,  2> xAIn, xBIn;
    AscendC::TQue<AscendC::TPosition::VECOUT, 2> xAOut, xBOut;
    AscendC::TQue<AscendC::TPosition::VECIN,  2> wrIn, wiIn;

    AscendC::TBuf<AscendC::TPosition::VECCALC> arBuf, aiBuf, brBuf, biBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> t1Buf, t2Buf, t3Buf, t4Buf;

    uint32_t n, h, tile, aligned;
    uint32_t blkNum;                 // tile stride across blocks (cores)
    uint32_t curValid, alignedValid, computeAligned;
};

extern "C" __global__ __aicore__ void fft_interleaved(GM_ADDR X, GM_ADDR W,
    uint32_t full_size, uint32_t half_size, uint32_t block_size, uint32_t aligned,
    uint32_t blk_num)
{
    KernelFFTInterleaved op;
    op.Init(X, W, full_size, half_size, aligned, blk_num);
    op.Process(block_size);
}

#ifndef ASCENDC_CPU_DEBUG
void fft_interleaved_do(uint32_t blockDim, void *stream,
                        void *x, void *w,
                        uint32_t full_size, uint32_t half_size,
                        uint32_t block_size, uint32_t aligned, uint32_t blk_num)
{
    fft_interleaved<<<blockDim, nullptr, stream>>>(x, w,
        full_size, half_size, block_size, aligned, blk_num);
}
#endif