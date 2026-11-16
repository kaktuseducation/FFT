#include "kernel_operator.h"
#include <cstdint>

class KernelFFTHybrid {
public:
    __aicore__ inline KernelFFTHybrid() {}

    __aicore__ inline void Init(GM_ADDR XR, GM_ADDR XI, GM_ADDR W,
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

        xrGm.SetGlobalBuffer((__gm__ float *)XR, this->n);
        xiGm.SetGlobalBuffer((__gm__ float *)XI, this->n);
        // W layout: [wr0..wr_{aligned-1} | wi0..wi_{aligned-1}]
        wGm.SetGlobalBuffer((__gm__ float *)W, aligned * 2);

        pipe.InitBuffer(xrAIn,  2, bufSizeSingle);
        pipe.InitBuffer(xiAIn,  2, bufSizeSingle);
        pipe.InitBuffer(xrBIn,  2, bufSizeSingle);
        pipe.InitBuffer(xiBIn,  2, bufSizeSingle);
        pipe.InitBuffer(xrAOut, 2, bufSizeSingle);
        pipe.InitBuffer(xiAOut, 2, bufSizeSingle);
        pipe.InitBuffer(xrBOut, 2, bufSizeSingle);
        pipe.InitBuffer(xiBOut, 2, bufSizeSingle);

        // FIX: wr/wi are now TQue<VECIN, 2> instead of TBuf<VECCALC>.
        // Without that, CopyIn(tile N+1) overwrote wrBuf/wiBuf while
        // Compute(tile N) was still reading from them — the classic data
        // race in a double-buffered AscendC pipeline.
        pipe.InitBuffer(wrIn, 2, bufSizeSingle);
        pipe.InitBuffer(wiIn, 2, bufSizeSingle);

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
    // exactly its own two segments idxA/idxB of length valid, and segments of
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
        AscendC::LocalTensor<float> ar = xrAIn.AllocTensor<float>();
        AscendC::LocalTensor<float> ai = xiAIn.AllocTensor<float>();
        AscendC::LocalTensor<float> br = xrBIn.AllocTensor<float>();
        AscendC::LocalTensor<float> bi = xiBIn.AllocTensor<float>();

        AscendC::DataCopy(ar, xrGm[idxA], this->alignedValid);
        AscendC::DataCopy(ai, xiGm[idxA], this->alignedValid);
        AscendC::DataCopy(br, xrGm[idxB], this->alignedValid);
        AscendC::DataCopy(bi, xiGm[idxB], this->alignedValid);

        // FIX: AllocTensor from a TQue — we get an isolated buffer slot
        AscendC::LocalTensor<float> wr = wrIn.AllocTensor<float>();
        AscendC::LocalTensor<float> wi = wiIn.AllocTensor<float>();

        // wr sits at offset=shift, wi — at offset=aligned+shift
        AscendC::DataCopy(wr, wGm[shift],                 this->alignedValid);
        AscendC::DataCopy(wi, wGm[this->aligned + shift], this->alignedValid);

        AscendC::PipeBarrier<PIPE_MTE2>();

        xrAIn.EnQue(ar); xiAIn.EnQue(ai);
        xrBIn.EnQue(br); xiBIn.EnQue(bi);
        wrIn.EnQue(wr);  wiIn.EnQue(wi);
    }

    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<float> ar = xrAIn.DeQue<float>();
        AscendC::LocalTensor<float> ai = xiAIn.DeQue<float>();
        AscendC::LocalTensor<float> br = xrBIn.DeQue<float>();
        AscendC::LocalTensor<float> bi = xiBIn.DeQue<float>();

        // FIX: DeQue from a TQue — guaranteed the same slot EnQue'd in CopyIn
        AscendC::LocalTensor<float> wr = wrIn.DeQue<float>();
        AscendC::LocalTensor<float> wi = wiIn.DeQue<float>();

        AscendC::LocalTensor<float> t1 = t1Buf.Get<float>();
        AscendC::LocalTensor<float> t2 = t2Buf.Get<float>();
        AscendC::LocalTensor<float> t3 = t3Buf.Get<float>();
        AscendC::LocalTensor<float> t4 = t4Buf.Get<float>();

        // Zero the tail of wr/wi if computeAligned > alignedValid
        if (this->computeAligned > this->alignedValid) {
            uint32_t tail = this->computeAligned - this->alignedValid;
            AscendC::Duplicate(wr[this->alignedValid], (float)0.0f, tail);
            AscendC::Duplicate(wi[this->alignedValid], (float)0.0f, tail);
            AscendC::PipeBarrier<PIPE_V>();
        }

        uint32_t len = this->computeAligned;

        // Butterfly: tr = br*wr - bi*wi,  ti = br*wi + bi*wr
        AscendC::Mul(t1, br, wr, len);
        AscendC::Mul(t2, bi, wi, len);
        AscendC::Mul(t3, br, wi, len);
        AscendC::Mul(t4, bi, wr, len);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Sub(wr, t1, t2, len);   // tr  (reuse wr slot)
        AscendC::Add(wi, t3, t4, len);   // ti  (reuse wi slot)
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::LocalTensor<float> outAr = xrAOut.AllocTensor<float>();
        AscendC::LocalTensor<float> outAi = xiAOut.AllocTensor<float>();
        AscendC::LocalTensor<float> outBr = xrBOut.AllocTensor<float>();
        AscendC::LocalTensor<float> outBi = xiBOut.AllocTensor<float>();

        AscendC::Add(outAr, ar, wr, len);
        AscendC::Add(outAi, ai, wi, len);
        AscendC::Sub(outBr, ar, wr, len);
        AscendC::Sub(outBi, ai, wi, len);  // FIX: wi here = ti (after the reuse above) ✓
        AscendC::PipeBarrier<PIPE_V>();

        xrAOut.EnQue(outAr); xiAOut.EnQue(outAi);
        xrBOut.EnQue(outBr); xiBOut.EnQue(outBi);

        xrAIn.FreeTensor(ar); xiAIn.FreeTensor(ai);
        xrBIn.FreeTensor(br); xiBIn.FreeTensor(bi);
        // FIX: FreeTensor for wr/wi coming from a TQue
        wrIn.FreeTensor(wr);  wiIn.FreeTensor(wi);
    }

    __aicore__ inline void CopyOut(uint32_t idxA, uint32_t idxB)
    {
        AscendC::LocalTensor<float> outAr = xrAOut.DeQue<float>();
        AscendC::LocalTensor<float> outAi = xiAOut.DeQue<float>();
        AscendC::LocalTensor<float> outBr = xrBOut.DeQue<float>();
        AscendC::LocalTensor<float> outBi = xiBOut.DeQue<float>();

        // Restore the tail so the DMA does not clobber neighboring elements
        if (this->alignedValid > this->curValid) {
            for (uint32_t i = this->curValid; i < this->alignedValid; i++) {
                outAr.SetValue(i, xrGm.GetValue(idxA + i));
                outAi.SetValue(i, xiGm.GetValue(idxA + i));
                outBr.SetValue(i, xrGm.GetValue(idxB + i));
                outBi.SetValue(i, xiGm.GetValue(idxB + i));
            }
        }

        AscendC::DataCopy(xrGm[idxA], outAr, this->alignedValid);
        AscendC::DataCopy(xiGm[idxA], outAi, this->alignedValid);
        AscendC::DataCopy(xrGm[idxB], outBr, this->alignedValid);
        AscendC::DataCopy(xiGm[idxB], outBi, this->alignedValid);
        AscendC::PipeBarrier<PIPE_MTE3>();

        xrAOut.FreeTensor(outAr); xiAOut.FreeTensor(outAi);
        xrBOut.FreeTensor(outBr); xiBOut.FreeTensor(outBi);
    }

private:
    AscendC::TPipe pipe;

    AscendC::GlobalTensor<float> xrGm, xiGm, wGm;

    AscendC::TQue<AscendC::TPosition::VECIN,  2> xrAIn, xiAIn, xrBIn, xiBIn;
    AscendC::TQue<AscendC::TPosition::VECOUT, 2> xrAOut, xiAOut, xrBOut, xiBOut;

    // FIX: wr/wi are now TQue<VECIN, 2> instead of TBuf<VECCALC>
    AscendC::TQue<AscendC::TPosition::VECIN, 2> wrIn, wiIn;

    AscendC::TBuf<AscendC::TPosition::VECCALC> t1Buf, t2Buf, t3Buf, t4Buf;

    uint32_t n, h, tile, aligned;
    uint32_t blkNum;                 // tile stride across blocks (cores)
    uint32_t curValid, alignedValid, computeAligned;
};

extern "C" __global__ __aicore__ void fft_hybrid(GM_ADDR XR, GM_ADDR XI, GM_ADDR W,
    uint32_t full_size, uint32_t half_size, uint32_t block_size, uint32_t aligned,
    uint32_t blk_num)
{
    KernelFFTHybrid op;
    op.Init(XR, XI, W, full_size, half_size, aligned, blk_num);
    op.Process(block_size);
}

#ifndef ASCENDC_CPU_DEBUG
void fft_hybrid_do(uint32_t blockDim, void *stream,
                   void *xr, void *xi, void *w,
                   uint32_t full_size, uint32_t half_size,
                   uint32_t block_size, uint32_t aligned, uint32_t blk_num)
{
    fft_hybrid<<<blockDim, nullptr, stream>>>(xr, xi, w,
        full_size, half_size, block_size, aligned, blk_num);
}
#endif