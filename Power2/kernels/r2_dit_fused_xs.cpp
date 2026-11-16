// kernels/r2_dit_fused_xs.cpp
// R2-DT-FS-XS  (DIT, fused, X=Split, W=Packed)
//
// All NPU stages of this radix-2 FFT run inside ONE kernel launch: outside
// (on the host) META = [halfSize, aligned, wOffset] x numStages is computed
// once (see build_fused_plan/fill_fused_buffers_packed in host/fft_host_r2.h)
// together with one shared twiddle buffer W (packed, all stages back to back).
// Inside — a loop over stages, and on each stage the same butterfly tiling as
// in r2_dit_perstage_xs.cpp.
//
// *** CANDIDATE FIX FOR BUG 2 (race at halfSize>=128, see handoff) ***
// An explicit AscendC::PipeBarrier<PIPE_ALL>() sits between the stages — the
// hypothesis: without it the double-buffered pipeline (queue depth=2) can
// start the CopyIn of stage s+1 for some of the tiles before the CopyOut of
// stage s to those same GM addresses has actually completed (in fused mode
// this is the ONLY place where a synchronization point between stages can
// exist at all — the per-stage variant got one for free through
// aclrtSynchronizeStream on the host). Needs a repro campaign on
// N=4096/TILE=128 and N=16384/TILE=128 (see handoff, item 6) — run WITH this
// barrier and WITHOUT it, compare the % failure. If the barrier does not help,
// the cause is something else, and that will have to go into the thesis as a
// negative experimental result rather than be kept quiet about.

#include "kernel_operator.h"
#include <cstdint>

class KernelFFTFusedXS {
public:
    __aicore__ inline KernelFFTFusedXS() {}

    __aicore__ inline void Init(GM_ADDR XR, GM_ADDR XI, GM_ADDR W, GM_ADDR META,
                                 uint32_t full_size, uint32_t num_stages, uint32_t blk_num)
    {
        this->n         = full_size;
        this->tile      = 64;
        this->numStages = num_stages;
        // Tile-distribution stride across blocks in multicore mode.
        this->blkNum    = blk_num > 0 ? blk_num : 1;

        uint32_t bufSizeSingle = this->tile * sizeof(float);

        xrGm.SetGlobalBuffer((__gm__ float *)XR, this->n);
        xiGm.SetGlobalBuffer((__gm__ float *)XI, this->n);
        // The size of W is not known to the kernel up front (it depends on the
        // sum of aligned*2 over all stages) — so we pass a large enough upper
        // bound via n (W is never more than a few n by how the plan is built).
        wGm.SetGlobalBuffer((__gm__ float *)W, this->n * 2);
        metaGm.SetGlobalBuffer((__gm__ uint32_t *)META, this->numStages * 3);

        pipe.InitBuffer(xrAIn,  2, bufSizeSingle);
        pipe.InitBuffer(xiAIn,  2, bufSizeSingle);
        pipe.InitBuffer(xrBIn,  2, bufSizeSingle);
        pipe.InitBuffer(xiBIn,  2, bufSizeSingle);
        pipe.InitBuffer(xrAOut, 2, bufSizeSingle);
        pipe.InitBuffer(xiAOut, 2, bufSizeSingle);
        pipe.InitBuffer(xrBOut, 2, bufSizeSingle);
        pipe.InitBuffer(xiBOut, 2, bufSizeSingle);
        pipe.InitBuffer(wrIn, 2, bufSizeSingle);
        pipe.InitBuffer(wiIn, 2, bufSizeSingle);

        pipe.InitBuffer(t1Buf, bufSizeSingle);
        pipe.InitBuffer(t2Buf, bufSizeSingle);
        pipe.InitBuffer(t3Buf, bufSizeSingle);
        pipe.InitBuffer(t4Buf, bufSizeSingle);
    }

    // ========================================================================
    // TWO OPERATING MODES — exactly as in kernels/r4_dit_fused_xs.cpp and
    // kernels/r8_dit_fused_xs.cpp, see the detailed rationale there.
    //
    // Process() — as it was: all stages in one launch, one core (variant
    // r2/dit_fs_xs, the numbers stayed bit-for-bit the same).
    // ProcessOneStage(stage) — one stage per launch, tiles across blocks
    // (variant r2/dit_fsm_xs). The barrier between stages is given by stream.
    //
    // Handing tiles out across cores is correct: tile (k, shift) reads and
    // writes its own two segments idxA/idxB of length valid, and segments of
    // different tiles do not overlap. The tail trick in CopyOut (when
    // alignedValid > curValid) never fires at present, because halfSize on the
    // NPU is a power of two no smaller than 8, and such numbers are multiples
    // of eight.
    // ========================================================================
    __aicore__ inline void Process()
    {
        for (uint32_t s = 0; s < this->numStages; ++s) {
            uint32_t halfSize    = metaGm.GetValue(s * 3 + 0);
            uint32_t aligned = metaGm.GetValue(s * 3 + 1);
            uint32_t wOffset = metaGm.GetValue(s * 3 + 2);
            uint32_t block   = halfSize << 1;

            ProcessStage(halfSize, aligned, wOffset, block);

            // --- CANDIDATE FIX for Bug 2: see the comment in the header ---
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }

    __aicore__ inline void ProcessOneStage(uint32_t stage)
    {
        uint32_t halfSize = metaGm.GetValue(stage * 3 + 0);
        uint32_t aligned  = metaGm.GetValue(stage * 3 + 1);
        uint32_t wOffset  = metaGm.GetValue(stage * 3 + 2);
        uint32_t block    = halfSize << 1;

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
        AscendC::LocalTensor<float> ar = xrAIn.AllocTensor<float>();
        AscendC::LocalTensor<float> ai = xiAIn.AllocTensor<float>();
        AscendC::LocalTensor<float> br = xrBIn.AllocTensor<float>();
        AscendC::LocalTensor<float> bi = xiBIn.AllocTensor<float>();

        AscendC::DataCopy(ar, xrGm[idxA], this->alignedValid);
        AscendC::DataCopy(ai, xiGm[idxA], this->alignedValid);
        AscendC::DataCopy(br, xrGm[idxB], this->alignedValid);
        AscendC::DataCopy(bi, xiGm[idxB], this->alignedValid);

        AscendC::LocalTensor<float> wr = wrIn.AllocTensor<float>();
        AscendC::LocalTensor<float> wi = wiIn.AllocTensor<float>();
        AscendC::DataCopy(wr, wGm[wOffset + shift],           this->alignedValid);
        AscendC::DataCopy(wi, wGm[wOffset + aligned + shift], this->alignedValid);

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

        AscendC::Sub(wr, t1, t2, len);   // tr
        AscendC::Add(wi, t3, t4, len);   // ti
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::LocalTensor<float> outAr = xrAOut.AllocTensor<float>();
        AscendC::LocalTensor<float> outAi = xiAOut.AllocTensor<float>();
        AscendC::LocalTensor<float> outBr = xrBOut.AllocTensor<float>();
        AscendC::LocalTensor<float> outBi = xiBOut.AllocTensor<float>();

        AscendC::Add(outAr, ar, wr, len);
        AscendC::Add(outAi, ai, wi, len);
        AscendC::Sub(outBr, ar, wr, len);
        AscendC::Sub(outBi, ai, wi, len);
        AscendC::PipeBarrier<PIPE_V>();

        xrAOut.EnQue(outAr); xiAOut.EnQue(outAi);
        xrBOut.EnQue(outBr); xiBOut.EnQue(outBi);

        xrAIn.FreeTensor(ar); xiAIn.FreeTensor(ai);
        xrBIn.FreeTensor(br); xiBIn.FreeTensor(bi);
        wrIn.FreeTensor(wr);  wiIn.FreeTensor(wi);
    }

    __aicore__ inline void CopyOut(uint32_t idxA, uint32_t idxB)
    {
        AscendC::LocalTensor<float> outAr = xrAOut.DeQue<float>();
        AscendC::LocalTensor<float> outAi = xiAOut.DeQue<float>();
        AscendC::LocalTensor<float> outBr = xrBOut.DeQue<float>();
        AscendC::LocalTensor<float> outBi = xiBOut.DeQue<float>();

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
    AscendC::GlobalTensor<float>    xrGm, xiGm, wGm;
    AscendC::GlobalTensor<uint32_t> metaGm;

    AscendC::TQue<AscendC::TPosition::VECIN,  2> xrAIn, xiAIn, xrBIn, xiBIn;
    AscendC::TQue<AscendC::TPosition::VECOUT, 2> xrAOut, xiAOut, xrBOut, xiBOut;
    AscendC::TQue<AscendC::TPosition::VECIN,  2> wrIn, wiIn;
    AscendC::TBuf<AscendC::TPosition::VECCALC> t1Buf, t2Buf, t3Buf, t4Buf;

    uint32_t n, tile, numStages;
    uint32_t blkNum;                 // tile stride across blocks (multicore mode)
    uint32_t curValid, alignedValid, computeAligned;
};

// stage == FFT2_ALL_STAGES — single-core legacy mode (variant r2/dit_fs_xs);
// any other value means multicore: exactly this stage, tiles across blocks
// (variant r2/dit_fsm_xs). One kernel with two modes rather than a copy of the
// file, so that the math of both variants is literally one and the same.
#define FFT2_ALL_STAGES 0xFFFFFFFFu

extern "C" __global__ __aicore__ void fft_dit_fused_xs(
    GM_ADDR XR, GM_ADDR XI, GM_ADDR W, GM_ADDR META,
    uint32_t full_size, uint32_t num_stages, uint32_t stage, uint32_t blk_num)
{
    KernelFFTFusedXS op;
    op.Init(XR, XI, W, META, full_size, num_stages, blk_num);
    if (stage == FFT2_ALL_STAGES) {
        op.Process();
    } else {
        op.ProcessOneStage(stage);
    }
}

#ifndef ASCENDC_CPU_DEBUG
void fft_dit_fused_xs_do(uint32_t blockDim, void *stream,
                          void *xr, void *xi, void *w, void *meta,
                          uint32_t full_size, uint32_t num_stages,
                          uint32_t stage, uint32_t blk_num)
{
    fft_dit_fused_xs<<<blockDim, nullptr, stream>>>(xr, xi, w, meta,
        full_size, num_stages, stage, blk_num);
}
#endif