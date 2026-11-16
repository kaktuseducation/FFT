// kernels/r2_stockham_perstage_xi.cpp
// R2-ST-PS-XI  (Stockham, per-stage, X=Interleaved, W=Packed)
//
// The same Stockham addressing as in r2_stockham_perstage_xs.cpp (fixed N/2
// split on the read, out-of-place write with stride L=2*halfSize) — I am not
// repeating the derivation of the formula here, see the comment in XS.cpp. The
// only difference is that X is stored interleaved (one buffer, re/im back to
// back), so the same scalar de-interleave/interleave as in
// r2_dit_perstage_xi.cpp and r2_dit_fused_xi.cpp is added around the butterfly.
//
// IMPORTANT: the same warning about halfSize<8 and the absence of a
// tail-restore trick as in r2_stockham_perstage_xs.cpp — the host must keep
// halfSize>=8 on the NPU path.

#include "kernel_operator.h"
#include <cstdint>

class KernelFFTStockhamXI {
public:
    __aicore__ inline KernelFFTStockhamXI() {}

    __aicore__ inline void Init(GM_ADDR X_IN, GM_ADDR X_OUT, GM_ADDR W,
                                 uint32_t full_size, uint32_t half_size, uint32_t aligned,
                                 uint32_t blk_num)
    {
        this->n       = full_size;
        this->halfSize    = half_size;      // Lstar
        this->tile    = 64;
        this->aligned = aligned;
        // Tile-distribution stride across blocks (cores).
        this->blkNum  = blk_num > 0 ? blk_num : 1;

        uint32_t bufSizeSingle = this->tile * sizeof(float);
        uint32_t bufSizePair   = 2 * this->tile * sizeof(float);

        xInGm.SetGlobalBuffer((__gm__ float *)X_IN,  2 * this->n);
        xOutGm.SetGlobalBuffer((__gm__ float *)X_OUT, 2 * this->n);
        wGm.SetGlobalBuffer((__gm__ float *)W, aligned * 2);

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
    // Tiles of ONE stage are handed out across cores: the block with index
    // blockIdx takes tiles tid = blockIdx, blockIdx+blkNum, ... — that is,
    // with stride blkNum over the linear numbering of (k, shift). The
    // numbering is linear rather than nested because the number of shifts per
    // k-block drops to one on the late stages, and handing out by the outer
    // index would leave almost every core without work.
    //
    // For Stockham this is safe in the strongest sense: a stage READS from one
    // buffer and WRITES into ANOTHER (the ping-pong is on the host side), so
    // there is no in-place aliasing at all — neither within a tile nor between
    // tiles on different cores. In the DIT kernels the write lands on top of
    // the array being read, and there the correctness of the distribution has
    // to be proven through non-overlapping segments; here there is nothing to
    // prove.
    //
    // The barrier between stages is still given by the launch boundary: the
    // host runs the stages as separate calls and swaps IN/OUT.
    // ========================================================================
    __aicore__ inline void Process()
    {
        uint32_t L = this->halfSize << 1;
        uint32_t r = this->n / L;

        uint32_t shiftsPerBlk = (this->halfSize + this->tile - 1) / this->tile;
        uint32_t nTiles       = r * shiftsPerBlk;
        uint32_t bid          = (uint32_t)AscendC::GetBlockIdx();

        for (uint32_t tid = bid; tid < nTiles; tid += this->blkNum) {
            uint32_t k     = tid / shiftsPerBlk;          // NOT multiplied by block
            uint32_t shift = (tid % shiftsPerBlk) * this->tile;
            ProcessTile(k, shift);
        }
    }

private:
    // The body of what used to be the inner loop of Process, lifted out as is,
    // so the math and the addressing stay bit-for-bit the same.
    __aicore__ inline void ProcessTile(uint32_t k, uint32_t shift)
    {
        // L is a pure function of halfSize, so we recompute it instead of
        // passing it in: that way not one line of the body below had to change.
        uint32_t L = this->halfSize << 1;

        uint32_t valid = (this->halfSize - shift < this->tile)
                         ? (this->halfSize - shift) : this->tile;

        this->curValid       = valid;
        this->alignedValid   = ((valid + 7)  / 8)  * 8;
        this->computeAligned = ((valid + 63) / 64) * 64;

        uint32_t idxIn   = k * this->halfSize + shift;
        uint32_t idxIn2  = idxIn + this->n / 2;
        uint32_t idxOut1 = k * L + shift;
        uint32_t idxOut2 = idxOut1 + this->halfSize;

        CopyIn(idxIn, idxIn2, shift);
        Compute();
        CopyOut(idxOut1, idxOut2);
    }

    __aicore__ inline void CopyIn(uint32_t idxIn, uint32_t idxIn2, uint32_t shift)
    {
        AscendC::LocalTensor<float> a = aIn.AllocTensor<float>();
        AscendC::LocalTensor<float> b = bIn.AllocTensor<float>();

        AscendC::DataCopy(a, xInGm[2 * idxIn],  2 * this->alignedValid);
        AscendC::DataCopy(b, xInGm[2 * idxIn2], 2 * this->alignedValid);

        AscendC::LocalTensor<float> wr = wrIn.AllocTensor<float>();
        AscendC::LocalTensor<float> wi = wiIn.AllocTensor<float>();
        AscendC::DataCopy(wr, wGm[shift],                 this->alignedValid);
        AscendC::DataCopy(wi, wGm[this->aligned + shift], this->alignedValid);

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

        // wb = w*b
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

        // out1 = a + w*b -> t1/t2 (reuse), out2 = a - w*b -> br/bi (reuse)
        AscendC::Add(t1, ar, wr, len);
        AscendC::Add(t2, ai, wi, len);
        AscendC::Sub(br, ar, wr, len);
        AscendC::Sub(bi, ai, wi, len);
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

    __aicore__ inline void CopyOut(uint32_t idxOut1, uint32_t idxOut2)
    {
        AscendC::LocalTensor<float> out1 = out1Out.DeQue<float>();
        AscendC::LocalTensor<float> out2 = out2Out.DeQue<float>();

        // NO tail-restore trick — the same warning as in XS.cpp: the host
        // must keep halfSize>=8 on the NPU path.
        AscendC::DataCopy(xOutGm[2 * idxOut1], out1, 2 * this->alignedValid);
        AscendC::DataCopy(xOutGm[2 * idxOut2], out2, 2 * this->alignedValid);
        AscendC::PipeBarrier<PIPE_MTE3>();

        out1Out.FreeTensor(out1); out2Out.FreeTensor(out2);
    }

private:
    AscendC::TPipe pipe;
    AscendC::GlobalTensor<float> xInGm, xOutGm, wGm;

    AscendC::TQue<AscendC::TPosition::VECIN,  2> aIn, bIn, wrIn, wiIn;
    AscendC::TQue<AscendC::TPosition::VECOUT, 2> out1Out, out2Out;
    AscendC::TBuf<AscendC::TPosition::VECCALC> arBuf, aiBuf, brBuf, biBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> t1Buf, t2Buf, t3Buf, t4Buf;

    uint32_t n, halfSize, tile, aligned;
    uint32_t blkNum;                 // stage tile stride across blocks (cores)
    uint32_t curValid, alignedValid, computeAligned;
};

extern "C" __global__ __aicore__ void fft_stockham_xi(
    GM_ADDR X_IN, GM_ADDR X_OUT, GM_ADDR W,
    uint32_t full_size, uint32_t half_size, uint32_t aligned, uint32_t blk_num)
{
    KernelFFTStockhamXI op;
    op.Init(X_IN, X_OUT, W, full_size, half_size, aligned, blk_num);
    op.Process();
}

#ifndef ASCENDC_CPU_DEBUG
void fft_stockham_xi_do(uint32_t blockDim, void *stream,
                         void *x_in, void *x_out, void *w,
                         uint32_t full_size, uint32_t half_size, uint32_t aligned,
                         uint32_t blk_num)
{
    fft_stockham_xi<<<blockDim, nullptr, stream>>>(x_in, x_out, w,
        full_size, half_size, aligned, blk_num);
}
#endif