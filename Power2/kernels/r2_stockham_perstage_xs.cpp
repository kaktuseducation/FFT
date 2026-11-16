// kernels/r2_stockham_perstage_xs.cpp
// R2-ST-PS-XS  (Stockham, per-stage, X=Split, W=Packed)
//
// Algorithm (self-sorting radix-2 Stockham DIT, out-of-place):
//   On the stage with halfSize=Lstar=2^(s-1), L=2*Lstar, r=N/L:
//     for k in [0,r):
//       for j in [0,Lstar):
//         a = IN[k*Lstar + j]            <- ALWAYS from the input's 1st half
//         b = IN[k*Lstar + j + N/2]      <- ALWAYS from the second half (the
//                                            offset is fixed at N/2, it does
//                                            not change from stage to stage
//                                            — that is the difference from
//                                            DIT, where offset B = halfSize)
//         w = exp(-2pi*i*j/L)            <- the same formula as in DIT
//         OUT[k*L + j]         = a + w*b
//         OUT[k*L + j + Lstar] = a - w*b
//
// The host must alternate the IN/OUT buffers between stage calls (ping-pong):
// odd stages read from buffer 0 and write into buffer 1, even stages the other
// way round (or however suits the host side; the kernel keeps no state between
// calls). After m stages the final result sits in the buffer with index
// (m mod 2) — that is the responsibility of host main.cpp; the kernel does not
// know about it.
//
// IMPORTANT (the same kind of bug as bug 1 in DIT, see handoff): at halfSize<8
// the round8 padding of the out1 write (a+w*b) can run into the range of out2
// (a-w*b) of THE SAME block k. Unlike DIT, there is no "restore the old value"
// trick here — the output buffer for this stage holds nothing meaningful
// beforehand, so there is nothing to restore. That is why the host MUST keep
// halfSize<8 off the NPU path (the same min_npu_half>=8 threshold already used
// for DIT in build_fused_plan) — the kernel does not do this guarding itself.

#include "kernel_operator.h"
#include <cstdint>

class KernelFFTStockhamXS {
public:
    __aicore__ inline KernelFFTStockhamXS() {}

    __aicore__ inline void Init(GM_ADDR XR_IN, GM_ADDR XI_IN,
                                 GM_ADDR XR_OUT, GM_ADDR XI_OUT, GM_ADDR W,
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

        xrInGm.SetGlobalBuffer((__gm__ float *)XR_IN,  this->n);
        xiInGm.SetGlobalBuffer((__gm__ float *)XI_IN,  this->n);
        xrOutGm.SetGlobalBuffer((__gm__ float *)XR_OUT, this->n);
        xiOutGm.SetGlobalBuffer((__gm__ float *)XI_OUT, this->n);
        wGm.SetGlobalBuffer((__gm__ float *)W, aligned * 2);

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
    // to be proven through the non-overlap of segments i0..i3; here there is
    // nothing to prove.
    //
    // The barrier between stages is still given by the launch boundary: the
    // host runs the stages as separate calls and swaps IN/OUT (see the header).
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

        uint32_t idxIn  = k * this->halfSize + shift;              // a
        uint32_t idxIn2 = idxIn + this->n / 2;                 // b (fixed N/2!)
        uint32_t idxOut1 = k * L + shift;                      // a+w*b
        uint32_t idxOut2 = idxOut1 + this->halfSize;               // a-w*b

        CopyIn(idxIn, idxIn2, shift);
        Compute();
        CopyOut(idxOut1, idxOut2);
    }

    __aicore__ inline void CopyIn(uint32_t idxIn, uint32_t idxIn2, uint32_t shift)
    {
        AscendC::LocalTensor<float> ar = arIn.AllocTensor<float>();
        AscendC::LocalTensor<float> ai = aiIn.AllocTensor<float>();
        AscendC::LocalTensor<float> br = brIn.AllocTensor<float>();
        AscendC::LocalTensor<float> bi = biIn.AllocTensor<float>();

        AscendC::DataCopy(ar, xrInGm[idxIn],  this->alignedValid);
        AscendC::DataCopy(ai, xiInGm[idxIn],  this->alignedValid);
        AscendC::DataCopy(br, xrInGm[idxIn2], this->alignedValid);
        AscendC::DataCopy(bi, xiInGm[idxIn2], this->alignedValid);

        AscendC::LocalTensor<float> wr = wrIn.AllocTensor<float>();
        AscendC::LocalTensor<float> wi = wiIn.AllocTensor<float>();
        AscendC::DataCopy(wr, wGm[shift],                 this->alignedValid);
        AscendC::DataCopy(wi, wGm[this->aligned + shift], this->alignedValid);

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

        // wb = w*b :  wbr = br*wr - bi*wi,  wbi = br*wi + bi*wr
        AscendC::Mul(t1, br, wr, len);
        AscendC::Mul(t2, bi, wi, len);
        AscendC::Mul(t3, br, wi, len);
        AscendC::Mul(t4, bi, wr, len);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Sub(wr, t1, t2, len);   // wbr (reuse wr slot)
        AscendC::Add(wi, t3, t4, len);   // wbi (reuse wi slot)
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::LocalTensor<float> o1r = out1ROut.AllocTensor<float>();
        AscendC::LocalTensor<float> o1i = out1IOut.AllocTensor<float>();
        AscendC::LocalTensor<float> o2r = out2ROut.AllocTensor<float>();
        AscendC::LocalTensor<float> o2i = out2IOut.AllocTensor<float>();

        AscendC::Add(o1r, ar, wr, len);   // out1 = a + w*b
        AscendC::Add(o1i, ai, wi, len);
        AscendC::Sub(o2r, ar, wr, len);   // out2 = a - w*b
        AscendC::Sub(o2i, ai, wi, len);
        AscendC::PipeBarrier<PIPE_V>();

        out1ROut.EnQue(o1r); out1IOut.EnQue(o1i);
        out2ROut.EnQue(o2r); out2IOut.EnQue(o2i);

        arIn.FreeTensor(ar); aiIn.FreeTensor(ai);
        brIn.FreeTensor(br); biIn.FreeTensor(bi);
        wrIn.FreeTensor(wr); wiIn.FreeTensor(wi);
    }

    __aicore__ inline void CopyOut(uint32_t idxOut1, uint32_t idxOut2)
    {
        AscendC::LocalTensor<float> o1r = out1ROut.DeQue<float>();
        AscendC::LocalTensor<float> o1i = out1IOut.DeQue<float>();
        AscendC::LocalTensor<float> o2r = out2ROut.DeQue<float>();
        AscendC::LocalTensor<float> o2i = out2IOut.DeQue<float>();

        // NO tail-restore trick like in DIT — see the warning in the file
        // header. The host must guarantee this->halfSize >= 8 on this path.
        AscendC::DataCopy(xrOutGm[idxOut1], o1r, this->alignedValid);
        AscendC::DataCopy(xiOutGm[idxOut1], o1i, this->alignedValid);
        AscendC::DataCopy(xrOutGm[idxOut2], o2r, this->alignedValid);
        AscendC::DataCopy(xiOutGm[idxOut2], o2i, this->alignedValid);
        AscendC::PipeBarrier<PIPE_MTE3>();

        out1ROut.FreeTensor(o1r); out1IOut.FreeTensor(o1i);
        out2ROut.FreeTensor(o2r); out2IOut.FreeTensor(o2i);
    }

private:
    AscendC::TPipe pipe;
    AscendC::GlobalTensor<float> xrInGm, xiInGm, xrOutGm, xiOutGm, wGm;

    AscendC::TQue<AscendC::TPosition::VECIN,  2> arIn, aiIn, brIn, biIn, wrIn, wiIn;
    AscendC::TQue<AscendC::TPosition::VECOUT, 2> out1ROut, out1IOut, out2ROut, out2IOut;
    AscendC::TBuf<AscendC::TPosition::VECCALC> t1Buf, t2Buf, t3Buf, t4Buf;

    uint32_t n, halfSize, tile, aligned;
    uint32_t blkNum;                 // stage tile stride across blocks (cores)
    uint32_t curValid, alignedValid, computeAligned;
};

extern "C" __global__ __aicore__ void fft_stockham_xs(
    GM_ADDR XR_IN, GM_ADDR XI_IN, GM_ADDR XR_OUT, GM_ADDR XI_OUT, GM_ADDR W,
    uint32_t full_size, uint32_t half_size, uint32_t aligned, uint32_t blk_num)
{
    KernelFFTStockhamXS op;
    op.Init(XR_IN, XI_IN, XR_OUT, XI_OUT, W, full_size, half_size, aligned, blk_num);
    op.Process();
}

#ifndef ASCENDC_CPU_DEBUG
void fft_stockham_xs_do(uint32_t blockDim, void *stream,
                         void *xr_in, void *xi_in, void *xr_out, void *xi_out, void *w,
                         uint32_t full_size, uint32_t half_size, uint32_t aligned,
                         uint32_t blk_num)
{
    fft_stockham_xs<<<blockDim, nullptr, stream>>>(xr_in, xi_in, xr_out, xi_out, w,
        full_size, half_size, aligned, blk_num);
}
#endif