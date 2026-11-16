// kernels/r4_stockham_perstage_xs.cpp
// R4-ST-PS-XS  (radix4 Stockham, per-stage, X=Split, W=Packed-3)
//
// The same butterfly math (combining x0..x3 into y0..y3) as in
// r4_dit_perstage_xs.cpp — see the derivation/check in host/fft_host_r4.h.
// The difference — self-sorting read addressing (fixed offsets N/4, N/2,
// 3N/4, independent of the stage, in the same way N/2 was fixed in the
// radix2 Stockham) and an OUT-OF-PLACE write with stride L=4*quarter between
// blocks. There is no tail-restore trick (as in the radix2 Stockham) — the
// host must keep quarter>=8 on the NPU path.

#include "kernel_operator.h"
#include <cstdint>

class KernelFFT4StockhamXS {
public:
    __aicore__ inline KernelFFT4StockhamXS() {}

    __aicore__ inline void Init(GM_ADDR XR_IN, GM_ADDR XI_IN,
                                 GM_ADDR XR_OUT, GM_ADDR XI_OUT, GM_ADDR W,
                                 uint32_t full_size, uint32_t quarter_size, uint32_t aligned,
                                 uint32_t blk_num)
    {
        this->n       = full_size;
        this->q       = quarter_size;
        this->tile    = 64;
        this->aligned = aligned;
        // Tile-distribution stride across blocks (cores).
        this->blkNum  = blk_num > 0 ? blk_num : 1;

        uint32_t bufS = this->tile * sizeof(float);

        xrInGm.SetGlobalBuffer((__gm__ float *)XR_IN, this->n);
        xiInGm.SetGlobalBuffer((__gm__ float *)XI_IN, this->n);
        xrOutGm.SetGlobalBuffer((__gm__ float *)XR_OUT, this->n);
        xiOutGm.SetGlobalBuffer((__gm__ float *)XI_OUT, this->n);
        wGm.SetGlobalBuffer((__gm__ float *)W, aligned * 6);

        // IMPORTANT: one input and one output queue (not 14 + 8).
        // The number of hardware event IDs is limited (8 per pipeline pair), and
        // with more TQue than that the events collide -> deadlock in EnQue/DeQue
        // (a hang). All tensors live in one buffer and are carved up by offsets
        // of TILE_F float.
        pipe.InitBuffer(inQ,  2, NIN_T  * bufS);
        pipe.InitBuffer(outQ, 2, NOUT_T * bufS);
        pipe.InitBuffer(calcBuf, NCALC_T * bufS);
    }

    // ========================================================================
    // The tiles of ONE stage are handed out across cores: the block with index
    // blockIdx takes the tiles tid = blockIdx, blockIdx+blkNum, ... — that is,
    // with stride blkNum over the linear (k, shift) numbering. The numbering is
    // linear and not nested because the number of shifts per k-block drops to
    // one on the late stages (quarter shrinks by a factor of four per stage),
    // and handing out by the outer index would leave almost every core idle.
    //
    // For Stockham this is safe in the strongest sense: a stage READS from one
    // buffer and WRITES into ANOTHER (the ping-pong lives on the host side), so
    // there is no in-place aliasing at all — neither inside a tile nor between
    // the tiles of different cores. In the DIT kernels the write goes on top of
    // the array being read, and there the correctness of the distribution has
    // to be proven through the non-overlap of the segments i0..i3; here there
    // is nothing to prove.
    //
    // The barrier between stages is still given by the launch boundary: the
    // host drives the stages as separate calls and swaps IN/OUT around.
    // ========================================================================
    __aicore__ inline void Process()
    {
        uint32_t L = this->q << 2;
        uint32_t r = this->n / L;

        uint32_t shiftsPerBlk = (this->q + this->tile - 1) / this->tile;
        uint32_t nTiles       = r * shiftsPerBlk;
        uint32_t bid          = (uint32_t)AscendC::GetBlockIdx();

        for (uint32_t tid = bid; tid < nTiles; tid += this->blkNum) {
            uint32_t k     = tid / shiftsPerBlk;          // NOT multiplied by block
            uint32_t shift = (tid % shiftsPerBlk) * this->tile;
            ProcessTile(k, shift);
        }
    }

private:
    // The body of the former inner loop of Process, lifted out as is, so that
    // the math and the addressing stay bit-for-bit the same.
    __aicore__ inline void ProcessTile(uint32_t k, uint32_t shift)
    {
        // L is a pure function of q, so we recompute it instead of passing it
        // as a parameter: that way the body below needed no change on any line.
        uint32_t L = this->q << 2;

        uint32_t valid = (this->q - shift < this->tile) ? (this->q - shift) : this->tile;
        this->curValid       = valid;
        this->alignedValid   = ((valid + 7)  / 8)  * 8;
        this->computeAligned = ((valid + 63) / 64) * 64;

        uint32_t i0 = k * this->q + shift;
        uint32_t i1 = i0 + this->n / 4;
        uint32_t i2 = i0 + this->n / 2;
        uint32_t i3 = i0 + 3 * this->n / 4;
        uint32_t o0 = k * L + shift;
        uint32_t o1 = o0 + this->q, o2 = o1 + this->q, o3 = o2 + this->q;

        CopyIn(i0, i1, i2, i3, shift);
        Compute();
        CopyOut(o0, o1, o2, o3);
    }

    __aicore__ inline void CopyIn(uint32_t i0, uint32_t i1, uint32_t i2, uint32_t i3, uint32_t shift)
    {
        auto inT = inQ.AllocTensor<float>();
        auto x0r = inT[0 * TILE_F];
        auto x0i = inT[1 * TILE_F];
        auto x1r = inT[2 * TILE_F];
        auto x1i = inT[3 * TILE_F];
        auto x2r = inT[4 * TILE_F];
        auto x2i = inT[5 * TILE_F];
        auto x3r = inT[6 * TILE_F];
        auto x3i = inT[7 * TILE_F];

        AscendC::DataCopy(x0r, xrInGm[i0], this->alignedValid); AscendC::DataCopy(x0i, xiInGm[i0], this->alignedValid);
        AscendC::DataCopy(x1r, xrInGm[i1], this->alignedValid); AscendC::DataCopy(x1i, xiInGm[i1], this->alignedValid);
        AscendC::DataCopy(x2r, xrInGm[i2], this->alignedValid); AscendC::DataCopy(x2i, xiInGm[i2], this->alignedValid);
        AscendC::DataCopy(x3r, xrInGm[i3], this->alignedValid); AscendC::DataCopy(x3i, xiInGm[i3], this->alignedValid);

        auto w1r = inT[8 * TILE_F];
        auto w1i = inT[9 * TILE_F];
        auto w2r = inT[10 * TILE_F];
        auto w2i = inT[11 * TILE_F];
        auto w3r = inT[12 * TILE_F];
        auto w3i = inT[13 * TILE_F];

        AscendC::DataCopy(w1r, wGm[shift],                     this->alignedValid);
        AscendC::DataCopy(w1i, wGm[this->aligned + shift],     this->alignedValid);
        AscendC::DataCopy(w2r, wGm[2*this->aligned + shift],   this->alignedValid);
        AscendC::DataCopy(w2i, wGm[3*this->aligned + shift],   this->alignedValid);
        AscendC::DataCopy(w3r, wGm[4*this->aligned + shift],   this->alignedValid);
        AscendC::DataCopy(w3i, wGm[5*this->aligned + shift],   this->alignedValid);


        inQ.EnQue(inT);
    }

    __aicore__ inline void Compute()
    {
        auto inT = inQ.DeQue<float>();
        auto x0r = inT[0 * TILE_F];
        auto x0i = inT[1 * TILE_F];
        auto x1r = inT[2 * TILE_F];
        auto x1i = inT[3 * TILE_F];
        auto x2r = inT[4 * TILE_F];
        auto x2i = inT[5 * TILE_F];
        auto x3r = inT[6 * TILE_F];
        auto x3i = inT[7 * TILE_F];
        auto w1r = inT[8 * TILE_F];
        auto w1i = inT[9 * TILE_F];
        auto w2r = inT[10 * TILE_F];
        auto w2i = inT[11 * TILE_F];
        auto w3r = inT[12 * TILE_F];
        auto w3i = inT[13 * TILE_F];

        auto calcT = calcBuf.Get<float>();
        auto t1r = calcT[0 * TILE_F];
        auto t1i = calcT[1 * TILE_F];
        auto t2r = calcT[2 * TILE_F];
        auto t2i = calcT[3 * TILE_F];
        auto t3r = calcT[4 * TILE_F];
        auto t3i = calcT[5 * TILE_F];
        auto u = calcT[6 * TILE_F];
        auto v = calcT[7 * TILE_F];

        if (this->computeAligned > this->alignedValid) {
            uint32_t tail = this->computeAligned - this->alignedValid;
            AscendC::Duplicate(w1r[this->alignedValid], (float)0.0f, tail);
            AscendC::Duplicate(w1i[this->alignedValid], (float)0.0f, tail);
            AscendC::Duplicate(w2r[this->alignedValid], (float)0.0f, tail);
            AscendC::Duplicate(w2i[this->alignedValid], (float)0.0f, tail);
            AscendC::Duplicate(w3r[this->alignedValid], (float)0.0f, tail);
            AscendC::Duplicate(w3i[this->alignedValid], (float)0.0f, tail);
            AscendC::PipeBarrier<PIPE_V>();
        }
        uint32_t len = this->computeAligned;

        AscendC::Mul(u, x1r, w1r, len); AscendC::Mul(v, x1i, w1i, len); AscendC::Sub(t1r, u, v, len);
        AscendC::Mul(u, x1r, w1i, len); AscendC::Mul(v, x1i, w1r, len); AscendC::Add(t1i, u, v, len);
        AscendC::Mul(u, x2r, w2r, len); AscendC::Mul(v, x2i, w2i, len); AscendC::Sub(t2r, u, v, len);
        AscendC::Mul(u, x2r, w2i, len); AscendC::Mul(v, x2i, w2r, len); AscendC::Add(t2i, u, v, len);
        AscendC::Mul(u, x3r, w3r, len); AscendC::Mul(v, x3i, w3i, len); AscendC::Sub(t3r, u, v, len);
        AscendC::Mul(u, x3r, w3i, len); AscendC::Mul(v, x3i, w3r, len); AscendC::Add(t3i, u, v, len);
        AscendC::PipeBarrier<PIPE_V>();

        auto outT = outQ.AllocTensor<float>();
        auto y0r = outT[0 * TILE_F];
        auto y0i = outT[1 * TILE_F];
        auto y1r = outT[2 * TILE_F];
        auto y1i = outT[3 * TILE_F];
        auto y2r = outT[4 * TILE_F];
        auto y2i = outT[5 * TILE_F];
        auto y3r = outT[6 * TILE_F];
        auto y3i = outT[7 * TILE_F];

        AscendC::Add(y0r, x0r, t1r, len); AscendC::Add(y0r, y0r, t2r, len); AscendC::Add(y0r, y0r, t3r, len);
        AscendC::Add(y0i, x0i, t1i, len); AscendC::Add(y0i, y0i, t2i, len); AscendC::Add(y0i, y0i, t3i, len);
        AscendC::Add(y1r, x0r, t1i, len); AscendC::Sub(y1r, y1r, t2r, len); AscendC::Sub(y1r, y1r, t3i, len);
        AscendC::Sub(y1i, x0i, t1r, len); AscendC::Sub(y1i, y1i, t2i, len); AscendC::Add(y1i, y1i, t3r, len);
        AscendC::Sub(y2r, x0r, t1r, len); AscendC::Add(y2r, y2r, t2r, len); AscendC::Sub(y2r, y2r, t3r, len);
        AscendC::Sub(y2i, x0i, t1i, len); AscendC::Add(y2i, y2i, t2i, len); AscendC::Sub(y2i, y2i, t3i, len);
        AscendC::Sub(y3r, x0r, t1i, len); AscendC::Sub(y3r, y3r, t2r, len); AscendC::Add(y3r, y3r, t3i, len);
        AscendC::Add(y3i, x0i, t1r, len); AscendC::Sub(y3i, y3i, t2i, len); AscendC::Sub(y3i, y3i, t3r, len);
        AscendC::PipeBarrier<PIPE_V>();

        outQ.EnQue(outT);
        inQ.FreeTensor(inT);
    }

    __aicore__ inline void CopyOut(uint32_t o0, uint32_t o1, uint32_t o2, uint32_t o3)
    {
        auto outT = outQ.DeQue<float>();
        auto y0r = outT[0 * TILE_F];
        auto y0i = outT[1 * TILE_F];
        auto y1r = outT[2 * TILE_F];
        auto y1i = outT[3 * TILE_F];
        auto y2r = outT[4 * TILE_F];
        auto y2i = outT[5 * TILE_F];
        auto y3r = outT[6 * TILE_F];
        auto y3i = outT[7 * TILE_F];

        // NO tail-restore — the host keeps quarter>=8 (see the file header).
        AscendC::DataCopy(xrOutGm[o0], y0r, this->alignedValid); AscendC::DataCopy(xiOutGm[o0], y0i, this->alignedValid);
        AscendC::DataCopy(xrOutGm[o1], y1r, this->alignedValid); AscendC::DataCopy(xiOutGm[o1], y1i, this->alignedValid);
        AscendC::DataCopy(xrOutGm[o2], y2r, this->alignedValid); AscendC::DataCopy(xiOutGm[o2], y2i, this->alignedValid);
        AscendC::DataCopy(xrOutGm[o3], y3r, this->alignedValid); AscendC::DataCopy(xiOutGm[o3], y3i, this->alignedValid);

        outQ.FreeTensor(outT);
    }

private:
    AscendC::TPipe pipe;
    AscendC::GlobalTensor<float> xrInGm, xiInGm, xrOutGm, xiOutGm, wGm;

    static constexpr uint32_t TILE_F  = 64;   // float per tensor slice (= tile)
    static constexpr uint32_t NIN_T   = 14;   // x0..x3 (re,im) + w1..w3 (re,im)
    static constexpr uint32_t NOUT_T  = 8;    // y0..y3 (re,im)
    static constexpr uint32_t NCALC_T = 8;    // t1..t3 (re,im) + u + v

    AscendC::TQue<AscendC::TPosition::VECIN,  2> inQ;
    AscendC::TQue<AscendC::TPosition::VECOUT, 2> outQ;
    AscendC::TBuf<AscendC::TPosition::VECCALC>   calcBuf;

    uint32_t n, q, tile, aligned;
    uint32_t blkNum;                 // stage tile-distribution stride across blocks (cores)
    uint32_t curValid, alignedValid, computeAligned;
};

extern "C" __global__ __aicore__ void fft4_stockham_xs(
    GM_ADDR XR_IN, GM_ADDR XI_IN, GM_ADDR XR_OUT, GM_ADDR XI_OUT, GM_ADDR W,
    uint32_t full_size, uint32_t quarter_size, uint32_t aligned, uint32_t blk_num)
{
    KernelFFT4StockhamXS op;
    op.Init(XR_IN, XI_IN, XR_OUT, XI_OUT, W, full_size, quarter_size, aligned, blk_num);
    op.Process();
}

#ifndef ASCENDC_CPU_DEBUG
void fft4_stockham_xs_do(uint32_t blockDim, void *stream,
                          void *xr_in, void *xi_in, void *xr_out, void *xi_out, void *w,
                          uint32_t full_size, uint32_t quarter_size, uint32_t aligned,
                          uint32_t blk_num)
{
    fft4_stockham_xs<<<blockDim, nullptr, stream>>>(xr_in, xi_in, xr_out, xi_out, w,
        full_size, quarter_size, aligned, blk_num);
}
#endif
