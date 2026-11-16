// kernels/r4_dit_perstage_xs.cpp
// R4-DT-PS-XS  (radix4 DIT, per-stage, X=Split, W=Packed-3)
//
// The formula and its check at N=4 — see host/fft_host_r4.h (the file header).
// The same pattern as the radix-2 r2_dit_perstage_xs.cpp: tiling by 64, the
// round8 tail-restore trick for valid<8 (extended to 4 groups instead of 2 —
// the overrun of group i can reach into group i+1, and we restore it the same
// way as in radix2, only at 4 positions instead of 2). The host must keep
// quarter>=8 on the NPU path (the same threshold as in radix2, see
// build_fused_plan4).

#include "kernel_operator.h"
#include <cstdint>

class KernelFFT4Hybrid {
public:
    __aicore__ inline KernelFFT4Hybrid() {}

    __aicore__ inline void Init(GM_ADDR XR, GM_ADDR XI, GM_ADDR W,
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

        xrGm.SetGlobalBuffer((__gm__ float *)XR, this->n);
        xiGm.SetGlobalBuffer((__gm__ float *)XI, this->n);
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
    // The tiles of ONE stage are handed out across blocks (cores): a block
    // takes tiles starting at its own blockIdx and onwards with stride blkNum.
    // For that, the double loop over (k, shift) has been flattened into a
    // single linear tile numbering — otherwise only the outer loop could be
    // handed out across cores, and with a small number of k-blocks (the late
    // stages: block_size close to n) only a handful of cores would be at work.
    //
    // Why the distribution is correct: the tile (k, shift) reads and writes
    // exactly its own four segments i0..i3 of length valid, and the segments
    // of different tiles do not overlap (the shifts cut [0, q) into tile-sized
    // pieces, the k-blocks run back to back). There are no inter-tile
    // dependencies inside a stage, and the barrier between stages is given by
    // the ordering of the launches on the stream itself.
    //
    // NOTE for the future: the tail trick in CopyOut (the curValid..
    // alignedValid loop that reads GM back) writes OTHER tiles' elements back.
    // Right now it never fires, because q on the NPU path is a power of four no
    // lower than 8 (16, 64, ...), hence a multiple of eight, and alignedValid
    // is always equal to curValid. If q ever becomes smaller than 8, this trick
    // will turn into a race between cores: by that point the neighbour may
    // already have written its own segment, and its result will be clobbered
    // by the old value.
    // ========================================================================
    __aicore__ inline void Process(uint32_t block_size)
    {
        uint32_t shiftsPerBlk = (this->q + this->tile - 1) / this->tile;
        uint32_t nTiles       = (this->n / block_size) * shiftsPerBlk;
        uint32_t bid          = (uint32_t)AscendC::GetBlockIdx();

        for (uint32_t tid = bid; tid < nTiles; tid += this->blkNum) {
            uint32_t k     = (tid / shiftsPerBlk) * block_size;
            uint32_t shift = (tid % shiftsPerBlk) * this->tile;
            ProcessTile(k, shift);
        }
    }

private:
    // The body of the former inner loop of Process, lifted out as is, so that
    // the tile math stays provably the same as in the single-core version.
    __aicore__ inline void ProcessTile(uint32_t k, uint32_t shift)
    {
        uint32_t i0 = k + shift, i1 = i0 + this->q, i2 = i1 + this->q, i3 = i2 + this->q;
        uint32_t valid = (this->q - shift < this->tile) ? (this->q - shift) : this->tile;

        this->curValid       = valid;
        this->alignedValid   = ((valid + 7)  / 8)  * 8;
        this->computeAligned = ((valid + 63) / 64) * 64;

        CopyIn(i0, i1, i2, i3, shift);
        Compute();
        CopyOut(i0, i1, i2, i3);
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

        AscendC::DataCopy(x0r, xrGm[i0], this->alignedValid); AscendC::DataCopy(x0i, xiGm[i0], this->alignedValid);
        AscendC::DataCopy(x1r, xrGm[i1], this->alignedValid); AscendC::DataCopy(x1i, xiGm[i1], this->alignedValid);
        AscendC::DataCopy(x2r, xrGm[i2], this->alignedValid); AscendC::DataCopy(x2i, xiGm[i2], this->alignedValid);
        AscendC::DataCopy(x3r, xrGm[i3], this->alignedValid); AscendC::DataCopy(x3i, xiGm[i3], this->alignedValid);

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

        // t1 = x1*w1 ; t2 = x2*w2 ; t3 = x3*w3  (complex multiplication)
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

        // y0 = x0+t1+t2+t3
        AscendC::Add(y0r, x0r, t1r, len); AscendC::Add(y0r, y0r, t2r, len); AscendC::Add(y0r, y0r, t3r, len);
        AscendC::Add(y0i, x0i, t1i, len); AscendC::Add(y0i, y0i, t2i, len); AscendC::Add(y0i, y0i, t3i, len);
        // y1 = x0 + t1i - t2r - t3i (re) ; x0i - t1r - t2i + t3r (im)
        AscendC::Add(y1r, x0r, t1i, len); AscendC::Sub(y1r, y1r, t2r, len); AscendC::Sub(y1r, y1r, t3i, len);
        AscendC::Sub(y1i, x0i, t1r, len); AscendC::Sub(y1i, y1i, t2i, len); AscendC::Add(y1i, y1i, t3r, len);
        // y2 = x0 - t1 + t2 - t3
        AscendC::Sub(y2r, x0r, t1r, len); AscendC::Add(y2r, y2r, t2r, len); AscendC::Sub(y2r, y2r, t3r, len);
        AscendC::Sub(y2i, x0i, t1i, len); AscendC::Add(y2i, y2i, t2i, len); AscendC::Sub(y2i, y2i, t3i, len);
        // y3 = x0 - t1i - t2r + t3i (re) ; x0i + t1r - t2i - t3r (im)
        AscendC::Sub(y3r, x0r, t1i, len); AscendC::Sub(y3r, y3r, t2r, len); AscendC::Add(y3r, y3r, t3i, len);
        AscendC::Add(y3i, x0i, t1r, len); AscendC::Sub(y3i, y3i, t2i, len); AscendC::Sub(y3i, y3i, t3r, len);
        AscendC::PipeBarrier<PIPE_V>();

        outQ.EnQue(outT);
        inQ.FreeTensor(inT);
    }

    __aicore__ inline void CopyOut(uint32_t i0, uint32_t i1, uint32_t i2, uint32_t i3)
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

        // tail-restore (round8 overrun) — the same idea as in radix2, but over
        // 4 groups: the overrun of group i reaches into the start of group i+1,
        // and we restore the old value (for the last group — into the start of
        // the next block).
        if (this->alignedValid > this->curValid) {
            for (uint32_t i = this->curValid; i < this->alignedValid; i++) {
                y0r.SetValue(i, xrGm.GetValue(i0 + i)); y0i.SetValue(i, xiGm.GetValue(i0 + i));
                y1r.SetValue(i, xrGm.GetValue(i1 + i)); y1i.SetValue(i, xiGm.GetValue(i1 + i));
                y2r.SetValue(i, xrGm.GetValue(i2 + i)); y2i.SetValue(i, xiGm.GetValue(i2 + i));
                y3r.SetValue(i, xrGm.GetValue(i3 + i)); y3i.SetValue(i, xiGm.GetValue(i3 + i));
            }
        }

        AscendC::DataCopy(xrGm[i0], y0r, this->alignedValid); AscendC::DataCopy(xiGm[i0], y0i, this->alignedValid);
        AscendC::DataCopy(xrGm[i1], y1r, this->alignedValid); AscendC::DataCopy(xiGm[i1], y1i, this->alignedValid);
        AscendC::DataCopy(xrGm[i2], y2r, this->alignedValid); AscendC::DataCopy(xiGm[i2], y2i, this->alignedValid);
        AscendC::DataCopy(xrGm[i3], y3r, this->alignedValid); AscendC::DataCopy(xiGm[i3], y3i, this->alignedValid);

        outQ.FreeTensor(outT);
    }

private:
    AscendC::TPipe pipe;
    AscendC::GlobalTensor<float> xrGm, xiGm, wGm;

    static constexpr uint32_t TILE_F  = 64;   // float per tensor slice (= tile)
    static constexpr uint32_t NIN_T   = 14;   // x0..x3 (re,im) + w1..w3 (re,im)
    static constexpr uint32_t NOUT_T  = 8;    // y0..y3 (re,im)
    static constexpr uint32_t NCALC_T = 8;    // t1..t3 (re,im) + u + v

    AscendC::TQue<AscendC::TPosition::VECIN,  2> inQ;
    AscendC::TQue<AscendC::TPosition::VECOUT, 2> outQ;
    AscendC::TBuf<AscendC::TPosition::VECCALC>   calcBuf;

    uint32_t n, q, tile, aligned;
    uint32_t blkNum;                 // tile-distribution stride across blocks (cores)
    uint32_t curValid, alignedValid, computeAligned;
};

extern "C" __global__ __aicore__ void fft4_hybrid(GM_ADDR XR, GM_ADDR XI, GM_ADDR W,
    uint32_t full_size, uint32_t quarter_size, uint32_t block_size, uint32_t aligned,
    uint32_t blk_num)
{
    KernelFFT4Hybrid op;
    op.Init(XR, XI, W, full_size, quarter_size, aligned, blk_num);
    op.Process(block_size);
}

#ifndef ASCENDC_CPU_DEBUG
void fft4_hybrid_do(uint32_t blockDim, void *stream,
                     void *xr, void *xi, void *w,
                     uint32_t full_size, uint32_t quarter_size, uint32_t block_size, uint32_t aligned,
                     uint32_t blk_num)
{
    fft4_hybrid<<<blockDim, nullptr, stream>>>(xr, xi, w, full_size, quarter_size, block_size, aligned,
                                                blk_num);
}
#endif
