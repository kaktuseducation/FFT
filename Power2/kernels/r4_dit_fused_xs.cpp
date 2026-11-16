// kernels/r4_dit_fused_xs.cpp
// R4-DT-FS-XS  (radix4 DIT, fused, X=Split, W=Packed-3)
//
// The same butterfly math as in r4_dit_perstage_xs.cpp (for the formula and
// its check at N=4 see host/fft_host_r4.h). The difference — a loop over the
// stages inside one kernel launch with a meta buffer [quarter,aligned,wOffset]
// x numStages, and the same PipeBarrier<PIPE_ALL> candidate between stages for
// Bug 2 as in the radix-2 r2_dit_fused_xs.cpp.

#include "kernel_operator.h"
#include <cstdint>

class KernelFFT4FusedXS {
public:
    __aicore__ inline KernelFFT4FusedXS() {}

    __aicore__ inline void Init(GM_ADDR XR, GM_ADDR XI, GM_ADDR W, GM_ADDR META,
                                 uint32_t full_size, uint32_t num_stages, uint32_t blk_num)
    {
        this->n         = full_size;
        this->tile      = 64;
        this->numStages = num_stages;
        // Tile-distribution stride across blocks in multicore mode. In
        // single-core (legacy) mode it is not used at all.
        this->blkNum    = blk_num > 0 ? blk_num : 1;

        uint32_t bufS = this->tile * sizeof(float);

        xrGm.SetGlobalBuffer((__gm__ float *)XR, this->n);
        xiGm.SetGlobalBuffer((__gm__ float *)XI, this->n);
        wGm.SetGlobalBuffer((__gm__ float *)W, this->n * 6);   // upper bound, as in radix2
        metaGm.SetGlobalBuffer((__gm__ uint32_t *)META, this->numStages * 3);

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
    // TWO OPERATING MODES
    //
    // Process() — as before: all the stages inside ONE launch, one core. Kept
    // verbatim so that the r4/dit_fs_xs variant stays bit-for-bit the same and
    // its numbers can still be compared against the matrix already measured.
    //
    // ProcessOneStage(stage) — multicore mode: ONE stage per launch, the tiles
    // of the stage are handed out across blocks (cores) with stride blkNum.
    // Splitting the stages into separate launches is mandatory here: stage s+1
    // reads what stage s wrote across the WHOLE array, that is, between them we
    // need a barrier that every core sees. A hand-written cross-core barrier
    // inside the kernel is a source of hangs, whereas the ordering between
    // launches on one stream comes for free from the semantics of the stream
    // itself. The same trick as in r8_dit_fused_xs.cpp.
    //
    // Why handing the tiles out across cores is correct: the tile (k, shift)
    // reads and writes exactly its own four segments i0..i3 of length valid,
    // and the segments of different tiles do not overlap (the shifts cut
    // [0, quarter) into tile-sized pieces, the k-blocks run back to back).
    // There are no inter-tile dependencies inside a stage.
    // NOTE for the future: the tail trick in CopyOut (reading GM back when
    // alignedValid > curValid) writes OTHER tiles' elements back. Right now it
    // never fires, because quarter is a power of four no lower than 16, and 16
    // and 64 are multiples of eight. If batching over k-blocks ever appears and
    // quarter becomes smaller than 8, this trick will turn into a race between
    // cores.
    // ========================================================================
    __aicore__ inline void Process()
    {
        for (uint32_t s = 0; s < this->numStages; ++s) {
            uint32_t quarter = metaGm.GetValue(s * 3 + 0);
            uint32_t aligned = metaGm.GetValue(s * 3 + 1);
            uint32_t wOffset = metaGm.GetValue(s * 3 + 2);
            uint32_t block   = quarter << 2;

            ProcessStage(quarter, aligned, wOffset, block);

            // --- CANDIDATE FIX for Bug 2, the same principle as in radix2 ---
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }

    __aicore__ inline void ProcessOneStage(uint32_t stage)
    {
        uint32_t quarter = metaGm.GetValue(stage * 3 + 0);
        uint32_t aligned = metaGm.GetValue(stage * 3 + 1);
        uint32_t wOffset = metaGm.GetValue(stage * 3 + 2);
        uint32_t block   = quarter << 2;

        uint32_t shiftsPerBlk = (quarter + this->tile - 1) / this->tile;
        uint32_t nTiles       = (this->n / block) * shiftsPerBlk;
        uint32_t bid          = (uint32_t)AscendC::GetBlockIdx();

        for (uint32_t tid = bid; tid < nTiles; tid += this->blkNum) {
            uint32_t k     = (tid / shiftsPerBlk) * block;
            uint32_t shift = (tid % shiftsPerBlk) * this->tile;
            ProcessTile(quarter, aligned, wOffset, k, shift);
        }
    }

private:
    __aicore__ inline void ProcessStage(uint32_t quarter, uint32_t aligned, uint32_t wOffset, uint32_t block)
    {
        for (uint32_t k = 0; k < this->n; k += block) {
            for (uint32_t shift = 0; shift < quarter; shift += this->tile) {
                ProcessTile(quarter, aligned, wOffset, k, shift);
            }
        }
    }

    // The body of the former inner loop of ProcessStage, lifted out as is —
    // shared by both modes, so that the math is guaranteed to be one and the
    // same.
    __aicore__ inline void ProcessTile(uint32_t quarter, uint32_t aligned, uint32_t wOffset,
                                        uint32_t k, uint32_t shift)
    {
        uint32_t i0 = k + shift, i1 = i0 + quarter, i2 = i1 + quarter, i3 = i2 + quarter;
        uint32_t valid = (quarter - shift < this->tile) ? (quarter - shift) : this->tile;

        this->curValid       = valid;
        this->alignedValid   = ((valid + 7)  / 8)  * 8;
        this->computeAligned = ((valid + 63) / 64) * 64;

        CopyIn(i0, i1, i2, i3, shift, aligned, wOffset);
        Compute();
        CopyOut(i0, i1, i2, i3);
    }

    __aicore__ inline void CopyIn(uint32_t i0, uint32_t i1, uint32_t i2, uint32_t i3, uint32_t shift,
                                   uint32_t aligned, uint32_t wOffset)
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

        AscendC::DataCopy(w1r, wGm[wOffset + shift],                     this->alignedValid);
        AscendC::DataCopy(w1i, wGm[wOffset + aligned + shift],           this->alignedValid);
        AscendC::DataCopy(w2r, wGm[wOffset + 2*aligned + shift],         this->alignedValid);
        AscendC::DataCopy(w2i, wGm[wOffset + 3*aligned + shift],         this->alignedValid);
        AscendC::DataCopy(w3r, wGm[wOffset + 4*aligned + shift],         this->alignedValid);
        AscendC::DataCopy(w3i, wGm[wOffset + 5*aligned + shift],         this->alignedValid);


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
    AscendC::GlobalTensor<float>    xrGm, xiGm, wGm;
    AscendC::GlobalTensor<uint32_t> metaGm;

    static constexpr uint32_t TILE_F  = 64;   // float per tensor slice (= tile)
    static constexpr uint32_t NIN_T   = 14;   // x0..x3 (re,im) + w1..w3 (re,im)
    static constexpr uint32_t NOUT_T  = 8;    // y0..y3 (re,im)
    static constexpr uint32_t NCALC_T = 8;    // t1..t3 (re,im) + u + v

    AscendC::TQue<AscendC::TPosition::VECIN,  2> inQ;
    AscendC::TQue<AscendC::TPosition::VECOUT, 2> outQ;
    AscendC::TBuf<AscendC::TPosition::VECCALC>   calcBuf;

    uint32_t n, tile, numStages;
    uint32_t blkNum;                 // tile-distribution stride across blocks (multicore)
    uint32_t curValid, alignedValid, computeAligned;
};

// stage == FFT4_ALL_STAGES — single-core legacy mode: the kernel itself walks
// all the stages in one launch (the r4/dit_fs_xs variant, as it was before
// multicore). Any other value — multicore mode: exactly this one stage is
// computed, its tiles handed out across blocks (the r4/dit_fsm_xs variant).
//
// Done as one kernel with two modes rather than a copy of the file, and that
// is deliberate: the math (CopyIn/Compute/CopyOut) has to be literally one and
// the same in both variants, otherwise the comparison "one core against forty"
// stops being a comparison of the number of cores.
#define FFT4_ALL_STAGES 0xFFFFFFFFu

extern "C" __global__ __aicore__ void fft4_dit_fused_xs(
    GM_ADDR XR, GM_ADDR XI, GM_ADDR W, GM_ADDR META,
    uint32_t full_size, uint32_t num_stages, uint32_t stage, uint32_t blk_num)
{
    KernelFFT4FusedXS op;
    op.Init(XR, XI, W, META, full_size, num_stages, blk_num);
    if (stage == FFT4_ALL_STAGES) {
        op.Process();
    } else {
        op.ProcessOneStage(stage);
    }
}

#ifndef ASCENDC_CPU_DEBUG
void fft4_dit_fused_xs_do(uint32_t blockDim, void *stream,
                           void *xr, void *xi, void *w, void *meta,
                           uint32_t full_size, uint32_t num_stages,
                           uint32_t stage, uint32_t blk_num)
{
    fft4_dit_fused_xs<<<blockDim, nullptr, stream>>>(xr, xi, w, meta,
                                                      full_size, num_stages, stage, blk_num);
}
#endif
