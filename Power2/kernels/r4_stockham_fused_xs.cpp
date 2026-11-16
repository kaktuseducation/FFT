// kernels/r4_stockham_fused_xs.cpp
// R4-ST-FS-XS  (radix4 Stockham, fused, X=Split, W=Packed-3)
//
// The same butterfly math as in the rest of the radix4 files. The ping-pong
// between XR0/XI0 and XR1/XI1 goes by the parity of the stage — the same
// principle as in the radix-2 r2_stockham_fused_xs.cpp. The host must add the
// parity of the CPU prefix on its own side to the numStages of this launch in
// order to work out where to read the D2H from — the same warning as in the
// radix2 version applies here word for word.

#include "kernel_operator.h"
#include <cstdint>

class KernelFFT4StockhamFusedXS {
public:
    __aicore__ inline KernelFFT4StockhamFusedXS() {}

    __aicore__ inline void Init(GM_ADDR XR0, GM_ADDR XI0, GM_ADDR XR1, GM_ADDR XI1,
                                 GM_ADDR W, GM_ADDR META,
                                 uint32_t full_size, uint32_t num_stages, uint32_t blk_num)
    {
        this->n         = full_size;
        this->tile      = 64;
        this->numStages = num_stages;
        // Tile-distribution stride across blocks in multicore mode. In
        // single-core (legacy) mode it is not used at all.
        this->blkNum    = blk_num > 0 ? blk_num : 1;

        uint32_t bufS = this->tile * sizeof(float);

        xr0Gm.SetGlobalBuffer((__gm__ float *)XR0, this->n);
        xi0Gm.SetGlobalBuffer((__gm__ float *)XI0, this->n);
        xr1Gm.SetGlobalBuffer((__gm__ float *)XR1, this->n);
        xi1Gm.SetGlobalBuffer((__gm__ float *)XI1, this->n);
        wGm.SetGlobalBuffer((__gm__ float *)W, this->n * 6);
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
    // TWO OPERATING MODES — exactly as in kernels/r4_dit_fused_xs.cpp and
    // kernels/r2_stockham_fused_xs.cpp, see the detailed rationale there.
    //
    // Process() — as before: all the stages inside ONE launch, one core (the
    // r4/stk_fs_xs variant). Kept verbatim so that the numbers of this variant
    // stay bit-for-bit the same and compare against the matrix already taken.
    //
    // ProcessOneStage(stage) — multicore mode: ONE stage per launch, the tiles
    // of the stage are handed out across blocks (cores) with stride blkNum (the
    // r4/stk_fsm_xs variant). Splitting the stages into separate launches is
    // mandatory here: in Stockham the buffer that stage s+1 is going to READ
    // from is exactly the buffer that stage s was JUST NOW writing across the
    // WHOLE array, that is, between the stages we need a barrier that every
    // core sees. A hand-written cross-core barrier inside the kernel is a known
    // source of hangs, whereas the ordering between launches on one stream
    // comes for free from the semantics of the stream itself.
    //
    // Handing the tiles out across cores is correct: the tile (k, shift) writes
    // its own four segments o0..o3 of length alignedValid into the destination
    // buffer, the segments of different tiles do not overlap (the k-blocks of
    // stride L run back to back, the shifts cut [0, quarter) into tile-sized
    // pieces), and it reads from the OTHER buffer, the one nobody writes to on
    // this stage. There is no tail-restore trick here at all, so there are no
    // writes into other tiles' data by construction.
    //
    // IMPORTANT about the src/dst choice in multicore mode: the ping-pong is
    // determined by the PARITY OF THE STAGE NUMBER, not by the order of calls,
    // which is why ProcessOneStage(stage) computes srcIsBuf1 = (stage % 2 == 1)
    // — exactly what Process() would have computed for s == stage. The host
    // must pass absolute stage numbers (0..numStages-1), otherwise the ping-pong
    // will slip.
    // ========================================================================
    __aicore__ inline void Process()
    {
        for (uint32_t s = 0; s < this->numStages; ++s) {
            uint32_t quarter = metaGm.GetValue(s * 3 + 0);
            uint32_t aligned = metaGm.GetValue(s * 3 + 1);
            uint32_t wOffset = metaGm.GetValue(s * 3 + 2);

            bool srcIsBuf1 = (s % 2 == 1);
            AscendC::GlobalTensor<float> &srcR = srcIsBuf1 ? xr1Gm : xr0Gm;
            AscendC::GlobalTensor<float> &srcI = srcIsBuf1 ? xi1Gm : xi0Gm;
            AscendC::GlobalTensor<float> &dstR = srcIsBuf1 ? xr0Gm : xr1Gm;
            AscendC::GlobalTensor<float> &dstI = srcIsBuf1 ? xi0Gm : xi1Gm;

            ProcessStage(srcR, srcI, dstR, dstI, quarter, aligned, wOffset);

            // --- CANDIDATE FIX for Bug 2 ---
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }

    __aicore__ inline void ProcessOneStage(uint32_t stage)
    {
        uint32_t quarter = metaGm.GetValue(stage * 3 + 0);
        uint32_t aligned = metaGm.GetValue(stage * 3 + 1);
        uint32_t wOffset = metaGm.GetValue(stage * 3 + 2);

        // The same ping-pong by stage parity as in Process() — see above.
        bool srcIsBuf1 = (stage % 2 == 1);
        AscendC::GlobalTensor<float> &srcR = srcIsBuf1 ? xr1Gm : xr0Gm;
        AscendC::GlobalTensor<float> &srcI = srcIsBuf1 ? xi1Gm : xi0Gm;
        AscendC::GlobalTensor<float> &dstR = srcIsBuf1 ? xr0Gm : xr1Gm;
        AscendC::GlobalTensor<float> &dstI = srcIsBuf1 ? xi0Gm : xi1Gm;

        uint32_t L = quarter << 2;
        uint32_t r = this->n / L;

        // The tiles of the stage are numbered by a running tid: the outer loop
        // of ProcessStage went over k = 0..r-1 (here k is the block INDEX, not
        // an offset), the inner one over shift with stride tile (shiftsPerBlk
        // shifts in total).
        uint32_t shiftsPerBlk = (quarter + this->tile - 1) / this->tile;
        uint32_t nTiles       = r * shiftsPerBlk;
        uint32_t bid          = (uint32_t)AscendC::GetBlockIdx();

        for (uint32_t tid = bid; tid < nTiles; tid += this->blkNum) {
            uint32_t k     = tid / shiftsPerBlk;
            uint32_t shift = (tid % shiftsPerBlk) * this->tile;
            ProcessTile(srcR, srcI, dstR, dstI, quarter, aligned, wOffset, L, k, shift);
        }
    }

private:
    __aicore__ inline void ProcessStage(AscendC::GlobalTensor<float> &srcR,
                                         AscendC::GlobalTensor<float> &srcI,
                                         AscendC::GlobalTensor<float> &dstR,
                                         AscendC::GlobalTensor<float> &dstI,
                                         uint32_t quarter, uint32_t aligned, uint32_t wOffset)
    {
        uint32_t L = quarter << 2;
        uint32_t r = this->n / L;

        for (uint32_t k = 0; k < r; ++k) {
            for (uint32_t shift = 0; shift < quarter; shift += this->tile) {
                ProcessTile(srcR, srcI, dstR, dstI, quarter, aligned, wOffset, L, k, shift);
            }
        }
    }

    // The body of the former inner loop of ProcessStage, lifted out as is —
    // shared by both modes, so that the math is guaranteed to be one and the
    // same. L is passed as a parameter because in the original it was computed
    // in ProcessStage and the body relies on it (o0 = k * L + shift).
    __aicore__ inline void ProcessTile(AscendC::GlobalTensor<float> &srcR,
                                        AscendC::GlobalTensor<float> &srcI,
                                        AscendC::GlobalTensor<float> &dstR,
                                        AscendC::GlobalTensor<float> &dstI,
                                        uint32_t quarter, uint32_t aligned, uint32_t wOffset,
                                        uint32_t L, uint32_t k, uint32_t shift)
    {
        uint32_t valid = (quarter - shift < this->tile) ? (quarter - shift) : this->tile;
        this->curValid       = valid;
        this->alignedValid   = ((valid + 7)  / 8)  * 8;
        this->computeAligned = ((valid + 63) / 64) * 64;

        uint32_t i0 = k * quarter + shift;
        uint32_t i1 = i0 + this->n / 4;
        uint32_t i2 = i0 + this->n / 2;
        uint32_t i3 = i0 + 3 * this->n / 4;
        uint32_t o0 = k * L + shift;
        uint32_t o1 = o0 + quarter, o2 = o1 + quarter, o3 = o2 + quarter;

        CopyIn(srcR, srcI, i0, i1, i2, i3, shift, aligned, wOffset);
        Compute();
        CopyOut(dstR, dstI, o0, o1, o2, o3);
    }

    __aicore__ inline void CopyIn(AscendC::GlobalTensor<float> &srcR, AscendC::GlobalTensor<float> &srcI,
                                   uint32_t i0, uint32_t i1, uint32_t i2, uint32_t i3, uint32_t shift,
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

        AscendC::DataCopy(x0r, srcR[i0], this->alignedValid); AscendC::DataCopy(x0i, srcI[i0], this->alignedValid);
        AscendC::DataCopy(x1r, srcR[i1], this->alignedValid); AscendC::DataCopy(x1i, srcI[i1], this->alignedValid);
        AscendC::DataCopy(x2r, srcR[i2], this->alignedValid); AscendC::DataCopy(x2i, srcI[i2], this->alignedValid);
        AscendC::DataCopy(x3r, srcR[i3], this->alignedValid); AscendC::DataCopy(x3i, srcI[i3], this->alignedValid);

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

    __aicore__ inline void CopyOut(AscendC::GlobalTensor<float> &dstR, AscendC::GlobalTensor<float> &dstI,
                                    uint32_t o0, uint32_t o1, uint32_t o2, uint32_t o3)
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

        AscendC::DataCopy(dstR[o0], y0r, this->alignedValid); AscendC::DataCopy(dstI[o0], y0i, this->alignedValid);
        AscendC::DataCopy(dstR[o1], y1r, this->alignedValid); AscendC::DataCopy(dstI[o1], y1i, this->alignedValid);
        AscendC::DataCopy(dstR[o2], y2r, this->alignedValid); AscendC::DataCopy(dstI[o2], y2i, this->alignedValid);
        AscendC::DataCopy(dstR[o3], y3r, this->alignedValid); AscendC::DataCopy(dstI[o3], y3i, this->alignedValid);

        outQ.FreeTensor(outT);
    }

private:
    AscendC::TPipe pipe;
    AscendC::GlobalTensor<float>    xr0Gm, xi0Gm, xr1Gm, xi1Gm, wGm;
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

// stage == FFT4_STK_XS_ALL_STAGES — single-core legacy mode (the r4/stk_fs_xs
// variant); any other value — multicore: exactly this one stage is computed,
// its tiles handed out across blocks (the r4/stk_fsm_xs variant). One kernel
// with two modes rather than a copy of the file, so that the math
// (CopyIn/Compute/CopyOut) is literally one and the same in both variants —
// otherwise the comparison "one core against forty" stops being a comparison
// of the number of cores.
#define FFT4_STK_XS_ALL_STAGES 0xFFFFFFFFu

extern "C" __global__ __aicore__ void fft4_stockham_fused_xs(
    GM_ADDR XR0, GM_ADDR XI0, GM_ADDR XR1, GM_ADDR XI1, GM_ADDR W, GM_ADDR META,
    uint32_t full_size, uint32_t num_stages, uint32_t stage, uint32_t blk_num)
{
    KernelFFT4StockhamFusedXS op;
    op.Init(XR0, XI0, XR1, XI1, W, META, full_size, num_stages, blk_num);
    if (stage == FFT4_STK_XS_ALL_STAGES) {
        op.Process();
    } else {
        op.ProcessOneStage(stage);
    }
}

#ifndef ASCENDC_CPU_DEBUG
void fft4_stockham_fused_xs_do(uint32_t blockDim, void *stream,
                                void *xr0, void *xi0, void *xr1, void *xi1,
                                void *w, void *meta,
                                uint32_t full_size, uint32_t num_stages,
                                uint32_t stage, uint32_t blk_num)
{
    fft4_stockham_fused_xs<<<blockDim, nullptr, stream>>>(xr0, xi0, xr1, xi1, w, meta,
        full_size, num_stages, stage, blk_num);
}
#endif
