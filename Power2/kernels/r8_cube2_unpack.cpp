// kernels/r8_cube2_unpack.cpp
// R8-CUBE2-FS-XS — the Vector/AIV half of the cube2 variant: summing the terms
// of the precision scheme, the Re/Im COMBINE STEP and laying the result out
// over the k-blocks.
//
// There is exactly one difference from kernels/r8_cube1_unpack.cpp, but it is
// a substantive one: in cube1 the matmul hands out [Re(y); Im(y)] straight
// away (the ± signs are baked into the block-real 16x16 matrix), and all
// unpack has left to do is sum the precision terms. In cube2 the matmul hands
// out two HALF-PRODUCTS
//     P = [MR*Re(t); MI*Re(t)]   (from the Re B operand)
//     Q = [MR*Im(t); MI*Im(t)]   (from the Im B operand)
// and the combine step is done by the vector side:
//     Re(y_m) = P[m]   - Q[8+m]
//     Im(y_m) = P[8+m] + Q[m]
// This is what "the price of two matmuls" amounts to on the Vector engine —
// 16 extra Add/Sub per tile; the scheme is verified numerically
// (scratchpad/chk_cube2.py).
//
// !!! THE COMBINE STEP RUNS IN-PLACE OVER P, AND THAT IS SAFE !!!
// Row m of the result is written over P[m], and row 8+m over P[8+m]. Each of
// them takes part in exactly one formula and is not needed anywhere else, and
// Q is not touched at all. So no third buffer for Y is needed — a saving of
// 16*CB float in UB, which at CUBE_BATCH=256 is precisely what decides
// whether we fit.
//
// !!! QUEUE DEPTH 1, NOT 2 (AS IN cube1) !!!
// The Y buffer here is TWICE as large as cube1's (six blocks P0,Q0,P1,Q1,P2,Q2
// against three Y0,Y1,Y2). At CUBE_BATCH=256 that is 6*16*256*4 = 96 KB, and
// with depth 2 it would come to 192 KB — more than the 191 KB of UB per core,
// i.e. the build/run would fall over at exactly the sweep's largest batch.
// There is nothing to pipeline here: one tile and one pass per block,
// AllocTensor is called once. So depth 1 is not penny-pinching, it is the
// condition for working at all at B=256.
//
// !!! WHY THIS KERNEL IS IN A FILE OF ITS OWN !!!
// One rule for the whole project: one __global__ kernel per file. The history
// of the bug with the "_mix_aic"/"_mix_aiv" classification is in
// kernels/r8_cube1_repack.cpp.
//
// !!! TQue<VECIN> VERSUS TQue<VECOUT> — THESE ARE SYNC DIRECTIONS !!!
// Taking P/Q in from GM goes through VECIN (which gives MTE2 -> V, "MTE2
// brought it in, the vector reads it"), handing the result out goes through
// VECOUT (which gives V -> MTE3). Taking data in to a VECOUT buffer and then
// reading it with the vector is a race, which on cube1 showed up as one or two
// floating elements out of 4096 (max_abs wandered between 9.8e-02 and 1.5e-01
// with max_rel unchanged). Here there are EVEN MORE vector operations over the
// data taken in than there, so mixing up the directions is all the less
// forgivable.

#include "kernel_operator.h"
#include "cube1_common.h"
#include <cstdint>

class KernelFFT8Cube2Unpack {
public:
    __aicore__ inline KernelFFT8Cube2Unpack() {}

    __aicore__ inline void Init(GM_ADDR XR, GM_ADDR XI, GM_ADDR SCRATCH_Y, uint32_t n,
                                 uint32_t slot)
    {
        xrGm.SetGlobalBuffer((__gm__ float *)XR, n);
        xiGm.SetGlobalBuffer((__gm__ float *)XI, n);
        // SCRATCH_Y — SIX blocks of 16*CUBE_BATCH: P0,Q0,P1,Q1,P2,Q2
        // (the order is set in kernels/r8_cube2_matmul.cpp). What is actually
        // written is 2*terms blocks; we do not read the rest.
        // A slice of its own per core: a group of tiles per launch, one tile
        // per block, slot = blockIdx. We shift the GM pointer itself so that
        // the indexing inside does not change.
        scratchYGm.SetGlobalBuffer((__gm__ float *)SCRATCH_Y + slot * (6 * BLK), 6 * BLK);
        pipe.InitBuffer(yInQ, 1, 6 * BLK  * sizeof(float));   // P0,Q0,P1,Q1,P2,Q2
        pipe.InitBuffer(outQ, 1, 2 * BLK  * sizeof(float));   // [0..15]=Psum->Y, [16..31]=Qsum
    }

    // The tile is derived from blockIdx by exactly the same formula as in
    // kernels/r8_cube1_repack.cpp (repack is reused by cube2 without edits) —
    // otherwise unpack lays the result out in the wrong place.
    __aicore__ inline void Process(uint32_t tileBase, uint32_t kStride, uint32_t shiftStride,
                                    uint32_t shiftsPerBlk, uint32_t eighth,
                                    uint32_t gCount, uint32_t perBlock, uint32_t terms)
    {
        uint32_t tileIdx = tileBase + (uint32_t)AscendC::GetBlockIdx();
        uint32_t k0      = (tileIdx / shiftsPerBlk) * kStride;
        uint32_t shift0  = (tileIdx % shiftsPerBlk) * shiftStride;

        uint32_t validB = gCount * perBlock;

        // Read 2*terms blocks of 16 rows: the P and the Q of every term.
        auto yIn = yInQ.AllocTensor<float>();
        uint32_t rows = 32u * terms;                 // (P,Q) per term
        for (uint32_t r = 0; r < rows; ++r) {
            AscendC::DataCopy(yIn[r * CUBE_BATCH], scratchYGm[r * CUBE_BATCH], validB);
        }
        yInQ.EnQue(yIn);
        auto yR = yInQ.DeQue<float>();               // <- here we wait for MTE2

        auto out = outQ.AllocTensor<float>();
        auto P = [&](int m) { return out[m        * CUBE_BATCH]; };   // rows 0..15
        auto Q = [&](int m) { return out[(16 + m) * CUBE_BATCH]; };   // rows 16..31
        // The blocks in yR: 0=P0, 1=Q0, 2=P1, 3=Q1, 4=P2, 5=Q2.
        auto src = [&](int blk, int m) { return yR[((uint32_t)blk * 16u + (uint32_t)m) * CUBE_BATCH]; };

        // --- summing the terms of the precision scheme (fp32, on Vector) ---
        // Keeping an accumulator on the Cube is more expensive, and P/Q are
        // read here anyway for the combine step and the layout over k-blocks.
        for (int m = 0; m < 16; ++m) {
            if (terms >= 2) {
                AscendC::Add(P(m), src(0, m), src(2, m), validB);
                AscendC::Add(Q(m), src(1, m), src(3, m), validB);
            } else {
                AscendC::Adds(P(m), src(0, m), 0.0f, validB);
                AscendC::Adds(Q(m), src(1, m), 0.0f, validB);
            }
        }
        if (terms >= 3) {
            for (int m = 0; m < 16; ++m) {
                AscendC::Add(P(m), P(m), src(4, m), validB);
                AscendC::Add(Q(m), Q(m), src(5, m), validB);
            }
        }
        yInQ.FreeTensor(yR);

        // --- the Re/Im combine step (what cube1 does not have) -------------
        // Re(y_m) = P[m] - Q[8+m] ;  Im(y_m) = P[8+m] + Q[m].
        // In-place over P is safe — see the comment in the file header.
        for (int m = 0; m < 8; ++m) {
            AscendC::Sub(P(m),     P(m),     Q(8 + m), validB);
            AscendC::Add(P(8 + m), P(8 + m), Q(m),     validB);
        }

        outQ.EnQue(out);
        auto yOut = outQ.DeQue<float>();             // <- here we set V -> MTE3

        // The mirror image of the gather in kernels/r8_cube1_repack.cpp:
        // gCount chunks of perBlock float, the stride in the destination being
        // a full k-block (8*eighth).
        // The units of blockLen/srcStride/dstStride are 32 bytes, i.e. 8 float.
        AscendC::DataCopyParams scatter;
        scatter.blockCount = (uint16_t)gCount;
        scatter.blockLen   = (uint16_t)(perBlock / 8);
        scatter.srcStride  = 0;
        scatter.dstStride  = (uint16_t)(((eighth << 3) - perBlock) / 8);

        for (int m = 0; m < 8; ++m) {
            uint32_t dst = k0 + shift0 + (uint32_t)m * eighth;
            AscendC::DataCopy(xrGm[dst], yOut[m       * CUBE_BATCH], scatter);
            AscendC::DataCopy(xiGm[dst], yOut[(8 + m) * CUBE_BATCH], scatter);
        }
        outQ.FreeTensor(yOut);
    }

private:
    static constexpr uint32_t BLK = 16 * CUBE_BATCH;   // elements in one P/Q block

    AscendC::TPipe pipe;
    AscendC::GlobalTensor<float> xrGm, xiGm, scratchYGm;
    AscendC::TQue<AscendC::TPosition::VECIN,  1> yInQ;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQ;
};

extern "C" __global__ __aicore__ void fft8_cube2_unpack(
    GM_ADDR XR, GM_ADDR XI, GM_ADDR SCRATCH_Y,
    uint32_t n, uint32_t tileBase, uint32_t kStride, uint32_t shiftStride,
    uint32_t shiftsPerBlk, uint32_t eighth, uint32_t gCount, uint32_t perBlock,
    uint32_t terms)
{
    KernelFFT8Cube2Unpack op;
    op.Init(XR, XI, SCRATCH_Y, n, (uint32_t)AscendC::GetBlockIdx());
    op.Process(tileBase, kStride, shiftStride, shiftsPerBlk, eighth, gCount, perBlock, terms);
}

#ifndef ASCENDC_CPU_DEBUG
void fft8_cube2_unpack_do(uint32_t blockDim, void *stream,
                           void *xr, void *xi, void *scratchY,
                           uint32_t n, uint32_t tileBase, uint32_t kStride,
                           uint32_t shiftStride, uint32_t shiftsPerBlk, uint32_t eighth,
                           uint32_t gCount, uint32_t perBlock, uint32_t terms)
{
    fft8_cube2_unpack<<<blockDim, nullptr, stream>>>(xr, xi, scratchY,
                                                      n, tileBase, kStride, shiftStride,
                                                      shiftsPerBlk, eighth, gCount, perBlock,
                                                      terms);
}
#endif
