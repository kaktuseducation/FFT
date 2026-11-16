// kernels/r8_cube1_unpack.cpp
// R8-CUBE-FS-XS — the unpack half of the Vector part of the cube1_fs_xs
// combine step (pure Vector/AIV, without a single call into the Cube/Matmul
// API).
//
// !!! BATCHING OVER k-BLOCKS !!!
// The mirror image of the gather in kernels/r8_cube1_repack.cpp: a Y row out
// of scratchY holds gCount*perBlock results, gathered from gCount consecutive
// k-blocks taking perBlock butterflies out of each. They are laid back out in
// one strided write (DataCopyParams: gCount chunks of perBlock float, the
// stride in the destination being a full k-block, 8*eighth).
//
// A pleasant side effect of batching: perBlock is always a multiple of 8
// (eighth is a power of eight >= 8, or else exactly CUBE_BATCH=128), so the
// write always lands on the boundaries of the 32-byte DataCopy blocks. The
// former trick with AlignUp8 + reading the tail in through GetValue/SetValue
// (so that a write of length alignedValid would not clobber the neighbouring
// elements of someone else's group) became unnecessary and was removed —
// and with it went 2*(alignedValid-validB) scalar GM accesses per tile.
//
// !!! fp16 FIX !!!
// The Y result of the matmul (scratchYGm) stays float — only the X operand
// (repack -> matmul) is cast to half, see kernels/r8_cube1_repack.cpp and
// kernels/r8_cube_fused_xs.cpp.
//
// !!! WHY THIS KERNEL IS IN A FILE OF ITS OWN !!!
// See the detailed history of the bug in kernels/r8_cube1_repack.cpp — the
// short version: while fft8_cube1_repack and fft8_cube1_unpack lived in one
// file, both got a "_mix_aic"/"_mix_aiv" classification out of CANN's
// precompile scanner, unlike the proven-working kernels/r8_dit_fused_xs.cpp.
// They have been split up into one __global__ kernel per file.

#include "kernel_operator.h"
#include "cube1_common.h"
#include <cstdint>

// Y (scratchYGm, the same block-real layout as X, but float and in a buffer
// of its own) -> xrGm/xiGm at the positions of the current batch.
class KernelFFT8Cube1Unpack {
public:
    __aicore__ inline KernelFFT8Cube1Unpack() {}

    __aicore__ inline void Init(GM_ADDR XR, GM_ADDR XI, GM_ADDR SCRATCH_Y, uint32_t n,
                                 uint32_t slot)
    {
        xrGm.SetGlobalBuffer((__gm__ float *)XR, n);
        xiGm.SetGlobalBuffer((__gm__ float *)XI, n);
        // SCRATCH_Y — THREE blocks of 16*CUBE_BATCH: Y0, Y1, Y2 — the terms
        // of the precision-recovery scheme (see terms in
        // kernels/r8_cube_fused_xs.cpp). Only terms blocks are actually
        // written; the rest we do not read.
        // Its own slice per core, as in kernels/r8_cube1_repack.cpp: a group
        // of tiles per launch, one tile per block, slot = blockIdx. The GM
        // pointer itself is shifted so the indexing inside does not change.
        scratchYGm.SetGlobalBuffer((__gm__ float *)SCRATCH_Y + slot * (3 * 16 * CUBE_BATCH),
                                   3 * 16 * CUBE_BATCH);
        // TWO queues, and that is a matter of principle (why — below):
        //   yInQ  (VECIN)  — receives Y0..Y2 from GM: gives MTE2 -> V sync
        //   outQ  (VECOUT) — the result of the sum: gives V -> MTE3 sync
        pipe.InitBuffer(yInQ, 2, 3 * 16 * CUBE_BATCH * sizeof(float));
        pipe.InitBuffer(outQ, 2,     16 * CUBE_BATCH * sizeof(float));
    }

    // The tile follows from blockIdx by exactly the same formula as in repack
    // (see the comment there) — otherwise unpack will lay the result out in
    // the wrong place.
    __aicore__ inline void Process(uint32_t tileBase, uint32_t kStride, uint32_t shiftStride,
                                    uint32_t shiftsPerBlk, uint32_t eighth,
                                    uint32_t gCount, uint32_t perBlock, uint32_t terms)
    {
        uint32_t tileIdx = tileBase + (uint32_t)AscendC::GetBlockIdx();
        uint32_t k0      = (tileIdx / shiftsPerBlk) * kStride;
        uint32_t shift0  = (tileIdx % shiftsPerBlk) * shiftStride;

        uint32_t validB = gCount * perBlock;

        // !!! WHY Y IS RECEIVED THROUGH VECIN AND NOT THROUGH VECOUT !!!
        // The previous version read Y into the buffer of a VECOUT queue and
        // summed the blocks with the vector unit right there. EnQue/DeQue on
        // a VECOUT queue installs synchronization in the V -> MTE3 direction
        // ("computed it in UB — now we unload it to GM"), whereas what is
        // needed here is the OPPOSITE one, MTE2 -> V: MTE2 brings the data in
        // and the vector unit reads it. Only VECIN gives that.
        // While terms = 1 there were no vector operations over this data at
        // all (the buffer was just shuffled GM -> UB -> GM), and the mismatch
        // did not show up — cube1_fs_xs reproduced bit for bit. The moment
        // terms >= 2 brought in the Add for the sum Y0+Y1+Y2, the vector unit
        // started reading UB without waiting for the load: the result stayed
        // almost right, but ONE OR TWO elements out of 4096 drifted from run
        // to run (max_abs wandered between 9.8e-02 and 1.5e-01 while max_rel
        // stayed put), and a bit-for-bit comparison of two outputs diverged.
        auto yIn = yInQ.AllocTensor<float>();
        uint32_t rows = 16u * terms;                 // Y0 [, Y1 [, Y2]]
        for (uint32_t r = 0; r < rows; ++r) {
            AscendC::DataCopy(yIn[r * CUBE_BATCH], scratchYGm[r * CUBE_BATCH], validB);
        }
        yInQ.EnQue(yIn);
        auto yR = yInQ.DeQue<float>();               // <- this is where we wait for MTE2

        // The sum of the precision-recovery terms — in fp32, on Vector:
        // Y = Y0 + Y1 + Y2. Keeping the accumulator on Cube costs more, and Y
        // is read here anyway to be laid out across the k-blocks.
        auto y2 = outQ.AllocTensor<float>();
        for (int m = 0; m < 16; ++m) {
            if (terms >= 2) {
                AscendC::Add(y2[m * CUBE_BATCH], yR[m * CUBE_BATCH],
                             yR[(16 + m) * CUBE_BATCH], validB);
            } else {
                AscendC::Adds(y2[m * CUBE_BATCH], yR[m * CUBE_BATCH], 0.0f, validB);
            }
        }
        if (terms >= 3) {
            for (int m = 0; m < 16; ++m) {
                AscendC::Add(y2[m * CUBE_BATCH], y2[m * CUBE_BATCH],
                             yR[(32 + m) * CUBE_BATCH], validB);
            }
        }
        yInQ.FreeTensor(yR);
        outQ.EnQue(y2);
        auto yOut = outQ.DeQue<float>();             // <- this is where V -> MTE3 is installed

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
    AscendC::TPipe pipe;
    AscendC::GlobalTensor<float> xrGm, xiGm, scratchYGm;
    AscendC::TQue<AscendC::TPosition::VECIN,  2> yInQ;
    AscendC::TQue<AscendC::TPosition::VECOUT, 2> outQ;
};

extern "C" __global__ __aicore__ void fft8_cube1_unpack(
    GM_ADDR XR, GM_ADDR XI, GM_ADDR SCRATCH_Y,
    uint32_t n, uint32_t tileBase, uint32_t kStride, uint32_t shiftStride,
    uint32_t shiftsPerBlk, uint32_t eighth, uint32_t gCount, uint32_t perBlock,
    uint32_t terms)
{
    KernelFFT8Cube1Unpack op;
    op.Init(XR, XI, SCRATCH_Y, n, (uint32_t)AscendC::GetBlockIdx());
    op.Process(tileBase, kStride, shiftStride, shiftsPerBlk, eighth, gCount, perBlock, terms);
}

#ifndef ASCENDC_CPU_DEBUG
void fft8_cube1_unpack_do(uint32_t blockDim, void *stream,
                           void *xr, void *xi, void *scratchY,
                           uint32_t n, uint32_t tileBase, uint32_t kStride,
                           uint32_t shiftStride, uint32_t shiftsPerBlk, uint32_t eighth,
                           uint32_t gCount, uint32_t perBlock, uint32_t terms)
{
    fft8_cube1_unpack<<<blockDim, nullptr, stream>>>(xr, xi, scratchY,
                                                      n, tileBase, kStride, shiftStride,
                                                      shiftsPerBlk, eighth, gCount, perBlock,
                                                      terms);
}
#endif