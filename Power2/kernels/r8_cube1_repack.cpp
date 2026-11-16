// kernels/r8_cube1_repack.cpp
// R8-CUBE-FS-XS — the repack/twiddle half of the Vector part of the
// cube1_fs_xs combine step (pure Vector/AIV, without a single call into the
// Cube/Matmul API).
//
// !!! ALIGNMENT OF THE WRITE INTO scratchX (a consequence of the fp16 fix) !!!
// The write into scratchX uses a length of AlignUp16(validB), not validB:
// DataCopy UB<->GM works in blocks of 32 bytes, and for half that is 16
// elements (for float it was 8 elements). The detailed analysis of the bug
// is at the AlignUp16 function below.
//
// !!! fp16 FIX (the X operand of the matmul is half now) !!!
// An isolated test of the official Huawei reference (matmul_custom.cpp) on
// this same hardware showed: fp32×fp32 through the AscendC Matmul high-level
// API either computes pure zeros (on a "normal" 512x512x1024, with no
// deadlock) or hangs dead (on our tiny 16x16x128 — visible from
// SyncTask/LogicCqReportV2 in the device logs: a real hardware deadlock,
// not a software bug). The official pattern — A_T=half, B_T=half, C_T=float
// (fp16 inputs, fp32 accumulator) — is exactly how the Cube engine is
// physically meant to work. So the twiddle-multiplied X operand is now
// computed in fp32 (for precision), and before the write into the scratch
// buffer for the matmul it is CAST to half (AscendC::Cast). The small
// precision loss at that one step is a deliberate trade-off; the Y result
// (the matmul output) and the whole rest of the pipeline (the repack
// arithmetic, unpack, the DIT path) stay fp32.
//
// !!! WHY THIS KERNEL IS IN A FILE OF ITS OWN !!!
// A structural bug with the AIC/AIV/MIX classification (see the history):
// this file uses TQue<VECIN>+TQue<VECOUT> (without a "bare" VECCALC-only
// final write) — a combination which, unlike the previous version with
// TBuf<VECCALC>, the compiler classifies purely as AIV, with no
// "_mix_aic"/"_mix_aiv" — confirmed by disassembling trial aic/aiv
// object files (llvm-objdump + the channel.py classifier out of
// extract_host_stub.py).

#include "kernel_operator.h"
#include "cube1_common.h"
#include <cstdint>

namespace {
// For half: a DataCopy block (32 bytes) is 16 elements, not 8.
// THE BUG THIS FIXED: after the X operand moved to fp16, the write into
// scratchX used a length of validB elements. On the stage eighth=8 that is
// validB=8, i.e. 16 bytes — HALF A BLOCK, and DataCopy copied nothing at
// all. scratchX stayed zero, the matmul computed Y = M16 * 0 = 0, unpack
// wrote zeros into xr/xi, and the following stages (eighth=64, 512) read
// those zeros and faithfully spread them over the whole array — the output
// was exactly zeros, all 4096 of them.
// While X was float, those same 8 elements came to exactly 32 bytes, which
// is why the problem did not show up before the fp16 fix.
__aicore__ inline uint32_t AlignUp16(uint32_t v)
{
    return ((v + 15) / 16) * 16;
}
}

// Twiddle multiplication (t0=x0; t1..t7=x_m*w_m) with the result in the
// block-real layout [Re(t0..t7) rows 0..7; Im(t0..t7) rows 8..15] — exactly
// what build_dft8_matrix16_blockreal() expects of the X operand of the Cube
// matmul. The result is cast to half immediately before the write into
// scratchXGm.
class KernelFFT8Cube1Repack {
public:
    __aicore__ inline KernelFFT8Cube1Repack() {}

    __aicore__ inline void Init(GM_ADDR XR, GM_ADDR XI, GM_ADDR WC, GM_ADDR SCRATCH_X,
                                 uint32_t n, uint32_t wcTotal, uint32_t slot)
    {
        xrGm.SetGlobalBuffer((__gm__ float *)XR, n);
        xiGm.SetGlobalBuffer((__gm__ float *)XI, n);
        // wcTotal (not n*14): on the host only wc_total floats are actually
        // allocated for the cube twiddles (build_cube_twiddle_plan8 sums
        // rowLen*14 over the NPU stages only) — a mismatch between the
        // declared size of the GM tensor and the real allocation produced an
        // aicore exception when DataCopy went past the buffer's bounds.
        wcGm.SetGlobalBuffer((__gm__ float *)WC, wcTotal);
        // SCRATCH_X — the X operand of the matmul (half), TWO blocks of
        // 16*CUBE_BATCH: [0, 16*CB) = X_hi, [16*CB, 32*CB) = X_lo (the
        // correction term for the precision-recovery scheme, see terms in
        // kernels/r8_cube_fused_xs.cpp). X_lo is written only when terms >= 2.
        // ITS OWN SLICE PER CORE: one launch of a stage processes a group of
        // tiles, one per block, and each has a scratch slot of its own. The
        // GM pointer itself is shifted — then all the indexing inside the
        // kernel stays exactly as in the single-core version, with no slot
        // offsets strewn through the code.
        scratchXGm.SetGlobalBuffer((__gm__ half *)SCRATCH_X + slot * (2 * 16 * CUBE_BATCH),
                                   2 * 16 * CUBE_BATCH);

        // outQ: rows 0..15 — the result of the twiddle multiplication (fp32),
        // 16/17 — scratch for u/v, 18..33 — a buffer for the back-cast
        // half->float while computing X_lo (needed only when terms >= 2, but
        // allocated unconditionally: there is UB enough for it with room to
        // spare, and branching on the buffer size would be dead weight).
        pipe.InitBuffer(inQ,   2, (16 + 14)     * CUBE_BATCH * sizeof(float));
        pipe.InitBuffer(outQ,  2, (16 + 2 + 16) * CUBE_BATCH * sizeof(float));
        pipe.InitBuffer(castQ, 2, (16 + 16)     * CUBE_BATCH * sizeof(half));
    }

    // The batch is assembled out of gCount consecutive k-blocks, perBlock
    // butterflies out of each: batch column
    // b = jBlock*perBlock + (shift - shift0).
    //   eighth <= CUBE_BATCH:  perBlock = eighth, shift0 = 0, gCount = how many
    //                          k-blocks fit (CUBE_BATCH/eighth or the remainder)
    //   eighth >  CUBE_BATCH:  gCount = 1, the batch is the window
    //                          [shift0, shift0+perBlock) inside one k-block
    //                          (the way it was before batching)
    // perBlock is always a multiple of 8 (eighth is a power of eight >= 8, or
    // else 128), so neither alignedValid nor the tail GetValue/SetValue are
    // needed any more: both the read and the write land exactly on the
    // boundaries of the 32-byte DataCopy blocks.
    // Which tile this block computes follows from its blockIdx:
    //   tileIdx = tileBase + blockIdx
    //   k0     = (tileIdx / shiftsPerBlk) * kStride
    //   shift0 = (tileIdx % shiftsPerBlk) * shiftStride
    // The same formula on the host (host/impl_r8.cpp) — the enumeration of a
    // stage's tiles has to agree down to the index, or repack and unpack will
    // diverge.
    //   eighth <= CUBE_BATCH:  shiftsPerBlk=1, kStride=gMax*block, shiftStride=0
    //   eighth >  CUBE_BATCH:  shiftsPerBlk=eighth/CUBE_BATCH, kStride=block,
    //                          shiftStride=CUBE_BATCH
    // The launch's blockDim equals the number of tiles in the group, so there
    // are never any "extra" blocks and no index guard is needed.
    __aicore__ inline void Process(uint32_t tileBase, uint32_t kStride, uint32_t shiftStride,
                                    uint32_t shiftsPerBlk, uint32_t eighth,
                                    uint32_t gCount, uint32_t perBlock,
                                    uint32_t wcOffset, uint32_t wcRowLen,
                                    uint32_t terms)
    {
        uint32_t tileIdx = tileBase + (uint32_t)AscendC::GetBlockIdx();
        uint32_t k0      = (tileIdx / shiftsPerBlk) * kStride;
        uint32_t shift0  = (tileIdx % shiftsPerBlk) * shiftStride;

        uint32_t validB = gCount * perBlock;
        CopyIn(k0, shift0, eighth, gCount, perBlock, wcOffset, wcRowLen, validB);
        Compute(validB, terms);
    }

private:
    __aicore__ inline void CopyIn(uint32_t k0, uint32_t shift0, uint32_t eighth,
                                   uint32_t gCount, uint32_t perBlock,
                                   uint32_t wcOffset, uint32_t wcRowLen, uint32_t validB)
    {
        auto inT = inQ.AllocTensor<float>();

        // The gather over k-blocks in one strided operation: gCount chunks of
        // perBlock float, the stride between chunks in the source being a
        // full k-block (8*eighth).
        // The units of blockLen/srcStride/dstStride in DataCopyParams are
        // 32 bytes, that is, 8 float.
        AscendC::DataCopyParams gather;
        gather.blockCount = (uint16_t)gCount;
        gather.blockLen   = (uint16_t)(perBlock / 8);
        gather.srcStride  = (uint16_t)(((eighth << 3) - perBlock) / 8);
        gather.dstStride  = 0;

        for (int m = 0; m < 8; ++m) {
            uint32_t src = k0 + shift0 + (uint32_t)m * eighth;
            AscendC::DataCopy(inT[(2*m)   * CUBE_BATCH], xrGm[src], gather);
            AscendC::DataCopy(inT[(2*m+1) * CUBE_BATCH], xiGm[src], gather);
        }

        // The twiddles have already been replicated across the columns on the
        // host (fill_cube_twiddle8), so here it is one contiguous read of
        // validB values per row.
        for (int m = 1; m <= 7; ++m) {
            uint32_t base = wcOffset + (uint32_t)(2*m - 2) * wcRowLen + shift0;
            AscendC::DataCopy(inT[(16 + 2*(m-1))   * CUBE_BATCH], wcGm[base],             validB);
            AscendC::DataCopy(inT[(16 + 2*(m-1)+1) * CUBE_BATCH], wcGm[base + wcRowLen],  validB);
        }
        inQ.EnQue(inT);
    }

    __aicore__ inline void Compute(uint32_t validB, uint32_t terms)
    {
        auto in2 = inQ.DeQue<float>();
        auto x = [&](int m, int ri) { return in2[(2*m+ri) * CUBE_BATCH]; };
        auto w = [&](int m, int ri) { return in2[(16 + 2*(m-1)+ri) * CUBE_BATCH]; };

        auto outT = outQ.AllocTensor<float>();
        // Block-real layout: Re(t_m) -> row m, Im(t_m) -> row 8+m.
        auto t = [&](int m, int ri) { return outT[((ri == 0) ? m : (8+m)) * CUBE_BATCH]; };
        auto u = outT[16 * CUBE_BATCH];
        auto v = outT[17 * CUBE_BATCH];

        // t0 = x0 (no twiddle)
        AscendC::Adds(t(0,0), x(0,0), 0.0f, validB);
        AscendC::Adds(t(0,1), x(0,1), 0.0f, validB);
        // t_m = x_m * w_m, m=1..7
        for (int m = 1; m <= 7; ++m) {
            AscendC::Mul(u, x(m,0), w(m,0), validB); AscendC::Mul(v, x(m,1), w(m,1), validB);
            AscendC::Sub(t(m,0), u, v, validB);
            AscendC::Mul(u, x(m,0), w(m,1), validB); AscendC::Mul(v, x(m,1), w(m,0), validB);
            AscendC::Add(t(m,1), u, v, validB);
        }
        AscendC::PipeBarrier<PIPE_V>();
        inQ.FreeTensor(in2);
        outQ.EnQue(outT);

        auto out2 = outQ.DeQue<float>();

        // X_hi — a plain fp32 -> fp16 cast in ONE call over all 16*CUBE_BATCH
        // elements (Re(t0..7) and Im(t0..7) lie consecutively, rows 0..15,
        // CUBE_BATCH each — exactly 16*CUBE_BATCH contiguous float ahead of
        // the u/v scratch in rows 16/17, which are not part of this).
        auto halfT = castQ.AllocTensor<half>();
        AscendC::Cast(halfT, out2[0], AscendC::RoundMode::CAST_RINT, 16 * CUBE_BATCH);

        if (terms >= 2) {
            // X_lo = X - float(X_hi): what did not fit into the half mantissa.
            // The back-cast half->float is RoundMode::CAST_NONE (a widening,
            // there is nothing to round). back is used as scratch (rows
            // 18..33 of the outQ buffer); after the Sub it holds the
            // correction term itself.
            auto back = out2[18 * CUBE_BATCH];
            AscendC::Cast(back, halfT, AscendC::RoundMode::CAST_NONE, 16 * CUBE_BATCH);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Sub(back, out2[0], back, 16 * CUBE_BATCH);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(halfT[16 * CUBE_BATCH], back, AscendC::RoundMode::CAST_RINT,
                          16 * CUBE_BATCH);
        }

        castQ.EnQue(halfT);
        outQ.FreeTensor(out2);

        auto halfOut = castQ.DeQue<half>();
        // The write length is AlignUp16(validB) and NOT validB: DataCopy
        // UB<->GM works in blocks of 32 bytes, which for half equals 16
        // elements (see the comment at AlignUp16 above — at validB=8 zero
        // blocks were copied, and the whole pipeline put out zeros).
        // Running out of bounds is impossible here: validB <= CUBE_BATCH = 128
        // and AlignUp16(128) = 128, that is, the only positions written over
        // are ones inside the same row, and the row address (m * 128 half =
        // m * 256 bytes) is a multiple of 32 bytes as well.
        // The surplus columns [validB, AlignUp16(validB)) hold garbage out of
        // UB, but they do not affect the result: in a GEMM, column j of matrix
        // C depends only on column j of matrix B, and unpack reads back
        // exactly validB columns.
        uint32_t copyB = AlignUp16(validB);
        uint32_t rows  = (terms >= 2) ? 32u : 16u;   // X_hi, plus X_lo when terms>=2
        for (uint32_t r = 0; r < rows; ++r) {
            AscendC::DataCopy(scratchXGm[r * CUBE_BATCH], halfOut[r * CUBE_BATCH], copyB);
        }
        castQ.FreeTensor(halfOut);
    }

private:
    AscendC::TPipe pipe;
    AscendC::GlobalTensor<float> xrGm, xiGm, wcGm;
    AscendC::GlobalTensor<half>  scratchXGm;
    AscendC::TQue<AscendC::TPosition::VECIN, 2>  inQ;
    AscendC::TQue<AscendC::TPosition::VECOUT, 2> outQ;
    AscendC::TQue<AscendC::TPosition::VECOUT, 2> castQ;
};

// Tile parameters: k0 — the start of the batch's first k-block, shift0 — the
// start of the window over shift inside the block, gCount/perBlock — how many
// blocks and how many butterflies out of each (see the comment at Process).
// wcOffset/wcRowLen — the offset and the row length of the stage in the cube
// twiddle buffer (build_cube_twiddle_plan8).
extern "C" __global__ __aicore__ void fft8_cube1_repack(
    GM_ADDR XR, GM_ADDR XI, GM_ADDR WC, GM_ADDR SCRATCH_X,
    uint32_t n, uint32_t tileBase, uint32_t kStride, uint32_t shiftStride,
    uint32_t shiftsPerBlk, uint32_t eighth, uint32_t gCount, uint32_t perBlock,
    uint32_t wcOffset, uint32_t wcRowLen, uint32_t wcTotal, uint32_t terms)
{
    KernelFFT8Cube1Repack op;
    op.Init(XR, XI, WC, SCRATCH_X, n, wcTotal, (uint32_t)AscendC::GetBlockIdx());
    op.Process(tileBase, kStride, shiftStride, shiftsPerBlk, eighth,
               gCount, perBlock, wcOffset, wcRowLen, terms);
}

#ifndef ASCENDC_CPU_DEBUG
void fft8_cube1_repack_do(uint32_t blockDim, void *stream,
                           void *xr, void *xi, void *wc, void *scratchX,
                           uint32_t n, uint32_t tileBase, uint32_t kStride,
                           uint32_t shiftStride, uint32_t shiftsPerBlk, uint32_t eighth,
                           uint32_t gCount, uint32_t perBlock, uint32_t wcOffset,
                           uint32_t wcRowLen, uint32_t wcTotal, uint32_t terms)
{
    fft8_cube1_repack<<<blockDim, nullptr, stream>>>(xr, xi, wc, scratchX,
                                                       n, tileBase, kStride, shiftStride,
                                                       shiftsPerBlk, eighth, gCount, perBlock,
                                                       wcOffset, wcRowLen, wcTotal, terms);
}
#endif