// kernels/r8_cube2_matmul.cpp
// R8-CUBE2-FS-XS — the Cube/AIC half of the cube2 variant: TWO matmuls per
// tile instead of one (the answer to the thesis question "one 16x16 matmul or
// two separate real ones").
//
// !!! WHY A SEPARATE FILE AND NOT NEXT TO fft8_cube1_matmul !!!
// The iron rule of this project, earned by debugging: ONE __global__ kernel
// per file. While fft8_cube1_repack and fft8_cube1_unpack lay in one file,
// CANN's precompile scanner marked both "_mix_aic"/"_mix_aiv" and they caught
// an aicore exception out of nowhere (see the history in
// kernels/r8_cube1_repack.cpp). We are not checking whether we get lucky this
// time.
// The file is added to KERNEL_FILES_CUBE (CMakeLists.txt) — it is built by a
// separate ascendc_library() with -DHAVE_WORKSPACE/-DHAVE_TILING, just like
// kernels/r8_cube_fused_xs.cpp.
//
// ============================================================================
// THE SCHEME
// ============================================================================
// cube1 computes combine with a single block-real 16x16 matmul:
//     [Re(y)]   [ MR  -MI ] [Re(t)]
//     [Im(y)] = [ MI   MR ] [Im(t)]
// cube2 keeps Re and Im SEPARATE and computes two matmuls with one and the
// same "vertical" weight matrix A2 = [MR; MI]:
//     P = A2 * Re(t) = [MR*Re(t); MI*Re(t)]
//     Q = A2 * Im(t) = [MR*Im(t); MI*Im(t)]
// and the combine step is done by Vector in unpack
// (kernels/r8_cube2_unpack.cpp):
//     Re(y) = P[0..7]  - Q[8..15]      ( = MR*Re - MI*Im )
//     Im(y) = P[8..15] + Q[0..7]       ( = MI*Re + MR*Im )
// Verified numerically before a line of code was written
// (scratchpad/chk_cube2.py): it matches M*t to 1e-15.
//
// !!! THE KEY DECISION: A2 IS STORED AS 16x16 WITH ZERO COLUMNS 8..15 !!!
// Mathematically A2 is 16x8 (K=8). Storing it that way and declaring the
// tiling at K=8 would be "more honest", but it brings three risks at once,
// every one of which has already gone off in this project:
//   1. A row of A 8 half long is 16 bytes — NOT a multiple of 32 bytes. The
//      granularity of DataCopy UB<->GM is exactly 32 bytes (see AlignUp16 in
//      kernels/r8_cube1_repack.cpp — the pipeline once produced pure zeros
//      over that), and the ND->NZ converter of the A operand goes the same
//      way.
//   2. We would have to build and debug a NEW tiling (16, 8, B). The current
//      one (16, 16, B) is the only shape tested on hardware; it has the
//      history with the hardware deadlock, the zeros and 507015 behind it.
//   3. The Cube engine's fractal for fp16 is 16x16 along K. At K=8 the
//      library will round baseK up to 16 anyway and read A columns 8..15 and
//      B rows 8..15. Our A columns 8..15 are ZERO, so whatever lies in B rows
//      8..15 contributes exactly nothing. That is, this storage is not merely
//      safer, it is IMMUNE to that rounding.
// The price: twice as many multiplications as necessary. But that is the real
// price of K=8 on a 16-wide fractal — an "honest" K=8 would have paid it too,
// only implicitly. So the measurement stays honest: cube2 does 2x the padded
// MACs of cube1, and that is exactly the comparison being measured.
//
// From the same decision follows something convenient: REPACK IS REUSED
// WITHOUT A SINGLE EDIT. fft8_cube1_repack already puts Re(t) in rows 0..7
// and Im(t) in 8..15 of one and the same buffer. So the B operand of the
// first matmul is scratchX + 0 (rows 0..15 = [Re(t); Im(t)], only Re(t)
// contributes), and of the second it is scratchX + 8*CUBE_BATCH (rows 8..23,
// only Im(t) contributes).
//
// Buffer layout (offsets in elements):
//   W2        : [0, 256) = A2_hi, [256, 512) = A2_lo    (16x16 half, both)
//   SCRATCH_X : rows 0..15  = X_hi ([Re; Im]),
//               rows 16..31 = X_lo ([Re_lo; Im_lo], only when terms >= 2)
//               — exactly what fft8_cube1_repack writes
//   SCRATCH_Y : SIX blocks of 16*CB float: P0,Q0,P1,Q1,P2,Q2
//               (cube1 had three: Y0,Y1,Y2)
//
// !!! ABOUT THE PADDING AT THE END OF SCRATCH_X (IMPORTANT FOR THE HOST) !!!
// The B operand of the Q1 matmul starts at row 24 and 16 rows are read, that
// is, up to row 40 — 8 rows PAST the end of the slot (32 rows). Where those
// 8 rows land depends on the slot, and both cases are benign for the same
// reason, but for DIFFERENT reasons they are safe:
//   * interior slots: the over-read falls into the NEXT slot's X_hi rows
//     0..7, i.e. a neighbouring tile's live Re_hi. It is real data, fully
//     inside the allocation, and it contributes nothing because it lands on
//     K rows 8..15, against which A2's columns are zero. No tile can corrupt
//     another: this is a read, never a write.
//   * the last slot: there is no next slot, so without help this would be a
//     read past the end of the buffer, i.e. a potential aicore exception.
//     That is why the host allocates scratchX with an extra 8*CUBE_BATCH
//     half at the tail (see run_cube_fs_xs in host/impl_r8.cpp).
// Do not "simplify" this by assuming the over-read always lands in dead
// padding — for 19 of the 20 slots it does not.
//
// !!! AND FOR THE SAME REASON THE HOST ZERO-FILLS scratchX BEFORE THE RUN !!!
// The matmuls inevitably read rows that repack never touched in this run: at
// terms=1 Q0 takes in X_lo (rows 16..23), and Q1 takes in the tail padding
// of the last slot.
// MATHEMATICALLY their contribution is zero (columns 8..15 of the matrix A2
// are zeros), but 0 * NaN = NaN, and a fresh aclrtMalloc holds garbage from
// previous tasks, in which NaN/Inf for half turns up all the time (an
// exponent of all ones is enough). Without the zero-fill this would give NaN
// in the output every other time, depending on what was left in DDR. The
// zero-fill is done once, outside the timing.

#include "kernel_operator.h"
#include "cube1_common.h"
#include <cstdint>

#ifndef ASCENDC_CPU_DEBUG
#include "lib/matmul_intf.h"

using MM2AType = matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half>;
using MM2BType = matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half>;
using MM2CType = matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>;

// terms — the number of terms of the precision-recovery scheme
// (Ozaki/split-matmul), spelled out in detail at fft8_cube1_matmul in
// kernels/r8_cube_fused_xs.cpp. There are just as many terms here, but each
// one costs TWO matmuls:
//   terms=1:  P0 = A_hi*Re_hi,  Q0 = A_hi*Im_hi
//   terms=2:  + P1 = A_hi*Re_lo, Q1 = A_hi*Im_lo
//   terms=3:  + P2 = A_lo*Re_hi, Q2 = A_lo*Im_hi
// They all still go through ONE Cube object and ONE launch per tile — which
// matters, because the time is bound by the launch overhead and not by the
// arithmetic (see [CUBE-PLAN] in host/impl_r8.cpp).
extern "C" __global__ __aicore__ void fft8_cube2_matmul(
    GM_ADDR W2, GM_ADDR SCRATCH_X, GM_ADDR SCRATCH_Y, uint32_t terms,
    GM_ADDR workspace, GM_ADDR tilingGm)
{
    constexpr uint32_t A_SZ  = 16 * 16;            // elements in one 16x16 matrix
    constexpr uint32_t ROW   = CUBE_BATCH;         // elements in one X/Y row
    constexpr uint32_t BLK   = 16 * CUBE_BATCH;    // elements in one X/Y block

    // One block = one tile, as in cube1: every AIC block has its own scratch
    // slot (slot = blockIdx), and each computes ITS OWN full matmuls. We
    // shift the GM pointers themselves — all the indexing below stays as in
    // the single-block version. W2 is shared by all blocks and is not
    // shifted. The slot is 2 blocks along X (X_hi, X_lo) and SIX along Y
    // (P0,Q0,P1,Q1,P2,Q2).
    const uint32_t slot = (uint32_t)AscendC::GetBlockIdx();

    AscendC::GlobalTensor<half>  w2Gm, scratchXGm;
    AscendC::GlobalTensor<float> scratchYGm;
    w2Gm.SetGlobalBuffer((__gm__ half *)W2, 2 * A_SZ);
    // The declared size is the slot's 2 blocks PLUS 8 rows of padding:
    // exactly what the host physically allocated, and exactly how far the B
    // operand of Q1 reads (see the warning in the file header).
    scratchXGm.SetGlobalBuffer((__gm__ half *)SCRATCH_X + slot * (2 * BLK),
                               2 * BLK + 8 * ROW);
    scratchYGm.SetGlobalBuffer((__gm__ float *)SCRATCH_Y + slot * (6 * BLK), 6 * BLK);

    TCubeTiling tiling;
    auto tilingSrc = (__gm__ uint32_t *)tilingGm;
    auto tilingDst = (uint32_t *)&tiling;
    for (uint32_t idx = 0; idx < sizeof(TCubeTiling) / sizeof(uint32_t); ++idx) {
        tilingDst[idx] = tilingSrc[idx];
    }

    AscendC::TPipe pipe;
    // Do NOT call SetSysWorkspaceForce(workspace) here — the auto-wrapper
    // (HAVE_WORKSPACE/HAVE_TILING, see common/cmake/npu_lib.cmake) does it
    // itself and does it right, together with matmul::clearWorkspace(). The
    // detailed account of why calling it a second time is dangerous, and why
    // it was precisely the absence of clearWorkspace() that gave the real
    // hardware deadlock, is in the comment at fft8_cube1_matmul
    // (kernels/r8_cube_fused_xs.cpp).
    (void)workspace;

    matmul::Matmul<MM2AType, MM2BType, MM2CType> mm;
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tiling);

    // There is no usedCoreNum guard, deliberately, for the same reason as in
    // cube1: the tiling describes ONE full 16x16x B matmul, and each block
    // computes it whole over its own slot, not a cut-out part of a shared
    // one. The launch's blockDim equals the number of tiles in the group, so
    // there are never any "spare" blocks.
    mm.SetOrgShape(tiling.M, tiling.N, tiling.Ka, tiling.Kb);

    // --- term 0: A_hi x X_hi -----------------------------------------------
    mm.SetTensorA(w2Gm[0]);
    mm.SetTensorB(scratchXGm[0]);                 // rows 0..15: Re_hi contributes
    mm.SetTail(tiling.M, tiling.N);
    mm.IterateAll(scratchYGm[0]);                 // P0

    mm.SetTensorA(w2Gm[0]);
    mm.SetTensorB(scratchXGm[8 * ROW]);           // rows 8..23: Im_hi contributes
    mm.SetTail(tiling.M, tiling.N);
    mm.IterateAll(scratchYGm[BLK]);               // Q0

    // --- term 1: A_hi x X_lo -----------------------------------------------
    if (terms >= 2) {
        mm.SetTensorA(w2Gm[0]);
        mm.SetTensorB(scratchXGm[16 * ROW]);      // rows 16..31: Re_lo contributes
        mm.SetTail(tiling.M, tiling.N);
        mm.IterateAll(scratchYGm[2 * BLK]);       // P1

        mm.SetTensorA(w2Gm[0]);
        mm.SetTensorB(scratchXGm[24 * ROW]);      // rows 24..39: Im_lo contributes
        mm.SetTail(tiling.M, tiling.N);
        mm.IterateAll(scratchYGm[3 * BLK]);       // Q1
    }

    // --- term 2: A_lo x X_hi -----------------------------------------------
    if (terms >= 3) {
        mm.SetTensorA(w2Gm[A_SZ]);
        mm.SetTensorB(scratchXGm[0]);
        mm.SetTail(tiling.M, tiling.N);
        mm.IterateAll(scratchYGm[4 * BLK]);       // P2

        mm.SetTensorA(w2Gm[A_SZ]);
        mm.SetTensorB(scratchXGm[8 * ROW]);
        mm.SetTail(tiling.M, tiling.N);
        mm.IterateAll(scratchYGm[5 * BLK]);       // Q2
    }
    mm.End();
}
// There is NO manual _do()/<<<>>> wrapper here and there must not be: the
// Matmul high-level API requires ACLRT_LAUNCH_KERNEL, and a manual wrapper in
// one file with a Matmul kernel breaks the build at
// MatmulChipCap::GetFeatures().
// It is launched from host/impl_r8.cpp as
//     ACLRT_LAUNCH_KERNEL(fft8_cube2_matmul)(cnt, stream, ...)
// through the auto-generated aclrtlaunch_fft8_cube2_matmul.h.
#endif // !ASCENDC_CPU_DEBUG
