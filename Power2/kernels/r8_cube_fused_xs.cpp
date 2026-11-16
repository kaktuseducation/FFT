// kernels/r8_cube_fused_xs.cpp
// R8-CUBE1-FS-XS — the radix-8 butterfly combine stage on the Cube engine
// (AIC): ONE block-real 16x16 matmul per tile.
//
// !!! IMPORTANT (file structure) !!!
// One __global__ kernel per file — otherwise the CANN precompile classifier
// marks the file as "mix" AIC+AIV and the kernels catch an aicore exception
// out of nowhere. That is why the whole Cube path is laid out like this:
//   kernels/r8_cube1_repack.cpp   — Vector: twiddles + block-real repack
//                                   into half (SHARED by cube1 and cube2)
//   kernels/r8_cube_fused_xs.cpp  — this file: the 16x16 Matmul of cube1
//   kernels/r8_cube1_unpack.cpp   — Vector: sum of terms + layout (cube1)
//   kernels/r8_cube2_matmul.cpp   — Matmul: the TWO matmuls of cube2
//   kernels/r8_cube2_unpack.cpp   — Vector: sum + Re/Im combine step + layout
// lib/matmul_intf.h is really only needed by the two Matmul files, and they
// are also the only ones in KERNEL_FILES_CUBE.
//
// !!! THE CAUSE OF THE HARDWARE DEADLOCK IS FOUND — IT IS NOT THE DTYPES !!!
// The real reason the matmul hung forever: the library
// ascendc_kernels_npu_cube was built WITHOUT -DHAVE_WORKSPACE/-DHAVE_TILING,
// and so the auto-generated wrapper never called matmul::clearWorkspace() —
// the message-exchange area between the AIC and AIV halves of the MIX kernel
// stayed uninitialised garbage, and the handshake between the halves never
// completed. Fixed in common/cmake/npu_lib.cmake (the defines added
// surgically, ONLY to the cube library) + the manual SetSysWorkspaceForce()
// below removed. Proof: this very same kernel, built inside the harness of the
// official reference (where those defines are present), does not hang at all.
// What was checked along the way and REJECTED as causes of the deadlock:
// dtype (fp32 vs fp16), the 16x16x128 shape, the order/names of the kernel
// parameters, SetDim(1) vs SetDim(GetCoreNumAiv()), KERNEL_TASK_TYPE_DEFAULT,
// SetSingleShape, two AscendC libraries in one process, the aclrtMalloc
// flags, an explicit aclrtCreateContext, the order/volume of the preceding
// memory allocations, the way the tiling is copied (a manual loop vs
// CopyTiling).
//
// !!! fp16 FIX (needed for PRECISION, not because of the deadlock) !!!
// An isolated test of the official Huawei reference (matmul_custom.cpp,
// samples/.../11_matmul_kernellaunch) on this same hardware (910B4) showed:
//   - The reference "as is" (A_T=half, B_T=half, C_T=float, 512x512x1024):
//     error ratio 0.0000, test pass — the MIX mechanism/environment is fine.
//   - The same reference with A_T/B_T swapped out for float (the same shape,
//     512x512x1024): does NOT hang, but computes pure zeros (error ratio 1.0,
//     actual=0 everywhere) — fp32×fp32 through this high-level API path is
//     simply not fully supported on this architecture (the internal L0A/L0B
//     layout is cut for fp16 granularity).
//   - The same swap to fp32 in OUR kernel also does not hang, if it is built
//     correctly (with HAVE_WORKSPACE) — i.e. fp32 would have given not a
//     deadlock but a silently wrong result (zeros), which is worse.
// Conclusion: fp32×fp32 through this path computes garbage, so A/B stay fp16
// regardless of the deadlock story.
// The fix: A (W16) and B (the X operand out of repack) are half (fp16) now;
// C (the Y output) stays float — exactly the same type pattern as in the
// proven-working official reference. The twiddle arithmetic in repack and the
// whole rest of the vector part of the pipeline stay fp32; the fp32->fp16
// cast happens ONCE, immediately before the X operand is written into
// scratch (see kernels/r8_cube1_repack.cpp) — a deliberate, documented
// precision trade-off for the sake of the Cube path working at all,
// discussed in the trade-offs chapter of the thesis.
//
// cube2_fs_xs IS IMPLEMENTED and lives in kernels/r8_cube2_matmul.cpp +
// kernels/r8_cube2_unpack.cpp: two matmuls with the weight matrix [MR; MI]
// instead of a single 16x16, plus Vector Add/Sub for the Re/Im combine step.
// Repack is SHARED by cube1 and cube2 (it already puts Re in rows 0..7 and Im
// in 8..15 — exactly what both need); the detailed walkthrough of the scheme
// is in the header of kernels/r8_cube2_matmul.cpp.
//
// ----------------------------------------------------------------------
// The idea (spelled out in detail in host/fft_host_r8.h, section "Cube
// variants"): the twiddle multiplication t_m = x_m * W_block^(mj) is
// data-dependent (it depends on j and on the stage) and stays on the Vector
// engine. COMBINE, though — the 8-point DFT of the vector t0..t7 by the fixed
// roots W8^k — is ONE AND THE SAME matrix for absolutely all groups of all
// stages. Combine is exactly what goes onto the Cube:
//   cube1: [Re(y);Im(y)] = M16 * [Re(t);Im(t)],  M16 is a fixed 16x16 real
//          matrix (see build_dft8_matrix16_blockreal).
//   cube2: Re(y) = MR*Re(t) - MI*Im(t) ;  Im(y) = MR*Im(t) + MI*Re(t),
//          MR,MI are fixed 8x8 (see build_dft8_matrix). That is four
//          PRODUCTS, but they are computed as TWO matmuls, not four: the
//          weight matrix is stacked as A2 = [MR; MI] and the two operands
//          Re(t) and Im(t) are fed to it in turn, so one matmul yields two
//          of the products at once (see kernels/r8_cube2_matmul.cpp). Plus
//          Vector Add/Sub for the combine step, which cube1 does not need.
//
// BATCHING: 8 (or 16) points is LESS than a typical Cube tile (usually 16x16
// for fp16; for fp32 the Cube unit's tiles on the 910B4 are also expected to
// be multiples of 16). So instead of one matmul per group we batch B
// independent groups of a stage into ONE call: an X matrix of size [16, B]
// (cube1) or [8, B] (cube2), multiplied by the fixed M16 (16x16) / MR,MI
// (8x8) — giving Y [16,B] / [8,B]. What this RUNS is not an honestly
// block-diagonal batched matmul but one "wide" GEMM: M (16x16 or 8x8, FIXED)
// x X (16xB or 8xB) -> Y. There is no loss on the zeros of a block-diagonal
// matrix here, because M is ONE AND THE SAME for all B groups at once (unlike
// an honest per-group weight) — and that is exactly what gives the Cube
// approach a chance to pay back the fixed overhead of launching the
// instruction.

#include "kernel_operator.h"
#include "cube1_common.h"
#include <cstdint>

#ifndef ASCENDC_CPU_DEBUG
#include "lib/matmul_intf.h"   // AscendC Matmul high-level API (see the warning above)
#endif

#ifndef ASCENDC_CPU_DEBUG
using MMAType = matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half>;
using MMBType = matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half>;
using MMCType = matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>;

// terms — how many terms of the "precision recovery" scheme to compute in one
// launch (they are all done by ONE Cube object, that is, there is still one
// launch per tile — which matters, because the run time is bound by the launch
// overhead and not by the arithmetic, see [CUBE-PLAN] in host/impl_r8.cpp):
//
//   terms=1:  Y = A_hi*X_hi
//             Pure fp16. Error ~5e-4 relative to the amplitude of the data
//             (11 bits of mantissa); at N=4096 that gives max_abs ~0.5.
//   terms=2:  Y = A_hi*X_hi + A_hi*X_lo,   X = X_hi + X_lo (both half)
//             Removes the DOMINANT error — the rounding of the data X. Only
//             the error of the constant A is left (~1e-4 relative).
//   terms=3:  Y = A_hi*X_hi + A_hi*X_lo + A_lo*X_hi,  A = A_hi + A_lo
//             Only A_lo*X_lo is dropped (the product of two corrections,
//             ~1e-7), that is, practically fp32 precision on an fp16 engine.
//
// This is the classic Ozaki/split-matmul scheme for matrix engines without
// fp32: a number not representable in half is split into a sum of two half,
// and the product is expanded term by term. The accumulation is fp32
// everywhere (MMCType), so the sums themselves are exact — the only thing
// lost is what was dropped.
//
// Buffer layout (offsets in elements):
//   W16       : [0, 256)      = A_hi,  [256, 512)      = A_lo
//   SCRATCH_X : [0, 16*CB)    = X_hi,  [16*CB, 32*CB)  = X_lo
//   SCRATCH_Y : [0, 16*CB)=Y0, [16*CB, 32*CB)=Y1, [32*CB, 48*CB)=Y2
// Y0..Y2 are summed by the vector half (kernels/r8_cube1_unpack.cpp): keeping
// an accumulator on the Cube is more expensive, and Vector reads Y for the
// layout anyway.
extern "C" __global__ __aicore__ void fft8_cube1_matmul(
    GM_ADDR W16, GM_ADDR SCRATCH_X, GM_ADDR SCRATCH_Y, uint32_t terms,
    GM_ADDR workspace, GM_ADDR tilingGm)
{
    constexpr uint32_t A_SZ = 16 * 16;            // elements in one 16x16 matrix
    constexpr uint32_t X_SZ = 16 * CUBE_BATCH;    // elements in one X/Y block

    // ONE BLOCK = ONE TILE. A launch processes a group of tiles of a stage,
    // one per AIC block; each has its own scratch slot (slot = blockIdx), and
    // each block computes ITS OWN full 16x16x128 matmul — not a part of a
    // shared one. So we shift the GM pointers themselves to the slot: all the
    // indexing below stays exactly as in the single-block version.
    // The weight matrix W16 is shared by all blocks and is not shifted.
    const uint32_t slot = (uint32_t)AscendC::GetBlockIdx();

    AscendC::GlobalTensor<half>  w16Gm, scratchXGm;
    AscendC::GlobalTensor<float> scratchYGm;
    w16Gm.SetGlobalBuffer((__gm__ half *)W16, 2 * A_SZ);
    scratchXGm.SetGlobalBuffer((__gm__ half *)SCRATCH_X + slot * (2 * X_SZ), 2 * X_SZ);
    scratchYGm.SetGlobalBuffer((__gm__ float *)SCRATCH_Y + slot * (3 * X_SZ), 3 * X_SZ);

    TCubeTiling tiling;
    auto tilingSrc = (__gm__ uint32_t *)tilingGm;
    auto tilingDst = (uint32_t *)&tiling;
    for (uint32_t idx = 0; idx < sizeof(TCubeTiling) / sizeof(uint32_t); ++idx) {
        tilingDst[idx] = tilingSrc[idx];
    }

    AscendC::TPipe pipe;
    // !!! DO NOT call SetSysWorkspaceForce(workspace) here !!!
    // With HAVE_WORKSPACE/HAVE_TILING enabled (see common/cmake/npu_lib.cmake)
    // the auto-generated wrapper auto_gen_r8_cube_fused_xs.cpp does this
    // ITSELF, and does it correctly and in the right order:
    //     AscendC::SetSysWorkspaceForce(workspace_param);
    //     workspace_usr = AscendC::GetUserWorkspace(workspace_param);
    //     if constexpr (g_coreType == AscendC::AIC) matmul::clearWorkspace(workspace_param);
    //     workspace = workspace_usr;           // the kernel receives the USER workspace
    // The key one of those calls is matmul::clearWorkspace(): it zero-fills,
    // inside the system workspace, the message-exchange area between the AIC
    // and AIV halves of the MIX kernel (KFC: MatmulClient on AIV <->
    // MatmulServer on AIC). Without it there is garbage lying there from
    // previous tasks, and the handshake between the halves NEVER completes —
    // that was the real hardware deadlock (device log: SyncTask/
    // LogicCqReportV2 round and round, CPU% ~0).
    // Calling SetSysWorkspaceForce(workspace) a second time here is
    // DANGEROUS: by this point the workspace parameter already holds the user
    // (shifted) pointer, and the system base of the workspace would be
    // corrupted. That is exactly why the official reference
    // (matmul_custom.cpp) does not have this call either — its absence there
    // is not an accident but a consequence of their build defining
    // HAVE_WORKSPACE/HAVE_TILING.
    (void)workspace;

    matmul::Matmul<MMAType, MMBType, MMCType> mm;
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tiling);

    // !!! THERE IS NO usedCoreNum GUARD HERE, AND THAT IS DELIBERATE !!!
    // In the reference matmul_custom.cpp the guard `if (GetBlockIdx() >=
    // usedCoreNum) return;` is needed because there ONE matmul is cut by the
    // tiling across usedCoreNum cores, and the blocks beyond that number have
    // nothing to do. Our task is arranged the other way round: every block
    // computes its own full 16x16x128 matmul over its own scratch slot, and
    // the tiling describes exactly one such task (usedCoreNum=1,
    // singleCoreM=16, singleCoreN=128 — that is, the full size). With a guard
    // here every block but block zero would silently return, and the group of
    // tiles would be computed to one eighth. The launch's blockDim equals the
    // number of tiles in the group, so there are never any "spare" blocks and
    // nothing to cut off.
    mm.SetOrgShape(tiling.M, tiling.N, tiling.Ka, tiling.Kb);

    // A_hi * X_hi -> Y0
    mm.SetTensorA(w16Gm[0]);
    mm.SetTensorB(scratchXGm[0]);
    mm.SetTail(tiling.M, tiling.N);
    mm.IterateAll(scratchYGm[0]);

    if (terms >= 2) {          // A_hi * X_lo -> Y1
        mm.SetTensorA(w16Gm[0]);
        mm.SetTensorB(scratchXGm[X_SZ]);
        mm.SetTail(tiling.M, tiling.N);
        mm.IterateAll(scratchYGm[X_SZ]);
    }
    if (terms >= 3) {          // A_lo * X_hi -> Y2
        mm.SetTensorA(w16Gm[A_SZ]);
        mm.SetTensorB(scratchXGm[0]);
        mm.SetTail(tiling.M, tiling.N);
        mm.IterateAll(scratchYGm[2 * X_SZ]);
    }
    mm.End();
}
// There is NO manual _do()/<<<>>> wrapper — host/impl_r8.cpp calls
// fft8_cube1_matmul through ACLRT_LAUNCH_KERNEL(fft8_cube1_matmul)(...)
// (mandatory for the Matmul high-level API — a manual _do() wrapper in this
// same file breaks the build at MatmulChipCap::GetFeatures()).
// fft8_cube1_repack/unpack are in files of their own
// (kernels/r8_cube1_repack.cpp / kernels/r8_cube1_unpack.cpp) and are
// launched through their own _do() wrappers, on the same stream as this
// kernel; the order between launches is guaranteed by stream semantics,
// without any cross-core sync inside a single kernel.
#endif // !ASCENDC_CPU_DEBUG — end of the NPU-only fft8_cube1_matmul implementation
// ============================================================================
// cube2_fs_xs — TWO matmuls instead of one. Moved out into its own files:
//   kernels/r8_cube2_matmul.cpp — the Cube half (A2 = [MR; MI], two matmuls
//                                 per term of the precision scheme)
//   kernels/r8_cube2_unpack.cpp — the Vector half (sum of terms + the combine
//                                 step Re(y)=P-Q', Im(y)=P'+Q + layout)
// class KernelFFT8Cube2FusedXS and __global__ fft8_cube2_fused_xs used to live
// here — empty stubs that were called from nowhere. They were deleted rather
// than finished in place, for two reasons:
//   1. ONE __global__ PER FILE. A second Matmul kernel next to
//      fft8_cube1_matmul is exactly the configuration on which this project
//      has already caught the "_mix_aic"/"_mix_aiv" classification and an
//      aicore exception (the history is in kernels/r8_cube1_repack.cpp). We
//      did not go and check whether we would get lucky this time.
//   2. The stub was arranged differently from the working cube1: one kernel
//      for all stages with the loop over groups INSIDE (Process() over s and
//      g0). The working scheme is a loop on the host, three launches per
//      group of tiles (repack on AIV, matmul on AIC, unpack on AIV), with the
//      order given by the stream. Finishing the stub would have meant first
//      rewriting it into that scheme, that is, throwing it away whole — which
//      is what was done.
// ============================================================================
