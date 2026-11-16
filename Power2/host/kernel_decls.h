// host/kernel_decls.h  (FFTBench)
//
// Declarations of ALL the project's kernels. Assembled out of the three
// kernel_decls.h of the original projects by plain concatenation: the kernel
// names never overlapped to begin with (the prefixes fft_ / fft4_ / fft8_), so
// there are no conflicts and nothing had to be edited. The historical comments
// are kept — they describe the pitfalls we have already stepped on.
#pragma once
#include <cstdint>

// FIX (multicore in ALL the variants): the signatures below are brought into
// exact correspondence with the kernels in kernels/ — generated out of them,
// so that host and device cannot diverge. The general scheme:
//   * the per-stage kernels got ONLY a trailing uint32_t blk_num: they already
//     computed one stage per launch, the only thing missing was handing the
//     tiles out across the cores;
//   * the fused kernels got a trailing uint32_t stage, uint32_t blk_num;
//     stage == 0xFFFFFFFFu means single-core legacy mode (the kernel walks all
//     the stages itself), any other value means compute exactly this stage.
//     That way the single-core variants stayed bit-for-bit the same and
//     comparable with the matrix already taken.
//   * blk_num MUST match the blockDim of the launch: less — and part of the
//     tiles gets computed twice by different cores, more — and part does not
//     get computed at all.


#ifdef ASCENDC_CPU_DEBUG
#include "kernel_operator.h"
#include "tikicpulib.h"
#endif

// ###########################################################################
// ##  radix-2  (r2/Radix2/host/kernel_decls.h)
// ###########################################################################
// ---- r2_dit_perstage_xs.cpp ----
#ifdef ASCENDC_CPU_DEBUG
extern "C" __global__ __aicore__ void fft_hybrid(GM_ADDR XR, GM_ADDR XI, GM_ADDR W, uint32_t full_size, uint32_t half_size, uint32_t block_size, uint32_t aligned, uint32_t blk_num);
#endif
#ifndef ASCENDC_CPU_DEBUG
void fft_hybrid_do(uint32_t blockDim, void *stream, void *xr, void *xi, void *w, uint32_t full_size, uint32_t half_size, uint32_t block_size, uint32_t aligned, uint32_t blk_num);
#endif

// ---- r2_dit_perstage_xi.cpp ----
#ifdef ASCENDC_CPU_DEBUG
extern "C" __global__ __aicore__ void fft_interleaved(GM_ADDR X, GM_ADDR W, uint32_t full_size, uint32_t half_size, uint32_t block_size, uint32_t aligned, uint32_t blk_num);
#endif
#ifndef ASCENDC_CPU_DEBUG
void fft_interleaved_do(uint32_t blockDim, void *stream, void *x, void *w, uint32_t full_size, uint32_t half_size, uint32_t block_size, uint32_t aligned, uint32_t blk_num);
#endif

// ---- r2_dit_fused_xs.cpp ----
#ifdef ASCENDC_CPU_DEBUG
extern "C" __global__ __aicore__ void fft_dit_fused_xs(GM_ADDR XR, GM_ADDR XI, GM_ADDR W, GM_ADDR META, uint32_t full_size, uint32_t num_stages, uint32_t stage, uint32_t blk_num);
#endif
// FIX (multicore radix-2): stage and blk_num, as in radix-4 and radix-8.
//   stage == 0xFFFFFFFFu — single-core legacy mode (variant r2/dit_fs_xs),
//     the numbers stayed bit-for-bit the same;
//   otherwise — exactly this stage, tiles across blocks (variant
//     r2/dit_fsm_xs).
#ifndef ASCENDC_CPU_DEBUG
void fft_dit_fused_xs_do(uint32_t blockDim, void *stream, void *xr, void *xi, void *w, void *meta, uint32_t full_size, uint32_t num_stages, uint32_t stage, uint32_t blk_num);
#endif

// ---- r2_dit_fused_xi.cpp ----
#ifdef ASCENDC_CPU_DEBUG
extern "C" __global__ __aicore__ void fft_dit_fused_xi(GM_ADDR X, GM_ADDR W, GM_ADDR META, uint32_t full_size, uint32_t num_stages, uint32_t stage, uint32_t blk_num);
#endif
#ifndef ASCENDC_CPU_DEBUG
void fft_dit_fused_xi_do(uint32_t blockDim, void *stream, void *x, void *w, void *meta, uint32_t full_size, uint32_t num_stages, uint32_t stage, uint32_t blk_num);
#endif

// ---- r2_stockham_perstage_xs.cpp ----
#ifdef ASCENDC_CPU_DEBUG
extern "C" __global__ __aicore__ void fft_stockham_xs(GM_ADDR XR_IN, GM_ADDR XI_IN, GM_ADDR XR_OUT, GM_ADDR XI_OUT, GM_ADDR W, uint32_t full_size, uint32_t half_size, uint32_t aligned, uint32_t blk_num);
#endif
#ifndef ASCENDC_CPU_DEBUG
void fft_stockham_xs_do(uint32_t blockDim, void *stream, void *xr_in, void *xi_in, void *xr_out, void *xi_out, void *w, uint32_t full_size, uint32_t half_size, uint32_t aligned, uint32_t blk_num);
#endif

// ---- r2_stockham_perstage_xi.cpp ----
#ifdef ASCENDC_CPU_DEBUG
extern "C" __global__ __aicore__ void fft_stockham_xi(GM_ADDR X_IN, GM_ADDR X_OUT, GM_ADDR W, uint32_t full_size, uint32_t half_size, uint32_t aligned, uint32_t blk_num);
#endif
#ifndef ASCENDC_CPU_DEBUG
void fft_stockham_xi_do(uint32_t blockDim, void *stream, void *x_in, void *x_out, void *w, uint32_t full_size, uint32_t half_size, uint32_t aligned, uint32_t blk_num);
#endif

// ---- r2_stockham_fused_xs.cpp ----
#ifdef ASCENDC_CPU_DEBUG
extern "C" __global__ __aicore__ void fft_stockham_fused_xs(GM_ADDR XR0, GM_ADDR XI0, GM_ADDR XR1, GM_ADDR XI1, GM_ADDR W, GM_ADDR META, uint32_t full_size, uint32_t num_stages, uint32_t stage, uint32_t blk_num);
#endif
#ifndef ASCENDC_CPU_DEBUG
void fft_stockham_fused_xs_do(uint32_t blockDim, void *stream, void *xr0, void *xi0, void *xr1, void *xi1, void *w, void *meta, uint32_t full_size, uint32_t num_stages, uint32_t stage, uint32_t blk_num);
#endif

// ---- r2_stockham_fused_xi.cpp ----
#ifdef ASCENDC_CPU_DEBUG
extern "C" __global__ __aicore__ void fft_stockham_fused_xi(GM_ADDR X0, GM_ADDR X1, GM_ADDR W, GM_ADDR META, uint32_t full_size, uint32_t num_stages, uint32_t stage, uint32_t blk_num);
#endif
#ifndef ASCENDC_CPU_DEBUG
void fft_stockham_fused_xi_do(uint32_t blockDim, void *stream, void *x0, void *x1, void *w, void *meta, uint32_t full_size, uint32_t num_stages, uint32_t stage, uint32_t blk_num);
#endif

// ###########################################################################
// ##  radix-4  (r4/Radix4/host/kernel_decls.h)
// ###########################################################################
// ---- r4_dit_perstage_xs.cpp ----
#ifdef ASCENDC_CPU_DEBUG
extern "C" __global__ __aicore__ void fft4_hybrid(GM_ADDR XR, GM_ADDR XI, GM_ADDR W, uint32_t full_size, uint32_t quarter_size, uint32_t block_size, uint32_t aligned, uint32_t blk_num);
#endif
#ifndef ASCENDC_CPU_DEBUG
void fft4_hybrid_do(uint32_t blockDim, void *stream, void *xr, void *xi, void *w, uint32_t full_size, uint32_t quarter_size, uint32_t block_size, uint32_t aligned, uint32_t blk_num);
#endif

// ---- r4_dit_fused_xs.cpp ----
#ifdef ASCENDC_CPU_DEBUG
extern "C" __global__ __aicore__ void fft4_dit_fused_xs(GM_ADDR XR, GM_ADDR XI, GM_ADDR W, GM_ADDR META, uint32_t full_size, uint32_t num_stages, uint32_t stage, uint32_t blk_num);
#endif
// FIX (multicore radix-4): stage and blk_num added.
//   stage == 0xFFFFFFFFu — single-core legacy mode, the kernel walks all the
//     stages itself in one launch. That is the variant r4/dit_fs_xs, its
//     numbers stayed bit-for-bit what they were before the fix.
//   otherwise — multicore mode: exactly this stage, tiles across blocks with
//     a stride of blk_num (= blockDim). That is the variant r4/dit_fsm_xs.
// The detailed reasoning for why the stages are spread over separate launches
// is in the header of the two Process methods in kernels/r4_dit_fused_xs.cpp.
#ifndef ASCENDC_CPU_DEBUG
void fft4_dit_fused_xs_do(uint32_t blockDim, void *stream, void *xr, void *xi, void *w, void *meta, uint32_t full_size, uint32_t num_stages, uint32_t stage, uint32_t blk_num);
#endif

// ---- r4_stockham_perstage_xs.cpp ----
#ifdef ASCENDC_CPU_DEBUG
extern "C" __global__ __aicore__ void fft4_stockham_xs(GM_ADDR XR_IN, GM_ADDR XI_IN, GM_ADDR XR_OUT, GM_ADDR XI_OUT, GM_ADDR W, uint32_t full_size, uint32_t quarter_size, uint32_t aligned, uint32_t blk_num);
#endif
#ifndef ASCENDC_CPU_DEBUG
void fft4_stockham_xs_do(uint32_t blockDim, void *stream, void *xr_in, void *xi_in, void *xr_out, void *xi_out, void *w, uint32_t full_size, uint32_t quarter_size, uint32_t aligned, uint32_t blk_num);
#endif

// ---- r4_stockham_fused_xs.cpp ----
#ifdef ASCENDC_CPU_DEBUG
extern "C" __global__ __aicore__ void fft4_stockham_fused_xs(GM_ADDR XR0, GM_ADDR XI0, GM_ADDR XR1, GM_ADDR XI1, GM_ADDR W, GM_ADDR META, uint32_t full_size, uint32_t num_stages, uint32_t stage, uint32_t blk_num);
#endif
#ifndef ASCENDC_CPU_DEBUG
void fft4_stockham_fused_xs_do(uint32_t blockDim, void *stream, void *xr0, void *xi0, void *xr1, void *xi1, void *w, void *meta, uint32_t full_size, uint32_t num_stages, uint32_t stage, uint32_t blk_num);
#endif

// ###########################################################################
// ##  radix-8  (live/Radix8/host/kernel_decls.h)
// ###########################################################################
// ---- kernels/r8_dit_fused_xs.cpp : dit_fs_xs (Vector/AIV, real radix-8 DIT) ----
#ifdef ASCENDC_CPU_DEBUG
extern "C" __global__ __aicore__ void fft8_dit_fused_xs(GM_ADDR XR, GM_ADDR XI, GM_ADDR W, GM_ADDR META, uint32_t full_size, uint32_t num_stages, uint32_t stage, uint32_t blk_num);
#endif
// FIX (multicore): one call = ONE stage (the stage argument), not the whole
// FFT. blk_num is the same number that is passed as blockDim: the kernel hands
// the stage's tiles out across the cores with that stride. The details and the
// reasoning for why the stages are spread over separate launches are in the
// header of Process() in kernels/r8_dit_fused_xs.cpp.
#ifndef ASCENDC_CPU_DEBUG
void fft8_dit_fused_xs_do(uint32_t blockDim, void *stream, void *xr, void *xi, void *w, void *meta, uint32_t full_size, uint32_t num_stages, uint32_t stage, uint32_t blk_num);
#endif

// ---- kernels/r8_cube1_repack.cpp + kernels/r8_cube1_unpack.cpp : cube1_fs_xs,
// the Vector/AIV half (repack/twiddle + unpack) ----
// (the former shared file kernels/cube1_repack_unpack.cpp was split into
// two — one __global__ kernel per file; if it is still sitting in kernels/,
// it has to be deleted: it is a dead duplicate with the same kernel names.)
// FIX: these used to be declared through ACLRT_LAUNCH_KERNEL + the
// autogenerated aclrtlaunch_fft8_cube1_repack.h/unpack.h — the same mechanism
// fft8_cube1_matmul MUST be launched by (see below). In practice it was exactly
// that launch path which, for pure Vector kernels, gave a deterministic aicore
// exception (0x7150026, "unaligned UUB") that did not depend on the contents of
// the kernel body at all (checked with three different rewrites). The one way
// of launching an AIV kernel proven to work anywhere in this project is a
// manual _do() wrapper with <<<blockDim,nullptr,stream>>> (like
// fft8_dit_fused_xs_do above) — they were moved onto that same one, see
// kernels/r8_cube1_repack.cpp and kernels/r8_cube1_unpack.cpp.
// FIX (batching over k-blocks): instead of (shift, validB) it is now
// (shift0, gCount, perBlock) — the batch is assembled out of gCount
// consecutive k-blocks taking perBlock butterflies from each,
// validB = gCount*perBlock. The twiddles come out of a separate cube buffer
// (wc) whose layout suits that batching: wcOffset/wcRowLen are the offset and
// the row length of the stage inside it (build_cube_twiddle_plan8 in
// host/fft_host_r8.h).
// terms is the number of terms of the precision-recovery scheme (1/2/3), see
// kernels/r8_cube_fused_xs.cpp: repack writes X_lo only when terms>=2, unpack
// sums the corresponding number of Y blocks.
#ifdef ASCENDC_CPU_DEBUG
extern "C" __global__ __aicore__ void fft8_cube1_repack(GM_ADDR XR, GM_ADDR XI, GM_ADDR WC, GM_ADDR SCRATCH_X, uint32_t n, uint32_t tileBase, uint32_t kStride, uint32_t shiftStride, uint32_t shiftsPerBlk, uint32_t eighth, uint32_t gCount, uint32_t perBlock, uint32_t wcOffset, uint32_t wcRowLen, uint32_t wcTotal, uint32_t terms);
extern "C" __global__ __aicore__ void fft8_cube1_unpack(GM_ADDR XR, GM_ADDR XI, GM_ADDR SCRATCH_Y, uint32_t n, uint32_t tileBase, uint32_t kStride, uint32_t shiftStride, uint32_t shiftsPerBlk, uint32_t eighth, uint32_t gCount, uint32_t perBlock, uint32_t terms);
#endif
#ifndef ASCENDC_CPU_DEBUG
// FIX (multicore Cube): one call = a GROUP of a stage's tiles, one tile per
// block, blockDim = the number of tiles in the group. Instead of ready-made
// (k0, shift0) what is passed is the enumeration rule (tileBase + strides), by
// which a block finds its own tile out of blockIdx; the formula and the
// invariants are in the header of Process() in kernels/r8_cube1_repack.cpp.
// The slot in the scratch buffers = blockIdx.
void fft8_cube1_repack_do(uint32_t blockDim, void *stream, void *xr, void *xi, void *wc, void *scratchX, uint32_t n, uint32_t tileBase, uint32_t kStride, uint32_t shiftStride, uint32_t shiftsPerBlk, uint32_t eighth, uint32_t gCount, uint32_t perBlock, uint32_t wcOffset, uint32_t wcRowLen, uint32_t wcTotal, uint32_t terms);
void fft8_cube1_unpack_do(uint32_t blockDim, void *stream, void *xr, void *xi, void *scratchY, uint32_t n, uint32_t tileBase, uint32_t kStride, uint32_t shiftStride, uint32_t shiftsPerBlk, uint32_t eighth, uint32_t gCount, uint32_t perBlock, uint32_t terms);
#endif

// ---- kernels/r8_cube_fused_xs.cpp : cube1_fs_xs (Cube/AIC, a single 16x16
// block-real matmul per group) ----
// META is the same format (eighth, aligned, wOffset) as in dit_fs_xs; the
// per-group twiddles are read out of the same W buffer as in the Vector variant
// (see the comment in the header of kernels/r8_cube_fused_xs.cpp). NPU-only: a
// CPU-debug (tikicpulib) simulation of the Cube unit is not implemented here.
//
// fft8_cube1_matmul is one of the two kernels (the other is fft8_cube2_matmul
// below) for which ACLRT_LAUNCH_KERNEL(...) + the autogenerated
// aclrtlaunch_fft8_cube1_matmul.h is MANDATORY (after the pattern of the
// official 11_matmul_kernellaunch): a manual _do() wrapper for it breaks the
// build once the Matmul high-level API is involved inside (see the exchange
// about MatmulChipCap::GetFeatures() in CMakeLists.txt). Repack/unpack above do
// NOT rely on that mechanism any more — see the fix above.
#ifndef ASCENDC_CPU_DEBUG
#include "aclrtlaunch_fft8_cube1_matmul.h"
#endif

// ---- kernels/r8_cube2_matmul.cpp + kernels/r8_cube2_unpack.cpp : cube2_fs_xs
// (Cube/AIC, TWO matmuls per term: P = A2*Re(t), Q = A2*Im(t), where
// A2 = [Re(M); Im(M)]; the combine Re(y)=P-Q', Im(y)=P'+Q — on Vector in
// unpack) ----
//
// Repack is SHARED by cube1 and cube2 (fft8_cube1_repack above): it already
// puts Re(t) into rows 0..7 and Im(t) into 8..15, that is, both B operands of
// cube2 are slices of one and the same buffer. The scheme and the reasoning are
// in the header of kernels/r8_cube2_matmul.cpp.
//
// fft8_cube2_matmul is launched ONLY through ACLRT_LAUNCH_KERNEL + its own
// autogenerated aclrtlaunch_fft8_cube2_matmul.h (like fft8_cube1_matmul): a
// manual _do() wrapper in one file with the Matmul high-level API breaks the
// build on MatmulChipCap::GetFeatures(). The last two parameters — GM_ADDR
// workspace, GM_ADDR tilingGm — are the mandatory convention of the cube
// library, which is built with -DHAVE_WORKSPACE/-DHAVE_TILING (see
// common/cmake/npu_lib.cmake).
// fft8_cube2_unpack is an ordinary Vector kernel with a _do() wrapper of its
// own.
#ifdef ASCENDC_CPU_DEBUG
extern "C" __global__ __aicore__ void fft8_cube2_unpack(GM_ADDR XR, GM_ADDR XI, GM_ADDR SCRATCH_Y, uint32_t n, uint32_t tileBase, uint32_t kStride, uint32_t shiftStride, uint32_t shiftsPerBlk, uint32_t eighth, uint32_t gCount, uint32_t perBlock, uint32_t terms);
#endif
#ifndef ASCENDC_CPU_DEBUG
void fft8_cube2_unpack_do(uint32_t blockDim, void *stream, void *xr, void *xi, void *scratchY, uint32_t n, uint32_t tileBase, uint32_t kStride, uint32_t shiftStride, uint32_t shiftsPerBlk, uint32_t eighth, uint32_t gCount, uint32_t perBlock, uint32_t terms);
#include "aclrtlaunch_fft8_cube2_matmul.h"
#endif
