// kernels/cube1_common.h  (radix8, cube1_fs_xs)
//
// The constant shared between kernels/r8_cube1_repack.cpp +
// kernels/r8_cube1_unpack.cpp (pure Vector; this used to be a single file
// cube1_repack_unpack.cpp, see below) and kernels/r8_cube_fused_xs.cpp
// (Matmul/Cube) — pulled out into a header of its own so as not to duplicate
// the literal and not to make the files #include each other: they are split
// DELIBERATELY (see CMakeLists.txt — KERNEL_FILES_CUBE is compiled through a
// separate ascendc_library() pipeline, otherwise host_bisheng_obj dies in
// MatmulChipCap::GetFeatures(); exactly the same reason repack/unpack had to
// be moved out of r8_cube_fused_xs.cpp — while physically in one file with
// the Matmul kernel they were built through that same "cube" pipeline and
// caught an aicore exception out of nowhere).
#pragma once
#include <cstdint>

// Batch size in groups per Cube call (the N dimension of the 16 x 16 x B
// matmul).
//
// !!! THIS IS THE ONLY PLACE WHERE THE CONSTANT LIVES !!!
// host/main.cpp INCLUDES this header (it is host-safe: only <cstdint> and
// constexpr) so that the device and host sides physically cannot drift apart.
// Formerly main.cpp had a CUBE_BATCH_HOST = 128 of its own plus one more
// literal 128 in the BuildFixedWeightCubeTiling call — three independent
// copies of one number.
//
// The value is set from CMake (-DCUBE_BATCH=...; run.sh -B <N>, CAPITAL — in
// run.sh the lowercase -b is the number of timing runs) and arrives
// here through the CUBE_BATCH_CFG macro, so a sweep over batch width needs no
// source edits. Why this matters: B determines how many tiles a stage is cut
// into, that is, how many AIC cores are actually busy. A stage has N/8
// butterflies and N/(8*B) tiles — at N=4096 and B=128 that is all of 4 tiles
// out of 20 cores. A smaller B gives more tiles (more cores, but a smaller
// matmul on each), a larger one the opposite. Where the optimum lies is an
// empirical question, hence the sweep.
#ifndef CUBE_BATCH_CFG
#define CUBE_BATCH_CFG 128
#endif

// Constraints on B (checked statically below):
//  * B >= 16   — the X operand is written to GM in half, and DataCopy
//                granularity is 32 bytes = 16 half (see AlignUp16 in
//                kernels/r8_cube1_repack.cpp);
//  * B = 2^k   — eighth is always a power of eight, hence a power of two;
//                the divisions eighth/B and B/eighth must be exact in both
//                layout modes (see launchCubeGroup in host/impl_r8.cpp).
static constexpr uint32_t CUBE_BATCH = CUBE_BATCH_CFG;

static_assert(CUBE_BATCH >= 16, "CUBE_BATCH < 16: a half write to GM shorter than 32 bytes — DataCopy will silently copy nothing");
static_assert((CUBE_BATCH & (CUBE_BATCH - 1)) == 0, "CUBE_BATCH must be a power of two");
