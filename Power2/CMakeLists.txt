cmake_minimum_required(VERSION 3.16)
project(FFTBench)

# ===========================================================================
# FFTBench — one project across the FFT kernels for the Ascend NPU.
# All the power-of-two radices (2, 4, 8), all the algorithms (DIT, Stockham),
# both granularities (fused / per-stage), both layouts (xs / xi) and both
# compute units (Vector/AIV, Cube/AIC) live in one binary and write one
# table output/matrix.csv. The list: ./build/ascendc_kernels_bbit --list
# ===========================================================================

set(RUN_MODE "npu" CACHE STRING "cpu/sim/npu")
set(SOC_VERSION "Ascend910B4" CACHE STRING "system on chip type")
set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/ascend-toolkit/latest"
    CACHE STRING "ASCEND CANN package installation directory"
)
# The batch width of a Cube call (the N dimension of the 16 x 16 x CUBE_BATCH
# matmul). A compile-time constant: it is passed by one define into the device
# kernels and into the host code alike, where kernels/cube1_common.h takes it
# in. Sweeping: run.sh -B <N> or run_matrix.sh -B. It affects the cube variants
# only.
set(CUBE_BATCH "128" CACHE STRING "Cube batch width (N of the 16x16xN matmul); power of two, >= 16")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type Release/Debug (default Debug)" FORCE)
endif()
if(CMAKE_INSTALL_PREFIX STREQUAL /usr/local)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_LIST_DIR}/out" CACHE STRING "path for install()" FORCE)
endif()

include(GNUInstallDirs)

# All the kernel files are picked up automatically: a new .cpp in kernels/ does
# not need an edit to CMakeLists.txt. The file names carry a radix prefix
# (r2_/r4_/r8_), because in the original projects they coincided
# (dit_fused_xs.cpp belonged to radix-4 and to radix-8 both).
file(GLOB_RECURSE KERNEL_FILES ${CMAKE_CURRENT_SOURCE_DIR}/kernels/*.cpp)
list(FILTER KERNEL_FILES EXCLUDE REGEX "\\.ipynb_checkpoints/")

# The matmul kernels have to be built into an ascendc_library() target of THEIR
# OWN, apart from the ordinary Vector kernels: when the two are mixed in one
# target, host_bisheng_obj falls over on MatmulChipCap::GetFeatures(). The
# details are in common/cmake/npu_lib.cmake.
# Listed here are exactly those files that have an #include "lib/matmul_intf.h":
# r8_cube_fused_xs.cpp (the 16x16 matmul of the cube1 variant) and
# r8_cube2_matmul.cpp (the two matmuls of the cube2 variant). Their Vector
# neighbours (r8_cube1_repack.cpp, r8_cube1_unpack.cpp, r8_cube2_unpack.cpp)
# stay in the vec target — that is precisely why they were moved out into
# separate files.
set(KERNEL_FILES_CUBE
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/r8_cube_fused_xs.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/r8_cube2_matmul.cpp)
set(KERNEL_FILES_VEC ${KERNEL_FILES})
list(REMOVE_ITEM KERNEL_FILES_VEC ${KERNEL_FILES_CUBE})

if("${RUN_MODE}" STREQUAL "cpu")
    include(common/cmake/cpu_lib.cmake)
elseif("${RUN_MODE}" STREQUAL "sim" OR "${RUN_MODE}" STREQUAL "npu")
    include(common/cmake/npu_lib.cmake)
else()
    message(FATAL_ERROR "invalid RUN_MODE: ${RUN_MODE}")
endif()

# The host part: main.cpp (the CLI, the registry, verification, the table) plus
# one file per radix. The split into files is not cosmetic: every impl_*.cpp has
# a `using namespace rN` of its own, while the three radices share the names
# FusedStageInfo and some of the helpers — in one translation unit they would
# conflict.
add_executable(ascendc_kernels_bbit
    ${CMAKE_CURRENT_SOURCE_DIR}/host/main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/host/impl_r2.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/host/impl_r4.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/host/impl_r8.cpp
)

target_include_directories(ascendc_kernels_bbit PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/common
    ${CMAKE_CURRENT_SOURCE_DIR}/host
    # the host code includes kernels/cube1_common.h — the one device header
    # that is host-safe (only <cstdint> and constexpr) and that holds
    # CUBE_BATCH, shared by both sides.
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels
)

target_compile_options(ascendc_kernels_bbit PRIVATE
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},cpu>:-g>>
    -O2 -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 -Wall -Werror
)

# SOC_VERSION as a string on the host side is needed to build the TCubeTiling
# through platform_ascendc::PlatformAscendCManager::GetInstance(socVersion)
# in host/cube_tiling.h (the radix-8 cube variants).
target_compile_definitions(ascendc_kernels_bbit PRIVATE
    SOC_VERSION_STR="${SOC_VERSION}"
    CUBE_BATCH_CFG=${CUBE_BATCH}
)

target_link_libraries(ascendc_kernels_bbit PRIVATE
    $<BUILD_INTERFACE:$<$<OR:$<STREQUAL:${RUN_MODE},npu>,$<STREQUAL:${RUN_MODE},sim>>:host_intf_pub>>
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},cpu>:ascendcl>>
    $<IF:$<STREQUAL:${RUN_MODE},cpu>,ascendc_kernels_cpu,ascendc_kernels_${RUN_MODE}_vec>
    $<BUILD_INTERFACE:$<$<OR:$<STREQUAL:${RUN_MODE},npu>,$<STREQUAL:${RUN_MODE},sim>>:ascendc_kernels_${RUN_MODE}_cube>>
    # Needed on the host side by host/cube_tiling.h
    # (platform_ascendc::PlatformAscendCManager,
    # matmul_tiling::MultiCoreMatmulTiling) — the set is taken from the
    # official 11_matmul_kernellaunch example.
    $<BUILD_INTERFACE:$<$<OR:$<STREQUAL:${RUN_MODE},npu>,$<STREQUAL:${RUN_MODE},sim>>:tiling_api>>
    $<BUILD_INTERFACE:$<$<OR:$<STREQUAL:${RUN_MODE},npu>,$<STREQUAL:${RUN_MODE},sim>>:register>>
    $<BUILD_INTERFACE:$<$<OR:$<STREQUAL:${RUN_MODE},npu>,$<STREQUAL:${RUN_MODE},sim>>:platform>>
    $<BUILD_INTERFACE:$<$<OR:$<STREQUAL:${RUN_MODE},npu>,$<STREQUAL:${RUN_MODE},sim>>:ascendalog>>
    $<BUILD_INTERFACE:$<$<OR:$<STREQUAL:${RUN_MODE},npu>,$<STREQUAL:${RUN_MODE},sim>>:dl>>
)

install(TARGETS ascendc_kernels_bbit
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
