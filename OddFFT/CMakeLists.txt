cmake_minimum_required(VERSION 3.16)
project(OddFFT)

# ===========================================================================
# OddFFT — FFT kernels for the ODD radices 3, 5 and 7 on the Ascend NPU.
#
# Every radix, every butterfly algorithm (direct / sym / winograd / rader),
# every scheme (DIT / Stockham), both granularities and layouts live in one
# binary and write one table, output/matrix.csv.
# The list: ./build/oddfft --list
#
# NOT INHERITED FROM THE POWER-OF-TWO PROJECT: the second ascendc_library()
# target for matmul kernels, and the -DHAVE_WORKSPACE/-DHAVE_TILING define
# scoping that came with it. There is no matmul here yet. When a `cube`
# butterfly variant is added it will need that whole arrangement back, and the
# reasons are recorded in common/cmake/npu_lib.cmake rather than being
# rediscovered.
# ===========================================================================

set(RUN_MODE "npu" CACHE STRING "cpu/sim/npu")
set(SOC_VERSION "Ascend910B4" CACHE STRING "system on chip type")
set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/ascend-toolkit/latest"
    CACHE STRING "ASCEND CANN package installation directory"
)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type Release/Debug (default Debug)" FORCE)
endif()
if(CMAKE_INSTALL_PREFIX STREQUAL /usr/local)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_LIST_DIR}/out" CACHE STRING "path for install()" FORCE)
endif()

include(GNUInstallDirs)

# Kernels are picked up automatically: a new .cpp in kernels/ needs no edit
# here. File names carry a radix prefix (r3_/r5_/r7_) because the same
# butterfly name will exist for all three.
file(GLOB_RECURSE KERNEL_FILES ${CMAKE_CURRENT_SOURCE_DIR}/kernels/*.cpp)
list(FILTER KERNEL_FILES EXCLUDE REGEX "\\.ipynb_checkpoints/")

# No cube kernels yet, so every kernel is a vector kernel. The split is kept as
# a variable so adding the first matmul kernel is a one-line change here plus
# re-enabling the cube target in npu_lib.cmake.
set(KERNEL_FILES_CUBE "")
set(KERNEL_FILES_VEC ${KERNEL_FILES})

if("${RUN_MODE}" STREQUAL "cpu")
    include(common/cmake/cpu_lib.cmake)
elseif("${RUN_MODE}" STREQUAL "sim" OR "${RUN_MODE}" STREQUAL "npu")
    include(common/cmake/npu_lib.cmake)
else()
    message(FATAL_ERROR "invalid RUN_MODE: ${RUN_MODE}")
endif()

# The host part: main.cpp (CLI, registry, verification, the table) and
# impl_odd.cpp. There is no file per radix: the kernels are generated from one
# template with one launch signature, so the host registers all of them from
# kernels/kernel_table.h in a single loop.
add_executable(oddfft
    ${CMAKE_CURRENT_SOURCE_DIR}/host/main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/host/impl_odd.cpp
    # Bluestein and Rader. A separate translation unit because its
    # run-functions are static and only it can take their address -- the same
    # reason impl_odd.cpp exists -- and because the inner transform it drives
    # is a power-of-two plan, which has nothing to do with the odd radices.
    ${CMAKE_CURRENT_SOURCE_DIR}/host/impl_arblen.cpp
)

target_include_directories(oddfft PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/common
    ${CMAKE_CURRENT_SOURCE_DIR}/host
    # the host includes kernels/kernel_table.h and kernels/kernel_limits.h,
    # both generated and both host-safe (cstdint and macros only)
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels
)

# OpenMP, for the HOST target only. The kernel library is compiled by bisheng
# for the device and must not see it.
#
# WHY: the sub-floor prefix in host/host_prefix_fast.h was single-threaded on a
# 192-thread Kunpeng-920. At N = 100800 those stages are 8370 us against 401.67
# us for the six device stages after them -- 94% of the transform, on one core.
# The loop is parallel without any synchronisation and the result is
# bit-identical, which tests/host_selftest.cpp checks.
#
# find_package, not a bare -fopenmp: the flag and the runtime library differ
# between compilers, and a missing -lgomp shows up as an undefined reference at
# link time rather than as a clear message. If OpenMP is not found the build
# still works, serially, and says so.
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found: the host prefix will run multi-threaded "
                   "(set OMP_NUM_THREADS to tune; this pass is memory-bound, "
                   "so the useful count is well below 192)")
else()
    message(WARNING "OpenMP NOT found: the host prefix stays single-threaded, "
                    "which the matrix measured as 94% of the runtime at N=100800")
endif()

target_compile_options(oddfft PRIVATE
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},cpu>:-g>>
    -O2 -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 -Wall -Werror
)
if(OpenMP_CXX_FOUND)
    target_link_libraries(oddfft PRIVATE OpenMP::OpenMP_CXX)
endif()

# SOC_VERSION as a host-side string is what host/platform_cores.h needs to ask
# the toolkit for the AIV/AIC core counts instead of hardcoding 40 and 20.
target_compile_definitions(oddfft PRIVATE
    SOC_VERSION_STR="${SOC_VERSION}"
)

target_link_libraries(oddfft PRIVATE
    $<BUILD_INTERFACE:$<$<OR:$<STREQUAL:${RUN_MODE},npu>,$<STREQUAL:${RUN_MODE},sim>>:host_intf_pub>>
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},cpu>:ascendcl>>
    $<IF:$<STREQUAL:${RUN_MODE},cpu>,ascendc_kernels_cpu,ascendc_kernels_${RUN_MODE}_vec>
    # platform_ascendc::PlatformAscendCManager (host/platform_cores.h) lives in
    # these; the set is the one the official matmul example uses, minus nothing
    # — dropping any of them gives an undefined-reference at link time rather
    # than a clear message.
    $<BUILD_INTERFACE:$<$<OR:$<STREQUAL:${RUN_MODE},npu>,$<STREQUAL:${RUN_MODE},sim>>:tiling_api>>
    $<BUILD_INTERFACE:$<$<OR:$<STREQUAL:${RUN_MODE},npu>,$<STREQUAL:${RUN_MODE},sim>>:register>>
    $<BUILD_INTERFACE:$<$<OR:$<STREQUAL:${RUN_MODE},npu>,$<STREQUAL:${RUN_MODE},sim>>:platform>>
    $<BUILD_INTERFACE:$<$<OR:$<STREQUAL:${RUN_MODE},npu>,$<STREQUAL:${RUN_MODE},sim>>:ascendalog>>
    $<BUILD_INTERFACE:$<$<OR:$<STREQUAL:${RUN_MODE},npu>,$<STREQUAL:${RUN_MODE},sim>>:dl>>
)

install(TARGETS oddfft
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
