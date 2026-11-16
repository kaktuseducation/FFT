cmake_minimum_required(VERSION 3.16)
project(AlignProbe)

set(RUN_MODE "npu" CACHE STRING "npu")
set(SOC_VERSION "Ascend910B4" CACHE STRING "system on chip type")
set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/ascend-toolkit/latest"
    CACHE STRING "ASCEND CANN package installation directory")
# Set to 0 if DataCopyPad / DataCopyExtParams do not exist in this toolkit:
# the other probe modes still build and run, which is the point of the switch.
set(HAVE_DATACOPYPAD "1" CACHE STRING "build the DataCopyPad probe mode")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "build type" FORCE)
endif()
include(GNUInstallDirs)

file(GLOB KERNEL_FILES ${CMAKE_CURRENT_SOURCE_DIR}/kernels/*.cpp)
include(common/cmake/npu_lib.cmake)

add_executable(align_probe ${CMAKE_CURRENT_SOURCE_DIR}/host/main.cpp)
target_include_directories(align_probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/host)
target_compile_options(align_probe PRIVATE -O2 -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 -Wall -Werror)
target_compile_definitions(align_probe PRIVATE HAVE_DATACOPYPAD=${HAVE_DATACOPYPAD})
target_link_libraries(align_probe PRIVATE
    host_intf_pub ascendc_probe_vec ascendalog dl)
