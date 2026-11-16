# Install script for directory: /usr/local/Ascend/ascend-toolkit/latest/compiler/tikcpp/ascendc_kernel_cmake/host_project

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/ma-user/work/OddFFT/build/ascendc_kernels_npu_vec_host_dir")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Debug")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "0")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/./objects-Debug/host_bisheng_obj" TYPE FILE FILES
    "home/ma-user/work/OddFFT/kernels/arb_cmul.cpp.o"
    "home/ma-user/work/OddFFT/kernels/arb_cscale.cpp.o"
    "home/ma-user/work/OddFFT/kernels/r2_direct_dit_perstage_xs.cpp.o"
    "home/ma-user/work/OddFFT/kernels/r3_direct_dit_perstage_xs.cpp.o"
    "home/ma-user/work/OddFFT/kernels/r3_rader_dit_perstage_xs.cpp.o"
    "home/ma-user/work/OddFFT/kernels/r3_sym_dit_perstage_xs.cpp.o"
    "home/ma-user/work/OddFFT/kernels/r3_winograd_dit_perstage_xs.cpp.o"
    "home/ma-user/work/OddFFT/kernels/r4_direct_dit_perstage_xs.cpp.o"
    "home/ma-user/work/OddFFT/kernels/r5_direct_dit_perstage_xs.cpp.o"
    "home/ma-user/work/OddFFT/kernels/r5_rader_dit_perstage_xs.cpp.o"
    "home/ma-user/work/OddFFT/kernels/r5_sym_dit_perstage_xs.cpp.o"
    "home/ma-user/work/OddFFT/kernels/r5_winograd_dit_perstage_xs.cpp.o"
    "home/ma-user/work/OddFFT/kernels/r7_direct_dit_perstage_xs.cpp.o"
    "home/ma-user/work/OddFFT/kernels/r7_rader_dit_perstage_xs.cpp.o"
    "home/ma-user/work/OddFFT/kernels/r7_sym_dit_perstage_xs.cpp.o"
    "home/ma-user/work/OddFFT/kernels/r7_winograd_dit_perstage_xs.cpp.o"
    FILES_FROM_DIR "/home/ma-user/work/OddFFT/build/ascendc_kernels_npu_vec_host-prefix/src/ascendc_kernels_npu_vec_host-build/CMakeFiles/host_bisheng_obj.dir/")
endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/home/ma-user/work/OddFFT/build/ascendc_kernels_npu_vec_host-prefix/src/ascendc_kernels_npu_vec_host-build/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
