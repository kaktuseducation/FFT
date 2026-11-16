if(NOT DEFINED ENV{CMAKE_PREFIX_PATH})
    set(CMAKE_PREFIX_PATH ${ASCEND_CANN_PACKAGE_PATH}/tools/tikicpulib/lib/cmake)
endif()
find_package(tikicpulib REQUIRED)

add_library(ascendc_kernels_${RUN_MODE} SHARED ${KERNEL_FILES})
target_link_libraries(ascendc_kernels_${RUN_MODE} PUBLIC tikicpulib::${SOC_VERSION})
target_compile_options(ascendc_kernels_${RUN_MODE} PRIVATE -g -O0 -std=c++17)
# The same Cube batch width as the host code and the NPU build have —
# otherwise CPU-debug would lay out batches of a different size than main.cpp
# expects (see CUBE_BATCH in CMakeLists.txt and kernels/cube1_common.h).
target_compile_definitions(ascendc_kernels_${RUN_MODE} PRIVATE CUBE_BATCH_CFG=${CUBE_BATCH})
install(TARGETS ascendc_kernels_${RUN_MODE} DESTINATION ${CMAKE_INSTALL_LIBDIR})
