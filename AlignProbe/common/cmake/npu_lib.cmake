# Minimal vec-only AscendC library build, lifted from FFTBench and stripped of
# the cube target: this probe has no matmul kernels, so it needs neither the
# separate ascendc_library() nor -DHAVE_WORKSPACE/-DHAVE_TILING.
if(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
else()
    message(FATAL_ERROR "ascendc_kernel_cmake does not exist, check the CANN installation")
endif()
include(${ASCENDC_CMAKE_DIR}/ascendc.cmake)

ascendc_library(ascendc_probe_vec SHARED ${KERNEL_FILES})
if(COMMAND ascendc_compile_definitions)
    ascendc_compile_definitions(ascendc_probe_vec PRIVATE -DHAVE_DATACOPYPAD=${HAVE_DATACOPYPAD})
else()
    set_property(TARGET ascendc_probe_vec_interface APPEND PROPERTY
                 DEFINITIONS -DHAVE_DATACOPYPAD=${HAVE_DATACOPYPAD})
endif()
