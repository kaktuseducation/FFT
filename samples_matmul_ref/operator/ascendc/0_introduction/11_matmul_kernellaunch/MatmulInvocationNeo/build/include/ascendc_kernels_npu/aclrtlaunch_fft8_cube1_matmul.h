#ifndef HEADER_ACLRTLAUNCH_FFT8_CUBE1_MATMUL_H
#define HEADER_ACLRTLAUNCH_FFT8_CUBE1_MATMUL_H
#include "acl/acl_base.h"

#ifndef ACLRT_LAUNCH_KERNEL
#define ACLRT_LAUNCH_KERNEL(kernel_func) aclrtlaunch_##kernel_func
#endif

extern "C" uint32_t aclrtlaunch_fft8_cube1_matmul(uint32_t blockDim, aclrtStream stream, void* W16, void* SCRATCH_X, void* SCRATCH_Y, void* workspace, void* tilingGm);
#endif
