#ifndef HEADER_ACLRTLAUNCH_FFT8_CUBE1_REPACK_H
#define HEADER_ACLRTLAUNCH_FFT8_CUBE1_REPACK_H
#include "acl/acl_base.h"

#ifndef ACLRT_LAUNCH_KERNEL
#define ACLRT_LAUNCH_KERNEL(kernel_func) aclrtlaunch_##kernel_func
#endif

extern "C" uint32_t aclrtlaunch_fft8_cube1_repack(uint32_t blockDim, aclrtStream stream, void* XR, void* XI, void* W, void* SCRATCH_X, uint32_t n, uint32_t k0, uint32_t shift, uint32_t eighth, uint32_t validB, uint32_t wOffset, uint32_t aligned, uint32_t wTotal);
#endif
