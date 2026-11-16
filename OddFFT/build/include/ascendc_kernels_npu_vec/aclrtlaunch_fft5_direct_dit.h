#ifndef HEADER_ACLRTLAUNCH_FFT5_DIRECT_DIT_H
#define HEADER_ACLRTLAUNCH_FFT5_DIRECT_DIT_H
#include "acl/acl_base.h"

#ifndef ACLRT_LAUNCH_KERNEL
#define ACLRT_LAUNCH_KERNEL(kernel_func) aclrtlaunch_##kernel_func
#endif

extern "C" uint32_t aclrtlaunch_fft5_direct_dit(uint32_t blockDim, aclrtStream stream, void* XR, void* XI, void* W, uint32_t n, uint32_t t, uint32_t block, uint32_t wOffset, uint32_t rowLen, uint32_t tile, uint32_t blk_num);
#endif
