#ifndef HEADER_ACLRTLAUNCH_ARB_CSCALE_H
#define HEADER_ACLRTLAUNCH_ARB_CSCALE_H
#include "acl/acl_base.h"

#ifndef ACLRT_LAUNCH_KERNEL
#define ACLRT_LAUNCH_KERNEL(kernel_func) aclrtlaunch_##kernel_func
#endif

extern "C" uint32_t aclrtlaunch_arb_cscale(uint32_t blockDim, aclrtStream stream, void* AR, void* AI, uint32_t n, float s, float sign, uint32_t tile, uint32_t blk_num);
#endif
