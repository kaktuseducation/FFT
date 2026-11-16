#ifndef __R2_DIRECT_DIT_PERSTAGE_XS__KERNEL_FUN_H__
#define __R2_DIRECT_DIT_PERSTAGE_XS__KERNEL_FUN_H__

#undef __global__
#define __global__ inline
#define fft2_direct_dit fft2_direct_dit_origin
#include "/home/ma-user/work/OddFFT/kernels/r2_direct_dit_perstage_xs.cpp"

#undef fft2_direct_dit
#undef __global__
#if ASCENDC_CPU_DEBUG
#define __global__
#else
#define __global__ __attribute__((cce_kernel))
#endif

#ifndef ONE_CORE_DUMP_SIZE
#define ONE_CORE_DUMP_SIZE 1048576 * 1
#endif

extern "C" __global__ [aicore] void auto_gen_fft2_direct_dit_kernel(
__attribute__((cce_global)) uint8_t* XR, __attribute__((cce_global)) uint8_t* XI, __attribute__((cce_global)) uint8_t* W, uint32_t n, uint32_t t, uint32_t block, uint32_t wOffset, uint32_t rowLen, uint32_t tile, uint32_t blk_num) {
#if defined(HAVE_WORKSPACE)
    GM_ADDR workspace_param;
    GM_ADDR workspace_usr;
#if defined(HAVE_TILING)
    workspace_param = tile;
#else
    workspace_param = blk_num;
#endif
    AscendC::SetSysWorkspaceForce(workspace_param);
    workspace_usr = AscendC::GetUserWorkspace(workspace_param);
#if defined(HAVE_TILING)
    tile = workspace_usr;
#else
    blk_num = workspace_usr;
#endif
#endif
    fft2_direct_dit_origin(XR, XI, W, n, t, block, wOffset, rowLen, tile, blk_num);
}

#endif
#include "inner_interface/inner_kernel_operator_intf.h"
