#ifndef __ARB_CSCALE__KERNEL_FUN_H__
#define __ARB_CSCALE__KERNEL_FUN_H__

#undef __global__
#define __global__ inline
#define arb_cscale arb_cscale_origin
#include "/home/ma-user/work/OddFFT/kernels/arb_cscale.cpp"

#undef arb_cscale
#undef __global__
#if ASCENDC_CPU_DEBUG
#define __global__
#else
#define __global__ __attribute__((cce_kernel))
#endif

#ifndef ONE_CORE_DUMP_SIZE
#define ONE_CORE_DUMP_SIZE 1048576 * 1
#endif

extern "C" __global__ [aicore] void auto_gen_arb_cscale_kernel(
__attribute__((cce_global)) uint8_t* AR, __attribute__((cce_global)) uint8_t* AI, uint32_t n, float s, float sign, uint32_t tile, uint32_t blk_num) {
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
    arb_cscale_origin(AR, AI, n, s, sign, tile, blk_num);
}

#endif
#include "inner_interface/inner_kernel_operator_intf.h"
