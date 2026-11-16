#ifndef __ARB_CMUL__KERNEL_FUN_H__
#define __ARB_CMUL__KERNEL_FUN_H__

#undef __global__
#define __global__ inline
#define arb_cmul arb_cmul_origin
#include "/home/ma-user/work/OddFFT/kernels/arb_cmul.cpp"

#undef arb_cmul
#undef __global__
#if ASCENDC_CPU_DEBUG
#define __global__
#else
#define __global__ __attribute__((cce_kernel))
#endif

#ifndef ONE_CORE_DUMP_SIZE
#define ONE_CORE_DUMP_SIZE 1048576 * 1
#endif

extern "C" __global__ [aicore] void auto_gen_arb_cmul_kernel(
__attribute__((cce_global)) uint8_t* AR, __attribute__((cce_global)) uint8_t* AI, __attribute__((cce_global)) uint8_t* BR, __attribute__((cce_global)) uint8_t* BI, uint32_t n, uint32_t tile, uint32_t blk_num) {
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
    arb_cmul_origin(AR, AI, BR, BI, n, tile, blk_num);
}

#endif
#include "inner_interface/inner_kernel_operator_intf.h"
