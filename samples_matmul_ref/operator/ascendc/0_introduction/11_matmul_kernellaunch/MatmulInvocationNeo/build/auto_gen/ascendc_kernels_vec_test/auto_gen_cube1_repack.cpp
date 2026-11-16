#ifndef __CUBE1_REPACK__KERNEL_FUN_H__
#define __CUBE1_REPACK__KERNEL_FUN_H__

#undef __global__
#define __global__ inline
#define fft8_cube1_repack fft8_cube1_repack_origin
#include "/home/ma-user/work/samples_matmul_ref/operator/ascendc/0_introduction/11_matmul_kernellaunch/MatmulInvocationNeo/cube1_repack.cpp"

#undef fft8_cube1_repack
#undef __global__
#if ASCENDC_CPU_DEBUG
#define __global__
#else
#define __global__ __attribute__((cce_kernel))
#endif

#ifndef ONE_CORE_DUMP_SIZE
#define ONE_CORE_DUMP_SIZE 1048576 * 1
#endif

extern "C" __global__ [aicore] void auto_gen_fft8_cube1_repack_kernel(
__attribute__((cce_global)) uint8_t* XR, __attribute__((cce_global)) uint8_t* XI, __attribute__((cce_global)) uint8_t* W, __attribute__((cce_global)) uint8_t* SCRATCH_X, uint32_t n, uint32_t k0, uint32_t shift, uint32_t eighth, uint32_t validB, uint32_t wOffset, uint32_t aligned, uint32_t wTotal) {
#if defined(HAVE_WORKSPACE)
    GM_ADDR workspace_param;
    GM_ADDR workspace_usr;
#if defined(HAVE_TILING)
    workspace_param = aligned;
#else
    workspace_param = wTotal;
#endif
    AscendC::SetSysWorkspaceForce(workspace_param);
    workspace_usr = AscendC::GetUserWorkspace(workspace_param);
#if defined(HAVE_TILING)
    aligned = workspace_usr;
#else
    wTotal = workspace_usr;
#endif
#endif
    fft8_cube1_repack_origin(XR, XI, W, SCRATCH_X, n, k0, shift, eighth, validB, wOffset, aligned, wTotal);
}

#endif
#include "inner_interface/inner_kernel_operator_intf.h"
