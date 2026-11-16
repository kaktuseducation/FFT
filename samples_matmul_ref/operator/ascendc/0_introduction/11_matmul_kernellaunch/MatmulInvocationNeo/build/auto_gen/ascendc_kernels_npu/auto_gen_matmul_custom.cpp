#ifndef __MATMUL_CUSTOM__KERNEL_FUN_H__
#define __MATMUL_CUSTOM__KERNEL_FUN_H__

#undef __global__
#define __global__ inline
#define fft8_cube1_matmul fft8_cube1_matmul_origin
#include "/home/ma-user/work/samples_matmul_ref/operator/ascendc/0_introduction/11_matmul_kernellaunch/MatmulInvocationNeo/matmul_custom.cpp"

#undef fft8_cube1_matmul
#undef __global__
#if ASCENDC_CPU_DEBUG
#define __global__
#else
#define __global__ __attribute__((cce_kernel))
#endif

#ifndef ONE_CORE_DUMP_SIZE
#define ONE_CORE_DUMP_SIZE 1048576 * 1
#endif

extern "C" __global__ [aicore] void auto_gen_fft8_cube1_matmul_kernel(
GM_ADDR ffts_addr, __attribute__((cce_global)) uint8_t* W16, __attribute__((cce_global)) uint8_t* SCRATCH_X, __attribute__((cce_global)) uint8_t* SCRATCH_Y, __attribute__((cce_global)) uint8_t* workspace, __attribute__((cce_global)) uint8_t* tilingGm) {
    icache_preload(1);
    set_ffts_base_addr((uint64_t)ffts_addr);

#if defined(HAVE_WORKSPACE)
    GM_ADDR workspace_param;
    GM_ADDR workspace_usr;
#if defined(HAVE_TILING)
    workspace_param = workspace;
#else
    workspace_param = tilingGm;
#endif
    if (workspace_param == nullptr) {
        return;
    }
    AscendC::SetSysWorkspaceForce(workspace_param);
    workspace_usr = AscendC::GetUserWorkspace(workspace_param);
#if defined(REGIST_MATMUL_OBJ)
    if constexpr (g_coreType == AscendC::AIC) {
        matmul::clearWorkspace(workspace_param);
    }
#endif
#if defined(HAVE_TILING)
    workspace = workspace_usr;
#else
    tilingGm = workspace_usr;
#endif
#endif
    fft8_cube1_matmul_origin(W16, SCRATCH_X, SCRATCH_Y, workspace, tilingGm);
}

#endif
#include "inner_interface/inner_kernel_operator_intf.h"
