/**
 * @file matmul_custom.cpp — ВРЕМЕННО заменён на байт-в-байт копию нашего
 * kernels/cube_fused_xs.cpp::fft8_cube1_matmul для изолирующего теста.
 */
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include <cstdint>

static constexpr uint32_t CUBE_BATCH = 128;

using MMAType = matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half>;
using MMBType = matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half>;
using MMCType = matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>;

extern "C" __global__ __aicore__ void fft8_cube1_matmul(
    GM_ADDR W16, GM_ADDR SCRATCH_X, GM_ADDR SCRATCH_Y, GM_ADDR workspace, GM_ADDR tilingGm)
{
    AscendC::GlobalTensor<half>  w16Gm, scratchXGm;
    AscendC::GlobalTensor<float> scratchYGm;
    w16Gm.SetGlobalBuffer((__gm__ half *)W16, 16 * 16);
    scratchXGm.SetGlobalBuffer((__gm__ half *)SCRATCH_X, 16 * CUBE_BATCH);
    scratchYGm.SetGlobalBuffer((__gm__ float *)SCRATCH_Y, 16 * CUBE_BATCH);

    TCubeTiling tiling;
    auto tilingSrc = (__gm__ uint32_t *)tilingGm;
    auto tilingDst = (uint32_t *)&tiling;
    for (uint32_t idx = 0; idx < sizeof(TCubeTiling) / sizeof(uint32_t); ++idx) {
        tilingDst[idx] = tilingSrc[idx];
    }

    AscendC::TPipe pipe;
    AscendC::SetSysWorkspaceForce(workspace);

    matmul::Matmul<MMAType, MMBType, MMCType> mm;
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tiling);

    if (AscendC::GetBlockIdx() >= tiling.usedCoreNum) {
        return;
    }
    mm.SetOrgShape(tiling.M, tiling.N, tiling.Ka, tiling.Kb);
    mm.SetTensorA(w16Gm);
    mm.SetTensorB(scratchXGm);
    mm.SetTail(tiling.M, tiling.N);
    mm.IterateAll(scratchYGm);
    mm.End();
}
