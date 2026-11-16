// host/cube_tiling.h  (radix8, cube1/cube2 fs_xs)
//
// Modelled on the official Huawei example
// samples/operator/ascendc/0_introduction/11_matmul_kernellaunch/
// matmul_custom_tiling.cpp (ascend/samples) — sent by the user as a
// reference, so confidence in the API here is high. Differences from the
// reference: there MultiCoreMatmulTiling computes the tiling for an honest
// multi-core split (SetDim(GetCoreNumAiv())); our shape (16x16x128) is tiny,
// so the library decides usedCoreNum itself (in practice it comes out as 1) —
// a manual SetDim(1) was a MISTAKE (an edge case the reference never tested),
// replaced by SetDim(GetCoreNumAiv()), as in the official example for
// 910B/910B4.
//
// !!! fp16 FIX !!!
// A/B (W16 and the X operand out of repack) are now DataType::DT_FLOAT16 — an
// isolated test showed that fp32×fp32 through this high-level Matmul API path
// on this architecture either computes zeros or hangs dead (a deadlock on
// tiny shapes). C (the Y result) stays DT_FLOAT — exactly the same type
// pattern as in the official reference that is proven to work
// (matmul_custom.cpp: A_T=half, B_T=half, C_T=float).
#pragma once
#include <cstdint>
#include <vector>

#ifndef ASCENDC_CPU_DEBUG
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"

// Builds the TCubeTiling for ONE fixed "wide" GEMM:
// C[M,N] = A[M,K] * B[K,N] (A — the constant weight matrix M16/MR/MI in fp16,
// B — the repack buffer of a batch of groups in fp16, C — fp32) and
// serializes it into a byte buffer THE SAME WAY the official example does
// (GetDataSize() + SaveToBuffer()), not by a raw memcpy of the struct — that
// is safer if TCubeTiling ever stops being a trivially copyable POD.
inline bool BuildFixedWeightCubeTiling(const char *socVersion, uint32_t M, uint32_t K, uint32_t N,
                                        std::vector<uint8_t> &tilingBuf)
{
    optiling::TCubeTiling tilingData;
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance(socVersion);
    matmul_tiling::MultiCoreMatmulTiling tilingApi(*ascendcPlatform);

    // As in the official matmul_custom_tiling.cpp for 910B/910B4 (the
    // ASCEND310P branch in that same file uses
    // SetSingleShape+SetDim(mBlockNum*nBlockNum), which does not concern us) —
    // the library decides usedCoreNum itself from the real shape.
    tilingApi.SetDim(ascendcPlatform->GetCoreNumAiv());
    tilingApi.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
    tilingApi.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
    tilingApi.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    tilingApi.SetOrgShape(M, N, K);
    tilingApi.SetShape(M, N, K);
    tilingApi.SetBias(false);
    tilingApi.SetBufferSpace(-1, -1, -1);

    if (tilingApi.GetTiling(tilingData) == -1) return false;

    tilingBuf.resize(tilingData.GetDataSize());
    tilingData.SaveToBuffer(tilingBuf.data(), tilingBuf.size());
    return true;
}

// FIX: REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tiling) in
// kernels/r8_cube_fused_xs.cpp requires the system workspace for Matmul to be
// REALLY allocated on the device and handed to the kernel — as in the official
// matmul_custom (its main.cpp: systemWorkspaceSize = GetLibApiWorkSpaceSize(),
// aclrtMalloc, passed into the kernel as GM_ADDR workspace, and the kernel
// does AscendC::SetSysWorkspaceForce(workspace) before REGIST_MATMUL_OBJ).
// Without that, GetSysWorkSpacePtr() inside REGIST_MATMUL_OBJ returns
// garbage/an invalid address — matmul writes outside DDR,
// aclrtSynchronizeStream dies with 507015, and the result is meaningless
// numbers. This helper gives host/impl_r8.cpp the size it needs.
// The number of Vector cores (AIV) on the platform — the blockDim for the
// multicore Vector kernels (dit_fs_xs). On Ascend910B4 that is 40, against 20
// AIC and 191 KB of UB per core. Taken from the same platform_ascendc as the
// tiling, so as not to hardcode a chip characteristic in two places.
inline uint32_t GetAivCoreNum(const char *socVersion)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance(socVersion);
    return static_cast<uint32_t>(ascendcPlatform->GetCoreNumAiv());
}

// The number of Cube cores (AIC) on the platform — it is what bounds how many
// tiles of a stage can be computed in one launch (one tile per block). On
// Ascend910B4 that is 20.
inline uint32_t GetAicCoreNum(const char *socVersion)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance(socVersion);
    return static_cast<uint32_t>(ascendcPlatform->GetCoreNumAic());
}

inline uint64_t GetCubeSystemWorkspaceSize(const char *socVersion)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance(socVersion);
    return static_cast<uint64_t>(ascendcPlatform->GetLibApiWorkSpaceSize());
}
#endif