// host/platform_cores.h  (OddFFT)
//
// The core counts of the platform, read from the toolkit rather than
// hardcoded.
//
// The power-of-two project got these out of host/cube_tiling.h, which also
// pulled in tiling/tiling_api.h for the matmul TCubeTiling. There is no matmul
// here yet, and pulling the matmul tiling API in just to count cores would
// drag the whole Cube dependency (and the -DHAVE_WORKSPACE/-DHAVE_TILING
// scoping problem that came with it) into a project that does not need it.
// So: just the core counts.
#pragma once
#include <cstdint>

#ifndef ASCENDC_CPU_DEBUG
#include "tiling/platform/platform_ascendc.h"

// Vector cores (AIV) — blockDim for the multicore vector kernels.
// On Ascend910B4: 40, against 20 AIC, with 191 KB of UB per core.
inline uint32_t GetAivCoreNum(const char *socVersion)
{
    auto plat = platform_ascendc::PlatformAscendCManager::GetInstance(socVersion);
    return static_cast<uint32_t>(plat->GetCoreNumAiv());
}

// Cube cores (AIC) — for the cube butterfly variants, when they arrive.
inline uint32_t GetAicCoreNum(const char *socVersion)
{
    auto plat = platform_ascendc::PlatformAscendCManager::GetInstance(socVersion);
    return static_cast<uint32_t>(plat->GetCoreNumAic());
}
#endif
