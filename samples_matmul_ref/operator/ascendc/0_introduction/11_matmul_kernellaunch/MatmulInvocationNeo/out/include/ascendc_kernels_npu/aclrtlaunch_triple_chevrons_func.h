
#ifndef HEADER_ACLRTLAUNCH_FFT8_CUBE1_MATMUL_HKERNEL_H_
#define HEADER_ACLRTLAUNCH_FFT8_CUBE1_MATMUL_HKERNEL_H_



extern "C" uint32_t aclrtlaunch_fft8_cube1_matmul(uint32_t blockDim, void* stream, void* W16, void* SCRATCH_X, void* SCRATCH_Y, void* workspace, void* tilingGm);

inline uint32_t fft8_cube1_matmul(uint32_t blockDim, void* hold, void* stream, void* W16, void* SCRATCH_X, void* SCRATCH_Y, void* workspace, void* tilingGm)
{
    (void)hold;
    return aclrtlaunch_fft8_cube1_matmul(blockDim, stream, W16, SCRATCH_X, SCRATCH_Y, workspace, tilingGm);
}

#endif
