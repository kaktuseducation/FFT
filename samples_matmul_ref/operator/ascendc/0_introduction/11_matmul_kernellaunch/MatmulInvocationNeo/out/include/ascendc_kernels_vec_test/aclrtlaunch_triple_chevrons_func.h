
#ifndef HEADER_ACLRTLAUNCH_FFT8_CUBE1_REPACK_HKERNEL_H_
#define HEADER_ACLRTLAUNCH_FFT8_CUBE1_REPACK_HKERNEL_H_



extern "C" uint32_t aclrtlaunch_fft8_cube1_repack(uint32_t blockDim, void* stream, void* XR, void* XI, void* W, void* SCRATCH_X, uint32_t n, uint32_t k0, uint32_t shift, uint32_t eighth, uint32_t validB, uint32_t wOffset, uint32_t aligned, uint32_t wTotal);

inline uint32_t fft8_cube1_repack(uint32_t blockDim, void* hold, void* stream, void* XR, void* XI, void* W, void* SCRATCH_X, uint32_t n, uint32_t k0, uint32_t shift, uint32_t eighth, uint32_t validB, uint32_t wOffset, uint32_t aligned, uint32_t wTotal)
{
    (void)hold;
    return aclrtlaunch_fft8_cube1_repack(blockDim, stream, XR, XI, W, SCRATCH_X, n, k0, shift, eighth, validB, wOffset, aligned, wTotal);
}

#endif
