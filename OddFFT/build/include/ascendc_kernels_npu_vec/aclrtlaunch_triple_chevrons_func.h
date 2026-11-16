
#ifndef HEADER_ACLRTLAUNCH_ARB_CMUL_HKERNEL_H_
#define HEADER_ACLRTLAUNCH_ARB_CMUL_HKERNEL_H_



extern "C" uint32_t aclrtlaunch_arb_cmul(uint32_t blockDim, void* stream, void* AR, void* AI, void* BR, void* BI, uint32_t n, uint32_t tile, uint32_t blk_num);

inline uint32_t arb_cmul(uint32_t blockDim, void* hold, void* stream, void* AR, void* AI, void* BR, void* BI, uint32_t n, uint32_t tile, uint32_t blk_num)
{
    (void)hold;
    return aclrtlaunch_arb_cmul(blockDim, stream, AR, AI, BR, BI, n, tile, blk_num);
}

#endif

#ifndef HEADER_ACLRTLAUNCH_ARB_CSCALE_HKERNEL_H_
#define HEADER_ACLRTLAUNCH_ARB_CSCALE_HKERNEL_H_



extern "C" uint32_t aclrtlaunch_arb_cscale(uint32_t blockDim, void* stream, void* AR, void* AI, uint32_t n, float s, float sign, uint32_t tile, uint32_t blk_num);

inline uint32_t arb_cscale(uint32_t blockDim, void* hold, void* stream, void* AR, void* AI, uint32_t n, float s, float sign, uint32_t tile, uint32_t blk_num)
{
    (void)hold;
    return aclrtlaunch_arb_cscale(blockDim, stream, AR, AI, n, s, sign, tile, blk_num);
}

#endif

#ifndef HEADER_ACLRTLAUNCH_FFT2_DIRECT_DIT_HKERNEL_H_
#define HEADER_ACLRTLAUNCH_FFT2_DIRECT_DIT_HKERNEL_H_



extern "C" uint32_t aclrtlaunch_fft2_direct_dit(uint32_t blockDim, void* stream, void* XR, void* XI, void* W, uint32_t n, uint32_t t, uint32_t block, uint32_t wOffset, uint32_t rowLen, uint32_t tile, uint32_t blk_num);

inline uint32_t fft2_direct_dit(uint32_t blockDim, void* hold, void* stream, void* XR, void* XI, void* W, uint32_t n, uint32_t t, uint32_t block, uint32_t wOffset, uint32_t rowLen, uint32_t tile, uint32_t blk_num)
{
    (void)hold;
    return aclrtlaunch_fft2_direct_dit(blockDim, stream, XR, XI, W, n, t, block, wOffset, rowLen, tile, blk_num);
}

#endif

#ifndef HEADER_ACLRTLAUNCH_FFT3_DIRECT_DIT_HKERNEL_H_
#define HEADER_ACLRTLAUNCH_FFT3_DIRECT_DIT_HKERNEL_H_



extern "C" uint32_t aclrtlaunch_fft3_direct_dit(uint32_t blockDim, void* stream, void* XR, void* XI, void* W, uint32_t n, uint32_t t, uint32_t block, uint32_t wOffset, uint32_t rowLen, uint32_t tile, uint32_t blk_num);

inline uint32_t fft3_direct_dit(uint32_t blockDim, void* hold, void* stream, void* XR, void* XI, void* W, uint32_t n, uint32_t t, uint32_t block, uint32_t wOffset, uint32_t rowLen, uint32_t tile, uint32_t blk_num)
{
    (void)hold;
    return aclrtlaunch_fft3_direct_dit(blockDim, stream, XR, XI, W, n, t, block, wOffset, rowLen, tile, blk_num);
}

#endif

#ifndef HEADER_ACLRTLAUNCH_FFT3_RADER_DIT_HKERNEL_H_
#define HEADER_ACLRTLAUNCH_FFT3_RADER_DIT_HKERNEL_H_



extern "C" uint32_t aclrtlaunch_fft3_rader_dit(uint32_t blockDim, void* stream, void* XR, void* XI, void* W, uint32_t n, uint32_t t, uint32_t block, uint32_t wOffset, uint32_t rowLen, uint32_t tile, uint32_t blk_num);

inline uint32_t fft3_rader_dit(uint32_t blockDim, void* hold, void* stream, void* XR, void* XI, void* W, uint32_t n, uint32_t t, uint32_t block, uint32_t wOffset, uint32_t rowLen, uint32_t tile, uint32_t blk_num)
{
    (void)hold;
    return aclrtlaunch_fft3_rader_dit(blockDim, stream, XR, XI, W, n, t, block, wOffset, rowLen, tile, blk_num);
}

#endif

#ifndef HEADER_ACLRTLAUNCH_FFT3_SYM_DIT_HKERNEL_H_
#define HEADER_ACLRTLAUNCH_FFT3_SYM_DIT_HKERNEL_H_



extern "C" uint32_t aclrtlaunch_fft3_sym_dit(uint32_t blockDim, void* stream, void* XR, void* XI, void* W, uint32_t n, uint32_t t, uint32_t block, uint32_t wOffset, uint32_t rowLen, uint32_t tile, uint32_t blk_num);

inline uint32_t fft3_sym_dit(uint32_t blockDim, void* hold, void* stream, void* XR, void* XI, void* W, uint32_t n, uint32_t t, uint32_t block, uint32_t wOffset, uint32_t rowLen, uint32_t tile, uint32_t blk_num)
{
    (void)hold;
    return aclrtlaunch_fft3_sym_dit(blockDim, stream, XR, XI, W, n, t, block, wOffset, rowLen, tile, blk_num);
}

#endif

#ifndef HEADER_ACLRTLAUNCH_FFT3_WINOGRAD_DIT_HKERNEL_H_
#define HEADER_ACLRTLAUNCH_FFT3_WINOGRAD_DIT_HKERNEL_H_



extern "C" uint32_t aclrtlaunch_fft3_winograd_dit(uint32_t blockDim, void* stream, void* XR, void* XI, void* W, uint32_t n, uint32_t t, uint32_t block, uint32_t wOffset, uint32_t rowLen, uint32_t tile, uint32_t blk_num);

inline uint32_t fft3_winograd_dit(uint32_t blockDim, void* hold, void* stream, void* XR, void* XI, void* W, uint32_t n, uint32_t t, uint32_t block, uint32_t wOffset, uint32_t rowLen, uint32_t tile, uint32_t blk_num)
{
    (void)hold;
    return aclrtlaunch_fft3_winograd_dit(blockDim, stream, XR, XI, W, n, t, block, wOffset, rowLen, tile, blk_num);
}

#endif

#ifndef HEADER_ACLRTLAUNCH_FFT4_DIRECT_DIT_HKERNEL_H_
#define HEADER_ACLRTLAUNCH_FFT4_DIRECT_DIT_HKERNEL_H_



extern "C" uint32_t aclrtlaunch_fft4_direct_dit(uint32_t blockDim, void* stream, void* XR, void* XI, void* W, uint32_t n, uint32_t t, uint32_t block, uint32_t wOffset, uint32_t rowLen, uint32_t tile, uint32_t blk_num);

inline uint32_t fft4_direct_dit(uint32_t blockDim, void* hold, void* stream, void* XR, void* XI, void* W, uint32_t n, uint32_t t, uint32_t block, uint32_t wOffset, uint32_t rowLen, uint32_t tile, uint32_t blk_num)
{
    (void)hold;
    return aclrtlaunch_fft4_direct_dit(blockDim, stream, XR, XI, W, n, t, block, wOffset, rowLen, tile, blk_num);
}

#endif

#ifndef HEADER_ACLRTLAUNCH_FFT5_DIRECT_DIT_HKERNEL_H_
#define HEADER_ACLRTLAUNCH_FFT5_DIRECT_DIT_HKERNEL_H_



extern "C" uint32_t aclrtlaunch_fft5_direct_dit(uint32_t blockDim, void* stream, void* XR, void* XI, void* W, uint32_t n, uint32_t t, uint32_t block, uint32_t wOffset, uint32_t rowLen, uint32_t tile, uint32_t blk_num);

inline uint32_t fft5_direct_dit(uint32_t blockDim, void* hold, void* stream, void* XR, void* XI, void* W, uint32_t n, uint32_t t, uint32_t block, uint32_t wOffset, uint32_t rowLen, uint32_t tile, uint32_t blk_num)
{
    (void)hold;
    return aclrtlaunch_fft5_direct_dit(blockDim, stream, XR, XI, W, n, t, block, wOffset, rowLen, tile, blk_num);
}

#endif

#ifndef HEADER_ACLRTLAUNCH_FFT5_RADER_DIT_HKERNEL_H_
#define HEADER_ACLRTLAUNCH_FFT5_RADER_DIT_HKERNEL_H_



extern "C" uint32_t aclrtlaunch_fft5_rader_dit(uint32_t blockDim, void* stream, void* XR, void* XI, void* W, uint32_t n, uint32_t t, uint32_t block, uint32_t wOffset, uint32_t rowLen, uint32_t tile, uint32_t blk_num);

inline uint32_t fft5_rader_dit(uint32_t blockDim, void* hold, void* stream, void* XR, void* XI, void* W, uint32_t n, uint32_t t, uint32_t block, uint32_t wOffset, uint32_t rowLen, uint32_t tile, uint32_t blk_num)
{
    (void)hold;
    return aclrtlaunch_fft5_rader_dit(blockDim, stream, XR, XI, W, n, t, block, wOffset, rowLen, tile, blk_num);
}

#endif

#ifndef HEADER_ACLRTLAUNCH_FFT5_SYM_DIT_HKERNEL_H_
#define HEADER_ACLRTLAUNCH_FFT5_SYM_DIT_HKERNEL_H_



extern "C" uint32_t aclrtlaunch_fft5_sym_dit(uint32_t blockDim, void* stream, void* XR, void* XI, void* W, uint32_t n, uint32_t t, uint32_t block, uint32_t wOffset, uint32_t rowLen, uint32_t tile, uint32_t blk_num);

inline uint32_t fft5_sym_dit(uint32_t blockDim, void* hold, void* stream, void* XR, void* XI, void* W, uint32_t n, uint32_t t, uint32_t block, uint32_t wOffset, uint32_t rowLen, uint32_t tile, uint32_t blk_num)
{
    (void)hold;
    return aclrtlaunch_fft5_sym_dit(blockDim, stream, XR, XI, W, n, t, block, wOffset, rowLen, tile, blk_num);
}

#endif

#ifndef HEADER_ACLRTLAUNCH_FFT5_WINOGRAD_DIT_HKERNEL_H_
#define HEADER_ACLRTLAUNCH_FFT5_WINOGRAD_DIT_HKERNEL_H_



extern "C" uint32_t aclrtlaunch_fft5_winograd_dit(uint32_t blockDim, void* stream, void* XR, void* XI, void* W, uint32_t n, uint32_t t, uint32_t block, uint32_t wOffset, uint32_t rowLen, uint32_t tile, uint32_t blk_num);

inline uint32_t fft5_winograd_dit(uint32_t blockDim, void* hold, void* stream, void* XR, void* XI, void* W, uint32_t n, uint32_t t, uint32_t block, uint32_t wOffset, uint32_t rowLen, uint32_t tile, uint32_t blk_num)
{
    (void)hold;
    return aclrtlaunch_fft5_winograd_dit(blockDim, stream, XR, XI, W, n, t, block, wOffset, rowLen, tile, blk_num);
}

#endif

#ifndef HEADER_ACLRTLAUNCH_FFT7_DIRECT_DIT_HKERNEL_H_
#define HEADER_ACLRTLAUNCH_FFT7_DIRECT_DIT_HKERNEL_H_



extern "C" uint32_t aclrtlaunch_fft7_direct_dit(uint32_t blockDim, void* stream, void* XR, void* XI, void* W, uint32_t n, uint32_t t, uint32_t block, uint32_t wOffset, uint32_t rowLen, uint32_t tile, uint32_t blk_num);

inline uint32_t fft7_direct_dit(uint32_t blockDim, void* hold, void* stream, void* XR, void* XI, void* W, uint32_t n, uint32_t t, uint32_t block, uint32_t wOffset, uint32_t rowLen, uint32_t tile, uint32_t blk_num)
{
    (void)hold;
    return aclrtlaunch_fft7_direct_dit(blockDim, stream, XR, XI, W, n, t, block, wOffset, rowLen, tile, blk_num);
}

#endif

#ifndef HEADER_ACLRTLAUNCH_FFT7_RADER_DIT_HKERNEL_H_
#define HEADER_ACLRTLAUNCH_FFT7_RADER_DIT_HKERNEL_H_



extern "C" uint32_t aclrtlaunch_fft7_rader_dit(uint32_t blockDim, void* stream, void* XR, void* XI, void* W, uint32_t n, uint32_t t, uint32_t block, uint32_t wOffset, uint32_t rowLen, uint32_t tile, uint32_t blk_num);

inline uint32_t fft7_rader_dit(uint32_t blockDim, void* hold, void* stream, void* XR, void* XI, void* W, uint32_t n, uint32_t t, uint32_t block, uint32_t wOffset, uint32_t rowLen, uint32_t tile, uint32_t blk_num)
{
    (void)hold;
    return aclrtlaunch_fft7_rader_dit(blockDim, stream, XR, XI, W, n, t, block, wOffset, rowLen, tile, blk_num);
}

#endif

#ifndef HEADER_ACLRTLAUNCH_FFT7_SYM_DIT_HKERNEL_H_
#define HEADER_ACLRTLAUNCH_FFT7_SYM_DIT_HKERNEL_H_



extern "C" uint32_t aclrtlaunch_fft7_sym_dit(uint32_t blockDim, void* stream, void* XR, void* XI, void* W, uint32_t n, uint32_t t, uint32_t block, uint32_t wOffset, uint32_t rowLen, uint32_t tile, uint32_t blk_num);

inline uint32_t fft7_sym_dit(uint32_t blockDim, void* hold, void* stream, void* XR, void* XI, void* W, uint32_t n, uint32_t t, uint32_t block, uint32_t wOffset, uint32_t rowLen, uint32_t tile, uint32_t blk_num)
{
    (void)hold;
    return aclrtlaunch_fft7_sym_dit(blockDim, stream, XR, XI, W, n, t, block, wOffset, rowLen, tile, blk_num);
}

#endif

#ifndef HEADER_ACLRTLAUNCH_FFT7_WINOGRAD_DIT_HKERNEL_H_
#define HEADER_ACLRTLAUNCH_FFT7_WINOGRAD_DIT_HKERNEL_H_



extern "C" uint32_t aclrtlaunch_fft7_winograd_dit(uint32_t blockDim, void* stream, void* XR, void* XI, void* W, uint32_t n, uint32_t t, uint32_t block, uint32_t wOffset, uint32_t rowLen, uint32_t tile, uint32_t blk_num);

inline uint32_t fft7_winograd_dit(uint32_t blockDim, void* hold, void* stream, void* XR, void* XI, void* W, uint32_t n, uint32_t t, uint32_t block, uint32_t wOffset, uint32_t rowLen, uint32_t tile, uint32_t blk_num)
{
    (void)hold;
    return aclrtlaunch_fft7_winograd_dit(blockDim, stream, XR, XI, W, n, t, block, wOffset, rowLen, tile, blk_num);
}

#endif
