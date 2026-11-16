#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <dlfcn.h>
#include <securec.h>

#ifndef ASCENDC_DUMP
#define ASCENDC_DUMP 1
#endif

#if defined(ASCENDC_DUMP) && (ASCENDC_DUMP == 0)
    #undef ASCENDC_DUMP
#endif

static void *g_kernel_handle_aiv;

struct ascend_kernels {
    uint32_t version;
    uint32_t type_cnt;
    uint32_t aiv_type;
    uint32_t aiv_len;
    uint32_t aiv_file_len;
    uint8_t aiv_buf[1772664];
} __ascend_kernel_ascend910b4_ascendc_kernels_npu_vec __attribute__ ((section (".ascend.kernel.ascend910b4.ascendc_kernels_npu_vec"))) = {1,1,1,1772664,1772664,{0}};

extern "C" {
uint32_t RegisterAscendBinary(const char *fileBuf, size_t fileSize, uint32_t type, void **handle);
uint32_t LaunchAscendKernel(void *handle, const uint64_t key, const uint32_t blockDim, void **args,
                            uint32_t size, const void *stream);
uint32_t GetAscendCoreSyncAddr(void **addr);
int UnregisterAscendBinary(void *hdl);
void StartAscendProf(const char *name, uint64_t *startTime);
void ReportAscendProf(const char *name, uint32_t blockDim, uint32_t taskType, const uint64_t startTime);
bool GetAscendProfStatus();
uint32_t AllocAscendMemDevice(void **devMem, uint64_t size);
uint32_t FreeAscendMemDevice(void *devMem);
void AscendProfRegister();
uint32_t GetCoreNumForMixVectorCore(uint32_t *aiCoreNum, uint32_t *vectorCoreNum);
uint32_t LaunchAscendKernelForVectorCore(const char *opType, void *handle, const uint64_t key, void **args, uint32_t size,
    const void *stream, bool enbaleProf, uint32_t aicBlockDim, uint32_t aivBlockDim, uint32_t aivBlockDimOffset);
}

namespace Adx {
    void AdumpPrintWorkSpace(const void *workSpaceAddr, const size_t dumpWorkSpaceSize,
                            void *stream, const char *opType);
}

static void __unregister_kernels(void) __attribute__((destructor));
void __unregister_kernels(void)
{
    if (g_kernel_handle_aiv) {
        UnregisterAscendBinary(g_kernel_handle_aiv);
        g_kernel_handle_aiv = NULL;
    }

}

static void __register_kernels(void) __attribute__((constructor));
void __register_kernels(void)
{
    uint32_t ret;

    ret = RegisterAscendBinary(
        (const char *)__ascend_kernel_ascend910b4_ascendc_kernels_npu_vec.aiv_buf,
        __ascend_kernel_ascend910b4_ascendc_kernels_npu_vec.aiv_file_len,
        1,
        &g_kernel_handle_aiv);
    if (ret != 0) {
        printf("RegisterAscendBinary aiv ret %u \n", ret);
    }

    AscendProfRegister();
}





uint32_t launch_and_profiling_arb_cmul(uint32_t blockDim, void* stream, void **args, uint32_t size)
{
    uint64_t startTime;
    const char *name = "arb_cmul";
    bool profStatus = GetAscendProfStatus();
    if (profStatus) {
        StartAscendProf(name, &startTime);
    }
    uint32_t ret = LaunchAscendKernel(g_kernel_handle_aiv, 0, blockDim, args, size, stream);
    if (ret != 0) {
        printf("LaunchAscendKernel ret %u\n", ret);
    }
    if (profStatus) {
        ReportAscendProf(name, blockDim, 1, startTime);
    }
    return ret;
}

extern "C" uint32_t aclrtlaunch_arb_cmul(uint32_t blockDim, void* stream, void* AR, void* AI, void* BR, void* BI, uint32_t n, uint32_t tile, uint32_t blk_num)
{
    struct {
        void* AR;
        void* AI;
        void* BR;
        void* BI;
        uint32_t n;
        uint32_t tile;
        uint32_t blk_num;
    } args;

    uint32_t __ascendc_ret;
    args.AR = AR;
    args.AI = AI;
    args.BR = BR;
    args.BI = BI;
    args.n = n;
    args.tile = tile;
    args.blk_num = blk_num;

    __ascendc_ret = launch_and_profiling_arb_cmul(blockDim, stream, (void **)&args, sizeof(args));
    return __ascendc_ret;
}


uint32_t launch_and_profiling_arb_cscale(uint32_t blockDim, void* stream, void **args, uint32_t size)
{
    uint64_t startTime;
    const char *name = "arb_cscale";
    bool profStatus = GetAscendProfStatus();
    if (profStatus) {
        StartAscendProf(name, &startTime);
    }
    uint32_t ret = LaunchAscendKernel(g_kernel_handle_aiv, 1, blockDim, args, size, stream);
    if (ret != 0) {
        printf("LaunchAscendKernel ret %u\n", ret);
    }
    if (profStatus) {
        ReportAscendProf(name, blockDim, 1, startTime);
    }
    return ret;
}

extern "C" uint32_t aclrtlaunch_arb_cscale(uint32_t blockDim, void* stream, void* AR, void* AI, uint32_t n, float s, float sign, uint32_t tile, uint32_t blk_num)
{
    struct {
        void* AR;
        void* AI;
        uint32_t n;
        float s;
        float sign;
        uint32_t tile;
        uint32_t blk_num;
    } args;

    uint32_t __ascendc_ret;
    args.AR = AR;
    args.AI = AI;
    args.n = n;
    args.s = s;
    args.sign = sign;
    args.tile = tile;
    args.blk_num = blk_num;

    __ascendc_ret = launch_and_profiling_arb_cscale(blockDim, stream, (void **)&args, sizeof(args));
    return __ascendc_ret;
}


uint32_t launch_and_profiling_fft2_direct_dit(uint32_t blockDim, void* stream, void **args, uint32_t size)
{
    uint64_t startTime;
    const char *name = "fft2_direct_dit";
    bool profStatus = GetAscendProfStatus();
    if (profStatus) {
        StartAscendProf(name, &startTime);
    }
    uint32_t ret = LaunchAscendKernel(g_kernel_handle_aiv, 2, blockDim, args, size, stream);
    if (ret != 0) {
        printf("LaunchAscendKernel ret %u\n", ret);
    }
    if (profStatus) {
        ReportAscendProf(name, blockDim, 1, startTime);
    }
    return ret;
}

extern "C" uint32_t aclrtlaunch_fft2_direct_dit(uint32_t blockDim, void* stream, void* XR, void* XI, void* W, uint32_t n, uint32_t t, uint32_t block, uint32_t wOffset, uint32_t rowLen, uint32_t tile, uint32_t blk_num)
{
    struct {
        void* XR;
        void* XI;
        void* W;
        uint32_t n;
        uint32_t t;
        uint32_t block;
        uint32_t wOffset;
        uint32_t rowLen;
        uint32_t tile;
        uint32_t blk_num;
    } args;

    uint32_t __ascendc_ret;
    args.XR = XR;
    args.XI = XI;
    args.W = W;
    args.n = n;
    args.t = t;
    args.block = block;
    args.wOffset = wOffset;
    args.rowLen = rowLen;
    args.tile = tile;
    args.blk_num = blk_num;

    __ascendc_ret = launch_and_profiling_fft2_direct_dit(blockDim, stream, (void **)&args, sizeof(args));
    return __ascendc_ret;
}


uint32_t launch_and_profiling_fft3_direct_dit(uint32_t blockDim, void* stream, void **args, uint32_t size)
{
    uint64_t startTime;
    const char *name = "fft3_direct_dit";
    bool profStatus = GetAscendProfStatus();
    if (profStatus) {
        StartAscendProf(name, &startTime);
    }
    uint32_t ret = LaunchAscendKernel(g_kernel_handle_aiv, 3, blockDim, args, size, stream);
    if (ret != 0) {
        printf("LaunchAscendKernel ret %u\n", ret);
    }
    if (profStatus) {
        ReportAscendProf(name, blockDim, 1, startTime);
    }
    return ret;
}

extern "C" uint32_t aclrtlaunch_fft3_direct_dit(uint32_t blockDim, void* stream, void* XR, void* XI, void* W, uint32_t n, uint32_t t, uint32_t block, uint32_t wOffset, uint32_t rowLen, uint32_t tile, uint32_t blk_num)
{
    struct {
        void* XR;
        void* XI;
        void* W;
        uint32_t n;
        uint32_t t;
        uint32_t block;
        uint32_t wOffset;
        uint32_t rowLen;
        uint32_t tile;
        uint32_t blk_num;
    } args;

    uint32_t __ascendc_ret;
    args.XR = XR;
    args.XI = XI;
    args.W = W;
    args.n = n;
    args.t = t;
    args.block = block;
    args.wOffset = wOffset;
    args.rowLen = rowLen;
    args.tile = tile;
    args.blk_num = blk_num;

    __ascendc_ret = launch_and_profiling_fft3_direct_dit(blockDim, stream, (void **)&args, sizeof(args));
    return __ascendc_ret;
}


uint32_t launch_and_profiling_fft3_rader_dit(uint32_t blockDim, void* stream, void **args, uint32_t size)
{
    uint64_t startTime;
    const char *name = "fft3_rader_dit";
    bool profStatus = GetAscendProfStatus();
    if (profStatus) {
        StartAscendProf(name, &startTime);
    }
    uint32_t ret = LaunchAscendKernel(g_kernel_handle_aiv, 4, blockDim, args, size, stream);
    if (ret != 0) {
        printf("LaunchAscendKernel ret %u\n", ret);
    }
    if (profStatus) {
        ReportAscendProf(name, blockDim, 1, startTime);
    }
    return ret;
}

extern "C" uint32_t aclrtlaunch_fft3_rader_dit(uint32_t blockDim, void* stream, void* XR, void* XI, void* W, uint32_t n, uint32_t t, uint32_t block, uint32_t wOffset, uint32_t rowLen, uint32_t tile, uint32_t blk_num)
{
    struct {
        void* XR;
        void* XI;
        void* W;
        uint32_t n;
        uint32_t t;
        uint32_t block;
        uint32_t wOffset;
        uint32_t rowLen;
        uint32_t tile;
        uint32_t blk_num;
    } args;

    uint32_t __ascendc_ret;
    args.XR = XR;
    args.XI = XI;
    args.W = W;
    args.n = n;
    args.t = t;
    args.block = block;
    args.wOffset = wOffset;
    args.rowLen = rowLen;
    args.tile = tile;
    args.blk_num = blk_num;

    __ascendc_ret = launch_and_profiling_fft3_rader_dit(blockDim, stream, (void **)&args, sizeof(args));
    return __ascendc_ret;
}


uint32_t launch_and_profiling_fft3_sym_dit(uint32_t blockDim, void* stream, void **args, uint32_t size)
{
    uint64_t startTime;
    const char *name = "fft3_sym_dit";
    bool profStatus = GetAscendProfStatus();
    if (profStatus) {
        StartAscendProf(name, &startTime);
    }
    uint32_t ret = LaunchAscendKernel(g_kernel_handle_aiv, 5, blockDim, args, size, stream);
    if (ret != 0) {
        printf("LaunchAscendKernel ret %u\n", ret);
    }
    if (profStatus) {
        ReportAscendProf(name, blockDim, 1, startTime);
    }
    return ret;
}

extern "C" uint32_t aclrtlaunch_fft3_sym_dit(uint32_t blockDim, void* stream, void* XR, void* XI, void* W, uint32_t n, uint32_t t, uint32_t block, uint32_t wOffset, uint32_t rowLen, uint32_t tile, uint32_t blk_num)
{
    struct {
        void* XR;
        void* XI;
        void* W;
        uint32_t n;
        uint32_t t;
        uint32_t block;
        uint32_t wOffset;
        uint32_t rowLen;
        uint32_t tile;
        uint32_t blk_num;
    } args;

    uint32_t __ascendc_ret;
    args.XR = XR;
    args.XI = XI;
    args.W = W;
    args.n = n;
    args.t = t;
    args.block = block;
    args.wOffset = wOffset;
    args.rowLen = rowLen;
    args.tile = tile;
    args.blk_num = blk_num;

    __ascendc_ret = launch_and_profiling_fft3_sym_dit(blockDim, stream, (void **)&args, sizeof(args));
    return __ascendc_ret;
}


uint32_t launch_and_profiling_fft3_winograd_dit(uint32_t blockDim, void* stream, void **args, uint32_t size)
{
    uint64_t startTime;
    const char *name = "fft3_winograd_dit";
    bool profStatus = GetAscendProfStatus();
    if (profStatus) {
        StartAscendProf(name, &startTime);
    }
    uint32_t ret = LaunchAscendKernel(g_kernel_handle_aiv, 6, blockDim, args, size, stream);
    if (ret != 0) {
        printf("LaunchAscendKernel ret %u\n", ret);
    }
    if (profStatus) {
        ReportAscendProf(name, blockDim, 1, startTime);
    }
    return ret;
}

extern "C" uint32_t aclrtlaunch_fft3_winograd_dit(uint32_t blockDim, void* stream, void* XR, void* XI, void* W, uint32_t n, uint32_t t, uint32_t block, uint32_t wOffset, uint32_t rowLen, uint32_t tile, uint32_t blk_num)
{
    struct {
        void* XR;
        void* XI;
        void* W;
        uint32_t n;
        uint32_t t;
        uint32_t block;
        uint32_t wOffset;
        uint32_t rowLen;
        uint32_t tile;
        uint32_t blk_num;
    } args;

    uint32_t __ascendc_ret;
    args.XR = XR;
    args.XI = XI;
    args.W = W;
    args.n = n;
    args.t = t;
    args.block = block;
    args.wOffset = wOffset;
    args.rowLen = rowLen;
    args.tile = tile;
    args.blk_num = blk_num;

    __ascendc_ret = launch_and_profiling_fft3_winograd_dit(blockDim, stream, (void **)&args, sizeof(args));
    return __ascendc_ret;
}


uint32_t launch_and_profiling_fft4_direct_dit(uint32_t blockDim, void* stream, void **args, uint32_t size)
{
    uint64_t startTime;
    const char *name = "fft4_direct_dit";
    bool profStatus = GetAscendProfStatus();
    if (profStatus) {
        StartAscendProf(name, &startTime);
    }
    uint32_t ret = LaunchAscendKernel(g_kernel_handle_aiv, 7, blockDim, args, size, stream);
    if (ret != 0) {
        printf("LaunchAscendKernel ret %u\n", ret);
    }
    if (profStatus) {
        ReportAscendProf(name, blockDim, 1, startTime);
    }
    return ret;
}

extern "C" uint32_t aclrtlaunch_fft4_direct_dit(uint32_t blockDim, void* stream, void* XR, void* XI, void* W, uint32_t n, uint32_t t, uint32_t block, uint32_t wOffset, uint32_t rowLen, uint32_t tile, uint32_t blk_num)
{
    struct {
        void* XR;
        void* XI;
        void* W;
        uint32_t n;
        uint32_t t;
        uint32_t block;
        uint32_t wOffset;
        uint32_t rowLen;
        uint32_t tile;
        uint32_t blk_num;
    } args;

    uint32_t __ascendc_ret;
    args.XR = XR;
    args.XI = XI;
    args.W = W;
    args.n = n;
    args.t = t;
    args.block = block;
    args.wOffset = wOffset;
    args.rowLen = rowLen;
    args.tile = tile;
    args.blk_num = blk_num;

    __ascendc_ret = launch_and_profiling_fft4_direct_dit(blockDim, stream, (void **)&args, sizeof(args));
    return __ascendc_ret;
}


uint32_t launch_and_profiling_fft5_direct_dit(uint32_t blockDim, void* stream, void **args, uint32_t size)
{
    uint64_t startTime;
    const char *name = "fft5_direct_dit";
    bool profStatus = GetAscendProfStatus();
    if (profStatus) {
        StartAscendProf(name, &startTime);
    }
    uint32_t ret = LaunchAscendKernel(g_kernel_handle_aiv, 8, blockDim, args, size, stream);
    if (ret != 0) {
        printf("LaunchAscendKernel ret %u\n", ret);
    }
    if (profStatus) {
        ReportAscendProf(name, blockDim, 1, startTime);
    }
    return ret;
}

extern "C" uint32_t aclrtlaunch_fft5_direct_dit(uint32_t blockDim, void* stream, void* XR, void* XI, void* W, uint32_t n, uint32_t t, uint32_t block, uint32_t wOffset, uint32_t rowLen, uint32_t tile, uint32_t blk_num)
{
    struct {
        void* XR;
        void* XI;
        void* W;
        uint32_t n;
        uint32_t t;
        uint32_t block;
        uint32_t wOffset;
        uint32_t rowLen;
        uint32_t tile;
        uint32_t blk_num;
    } args;

    uint32_t __ascendc_ret;
    args.XR = XR;
    args.XI = XI;
    args.W = W;
    args.n = n;
    args.t = t;
    args.block = block;
    args.wOffset = wOffset;
    args.rowLen = rowLen;
    args.tile = tile;
    args.blk_num = blk_num;

    __ascendc_ret = launch_and_profiling_fft5_direct_dit(blockDim, stream, (void **)&args, sizeof(args));
    return __ascendc_ret;
}


uint32_t launch_and_profiling_fft5_rader_dit(uint32_t blockDim, void* stream, void **args, uint32_t size)
{
    uint64_t startTime;
    const char *name = "fft5_rader_dit";
    bool profStatus = GetAscendProfStatus();
    if (profStatus) {
        StartAscendProf(name, &startTime);
    }
    uint32_t ret = LaunchAscendKernel(g_kernel_handle_aiv, 9, blockDim, args, size, stream);
    if (ret != 0) {
        printf("LaunchAscendKernel ret %u\n", ret);
    }
    if (profStatus) {
        ReportAscendProf(name, blockDim, 1, startTime);
    }
    return ret;
}

extern "C" uint32_t aclrtlaunch_fft5_rader_dit(uint32_t blockDim, void* stream, void* XR, void* XI, void* W, uint32_t n, uint32_t t, uint32_t block, uint32_t wOffset, uint32_t rowLen, uint32_t tile, uint32_t blk_num)
{
    struct {
        void* XR;
        void* XI;
        void* W;
        uint32_t n;
        uint32_t t;
        uint32_t block;
        uint32_t wOffset;
        uint32_t rowLen;
        uint32_t tile;
        uint32_t blk_num;
    } args;

    uint32_t __ascendc_ret;
    args.XR = XR;
    args.XI = XI;
    args.W = W;
    args.n = n;
    args.t = t;
    args.block = block;
    args.wOffset = wOffset;
    args.rowLen = rowLen;
    args.tile = tile;
    args.blk_num = blk_num;

    __ascendc_ret = launch_and_profiling_fft5_rader_dit(blockDim, stream, (void **)&args, sizeof(args));
    return __ascendc_ret;
}


uint32_t launch_and_profiling_fft5_sym_dit(uint32_t blockDim, void* stream, void **args, uint32_t size)
{
    uint64_t startTime;
    const char *name = "fft5_sym_dit";
    bool profStatus = GetAscendProfStatus();
    if (profStatus) {
        StartAscendProf(name, &startTime);
    }
    uint32_t ret = LaunchAscendKernel(g_kernel_handle_aiv, 10, blockDim, args, size, stream);
    if (ret != 0) {
        printf("LaunchAscendKernel ret %u\n", ret);
    }
    if (profStatus) {
        ReportAscendProf(name, blockDim, 1, startTime);
    }
    return ret;
}

extern "C" uint32_t aclrtlaunch_fft5_sym_dit(uint32_t blockDim, void* stream, void* XR, void* XI, void* W, uint32_t n, uint32_t t, uint32_t block, uint32_t wOffset, uint32_t rowLen, uint32_t tile, uint32_t blk_num)
{
    struct {
        void* XR;
        void* XI;
        void* W;
        uint32_t n;
        uint32_t t;
        uint32_t block;
        uint32_t wOffset;
        uint32_t rowLen;
        uint32_t tile;
        uint32_t blk_num;
    } args;

    uint32_t __ascendc_ret;
    args.XR = XR;
    args.XI = XI;
    args.W = W;
    args.n = n;
    args.t = t;
    args.block = block;
    args.wOffset = wOffset;
    args.rowLen = rowLen;
    args.tile = tile;
    args.blk_num = blk_num;

    __ascendc_ret = launch_and_profiling_fft5_sym_dit(blockDim, stream, (void **)&args, sizeof(args));
    return __ascendc_ret;
}


uint32_t launch_and_profiling_fft5_winograd_dit(uint32_t blockDim, void* stream, void **args, uint32_t size)
{
    uint64_t startTime;
    const char *name = "fft5_winograd_dit";
    bool profStatus = GetAscendProfStatus();
    if (profStatus) {
        StartAscendProf(name, &startTime);
    }
    uint32_t ret = LaunchAscendKernel(g_kernel_handle_aiv, 11, blockDim, args, size, stream);
    if (ret != 0) {
        printf("LaunchAscendKernel ret %u\n", ret);
    }
    if (profStatus) {
        ReportAscendProf(name, blockDim, 1, startTime);
    }
    return ret;
}

extern "C" uint32_t aclrtlaunch_fft5_winograd_dit(uint32_t blockDim, void* stream, void* XR, void* XI, void* W, uint32_t n, uint32_t t, uint32_t block, uint32_t wOffset, uint32_t rowLen, uint32_t tile, uint32_t blk_num)
{
    struct {
        void* XR;
        void* XI;
        void* W;
        uint32_t n;
        uint32_t t;
        uint32_t block;
        uint32_t wOffset;
        uint32_t rowLen;
        uint32_t tile;
        uint32_t blk_num;
    } args;

    uint32_t __ascendc_ret;
    args.XR = XR;
    args.XI = XI;
    args.W = W;
    args.n = n;
    args.t = t;
    args.block = block;
    args.wOffset = wOffset;
    args.rowLen = rowLen;
    args.tile = tile;
    args.blk_num = blk_num;

    __ascendc_ret = launch_and_profiling_fft5_winograd_dit(blockDim, stream, (void **)&args, sizeof(args));
    return __ascendc_ret;
}


uint32_t launch_and_profiling_fft7_direct_dit(uint32_t blockDim, void* stream, void **args, uint32_t size)
{
    uint64_t startTime;
    const char *name = "fft7_direct_dit";
    bool profStatus = GetAscendProfStatus();
    if (profStatus) {
        StartAscendProf(name, &startTime);
    }
    uint32_t ret = LaunchAscendKernel(g_kernel_handle_aiv, 12, blockDim, args, size, stream);
    if (ret != 0) {
        printf("LaunchAscendKernel ret %u\n", ret);
    }
    if (profStatus) {
        ReportAscendProf(name, blockDim, 1, startTime);
    }
    return ret;
}

extern "C" uint32_t aclrtlaunch_fft7_direct_dit(uint32_t blockDim, void* stream, void* XR, void* XI, void* W, uint32_t n, uint32_t t, uint32_t block, uint32_t wOffset, uint32_t rowLen, uint32_t tile, uint32_t blk_num)
{
    struct {
        void* XR;
        void* XI;
        void* W;
        uint32_t n;
        uint32_t t;
        uint32_t block;
        uint32_t wOffset;
        uint32_t rowLen;
        uint32_t tile;
        uint32_t blk_num;
    } args;

    uint32_t __ascendc_ret;
    args.XR = XR;
    args.XI = XI;
    args.W = W;
    args.n = n;
    args.t = t;
    args.block = block;
    args.wOffset = wOffset;
    args.rowLen = rowLen;
    args.tile = tile;
    args.blk_num = blk_num;

    __ascendc_ret = launch_and_profiling_fft7_direct_dit(blockDim, stream, (void **)&args, sizeof(args));
    return __ascendc_ret;
}


uint32_t launch_and_profiling_fft7_rader_dit(uint32_t blockDim, void* stream, void **args, uint32_t size)
{
    uint64_t startTime;
    const char *name = "fft7_rader_dit";
    bool profStatus = GetAscendProfStatus();
    if (profStatus) {
        StartAscendProf(name, &startTime);
    }
    uint32_t ret = LaunchAscendKernel(g_kernel_handle_aiv, 13, blockDim, args, size, stream);
    if (ret != 0) {
        printf("LaunchAscendKernel ret %u\n", ret);
    }
    if (profStatus) {
        ReportAscendProf(name, blockDim, 1, startTime);
    }
    return ret;
}

extern "C" uint32_t aclrtlaunch_fft7_rader_dit(uint32_t blockDim, void* stream, void* XR, void* XI, void* W, uint32_t n, uint32_t t, uint32_t block, uint32_t wOffset, uint32_t rowLen, uint32_t tile, uint32_t blk_num)
{
    struct {
        void* XR;
        void* XI;
        void* W;
        uint32_t n;
        uint32_t t;
        uint32_t block;
        uint32_t wOffset;
        uint32_t rowLen;
        uint32_t tile;
        uint32_t blk_num;
    } args;

    uint32_t __ascendc_ret;
    args.XR = XR;
    args.XI = XI;
    args.W = W;
    args.n = n;
    args.t = t;
    args.block = block;
    args.wOffset = wOffset;
    args.rowLen = rowLen;
    args.tile = tile;
    args.blk_num = blk_num;

    __ascendc_ret = launch_and_profiling_fft7_rader_dit(blockDim, stream, (void **)&args, sizeof(args));
    return __ascendc_ret;
}


uint32_t launch_and_profiling_fft7_sym_dit(uint32_t blockDim, void* stream, void **args, uint32_t size)
{
    uint64_t startTime;
    const char *name = "fft7_sym_dit";
    bool profStatus = GetAscendProfStatus();
    if (profStatus) {
        StartAscendProf(name, &startTime);
    }
    uint32_t ret = LaunchAscendKernel(g_kernel_handle_aiv, 14, blockDim, args, size, stream);
    if (ret != 0) {
        printf("LaunchAscendKernel ret %u\n", ret);
    }
    if (profStatus) {
        ReportAscendProf(name, blockDim, 1, startTime);
    }
    return ret;
}

extern "C" uint32_t aclrtlaunch_fft7_sym_dit(uint32_t blockDim, void* stream, void* XR, void* XI, void* W, uint32_t n, uint32_t t, uint32_t block, uint32_t wOffset, uint32_t rowLen, uint32_t tile, uint32_t blk_num)
{
    struct {
        void* XR;
        void* XI;
        void* W;
        uint32_t n;
        uint32_t t;
        uint32_t block;
        uint32_t wOffset;
        uint32_t rowLen;
        uint32_t tile;
        uint32_t blk_num;
    } args;

    uint32_t __ascendc_ret;
    args.XR = XR;
    args.XI = XI;
    args.W = W;
    args.n = n;
    args.t = t;
    args.block = block;
    args.wOffset = wOffset;
    args.rowLen = rowLen;
    args.tile = tile;
    args.blk_num = blk_num;

    __ascendc_ret = launch_and_profiling_fft7_sym_dit(blockDim, stream, (void **)&args, sizeof(args));
    return __ascendc_ret;
}


uint32_t launch_and_profiling_fft7_winograd_dit(uint32_t blockDim, void* stream, void **args, uint32_t size)
{
    uint64_t startTime;
    const char *name = "fft7_winograd_dit";
    bool profStatus = GetAscendProfStatus();
    if (profStatus) {
        StartAscendProf(name, &startTime);
    }
    uint32_t ret = LaunchAscendKernel(g_kernel_handle_aiv, 15, blockDim, args, size, stream);
    if (ret != 0) {
        printf("LaunchAscendKernel ret %u\n", ret);
    }
    if (profStatus) {
        ReportAscendProf(name, blockDim, 1, startTime);
    }
    return ret;
}

extern "C" uint32_t aclrtlaunch_fft7_winograd_dit(uint32_t blockDim, void* stream, void* XR, void* XI, void* W, uint32_t n, uint32_t t, uint32_t block, uint32_t wOffset, uint32_t rowLen, uint32_t tile, uint32_t blk_num)
{
    struct {
        void* XR;
        void* XI;
        void* W;
        uint32_t n;
        uint32_t t;
        uint32_t block;
        uint32_t wOffset;
        uint32_t rowLen;
        uint32_t tile;
        uint32_t blk_num;
    } args;

    uint32_t __ascendc_ret;
    args.XR = XR;
    args.XI = XI;
    args.W = W;
    args.n = n;
    args.t = t;
    args.block = block;
    args.wOffset = wOffset;
    args.rowLen = rowLen;
    args.tile = tile;
    args.blk_num = blk_num;

    __ascendc_ret = launch_and_profiling_fft7_winograd_dit(blockDim, stream, (void **)&args, sizeof(args));
    return __ascendc_ret;
}
