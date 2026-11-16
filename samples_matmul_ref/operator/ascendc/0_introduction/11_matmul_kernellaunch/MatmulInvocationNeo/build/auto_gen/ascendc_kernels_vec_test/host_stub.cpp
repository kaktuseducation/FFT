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
    uint8_t aiv_buf[104776];
} __ascend_kernel_ascend910b4_ascendc_kernels_vec_test __attribute__ ((section (".ascend.kernel.ascend910b4.ascendc_kernels_vec_test"))) = {1,1,1,104776,104776,{0}};

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
        (const char *)__ascend_kernel_ascend910b4_ascendc_kernels_vec_test.aiv_buf,
        __ascend_kernel_ascend910b4_ascendc_kernels_vec_test.aiv_file_len,
        1,
        &g_kernel_handle_aiv);
    if (ret != 0) {
        printf("RegisterAscendBinary aiv ret %u \n", ret);
    }

    AscendProfRegister();
}





uint32_t launch_and_profiling_fft8_cube1_repack(uint32_t blockDim, void* stream, void **args, uint32_t size)
{
    uint64_t startTime;
    const char *name = "fft8_cube1_repack";
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

extern "C" uint32_t aclrtlaunch_fft8_cube1_repack(uint32_t blockDim, void* stream, void* XR, void* XI, void* W, void* SCRATCH_X, uint32_t n, uint32_t k0, uint32_t shift, uint32_t eighth, uint32_t validB, uint32_t wOffset, uint32_t aligned, uint32_t wTotal)
{
    struct {
        void* XR;
        void* XI;
        void* W;
        void* SCRATCH_X;
        uint32_t n;
        uint32_t k0;
        uint32_t shift;
        uint32_t eighth;
        uint32_t validB;
        uint32_t wOffset;
        uint32_t aligned;
        uint32_t wTotal;
    } args;

    uint32_t __ascendc_ret;
    args.XR = XR;
    args.XI = XI;
    args.W = W;
    args.SCRATCH_X = SCRATCH_X;
    args.n = n;
    args.k0 = k0;
    args.shift = shift;
    args.eighth = eighth;
    args.validB = validB;
    args.wOffset = wOffset;
    args.aligned = aligned;
    args.wTotal = wTotal;

    __ascendc_ret = launch_and_profiling_fft8_cube1_repack(blockDim, stream, (void **)&args, sizeof(args));
    return __ascendc_ret;
}
