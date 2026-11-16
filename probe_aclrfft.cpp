/*
 * probe_aclrfft.cpp — find the parameter combination aclRfft1D accepts.
 *
 * The first attempt returned 161002 (ACLNN_ERR_PARAM_INVALID), which says a
 * parameter was rejected but not which one.  The header documents the
 * signature and the data type but not the expected output shape, the
 * meaning of `norm`, or whether `dim` counts from the front.  Rather than
 * guess one combination at a time, this enumerates the plausible ones and
 * reports which are accepted.
 *
 * Combinations tried, for one length N:
 *   input  rank 1 [N]                 and rank 2 [1, N]
 *   output rank 1 [2*(N/2+1)]         (interleaved complex as float)
 *          rank 2 [N/2+1, 2]          (complex pairs as last axis)
 *          rank 1 [N+2]               (some APIs pad to N+2)
 *          rank 1 [N]                 (in case it writes in place-sized)
 *   dim    0, -1, 1 (for rank 2)
 *   norm   0, 1, 2
 *
 * Whatever is accepted is printed with its full parameter set, and that
 * combination goes into bench_aclrfft.cpp.
 *
 * BUILD
 *   g++ -std=c++17 -O2 probe_aclrfft.cpp -o probe_aclrfft \
 *       -I${ASCEND_HOME_PATH}/aarch64-linux/include \
 *       -L${ASCEND_HOME_PATH}/aarch64-linux/lib64 \
 *       -lascendcl -lopapi -lnnopbase \
 *       -Wl,-rpath,${ASCEND_HOME_PATH}/aarch64-linux/lib64
 */

#include <acl/acl.h>
#include <aclnn/aclnn_base.h>
#include <aclnnop/acl_rfft1d.h>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

struct Shape {
    std::vector<int64_t> dims;
    std::string label;
};

static aclTensor *mk(const std::vector<int64_t> &dims, void *data)
{
    std::vector<int64_t> stride(dims.size(), 1);
    for (int i = static_cast<int>(dims.size()) - 2; i >= 0; --i)
        stride[i] = stride[i + 1] * dims[i + 1];
    return aclCreateTensor(dims.data(), dims.size(), ACL_FLOAT, stride.data(),
                           0, ACL_FORMAT_ND, dims.data(), dims.size(), data);
}

int main(int argc, char **argv)
{
    int64_t N = 4096;
    int deviceId = 0;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--n" && i + 1 < argc) N = std::atoll(argv[++i]);
        else if (a == "--device" && i + 1 < argc) deviceId = std::atoi(argv[++i]);
    }

    if (aclInit(nullptr) != ACL_SUCCESS) { std::printf("aclInit failed\n"); return 1; }
    if (aclrtSetDevice(deviceId) != ACL_SUCCESS) { std::printf("setDevice failed\n"); return 1; }

    // Generous allocations: every candidate shape must fit.
    const size_t inFloats = static_cast<size_t>(N) * 2 + 16;
    const size_t outFloats = static_cast<size_t>(N) * 2 + 16;
    void *dIn = nullptr, *dOut = nullptr;
    aclrtMalloc(&dIn, inFloats * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&dOut, outFloats * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    std::vector<float> host(inFloats, 0.0f);
    for (size_t i = 0; i < static_cast<size_t>(N); ++i)
        host[i] = static_cast<float>(i % 17) * 0.1f;
    aclrtMemcpy(dIn, inFloats * sizeof(float), host.data(),
                inFloats * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);

    const int64_t half = N / 2 + 1;

    std::vector<Shape> inShapes = {
        {{N},     "[N]"},
        {{1, N},  "[1,N]"},
    };
    std::vector<Shape> outShapes = {
        {{2 * half},   "[2*(N/2+1)]"},
        {{half, 2},    "[N/2+1,2]"},
        {{N + 2},      "[N+2]"},
        {{N},          "[N]"},
        {{1, 2 * half},"[1,2*(N/2+1)]"},
        {{1, half, 2}, "[1,N/2+1,2]"},
    };
    std::vector<int64_t> dims = {0, -1, 1};
    std::vector<int64_t> norms = {0, 1, 2};

    std::printf("Probing aclRfft1D at N = %lld\n", static_cast<long long>(N));
    std::printf("%-8s %-14s %5s %5s   %s\n", "input", "output", "dim", "norm",
                "result");
    std::printf("--------------------------------------------------------------\n");

    int accepted = 0;
    for (auto &is : inShapes) {
        aclTensor *tIn = mk(is.dims, dIn);
        if (!tIn) continue;
        for (auto &os : outShapes) {
            aclTensor *tOut = mk(os.dims, dOut);
            if (!tOut) continue;
            for (int64_t d : dims) {
                // a dim of 1 only makes sense for a rank-2 input
                if (d == 1 && is.dims.size() < 2) continue;
                for (int64_t nm : norms) {
                    uint64_t ws = 0;
                    aclOpExecutor *ex = nullptr;
                    auto st = aclRfft1DGetWorkspaceSize(tIn, N, d, nm, tOut,
                                                        &ws, &ex);
                    if (st == ACL_SUCCESS) {
                        std::printf("%-8s %-14s %5lld %5lld   ACCEPTED  "
                                    "workspace=%llu\n",
                                    is.label.c_str(), os.label.c_str(),
                                    static_cast<long long>(d),
                                    static_cast<long long>(nm),
                                    static_cast<unsigned long long>(ws));
                        ++accepted;
                    }
                }
            }
            aclDestroyTensor(tOut);
        }
        aclDestroyTensor(tIn);
    }

    if (accepted == 0) {
        std::printf("\nNo combination accepted at N = %lld.\n",
                    static_cast<long long>(N));
        std::printf("Try other lengths: the operator may support only a\n");
        std::printf("restricted set. Run with --n 1024, --n 512, --n 256.\n");
    } else {
        std::printf("\n%d combination(s) accepted. Use the first one in\n",
                    accepted);
        std::printf("bench_aclrfft.cpp.\n");
    }

    aclrtFree(dIn);
    aclrtFree(dOut);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return accepted ? 0 : 2;
}
