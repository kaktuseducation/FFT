/*
 * bench_aclrfft.cpp — the vendor's own transform operator, measured directly.
 *
 * WHY THIS EXISTS
 *
 * Section 6.5 established that MindSpore's transform carries a fixed cost of
 * roughly 845 us independent of transform length, and attributed about 123 us
 * of it to dispatching any operation at all.  That attribution was made from
 * inside MindSpore, by timing an elementwise operation through the same
 * framework.  It therefore cannot distinguish a cost imposed by the framework
 * from one imposed by the CANN stack beneath it.
 *
 * This benchmark settles that question.  aclRfft1D is a transform operator
 * shipped in the CANN toolkit itself (aclnnop/acl_rfft1d.h), reached through
 * libopapi.so with no framework in between.  If it carries a fixed cost of
 * the same order, the cost belongs to the platform.  If it does not, the
 * cost belongs to MindSpore, which is a narrower claim and must be stated as
 * one.
 *
 * WHAT IT DOES NOT MEASURE
 *
 * aclRfft1D is real-to-complex.  The kernels of this thesis are
 * complex-to-complex, which is roughly twice the work, so the two times are
 * NOT directly comparable and no speedup should be computed between them.
 * What transfers is the shape of the cost curve: whether a plateau exists at
 * short lengths, and where it sits.  Keep that distinction in the text.
 *
 * Worth recording in its own right: in CANN 8.0.RC3 this is the ONLY
 * transform operator in aclnnop.  There is no complex-to-complex operator at
 * all, which is the gap this thesis fills and which the ops-fft library,
 * released in March 2026 for CANN 9.0, later fills on the vendor's side.
 *
 * BUILD
 *   bash build_aclrfft.sh
 * or by hand:
 *   g++ -std=c++17 bench_aclrfft.cpp -o bench_aclrfft \
 *       -I${ASCEND_HOME_PATH}/include \
 *       -L${ASCEND_HOME_PATH}/lib64 -lascendcl -lopapi -lnnopbase
 *
 * RUN
 *   ./bench_aclrfft --n 4096 --runs 20 --warmup 5 --csv output/aclrfft.csv
 */

#include <acl/acl.h>
#include <aclnn/aclnn_base.h>
#include <aclnnop/acl_rfft1d.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CHECK(expr, msg)                                                    \
    do {                                                                    \
        auto _st = (expr);                                                  \
        if (_st != ACL_SUCCESS) {                                           \
            std::fprintf(stderr, "[ERROR ] %s failed, code %d\n", msg,      \
                         static_cast<int>(_st));                            \
            return 1;                                                       \
        }                                                                   \
    } while (0)

static double now_us()
{
    using namespace std::chrono;
    return duration<double, std::micro>(
               steady_clock::now().time_since_epoch())
        .count();
}

struct Stats {
    double mean, sd, lo, hi;
    double median, p90;
    int n;
};

static Stats summarise(const std::vector<double> &v)
{
    Stats s{};
    s.n = static_cast<int>(v.size());
    if (s.n == 0) return s;
    double sum = 0.0;
    for (double x : v) sum += x;
    s.mean = sum / s.n;
    double acc = 0.0;
    for (double x : v) acc += (x - s.mean) * (x - s.mean);
    s.sd = s.n > 1 ? std::sqrt(acc / (s.n - 1)) : 0.0;
    s.lo = *std::min_element(v.begin(), v.end());
    s.hi = *std::max_element(v.begin(), v.end());

    // The mean is not robust here.  One stalled run among a hundred, which
    // this platform does produce, shifts it by tens of microseconds: at
    // N = 64 a single 3.8 ms outlier moved the mean from about 78 us to
    // 117.66.  The median is what the reported figures should rest on, so it
    // is computed alongside and a 90th percentile records the tail rather
    // than hiding it.
    std::vector<double> sorted(v);
    std::sort(sorted.begin(), sorted.end());
    s.median = (s.n % 2) ? sorted[s.n / 2]
                         : 0.5 * (sorted[s.n / 2 - 1] + sorted[s.n / 2]);
    s.p90 = sorted[static_cast<size_t>(0.9 * (s.n - 1) + 0.5)];
    return s;
}

int main(int argc, char **argv)
{
    int64_t N = 4096;
    int runs = 20, warmup = 5, deviceId = 0;
    int64_t norm = 1;
    std::string csvPath = "./output/aclrfft_baseline.csv";
    std::string inputPath;  // optional: read real input from a file
    bool reuseWs = false;   // allocate the workspace once, as real code does

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { return (i + 1 < argc) ? argv[++i] : nullptr; };
        if (a == "--n")            { if (auto v = next()) N = std::atoll(v); }
        else if (a == "--runs")    { if (auto v = next()) runs = std::atoi(v); }
        else if (a == "--warmup")  { if (auto v = next()) warmup = std::atoi(v); }
        else if (a == "--device")  { if (auto v = next()) deviceId = std::atoi(v); }
        else if (a == "--csv")     { if (auto v = next()) csvPath = v; }
        else if (a == "--input")   { if (auto v = next()) inputPath = v; }
        else if (a == "--norm")    { if (auto v = next()) norm = std::atoll(v); }
        else if (a == "--reuse-ws") { reuseWs = true; }
        else if (a == "--help") {
            std::printf("usage: %s [--n N] [--runs R] [--warmup W] "
                        "[--device D] [--csv PATH] [--input FILE] "
                        "[--norm 1|2] [--reuse-ws]\n", argv[0]);
            std::printf("  --reuse-ws  allocate the workspace once instead of "
                        "per call,\n              which is how production code "
                        "uses the two-stage API\n");
            return 0;
        }
    }

    // Real input of length N.  When --input is given the file is read as
    // interleaved complex float32 (the format the rest of this project uses)
    // and only the real parts are taken, so that the same generated data
    // drives this benchmark as drives every other one.
    std::vector<float> host(static_cast<size_t>(N));
    if (!inputPath.empty()) {
        FILE *f = std::fopen(inputPath.c_str(), "rb");
        if (!f) {
            std::fprintf(stderr, "[ERROR ] cannot open %s\n", inputPath.c_str());
            return 1;
        }
        std::vector<float> inter(static_cast<size_t>(N) * 2);
        size_t got = std::fread(inter.data(), sizeof(float), inter.size(), f);
        std::fclose(f);
        if (got < inter.size()) {
            std::fprintf(stderr, "[ERROR ] %s holds %zu floats, need %zu\n",
                         inputPath.c_str(), got, inter.size());
            return 1;
        }
        for (int64_t i = 0; i < N; ++i) host[i] = inter[2 * i];
    } else {
        for (int64_t i = 0; i < N; ++i)
            host[i] = static_cast<float>(std::sin(0.001 * i) * 10.0);
    }

    // rfft of a length-N real signal yields N/2 + 1 complex values.  The
    // operator's output tensor is declared FLOAT, so the complex pairs are
    // laid out as 2*(N/2+1) floats.
    const int64_t outComplex = N / 2 + 1;
    const int64_t outFloats = outComplex * 2;

    CHECK(aclInit(nullptr), "aclInit");
    CHECK(aclrtSetDevice(deviceId), "aclrtSetDevice");
    aclrtStream stream = nullptr;
    CHECK(aclrtCreateStream(&stream), "aclrtCreateStream");

    void *dIn = nullptr, *dOut = nullptr;
    CHECK(aclrtMalloc(&dIn, host.size() * sizeof(float),
                      ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc(in)");
    CHECK(aclrtMalloc(&dOut, outFloats * sizeof(float),
                      ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc(out)");
    CHECK(aclrtMemcpy(dIn, host.size() * sizeof(float), host.data(),
                      host.size() * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE),
          "aclrtMemcpy(h2d)");

    // The probe established which shapes this operator accepts: a rank-1
    // real input [N] and a rank-2 output [N/2+1, 2] holding the complex
    // results as (re, im) pairs along the last axis.  A flat [2*(N/2+1)]
    // output is rejected, as is norm = 0.
    int64_t inShape[1]  = {N};
    int64_t outShape[2] = {outComplex, 2};
    int64_t inStride[1]  = {1};
    int64_t outStride[2] = {2, 1};

    aclTensor *tIn = aclCreateTensor(inShape, 1, ACL_FLOAT, inStride, 0,
                                     ACL_FORMAT_ND, inShape, 1, dIn);
    aclTensor *tOut = aclCreateTensor(outShape, 2, ACL_FLOAT, outStride, 0,
                                      ACL_FORMAT_ND, outShape, 2, dOut);
    if (!tIn || !tOut) {
        std::fprintf(stderr, "[ERROR ] aclCreateTensor returned null\n");
        return 1;
    }

    // The probe found norm = 0 rejected and norms 1 and 2 accepted; which
    // of the two matches numpy.fft.rfft's unscaled convention is settled by
    // the comparison below rather than assumed.
    const int64_t dim = 0;

    std::printf("================================================================\n");
    std::printf("[BENCH ] impl=acl/rfft1d  N=%lld  warmup=%d  runs=%d\n",
                static_cast<long long>(N), warmup, runs);
    std::printf("[BENCH ] path=libopapi.so (CANN operator, no framework)\n");
    std::printf("[BENCH ] transform=real-to-complex, NOT comparable to the "
                "complex-to-complex kernels\n");
    std::printf("[BENCH ] norm=%lld  output shape=[%lld,2]\n",
                static_cast<long long>(norm),
                static_cast<long long>(outComplex));
    std::printf("----------------------------------------------------------------\n");

    std::vector<double> planUs, allocUs, execUs, totalUs;

    // The workspace size turned out not to depend on N, so a single
    // allocation serves every iteration.  Keeping both modes is deliberate:
    // per-call allocation is what a naive caller pays, reuse is what the
    // two-stage API is designed for, and the two differ by enough to matter.
    void *wsShared = nullptr;
    double wsFirstAllocUs = 0.0;
    uint64_t wsSizeSeen = 0;   // kept for the report after the loop

    for (int it = 0; it < warmup + runs; ++it) {
        uint64_t wsSize = 0;
        // (recorded into wsSizeSeen once the call has filled it in)
        aclOpExecutor *executor = nullptr;

        // t0 -> t1 planning, t1 -> t2 workspace allocation, t2 -> t3
        // execution.  An earlier version of this benchmark timed planning and
        // allocation together, which mixed a host-side plan with a
        // two-gigabyte malloc and made the planning figure meaningless.
        double t0 = now_us();
        auto st = aclRfft1DGetWorkspaceSize(tIn, N, dim, norm, tOut,
                                            &wsSize, &executor);
        if (st != ACL_SUCCESS) {
            std::fprintf(stderr,
                         "[ERROR ] aclRfft1DGetWorkspaceSize failed, code %d\n",
                         static_cast<int>(st));
            std::fprintf(stderr,
                         "[HINT  ] the operator may reject this length or "
                         "this norm value; try another N\n");
            return 1;
        }

        wsSizeSeen = wsSize;
        if (it == 0)
            std::printf("[BENCH ] workspace requested: %llu bytes (%.1f MB)\n",
                        static_cast<unsigned long long>(wsSize),
                        wsSize / 1048576.0);

        double t1 = now_us();

        void *ws = nullptr;
        if (reuseWs && wsShared) {
            ws = wsShared;                      // already paid for
        } else if (wsSize > 0) {
            auto mst = aclrtMalloc(&ws, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
            if (mst != ACL_SUCCESS) {
                std::fprintf(stderr,
                             "[ERROR ] could not allocate %llu bytes of "
                             "workspace (code %d).\n",
                             static_cast<unsigned long long>(wsSize),
                             static_cast<int>(mst));
                std::fprintf(stderr,
                             "[HINT  ] the operator asked for %.1f MB for a "
                             "transform of %lld points, which is itself worth "
                             "reporting.\n",
                             wsSize / 1048576.0, static_cast<long long>(N));
                return 1;
            }
            if (reuseWs) {
                wsShared = ws;
                wsFirstAllocUs = now_us() - t1;
            }
        }
        double t2 = now_us();

        st = aclRfft1D(ws, wsSize, executor, stream);
        if (st != ACL_SUCCESS) {
            std::fprintf(stderr, "[ERROR ] aclRfft1D failed, code %d\n",
                         static_cast<int>(st));
            return 1;
        }
        CHECK(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream");
        double t3 = now_us();

        if (ws && !reuseWs) aclrtFree(ws);

        if (it >= warmup) {
            planUs.push_back(t1 - t0);
            allocUs.push_back(t2 - t1);
            execUs.push_back(t3 - t2);
            totalUs.push_back(t3 - t0);
            std::printf("[RUN   ] %d/%d  plan_us=%9.2f  alloc_us=%9.2f  "
                        "exec_us=%9.2f  total_us=%9.2f\n",
                        it - warmup + 1, runs, t1 - t0, t2 - t1,
                        t3 - t2, t3 - t0);
        } else if (it == warmup - 1) {
            std::printf("[WARMUP] %d/%d done\n", warmup, warmup);
        }
    }

    Stats sp = summarise(planUs), sa = summarise(allocUs),
          se = summarise(execUs), stt = summarise(totalUs);

    std::printf("----------------------------------------------------------------\n");
    std::printf("           %10s %10s %8s %10s %10s %10s\n",
                "median", "mean", "sd", "min", "p90", "max");
    std::printf("PLAN:  %10.2f %10.2f %8.2f %10.2f %10.2f %10.2f\n",
                sp.median, sp.mean, sp.sd, sp.lo, sp.p90, sp.hi);
    std::printf("ALLOC: %10.2f %10.2f %8.2f %10.2f %10.2f %10.2f%s\n",
                sa.median, sa.mean, sa.sd, sa.lo, sa.p90, sa.hi,
                reuseWs ? "   (reused)" : "");
    if (reuseWs)
        std::printf("ALLOC: one-off cost of the single %.1f MB allocation: "
                    "%.2f us\n", wsSizeSeen / 1048576.0, wsFirstAllocUs);
    std::printf("EXEC:  %10.2f %10.2f %8.2f %10.2f %10.2f %10.2f\n",
                se.median, se.mean, se.sd, se.lo, se.p90, se.hi);
    std::printf("Total: %10.2f %10.2f %8.2f %10.2f %10.2f %10.2f\n",
                stt.median, stt.mean, stt.sd, stt.lo, stt.p90, stt.hi);
    if (se.mean > 1.15 * se.median)
        std::printf("[NOTE  ] the exec mean exceeds its median by %.0f%%, so a "
                    "few stalled runs are pulling it up; quote the median\n",
                    100.0 * (se.mean / se.median - 1.0));
    // Medians, so a single stalled run cannot move the headline figure.
    std::printf("[RESULT] impl=acl/rfft1d N=%lld npu_us=%.2f cpu_us=%.2f "
                "alloc_us=%.2f total_us=%.2f min_us=%.2f max_us=%.2f ws=%s "
                "stat=median\n",
                static_cast<long long>(N), se.median, sp.median, sa.median,
                stt.median, stt.lo, stt.hi, reuseWs ? "reused" : "per-call");

    // Spot-check the output so a silently wrong result is not reported as a
    // fast one.  A full verification against the project's golden file would
    // need a real-input reference, which gen_data.py does not currently
    // write; the magnitude check below catches an all-zero or NaN output.
    std::vector<float> back(static_cast<size_t>(outFloats));
    CHECK(aclrtMemcpy(back.data(), back.size() * sizeof(float), dOut,
                      back.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST),
          "aclrtMemcpy(d2h)");
    double maxAbs = 0.0;
    bool finite = true;
    for (float v : back) {
        if (!std::isfinite(v)) finite = false;
        maxAbs = std::max(maxAbs, std::fabs(static_cast<double>(v)));
    }
    std::printf("[CHECK ] output finite=%s  max|out|=%.6e  %s\n",
                finite ? "yes" : "NO", maxAbs,
                (finite && maxAbs > 0.0) ? "(non-trivial)"
                                         : "(SUSPECT: all zero or non-finite)");

    // A direct discrete transform of the same real input, computed on the
    // host in double precision, at three bins.  Three is enough to tell a
    // correct result from a scaled or reordered one, and cheap enough to do
    // inline: a full reference would cost O(N^2) here.
    {
        const int probeBins[3] = {0, 1, static_cast<int>(outComplex) / 2};
        double worstRel = 0.0;
        std::printf("[REF   ] bin        device(re,im)              "
                    "reference(re,im)\n");
        for (int b : probeBins) {
            if (b >= outComplex) continue;
            double re = 0.0, im = 0.0;
            for (int64_t t = 0; t < N; ++t) {
                double ang = -2.0 * 3.14159265358979323846 * b * t / N;
                re += host[t] * std::cos(ang);
                im += host[t] * std::sin(ang);
            }
            double dre = back[2 * b], dim_ = back[2 * b + 1];
            double mag = std::sqrt(re * re + im * im);
            double err = std::sqrt((dre - re) * (dre - re) +
                                   (dim_ - im) * (dim_ - im));
            double rel = mag > 1e-12 ? err / mag : err;
            worstRel = std::max(worstRel, rel);
            std::printf("[REF   ] %-6d (%12.4f,%12.4f)  (%12.4f,%12.4f)\n",
                        b, dre, dim_, re, im);
        }
        std::printf("[REF   ] worst relative deviation at probed bins: %.3e  %s\n",
                    worstRel,
                    worstRel < 1e-3 ? "(matches the unscaled convention)"
                                    : "(does NOT match: try the other norm)");
    }
    std::printf("================================================================\n");

    if (!csvPath.empty()) {
        FILE *f = std::fopen(csvPath.c_str(), "r");
        bool needHeader = (f == nullptr);
        if (f) std::fclose(f);
        f = std::fopen(csvPath.c_str(), "a");
        if (f) {
            if (needHeader)
                std::fprintf(f, "variant,N,tile_size,npu_avg_us,cpu_avg_us,"
                                "total_avg_us,total_min_us,total_max_us\n");
            std::fprintf(f, "acl_rfft1d_norm%lld%s,%lld,0,%.2f,%.2f,%.2f,"
                            "%.2f,%.2f\n",
                         static_cast<long long>(norm),
                         reuseWs ? "_reusews" : "",
                         static_cast<long long>(N), se.median, sp.median,
                         stt.median, stt.lo, stt.hi);
            std::fclose(f);
            std::printf("[WROTE ] %s\n", csvPath.c_str());
        }

        // The project's CSV schema has no column for the allocation stage, so
        // the three-way breakdown goes into a companion file rather than
        // bending a schema other scripts already read.
        std::string detailPath = csvPath + ".detail.csv";
        FILE *g = std::fopen(detailPath.c_str(), "r");
        bool detailHeader = (g == nullptr);
        if (g) std::fclose(g);
        g = std::fopen(detailPath.c_str(), "a");
        if (g) {
            if (detailHeader)
                std::fprintf(g, "N,norm,ws_mode,ws_bytes,runs,"
                                "plan_med_us,plan_avg_us,plan_sd_us,"
                                "alloc_med_us,alloc_avg_us,alloc_sd_us,"
                                "exec_med_us,exec_avg_us,exec_sd_us,"
                                "exec_p90_us,exec_max_us,"
                                "total_med_us,total_avg_us,total_sd_us\n");
            std::fprintf(g, "%lld,%lld,%s,%llu,%d,"
                            "%.2f,%.2f,%.2f,"
                            "%.2f,%.2f,%.2f,"
                            "%.2f,%.2f,%.2f,"
                            "%.2f,%.2f,"
                            "%.2f,%.2f,%.2f\n",
                         static_cast<long long>(N),
                         static_cast<long long>(norm),
                         reuseWs ? "reused" : "per-call",
                         static_cast<unsigned long long>(wsSizeSeen), runs,
                         sp.median, sp.mean, sp.sd,
                         sa.median, sa.mean, sa.sd,
                         se.median, se.mean, se.sd,
                         se.p90, se.hi,
                         stt.median, stt.mean, stt.sd);
            std::fclose(g);
            std::printf("[WROTE ] %s\n", detailPath.c_str());
        }
    }

    aclDestroyTensor(tIn);
    aclDestroyTensor(tOut);
    aclrtFree(dIn);
    aclrtFree(dOut);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
