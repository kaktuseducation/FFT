// host/main.cpp — AlignProbe
//
// One mode per process run (argv[1]), so a mode that faults the AI core does
// not take the other results down with it. run.sh loops over the modes.
//
// Every case prints its verdict immediately and flushes, so even a fault
// leaves a record of which case was in flight when it happened.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <string>
#include <vector>
#include "acl/acl.h"

#define CHECK(x) do { aclError _e = (x); if (_e != ACL_SUCCESS) { \
    fprintf(stderr, "[ERROR] ACL %d at %s:%d\n", (int)_e, __FILE__, __LINE__); exit(1); } } while (0)

extern void probe_contig_do(uint32_t, void *, void *, void *, uint32_t, uint32_t);
extern void probe_scalar_do(uint32_t, void *, void *, void *, uint32_t, uint32_t);
extern void probe_trunc_do(uint32_t, void *, void *, void *, uint32_t, uint32_t, uint32_t, uint32_t);
extern void probe_bench_do(uint32_t, void *, void *, void *, uint32_t, uint32_t, uint32_t, uint32_t);
#if HAVE_DATACOPYPAD
extern void probe_pad_do(uint32_t, void *, void *, void *, uint32_t, uint32_t);
extern void probe_padodd_do(uint32_t, void *, void *, void *, uint32_t, uint32_t);
extern void probe_vecunalign_do(uint32_t, void *, void *, void *, uint32_t, uint32_t);
#endif

static const uint32_t SRC_FLOATS = 1 << 16;
static const uint32_t DST_FLOATS = 1 << 14;

int main(int argc, char **argv)
{
    const std::string mode = (argc > 1) ? argv[1] : "aligned";

    CHECK(aclInit(nullptr));
    CHECK(aclrtSetDevice(0));
    aclrtStream stream; CHECK(aclrtCreateStream(&stream));

    // src[i] = i, so the expected value of every element is known exactly and
    // a wrong result says WHICH element was actually read.
    std::vector<float> host(SRC_FLOATS);
    for (uint32_t i = 0; i < SRC_FLOATS; ++i) host[i] = (float)i;
    void *srcDev, *dstDev;
    CHECK(aclrtMalloc(&srcDev, SRC_FLOATS * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK(aclrtMalloc(&dstDev, DST_FLOATS * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK(aclrtMemcpy(srcDev, SRC_FLOATS * sizeof(float), host.data(),
                      SRC_FLOATS * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE));
    std::vector<float> back(DST_FLOATS);

    auto readback = [&](uint32_t n) {
        CHECK(aclrtMemcpy(back.data(), n * sizeof(float), dstDev,
                          n * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST));
    };
    auto zero_dst = [&] {
        std::vector<float> z(DST_FLOATS, -1.0f);
        CHECK(aclrtMemcpy(dstDev, DST_FLOATS * sizeof(float), z.data(),
                          DST_FLOATS * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE));
    };

    int failures = 0;

    // ---- contiguous modes: vary the base offset, keep the length a multiple of 8
    if (mode == "aligned" || mode == "unaligned" || mode == "pad" || mode == "scalar") {
        // offs chosen as the actual odd strides a radix-3/5/7 stage produces
        const uint32_t offs[] = {0, 8, 1, 3, 5, 7, 9, 25, 27, 49, 125, 343};
        const uint32_t len = 64;
        printf("mode=%-10s len=%u floats\n", mode.c_str(), len);
        printf("  %-8s %-10s %-9s %s\n", "offset", "aligned?", "time_us", "verdict");
        for (uint32_t off : offs) {
            if (mode == "aligned" && (off % 8) != 0) continue;
            if (mode == "unaligned" && (off % 8) == 0) continue;
            zero_dst();
            printf("  %-8u %-10s ", off, (off % 8) == 0 ? "yes" : "no");
            fflush(stdout);
            auto t0 = std::chrono::high_resolution_clock::now();
            if (mode == "scalar")      probe_scalar_do(1, stream, srcDev, dstDev, off, len);
#if HAVE_DATACOPYPAD
            else if (mode == "pad")    probe_pad_do(1, stream, srcDev, dstDev, off, len);
#else
            else if (mode == "pad")  { printf("SKIP (built with HAVE_DATACOPYPAD=0)\n"); continue; }
#endif
            else                       probe_contig_do(1, stream, srcDev, dstDev, off, len);
            CHECK(aclrtSynchronizeStream(stream));
            auto t1 = std::chrono::high_resolution_clock::now();
            double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
            readback(len);
            uint32_t bad = 0; int firstBad = -1; float got = 0, want = 0;
            for (uint32_t i = 0; i < len; ++i) {
                if (back[i] != (float)(off + i)) {
                    if (firstBad < 0) { firstBad = (int)i; got = back[i]; want = (float)(off + i); }
                    ++bad;
                }
            }
            if (bad == 0) printf("%-9.1f OK\n", us);
            else {
                printf("%-9.1f WRONG: %u/%u elements; first at i=%d got %.0f want %.0f"
                       " (i.e. it read element %.0f)\n",
                       us, bad, len, firstBad, got, want, got);
                ++failures;
            }
            fflush(stdout);
        }
    }


    // ---- the odd-radix kernel pattern: byte-granular copy at an unaligned
    // base with a length that is NOT a multiple of 8, with and without an
    // unaligned vector-op offset. The two modes differ in exactly that one
    // thing, so whichever faults is the cause of aicore exception 507035 in
    // the odd-radix kernels. See the comment above the kernels in
    // kernels/probe.cpp.
    if (mode == "padodd" || mode == "vecunalign") {
#if !HAVE_DATACOPYPAD
        printf("SKIP (built with HAVE_DATACOPYPAD=0)\n");
#else
        // Bases and lengths taken from what a radix-3/5/7 stage really asks
        // for: t is a power of an odd number, so the base is unaligned and the
        // last tile of each k-block has a length that is not a multiple of 8.
        const uint32_t offs[] = {0, 1, 3, 7, 9, 27, 81, 243};
        // Lengths BELOW the 32-byte granule are in this list deliberately.
        // DataCopyPad is byte-granular by construction, but nothing had ever
        // tested it under 8 floats, and the answer decides how much of a
        // transform can run on the device at all: a radix-7 stage with t = 7
        // and a radix-5 stage with t = 5 are otherwise stranded on the host,
        // where the first measured matrix showed 80-93% of the runtime going.
        const uint32_t lens[] = {1, 2, 3, 5, 7, 8, 16, 17, 25, 51, 64, 65};
        printf("mode=%s%s\n", mode.c_str(),
               mode == "vecunalign"
                   ? "  (DataCopyPad + a Duplicate at an UNALIGNED offset)"
                   : "     (DataCopyPad only, every vector op aligned)");
        printf("  %-7s %-6s %-8s %-9s %s\n",
               "offset", "len", "len%8", "time_us", "verdict");
        for (uint32_t off : offs) {
            for (uint32_t len : lens) {
                zero_dst();
                printf("  %-7u %-6u %-8u ", off, len, len % 8);
                fflush(stdout);
                auto t0 = std::chrono::high_resolution_clock::now();
                if (mode == "padodd")
                    probe_padodd_do(1, stream, srcDev, dstDev, off, len);
                else
                    probe_vecunalign_do(1, stream, srcDev, dstDev, off, len);
                CHECK(aclrtSynchronizeStream(stream));
                auto t1 = std::chrono::high_resolution_clock::now();
                const double us =
                    std::chrono::duration<double, std::micro>(t1 - t0).count();
                readback(off + len + 8);
                uint32_t bad = 0;
                int firstBad = -1;
                float got = 0, want = 0;
                for (uint32_t i = 0; i < len; ++i) {
                    if (back[off + i] != (float)(off + i)) {
                        if (firstBad < 0) {
                            firstBad = (int)i;
                            got = back[off + i];
                            want = (float)(off + i);
                        }
                        ++bad;
                    }
                }
                // The write must not spill past `len`: that overrun is what
                // would race with the neighbouring tile on another core.
                uint32_t spill = 0;
                for (uint32_t i = len; i < len + 8; ++i)
                    if (back[off + i] != -1.0f) ++spill;

                if (bad == 0 && spill == 0) {
                    printf("%-9.1f OK\n", us);
                } else if (bad != 0) {
                    printf("%-9.1f WRONG: %u/%u; first at i=%d got %.0f want %.0f\n",
                           us, bad, len, firstBad, got, want);
                    ++failures;
                } else {
                    printf("%-9.1f SPILL: wrote %u element(s) past the end -- "
                           "this would race with the next tile\n", us, spill);
                    ++failures;
                }
                fflush(stdout);
            }
        }
#endif
    }

    // ---- the truncating power-of-two gather on an odd stride
    if (mode == "trunc") {
        printf("mode=trunc — the radix-2/4/8 gather pattern applied to an odd stride\n");
        printf("  %-6s %-4s %-9s %-9s %-11s %s\n",
               "t", "r", "perBlock", "srcStride", "truncated?", "verdict");
        struct Case { uint32_t t, r, perBlock, gCount; };
        const Case cases[] = {
            {9, 3, 8, 2}, {27, 3, 8, 2}, {25, 5, 8, 2}, {49, 7, 8, 2},
            {8, 2, 8, 2},                      // the power-of-two control: must be OK
        };
        for (const Case &c : cases) {
            const uint32_t exact = (c.r * c.t) - c.perBlock;
            const bool trunc = (exact % 8) != 0;
            zero_dst();
            printf("  %-6u %-4u %-9u %-9.3f %-11s ", c.t, c.r, c.perBlock,
                   exact / 8.0, trunc ? "YES" : "no");
            fflush(stdout);
            probe_trunc_do(1, stream, srcDev, dstDev, c.t, c.r, c.perBlock, c.gCount);
            CHECK(aclrtSynchronizeStream(stream));
            readback(c.r * 256);
            uint32_t bad = 0;
            for (uint32_t m = 0; m < c.r; ++m)
                for (uint32_t g = 0; g < c.gCount; ++g)
                    for (uint32_t i = 0; i < c.perBlock; ++i) {
                        const float want = (float)(g * c.r * c.t + m * c.t + i);
                        if (back[m * 256 + g * c.perBlock + i] != want) ++bad;
                    }
            if (bad == 0) printf("OK\n");
            else { printf("WRONG: %u of %u elements\n", bad, c.r * c.gCount * c.perBlock); ++failures; }
            fflush(stdout);
        }
    }

    // ---- throughput: the only mode whose timings mean anything -----------
    // The correctness modes above copy 64 floats per launch, which at a launch
    // cost of 20-55 us measures nothing but the launch. Here the copy is
    // repeated REP times inside one launch, so the per-copy cost comes out.
    // The number this is for: what an unaligned base costs against an aligned
    // one. If that is small, the odd-radix kernels can read operands straight
    // from their natural positions; if it is large, it is worth paying for a
    // layout that restores alignment.
    if (mode == "bench") {
        const uint32_t REP = 2000;
        struct Which { uint32_t id; const char *name; };
        const Which whichs[] = {{0, "DataCopy"}, {1, "DataCopyPad"}, {2, "scalar"}};
        const uint32_t offs[] = {0, 1, 9, 27};
        const uint32_t lens[] = {64, 512};
        printf("mode=bench  rep=%u per launch, 2 warmup launches, best of 3\n", REP);
        printf("  %-12s %-6s %-7s %-10s %-11s %s\n",
               "primitive", "len", "offset", "aligned?", "us/copy", "vs aligned");
        for (const Which &w : whichs) {
            for (uint32_t len : lens) {
                double basis = 0.0;
                for (uint32_t off : offs) {
                    if (w.id == 2 && len > 64) continue;   // scalar at 512 is pointless and slow
                    double best = 1e30;
                    for (int t = 0; t < 5; ++t) {          // 2 warmup + 3 measured
                        auto t0 = std::chrono::high_resolution_clock::now();
                        probe_bench_do(1, stream, srcDev, dstDev, off, len, REP, w.id);
                        CHECK(aclrtSynchronizeStream(stream));
                        auto t1 = std::chrono::high_resolution_clock::now();
                        if (t < 2) continue;
                        double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / REP;
                        if (us < best) best = us;
                    }
                    if (off == 0) basis = best;
                    char ratio[16];
                    if (basis > 0.0) snprintf(ratio, sizeof(ratio), "x%.3f", best / basis);
                    else             snprintf(ratio, sizeof(ratio), "-");
                    printf("  %-12s %-6u %-7u %-10s %-11.4f %s\n",
                           w.name, len, off, (off % 8) == 0 ? "yes" : "no", best, ratio);
                    fflush(stdout);
                    // sanity: the result must still be right
                    readback(len);
                    uint32_t bad = 0;
                    for (uint32_t i = 0; i < len; ++i)
                        if (back[i] != (float)(off + i) + 1.0f) ++bad;
                    if (bad) { printf("      ^ WRONG: %u of %u elements\n", bad, len); ++failures; }
                }
            }
        }
    }

    aclrtFree(srcDev); aclrtFree(dstDev);
    aclrtDestroyStream(stream);
    CHECK(aclrtResetDevice(0)); CHECK(aclFinalize());
    printf("[RESULT] mode=%s failures=%d\n", mode.c_str(), failures);
    return failures == 0 ? 0 : 3;
}
