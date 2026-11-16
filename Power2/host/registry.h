// host/registry.h  (FFTBench)
//
// The variant registry: the one place where everything the project can do at
// all is listed. main() takes its dispatch from here, and the metadata for the
// CSV, and the listing for `--list` that run_matrix.sh uses — that way the
// knowledge about the variants is not duplicated between C++ and bash.
//
// The registration itself lives in host/impl_r2.cpp / host/impl_r4.cpp /
// host/impl_r8.cpp: each of those files sees its own static run-functions, and
// only it can take their address.
#pragma once
#include "cell.h"
#include <string>
#include <vector>

// One signature for all the run-functions. It came out that way by history in
// all three projects, and that turned out to be a large piece of luck: the
// bodies carry over with no edits.
// The second argument is the number of stages in the radix's own base
// (log_radix(N)): in the sources it was called m, m2 and m3 respectively.
using RunFn = void (*)(uint32_t N, uint32_t m, const std::vector<float> &input,
                       uint32_t warmup, uint32_t bench_runs,
                       const std::string &variant_name);

struct VariantInfo {
    const char *impl;
    const char *name;
    uint32_t    radix;
    const char *algo;
    const char *granularity;
    const char *layout;
    const char *unit;
    uint32_t    terms;
    uint32_t    mm;           // matmuls per tile per term: 1 cube1, 2 cube2, 0 vector
    uint32_t    cores;        // the declared blockDim policy, see cell.h
    bool        npu_only;     // in CPU-debug the cube variants print a stub
    RunFn       run;
    uint32_t  (*prewarm)(uint32_t, uint32_t);

    std::string full() const { return std::string(impl) + "/" + name; }
};

struct Registry {
    std::vector<VariantInfo> all;

    void add(const VariantInfo &v) { all.push_back(v); }

    // Takes both the full name "r4/dit_fs_xs" and the short "dit_fs_xs" — the
    // latter only when it is unambiguous. Short names are convenient in manual
    // runs, the full ones are mandatory in the matrix: all three radices have a
    // dit_fs_xs.
    const VariantInfo *find(const std::string &q, std::string *err) const
    {
        const VariantInfo *hit = nullptr;
        int matches = 0;
        for (const auto &v : all) {
            if (q == v.full()) return &v;
            if (q == v.name) { hit = &v; ++matches; }
        }
        if (matches == 1) return hit;
        if (err) {
            *err = (matches > 1)
                 ? ("the name '" + q + "' is ambiguous — several radices have it, "
                    "spell it out in full, for example r4/" + q)
                 : ("unknown variant '" + q + "'");
        }
        return nullptr;
    }
};

void register_r2(Registry &reg);
void register_r4(Registry &reg);
void register_r8(Registry &reg);

inline Registry build_registry()
{
    Registry reg;
    register_r2(reg);
    register_r4(reg);
    register_r8(reg);
    return reg;
}
