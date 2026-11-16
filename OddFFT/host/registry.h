// host/registry.h  (OddFFT)
//
// The variant registry: the one place listing everything the project can do.
// main() takes its dispatch from here, the metadata for the CSV, and the
// listing for `--list` that run_matrix.sh reads — so the knowledge about the
// variants is not duplicated between C++ and bash.
//
// Registration lives in host/impl_r3.cpp / impl_r5.cpp / impl_r7.cpp: each of
// those sees its own static run-functions and only it can take their address.
#pragma once
#include "cell.h"
#include <string>
#include <vector>

using RunFn = void (*)(uint32_t N, uint32_t m, const std::vector<float> &input,
                       uint32_t warmup, uint32_t bench_runs,
                       const std::string &variant_name);

struct VariantInfo {
    const char *impl;
    const char *name;
    uint32_t    radix;
    const char *butterfly;
    const char *scheme;
    const char *granularity;
    const char *layout;
    const char *unit;
    uint32_t    terms;
    uint32_t    mm;
    uint32_t    cores;        // the declared blockDim policy, see cell.h
    bool        npu_only;
    RunFn       run;
    uint32_t  (*prewarm)(uint32_t, uint32_t);

    std::string full() const { return std::string(impl) + "/" + name; }
};

struct Registry {
    std::vector<VariantInfo> all;

    void add(const VariantInfo &v) { all.push_back(v); }

    // Takes the full name "r3/sym_psm_xs" and the short "sym_psm_xs", the
    // latter only when unambiguous — all three radices will have a sym_psm_xs.
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
                    "spell it out in full, for example r3/" + q)
                 : ("unknown variant '" + q + "'");
        }
        return nullptr;
    }
};

// One registration function, not one per radix: the kernels are generated from
// one template so the host has nothing per-radix to say. See host/impl_odd.cpp.
void register_odd(Registry &reg);

// Bluestein (any N) and Rader (prime N), each over a power-of-two inner
// transform built from this project's own plan and kernels. Separate from
// register_odd because the run-functions are static to host/impl_arblen.cpp.
void register_arblen(Registry &reg);

inline Registry build_registry()
{
    Registry reg;
    register_odd(reg);
    register_arblen(reg);
    return reg;
}
