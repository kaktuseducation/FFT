// host/verify.h  (FFTBench)
//
// Verification of output_x.bin against golden.bin, right inside the binary.
//
// WHY DUPLICATE scripts/verify_result.py. The matrix has thousands of cells,
// and every one of them has to carry BOTH a time AND a verification status —
// in ONE CSV row. If a separate Python process does the verification, its
// result then has to be glued onto the timing result by a key (impl, variant,
// N, tile, batch...), and that is one more source of errors: keys drifting
// apart quietly produces a wrong table. So the binary computes the
// verification itself and writes a finished row.
//
// The verification rule REPEATS numpy.isclose, the same as in
// scripts/verify_result.py, so that the numbers stay comparable with the whole
// history accumulated before now:
//
//     a mismatch if  |out - gold| >  atol + rtol * |gold|
//     atol = rtol = 1e-3,  PASS if the mismatch fraction is <= 1e-3
//
// What gets compared is complex points (|.| — the modulus), not individual
// floats: the files are laid out as [re0, im0, re1, im1, ...].
//
// scripts/verify_result.py stays in the project unchanged — it is convenient
// for re-checking a single run by hand, and it still writes validation.csv.
#pragma once
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <vector>

static constexpr double VERIFY_RTOL      = 1e-3;
static constexpr double VERIFY_ATOL      = 1e-3;
static constexpr double VERIFY_ERR_RATIO = 1e-3;

struct VerifyStats {
    bool     ok         = false;
    bool     available  = false;   // false — the files could not be read
    uint64_t mismatches = 0;
    uint64_t points     = 0;
    double   err_ratio  = 0.0;
    double   max_abs    = 0.0;
    double   max_rel    = 0.0;
    const char *note    = "";
};

inline bool read_floats(const char *path, std::vector<float> &out)
{
    FILE *f = fopen(path, "rb");
    if (!f) return false;
    fseek(f, 0, SEEK_END);
    long bytes = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (bytes <= 0 || (bytes % (long)sizeof(float)) != 0) { fclose(f); return false; }
    out.resize((size_t)bytes / sizeof(float));
    size_t got = fread(out.data(), sizeof(float), out.size(), f);
    fclose(f);
    return got == out.size();
}

inline VerifyStats verify_against_golden(const char *outPath = "./output/output_x.bin",
                                          const char *goldPath = "./output/golden.bin")
{
    VerifyStats st;
    std::vector<float> o, g;
    if (!read_floats(outPath, o))  { st.note = "output_x.bin not read";   return st; }
    if (!read_floats(goldPath, g)) { st.note = "golden.bin not read";     return st; }
    if (o.size() != g.size())      { st.note = "sizes did not match";     return st; }

    st.available = true;
    st.points = o.size() / 2;
    for (size_t i = 0; i + 1 < o.size(); i += 2) {
        const double dr = (double)o[i]     - (double)g[i];
        const double di = (double)o[i + 1] - (double)g[i + 1];
        const double absErr  = std::sqrt(dr * dr + di * di);
        const double goldMag = std::sqrt((double)g[i] * g[i] + (double)g[i + 1] * g[i + 1]);
        if (absErr > VERIFY_ATOL + VERIFY_RTOL * goldMag) ++st.mismatches;
        if (absErr > st.max_abs) st.max_abs = absErr;
        const double rel = absErr / (goldMag > 1e-30 ? goldMag : 1e-30);
        if (rel > st.max_rel) st.max_rel = rel;
    }
    st.err_ratio = st.points ? (double)st.mismatches / (double)st.points : 0.0;
    st.ok = st.err_ratio <= VERIFY_ERR_RATIO;
    return st;
}
