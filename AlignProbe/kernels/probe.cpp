// kernels/probe.cpp — AlignProbe
//
// WHAT THIS ANSWERS, AND WHY IT GATES THE WHOLE ODD-RADIX PROJECT
//
// Every kernel in the power-of-two project moves butterfly operands with
//     DataCopy(dst, src, DataCopyParams{blockCount, blockLen, srcStride, dstStride})
// whose blockLen and strides are in units of 32 BYTES, i.e. 8 float. For a
// radix-r stage with parameter t (t = r^s) the operands of one butterfly sit
// t elements apart and the k-blocks sit r*t apart. For r = 2, 4, 8 those are
// powers of two, so from the first stage that matters they are multiples of 8
// and the strides are representable. For r = 3, 5, 7 they are odd at EVERY
// stage — not "slow", not "misaligned", but impossible to write down in that
// struct at all. Checked arithmetically: radix-3 has 0 representable stages
// out of 12, radix-5 0 out of 8, radix-7 0 out of 6.
//
// So the odd-radix kernels need a different primitive for data movement, and
// which ones actually work on this device (CANN 8.0.RC3, Ascend 910B4) is a
// question about the hardware and the toolkit, not about the algorithm. This
// probe asks it directly, before any FFT code is written on top of a guess.
//
// Each mode is a separate __global__ kernel and a separate process run, so a
// mode that faults the AI core does not take the other results with it.
//
//   mode 0  aligned      DataCopy of `len` floats from an ALIGNED base.
//                        The control: if this fails, the probe itself is wrong.
//   mode 1  unaligned    The same DataCopy from base `off` where off is odd.
//                        Tests whether the plain API tolerates an unaligned
//                        GM base at all.
//   mode 2  pad          DataCopyPad with byte-granular blockLen/srcStride.
//                        This is the primitive that WOULD make odd strides
//                        expressible. Compiled only when HAVE_DATACOPYPAD=1
//                        (the default) — if it does not compile on this
//                        toolkit, rebuild with -DHAVE_DATACOPYPAD=0 and the
//                        other modes still run.
//   mode 3  scalar       GetValue/SetValue element by element. Certain to be
//                        correct, certain to be slow; this is the fallback
//                        baseline and the number the other modes have to beat.
//   mode 4  truncating   The power-of-two gather applied to an ODD stride, with
//                        srcStride = (r*t - perBlock)/8 truncated by integer
//                        division — which is what a careless port of the
//                        existing kernels would compute. Included because the
//                        failure mode matters: it does not fault, it silently
//                        reads the wrong elements. Expected to produce a wrong
//                        answer, and the probe is here to prove that it does.
#include "kernel_operator.h"
#include <cstdint>

#ifndef HAVE_DATACOPYPAD
#define HAVE_DATACOPYPAD 1
#endif

constexpr uint32_t UB_FLOATS = 2048;   // 8 KB — far inside UB, not the point here

// ---- mode 0 / 1: plain DataCopy from base `off` ---------------------------
// `len` is rounded up to a multiple of 8 by the host; what varies is the base.
extern "C" __global__ __aicore__ void probe_contig(
    GM_ADDR SRC, GM_ADDR DST, uint32_t off, uint32_t len)
{
    AscendC::GlobalTensor<float> srcGm, dstGm;
    // The declared size is passed explicitly and deliberately: a GlobalTensor
    // whose declared extent disagrees with what the host actually allocated
    // gave an aicore exception in the power-of-two project the moment a
    // DataCopy ran past the real end of the buffer. Here the host allocates
    // SRC_FLOATS = 1<<16 and DST_FLOATS = 1<<14.
    srcGm.SetGlobalBuffer((__gm__ float *)SRC, 1u << 16);
    dstGm.SetGlobalBuffer((__gm__ float *)DST, 1u << 14);
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQ;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQ;
    pipe.InitBuffer(inQ, 1, UB_FLOATS * sizeof(float));
    pipe.InitBuffer(outQ, 1, UB_FLOATS * sizeof(float));

    auto in = inQ.AllocTensor<float>();
    AscendC::DataCopy(in, srcGm[off], len);
    inQ.EnQue(in);
    auto a = inQ.DeQue<float>();
    auto o = outQ.AllocTensor<float>();
    AscendC::Adds(o, a, 0.0f, len);          // touch it with the vector unit
    inQ.FreeTensor(a);
    outQ.EnQue(o);
    auto b = outQ.DeQue<float>();
    AscendC::DataCopy(dstGm[0], b, len);     // destination base is aligned
    outQ.FreeTensor(b);
}

// ---- mode 2: DataCopyPad, byte-granular ----------------------------------
#if HAVE_DATACOPYPAD
extern "C" __global__ __aicore__ void probe_pad(
    GM_ADDR SRC, GM_ADDR DST, uint32_t off, uint32_t len)
{
    AscendC::GlobalTensor<float> srcGm, dstGm;
    // The declared size is passed explicitly and deliberately: a GlobalTensor
    // whose declared extent disagrees with what the host actually allocated
    // gave an aicore exception in the power-of-two project the moment a
    // DataCopy ran past the real end of the buffer. Here the host allocates
    // SRC_FLOATS = 1<<16 and DST_FLOATS = 1<<14.
    srcGm.SetGlobalBuffer((__gm__ float *)SRC, 1u << 16);
    dstGm.SetGlobalBuffer((__gm__ float *)DST, 1u << 14);
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQ;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQ;
    pipe.InitBuffer(inQ, 1, UB_FLOATS * sizeof(float));
    pipe.InitBuffer(outQ, 1, UB_FLOATS * sizeof(float));

    auto in = inQ.AllocTensor<float>();
    // blockLen and srcStride in BYTES — this is the whole point of the probe.
    AscendC::DataCopyExtParams ext{1, (uint32_t)(len * sizeof(float)), 0, 0, 0};
    AscendC::DataCopyPadExtParams<float> pad{false, 0, 0, 0.0f};
    AscendC::DataCopyPad(in, srcGm[off], ext, pad);
    inQ.EnQue(in);
    auto a = inQ.DeQue<float>();
    auto o = outQ.AllocTensor<float>();
    AscendC::Adds(o, a, 0.0f, len);
    inQ.FreeTensor(a);
    outQ.EnQue(o);
    auto b = outQ.DeQue<float>();
    AscendC::DataCopyExtParams oext{1, (uint32_t)(len * sizeof(float)), 0, 0, 0};
    AscendC::DataCopyPad(dstGm[0], b, oext);
    outQ.FreeTensor(b);
}
#endif

// ---- mode 3: scalar, element by element ---------------------------------
// The honest fallback. No alignment constraint of any kind, and no vector unit.
extern "C" __global__ __aicore__ void probe_scalar(
    GM_ADDR SRC, GM_ADDR DST, uint32_t off, uint32_t len)
{
    AscendC::GlobalTensor<float> srcGm, dstGm;
    // The declared size is passed explicitly and deliberately: a GlobalTensor
    // whose declared extent disagrees with what the host actually allocated
    // gave an aicore exception in the power-of-two project the moment a
    // DataCopy ran past the real end of the buffer. Here the host allocates
    // SRC_FLOATS = 1<<16 and DST_FLOATS = 1<<14.
    srcGm.SetGlobalBuffer((__gm__ float *)SRC, 1u << 16);
    dstGm.SetGlobalBuffer((__gm__ float *)DST, 1u << 14);
    for (uint32_t i = 0; i < len; ++i) {
        dstGm.SetValue(i, srcGm.GetValue(off + i));
    }
}

// ---- mode 4: the truncating power-of-two gather on an odd stride ---------
// r sub-runs of `perBlock` floats each, `gCount` k-blocks apart, exactly the
// shape the existing radix-2/4/8 kernels use — but with t odd, so
// (r*t - perBlock)/8 truncates. Kept deliberately wrong to document the trap.
extern "C" __global__ __aicore__ void probe_trunc(
    GM_ADDR SRC, GM_ADDR DST, uint32_t t, uint32_t r, uint32_t perBlock, uint32_t gCount)
{
    AscendC::GlobalTensor<float> srcGm, dstGm;
    srcGm.SetGlobalBuffer((__gm__ float *)SRC, 1u << 16);
    dstGm.SetGlobalBuffer((__gm__ float *)DST, 1u << 14);
    AscendC::TPipe pipe;
    // !!! TWO QUEUES, AND THE FIRST VERSION OF THIS PROBE GOT IT WRONG !!!
    // It read into a TQue<VECIN> tensor and then wrote to GM straight out of
    // that same tensor. A VECIN queue installs MTE2 -> V ("MTE2 brings it in,
    // the vector unit reads it"); the write out to GM is MTE3, and nothing
    // installs MTE2 -> MTE3, so the store could begin before the load had
    // finished. The symptom was partial, data-dependent corruption — 24 of 32
    // elements, 40 of 48 — and it fired even on the power-of-two control case
    // whose stride needs no truncation at all, which is how it was caught: a
    // control that fails is a broken probe, not a finding.
    // This is the same class of bug as the VECIN/VECOUT race documented in
    // the power-of-two project, and the fix is the same: VECIN for the read,
    // VECOUT for the write, a vector op between them so the dependency chain
    // MTE2 -> V -> MTE3 is complete.
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQ;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQ;
    pipe.InitBuffer(inQ, 1, UB_FLOATS * sizeof(float));
    pipe.InitBuffer(outQ, 1, UB_FLOATS * sizeof(float));

    AscendC::DataCopyParams g;
    g.blockCount = (uint16_t)gCount;
    g.blockLen   = (uint16_t)(perBlock / 8);
    g.srcStride  = (uint16_t)(((r * t) - perBlock) / 8);   // <-- truncates when odd
    g.dstStride  = 0;

    auto in = inQ.AllocTensor<float>();
    for (uint32_t m = 0; m < r; ++m) {
        AscendC::DataCopy(in[m * 256], srcGm[m * t], g);
    }
    inQ.EnQue(in);
    auto a = inQ.DeQue<float>();
    auto o = outQ.AllocTensor<float>();
    AscendC::Adds(o, a, 0.0f, r * 256);
    inQ.FreeTensor(a);
    outQ.EnQue(o);
    auto b = outQ.DeQue<float>();
    for (uint32_t m = 0; m < r; ++m) {
        AscendC::DataCopy(dstGm[m * 256], b[m * 256], gCount * perBlock);
    }
    outQ.FreeTensor(b);
}

// ---- mode 5: throughput, so the primitives can actually be compared ------
// The first version of this probe timed one 64-float copy per launch. At that
// size the measurement is entirely launch overhead — 20-55 us, the same fixed
// cost the power-of-two project fitted independently — and every mode came
// out indistinguishable, including the scalar one, which cannot be right.
// This kernel repeats the copy REP times inside a single launch over a
// realistic chunk, so what comes out is a per-copy cost rather than a launch
// cost. `which` selects the primitive: 0 plain DataCopy, 1 DataCopyPad,
// 2 scalar. The question it answers is the one that matters for the design:
// what does an unaligned base actually cost against an aligned one?
extern "C" __global__ __aicore__ void probe_bench(
    GM_ADDR SRC, GM_ADDR DST, uint32_t off, uint32_t len, uint32_t rep, uint32_t which)
{
    AscendC::GlobalTensor<float> srcGm, dstGm;
    srcGm.SetGlobalBuffer((__gm__ float *)SRC, 1u << 16);
    dstGm.SetGlobalBuffer((__gm__ float *)DST, 1u << 14);
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQ;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQ;
    pipe.InitBuffer(inQ, 1, UB_FLOATS * sizeof(float));
    pipe.InitBuffer(outQ, 1, UB_FLOATS * sizeof(float));

    for (uint32_t it = 0; it < rep; ++it) {
        auto in = inQ.AllocTensor<float>();
        if (which == 2) {
            for (uint32_t i = 0; i < len; ++i) in.SetValue(i, srcGm.GetValue(off + i));
        } else if (which == 1) {
#if HAVE_DATACOPYPAD
            AscendC::DataCopyExtParams ext{1, (uint32_t)(len * sizeof(float)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<float> pad{false, 0, 0, 0.0f};
            AscendC::DataCopyPad(in, srcGm[off], ext, pad);
#else
            AscendC::DataCopy(in, srcGm[off], len);
#endif
        } else {
            AscendC::DataCopy(in, srcGm[off], len);
        }
        inQ.EnQue(in);
        auto a = inQ.DeQue<float>();
        auto o = outQ.AllocTensor<float>();
        AscendC::Adds(o, a, 1.0f, len);
        inQ.FreeTensor(a);
        outQ.EnQue(o);
        auto b = outQ.DeQue<float>();
        AscendC::DataCopy(dstGm[0], b, len);
        outQ.FreeTensor(b);
    }
}


// ---- modes 6 and 7: the ODD-RADIX KERNEL PATTERN -------------------------
//
// These two exist to settle a specific hardware failure. The odd-radix kernels
// faulted on every launch with aicore exception 507035, and two things in them
// had never been tested by this probe:
//
//   (a) DataCopyPad with a length that is NOT a multiple of 32 bytes, to and
//       from an UNALIGNED base. Mode 2 above only ever used len=64 floats
//       (256 bytes) and wrote to an aligned destination, so it did not cover
//       this.
//   (b) a VECTOR OP at an unaligned offset into a tensor -- Duplicate(t[k],
//       ...) with k not a multiple of 8 -- which the power-of-two project
//       never did, because every such offset there was a multiple of 8 by
//       construction.
//
// The two modes differ in EXACTLY (b), so whichever faults is the answer:
//   probe_padodd    (a) only
//   probe_vecunalign (a) + (b)
//
// If padodd passes and vecunalign faults, the unaligned vector offset is the
// cause and byte-granular DataCopyPad is innocent. If both fault, DataCopyPad
// itself cannot do this and the whole odd-radix data-movement design needs a
// different primitive.

// Must match SRC_FLOATS / DST_FLOATS in host/main.cpp. Named constants rather
// than two more bare 1u<<16 literals, because this probe exists to find
// off-by-one and out-of-range faults and a wrong buffer length here would look
// exactly like the fault being hunted.
static constexpr uint32_t PROBE_SRC_FLOATS = 1u << 16;
static constexpr uint32_t PROBE_DST_FLOATS = 1u << 14;

#if HAVE_DATACOPYPAD
// (a) only: byte-granular read and write at an unaligned base, exact length.
extern "C" __global__ __aicore__ void probe_padodd(
    GM_ADDR SRC, GM_ADDR DST, uint32_t off, uint32_t len)
{
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQ;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQ;
    pipe.InitBuffer(inQ, 1, 512 * sizeof(float));
    pipe.InitBuffer(outQ, 1, 512 * sizeof(float));

    AscendC::GlobalTensor<float> srcGm, dstGm;
    srcGm.SetGlobalBuffer((__gm__ float *)SRC, PROBE_SRC_FLOATS);
    dstGm.SetGlobalBuffer((__gm__ float *)DST, PROBE_DST_FLOATS);

    auto in = inQ.AllocTensor<float>();
    AscendC::DataCopyExtParams ext{1, (uint32_t)(len * sizeof(float)), 0, 0, 0};
    AscendC::DataCopyPadExtParams<float> pad{false, 0, 0, 0.0f};
    AscendC::DataCopyPad(in, srcGm[off], ext, pad);
    inQ.EnQue(in);

    auto a = inQ.DeQue<float>();
    auto o = outQ.AllocTensor<float>();
    // The vector op runs at the EXACT length, from an aligned tensor start.
    AscendC::Adds(o, a, (float)0.0, len);
    outQ.EnQue(o);
    inQ.FreeTensor(a);

    auto b = outQ.DeQue<float>();
    AscendC::DataCopyExtParams oext{1, (uint32_t)(len * sizeof(float)), 0, 0, 0};
    AscendC::DataCopyPad(dstGm[off], b, oext);
    outQ.FreeTensor(b);
}

// (a) + (b): the same, plus a Duplicate at the unaligned offset `len`, which
// is what the odd-radix kernels used to do to zero the round-up tail.
extern "C" __global__ __aicore__ void probe_vecunalign(
    GM_ADDR SRC, GM_ADDR DST, uint32_t off, uint32_t len)
{
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQ;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQ;
    pipe.InitBuffer(inQ, 1, 512 * sizeof(float));
    pipe.InitBuffer(outQ, 1, 512 * sizeof(float));

    AscendC::GlobalTensor<float> srcGm, dstGm;
    srcGm.SetGlobalBuffer((__gm__ float *)SRC, PROBE_SRC_FLOATS);
    dstGm.SetGlobalBuffer((__gm__ float *)DST, PROBE_DST_FLOATS);

    const uint32_t padded = ((len + 7u) / 8u) * 8u;

    auto in = inQ.AllocTensor<float>();
    AscendC::DataCopyExtParams ext{1, (uint32_t)(len * sizeof(float)), 0, 0, 0};
    AscendC::DataCopyPadExtParams<float> pad{false, 0, 0, 0.0f};
    AscendC::DataCopyPad(in, srcGm[off], ext, pad);
    inQ.EnQue(in);

    auto a = inQ.DeQue<float>();
    // THE SUSPECT: a vector op whose tensor starts at element `len`, which is
    // not a multiple of 8, hence not on a 32-byte boundary.
    if (padded > len) {
        AscendC::Duplicate(a[len], (float)0.0, padded - len);
        AscendC::PipeBarrier<PIPE_V>();
    }
    auto o = outQ.AllocTensor<float>();
    AscendC::Adds(o, a, (float)0.0, len);
    outQ.EnQue(o);
    inQ.FreeTensor(a);

    auto b = outQ.DeQue<float>();
    AscendC::DataCopyExtParams oext{1, (uint32_t)(len * sizeof(float)), 0, 0, 0};
    AscendC::DataCopyPad(dstGm[off], b, oext);
    outQ.FreeTensor(b);
}
#endif

#ifndef ASCENDC_CPU_DEBUG
void probe_contig_do(uint32_t bd, void *s, void *src, void *dst, uint32_t off, uint32_t len)
{ probe_contig<<<bd, nullptr, s>>>(src, dst, off, len); }
#if HAVE_DATACOPYPAD
void probe_pad_do(uint32_t bd, void *s, void *src, void *dst, uint32_t off, uint32_t len)
{ probe_pad<<<bd, nullptr, s>>>(src, dst, off, len); }
void probe_padodd_do(uint32_t bd, void *s, void *src, void *dst, uint32_t off, uint32_t len)
{ probe_padodd<<<bd, nullptr, s>>>(src, dst, off, len); }
void probe_vecunalign_do(uint32_t bd, void *s, void *src, void *dst, uint32_t off, uint32_t len)
{ probe_vecunalign<<<bd, nullptr, s>>>(src, dst, off, len); }
#endif
void probe_scalar_do(uint32_t bd, void *s, void *src, void *dst, uint32_t off, uint32_t len)
{ probe_scalar<<<bd, nullptr, s>>>(src, dst, off, len); }
void probe_trunc_do(uint32_t bd, void *s, void *src, void *dst,
                    uint32_t t, uint32_t r, uint32_t perBlock, uint32_t gCount)
{ probe_trunc<<<bd, nullptr, s>>>(src, dst, t, r, perBlock, gCount); }
void probe_bench_do(uint32_t bd, void *s, void *src, void *dst,
                    uint32_t off, uint32_t len, uint32_t rep, uint32_t which)
{ probe_bench<<<bd, nullptr, s>>>(src, dst, off, len, rep, which); }
#endif
