// kernels/arb_cmul.cpp
//
// a := a * b, complex, split real/imaginary, in place over n elements.
//
// This one kernel serves every elementwise step of both arbitrary-length
// constructions:
//   Bluestein  chirp pre-multiply, frequency-domain multiply, chirp post-
//              multiply -- three uses, one kernel, because the host supplies
//              whichever table (conjugated or not) the step needs. The
//              conjugation lives in the precomputed table, not in a branch
//              here, so the kernel has no phase parameter at all.
//   Rader      the frequency-domain multiply against the transformed filter.
//
// The length is always a power of two (the inner transform's M), so every
// base and every tile length is a multiple of 8 and nothing here meets the
// unaligned-access machinery the odd-radix kernels need. DataCopyPad is used
// anyway: it costs the same as the plain copy (measured: 0.1655 vs 0.1670 us
// per 64-float copy) and keeps one transfer primitive across the project.
#include "kernel_operator.h"
#include "kernel_limits.h"
#include <cstdint>

class KernelArbCmul {
public:
    __aicore__ inline KernelArbCmul() {}

    __aicore__ inline void Init(GM_ADDR AR, GM_ADDR AI, GM_ADDR BR, GM_ADDR BI,
                                uint32_t n, uint32_t tile, uint32_t blk_num)
    {
        this->n      = n;
        this->tile   = (tile > MAX_TILE) ? MAX_TILE : (tile < 8u ? 8u : tile);
        this->blkNum = blk_num > 0 ? blk_num : 1;

        arGm.SetGlobalBuffer((__gm__ float *)AR, n);
        aiGm.SetGlobalBuffer((__gm__ float *)AI, n);
        brGm.SetGlobalBuffer((__gm__ float *)BR, n);
        biGm.SetGlobalBuffer((__gm__ float *)BI, n);

        const uint32_t bufS = TILE_F * sizeof(float);
        pipe.InitBuffer(inQ,  2, 4 * bufS);
        pipe.InitBuffer(outQ, 2, 2 * bufS);
        pipe.InitBuffer(calcBuf, 2 * bufS);
    }

    __aicore__ inline void Process()
    {
        const uint32_t nTiles = (this->n + this->tile - 1u) / this->tile;
        const uint32_t bid    = (uint32_t)AscendC::GetBlockIdx();
        for (uint32_t t = bid; t < nTiles; t += this->blkNum) {
            const uint32_t off  = t * this->tile;
            const uint32_t left = this->n - off;
            ProcessTile(off, (left < this->tile) ? left : this->tile);
        }
    }

private:
    __aicore__ inline void Load(const AscendC::LocalTensor<float> &dst,
                                const AscendC::GlobalTensor<float> &src,
                                uint32_t idx, uint32_t len)
    {
        AscendC::DataCopyExtParams ext{1, (uint32_t)(len * sizeof(float)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<float> pad{false, 0, 0, 0.0f};
        AscendC::DataCopyPad(dst, src[idx], ext, pad);
    }

    __aicore__ inline void ProcessTile(uint32_t off, uint32_t len)
    {
        auto inT = inQ.AllocTensor<float>();
        auto ar = inT[0 * TILE_F];  auto ai = inT[1 * TILE_F];
        auto br = inT[2 * TILE_F];  auto bi = inT[3 * TILE_F];
        Load(ar, arGm, off, len);  Load(ai, aiGm, off, len);
        Load(br, brGm, off, len);  Load(bi, biGm, off, len);
        inQ.EnQue(inT);

        auto in2 = inQ.DeQue<float>();
        auto a_r = in2[0 * TILE_F];  auto a_i = in2[1 * TILE_F];
        auto b_r = in2[2 * TILE_F];  auto b_i = in2[3 * TILE_F];
        auto c   = calcBuf.Get<float>();
        auto t0  = c[0 * TILE_F];    auto t1 = c[1 * TILE_F];

        auto outT = outQ.AllocTensor<float>();
        auto or_ = outT[0 * TILE_F]; auto oi = outT[1 * TILE_F];

        // (ar + i ai)(br + i bi) = (ar*br - ai*bi) + i(ar*bi + ai*br)
        AscendC::Mul(t0, a_r, b_r, len);
        AscendC::Mul(t1, a_i, b_i, len);
        AscendC::Sub(or_, t0, t1, len);
        AscendC::Mul(t0, a_r, b_i, len);
        AscendC::Mul(t1, a_i, b_r, len);
        AscendC::Add(oi, t0, t1, len);
        AscendC::PipeBarrier<PIPE_V>();

        outQ.EnQue(outT);
        inQ.FreeTensor(in2);

        auto o2 = outQ.DeQue<float>();
        AscendC::DataCopyExtParams ext{1, (uint32_t)(len * sizeof(float)), 0, 0, 0};
        AscendC::DataCopyPad(arGm[off], o2[0 * TILE_F], ext);
        AscendC::DataCopyPad(aiGm[off], o2[1 * TILE_F], ext);
        outQ.FreeTensor(o2);
    }

private:
    AscendC::TPipe pipe;
    AscendC::GlobalTensor<float> arGm, aiGm, brGm, biGm;
    static constexpr uint32_t MAX_TILE = ODDFFT_MAX_TILE;
    static constexpr uint32_t TILE_F   = MAX_TILE + 8;
    AscendC::TQue<AscendC::TPosition::VECIN,  2> inQ;
    AscendC::TQue<AscendC::TPosition::VECOUT, 2> outQ;
    AscendC::TBuf<AscendC::TPosition::VECCALC>   calcBuf;
    uint32_t n, tile, blkNum;
};

extern "C" __global__ __aicore__ void arb_cmul(
    GM_ADDR AR, GM_ADDR AI, GM_ADDR BR, GM_ADDR BI,
    uint32_t n, uint32_t tile, uint32_t blk_num)
{
    KernelArbCmul op;
    op.Init(AR, AI, BR, BI, n, tile, blk_num);
    op.Process();
}

#ifndef ASCENDC_CPU_DEBUG
void arb_cmul_do(uint32_t blockDim, void *stream,
                 void *ar, void *ai, void *br, void *bi,
                 uint32_t n, uint32_t tile, uint32_t blk_num)
{
    arb_cmul<<<blockDim, nullptr, stream>>>(ar, ai, br, bi, n, tile, blk_num);
}
#endif
