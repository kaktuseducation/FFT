// kernels/arb_cscale.cpp
//
// ar := ar * s ; ai := ai * s * sign, over n elements, split layout.
//
// Two uses, both inside the inverse transform, which is built out of the
// FORWARD one rather than written separately:
//     inverse(X) = conj( FFT( conj(X) ) ) / M
// so the imaginary part is negated before the forward pass (s = 1,
// sign = -1) and negated and scaled after it (s = 1/M, sign = -1).
//
// The earlier implementation did both of these on the host, which cost two
// device-to-host and two host-to-device transfers each time -- four PCIe
// round trips per inverse transform, inside the timed region. Two kernel
// launches replace them.
#include "kernel_operator.h"
#include "kernel_limits.h"
#include <cstdint>

class KernelArbCscale {
public:
    __aicore__ inline KernelArbCscale() {}

    __aicore__ inline void Init(GM_ADDR AR, GM_ADDR AI, uint32_t n,
                                uint32_t tile, uint32_t blk_num)
    {
        this->n      = n;
        this->tile   = (tile > MAX_TILE) ? MAX_TILE : (tile < 8u ? 8u : tile);
        this->blkNum = blk_num > 0 ? blk_num : 1;
        arGm.SetGlobalBuffer((__gm__ float *)AR, n);
        aiGm.SetGlobalBuffer((__gm__ float *)AI, n);
        const uint32_t bufS = TILE_F * sizeof(float);
        pipe.InitBuffer(inQ,  2, 2 * bufS);
        pipe.InitBuffer(outQ, 2, 2 * bufS);
    }

    __aicore__ inline void Process(float s, float sign)
    {
        const uint32_t nTiles = (this->n + this->tile - 1u) / this->tile;
        const uint32_t bid    = (uint32_t)AscendC::GetBlockIdx();
        for (uint32_t t = bid; t < nTiles; t += this->blkNum) {
            const uint32_t off  = t * this->tile;
            const uint32_t left = this->n - off;
            ProcessTile(off, (left < this->tile) ? left : this->tile, s, sign);
        }
    }

private:
    __aicore__ inline void ProcessTile(uint32_t off, uint32_t len, float s, float sign)
    {
        auto inT = inQ.AllocTensor<float>();
        AscendC::DataCopyExtParams ext{1, (uint32_t)(len * sizeof(float)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<float> pad{false, 0, 0, 0.0f};
        AscendC::DataCopyPad(inT[0 * TILE_F], arGm[off], ext, pad);
        AscendC::DataCopyPad(inT[1 * TILE_F], aiGm[off], ext, pad);
        inQ.EnQue(inT);

        auto in2 = inQ.DeQue<float>();
        auto outT = outQ.AllocTensor<float>();
        AscendC::Muls(outT[0 * TILE_F], in2[0 * TILE_F], s, len);
        AscendC::Muls(outT[1 * TILE_F], in2[1 * TILE_F], s * sign, len);
        AscendC::PipeBarrier<PIPE_V>();
        outQ.EnQue(outT);
        inQ.FreeTensor(in2);

        auto o2 = outQ.DeQue<float>();
        AscendC::DataCopyPad(arGm[off], o2[0 * TILE_F], ext);
        AscendC::DataCopyPad(aiGm[off], o2[1 * TILE_F], ext);
        outQ.FreeTensor(o2);
    }

private:
    AscendC::TPipe pipe;
    AscendC::GlobalTensor<float> arGm, aiGm;
    static constexpr uint32_t MAX_TILE = ODDFFT_MAX_TILE;
    static constexpr uint32_t TILE_F   = MAX_TILE + 8;
    AscendC::TQue<AscendC::TPosition::VECIN,  2> inQ;
    AscendC::TQue<AscendC::TPosition::VECOUT, 2> outQ;
    uint32_t n, tile, blkNum;
};

extern "C" __global__ __aicore__ void arb_cscale(
    GM_ADDR AR, GM_ADDR AI, uint32_t n, float s, float sign,
    uint32_t tile, uint32_t blk_num)
{
    KernelArbCscale op;
    op.Init(AR, AI, n, tile, blk_num);
    op.Process(s, sign);
}

#ifndef ASCENDC_CPU_DEBUG
void arb_cscale_do(uint32_t blockDim, void *stream, void *ar, void *ai,
                   uint32_t n, float s, float sign, uint32_t tile, uint32_t blk_num)
{
    arb_cscale<<<blockDim, nullptr, stream>>>(ar, ai, n, s, sign, tile, blk_num);
}
#endif
