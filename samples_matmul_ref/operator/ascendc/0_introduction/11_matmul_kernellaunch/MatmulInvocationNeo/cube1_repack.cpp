// kernels/cube1_repack.cpp
// R8-CUBE-FS-XS — repack/twiddle половина Vector-части combine-этапа
// cube1_fs_xs (чисто Vector/AIV, без единого обращения к Cube/Matmul API).
//
// !!! fp16-ПРАВКА (X-операнд матмула теперь half) !!!
// Изолированный тест официального Huawei-эталона (matmul_custom.cpp) на
// этом же железе показал: fp32×fp32 через AscendC Matmul high-level API
// либо считает чистые нули (на "нормальном" 512x512x1024 без дедлока),
// либо виснет насмерть (на нашем крошечном 16x16x128, видно по
// SyncTask/LogicCqReportV2 в device-логах — реальный hardware deadlock,
// а не software-баг). Официальный паттерн — A_T=half, B_T=half, C_T=float
// (fp16 входы, fp32 аккумулятор) — именно так Cube-движок физически
// рассчитан работать. Поэтому твиддл-умноженный X-операнд теперь
// считается в fp32 (точность), а перед записью в scratch-буфер для
// матмула — КАСТУЕТСЯ в half (AscendC::Cast). Небольшая потеря точности
// именно на этом шаге — осознанный trade-off, Y-результат (выход матмула)
// и весь остальной pipeline (repack-арифметика, unpack, DIT-путь) остаются
// fp32.
//
// !!! ПОЧЕМУ ЭТОТ КЕРНЕЛ В СВОЁМ СОБСТВЕННОМ ФАЙЛЕ !!!
// Структурный баг с классификацией AIC/AIV/MIX (см. историю): этот файл
// использует TQue<VECIN>+TQue<VECOUT> (без "голого" VECCALC-only финального
// райта) — комбинация, которая, в отличие от предыдущей версии с
// TBuf<VECCALC>, классифицируется компилятором чисто как AIV, без
// "_mix_aic"/"_mix_aiv" — подтверждено дизасмом пробных aic/aiv
// object-файлов (llvm-objdump + классификатор channel.py из
// extract_host_stub.py).

#include "kernel_operator.h"
#include "cube1_common.h"
#include <cstdint>

namespace {
__aicore__ inline uint32_t AlignUp8(uint32_t v)
{
    return ((v + 7) / 8) * 8;
}
}

// Твиддл-умножение (t0=x0; t1..t7=x_m*w_m) с результатом в блочно-вещественном
// layout [Re(t0..t7) строки 0..7; Im(t0..t7) строки 8..15] — ровно то, что
// build_dft8_matrix16_blockreal() ожидает от X-операнда Cube-matmul'а.
// Результат кастуется в half непосредственно перед записью в scratchXGm.
class KernelFFT8Cube1Repack {
public:
    __aicore__ inline KernelFFT8Cube1Repack() {}

    __aicore__ inline void Init(GM_ADDR XR, GM_ADDR XI, GM_ADDR W, GM_ADDR SCRATCH_X,
                                 uint32_t n, uint32_t wTotal)
    {
        xrGm.SetGlobalBuffer((__gm__ float *)XR, n);
        xiGm.SetGlobalBuffer((__gm__ float *)XI, n);
        // wTotal (не n*14): на хосте под W реально выделено только w_total
        // floats (build_fused_plan8 суммирует aligned*14 только по
        // NPU-стадиям) — несовпадение объявленного размера GM-тензора с
        // реальным выделением давало aicore exception при выходе DataCopy
        // за границы буфера.
        wGm.SetGlobalBuffer((__gm__ float *)W, wTotal);
        // SCRATCH_X — ТОЛЬКО X-операнд матмула (half), 16*CUBE_BATCH элементов
        // (Y теперь отдельный float-буфер scratchY, см. cube_fused_xs.cpp).
        scratchXGm.SetGlobalBuffer((__gm__ half *)SCRATCH_X, 16 * CUBE_BATCH);

        pipe.InitBuffer(inQ, 2, (16 + 14) * CUBE_BATCH * sizeof(float));
        pipe.InitBuffer(outQ, 2, (16 + 2) * CUBE_BATCH * sizeof(float));
        pipe.InitBuffer(castQ, 2, 16 * CUBE_BATCH * sizeof(half));
    }

    __aicore__ inline void Process(uint32_t k0, uint32_t shift, uint32_t eighth,
                                    uint32_t validB, uint32_t wOffset, uint32_t aligned)
    {
        uint32_t i[8];
        for (int m = 0; m < 8; ++m) i[m] = k0 + shift + (uint32_t)m * eighth;
        uint32_t alignedValid = AlignUp8(validB);

        CopyIn(i, shift, aligned, wOffset, alignedValid);
        Compute(validB);
    }

private:
    __aicore__ inline void CopyIn(uint32_t i[8], uint32_t shift, uint32_t aligned,
                                   uint32_t wOffset, uint32_t alignedValid)
    {
        auto inT = inQ.AllocTensor<float>();
        for (int m = 0; m < 8; ++m) {
            AscendC::DataCopy(inT[(2*m)   * CUBE_BATCH], xrGm[i[m]], alignedValid);
            AscendC::DataCopy(inT[(2*m+1) * CUBE_BATCH], xiGm[i[m]], alignedValid);
        }
        for (int m = 1; m <= 7; ++m) {
            uint32_t base = wOffset + (uint32_t)(2*m - 2) * aligned + shift;
            AscendC::DataCopy(inT[(16 + 2*(m-1))   * CUBE_BATCH], wGm[base],           alignedValid);
            AscendC::DataCopy(inT[(16 + 2*(m-1)+1) * CUBE_BATCH], wGm[base + aligned], alignedValid);
        }
        inQ.EnQue(inT);
    }

    __aicore__ inline void Compute(uint32_t validB)
    {
        auto in2 = inQ.DeQue<float>();
        auto x = [&](int m, int ri) { return in2[(2*m+ri) * CUBE_BATCH]; };
        auto w = [&](int m, int ri) { return in2[(16 + 2*(m-1)+ri) * CUBE_BATCH]; };

        auto outT = outQ.AllocTensor<float>();
        // Блочно-вещественная раскладка: Re(t_m) -> строка m, Im(t_m) -> строка 8+m.
        auto t = [&](int m, int ri) { return outT[((ri == 0) ? m : (8+m)) * CUBE_BATCH]; };
        auto u = outT[16 * CUBE_BATCH];
        auto v = outT[17 * CUBE_BATCH];

        // t0 = x0 (без твиддла)
        AscendC::Adds(t(0,0), x(0,0), 0.0f, validB);
        AscendC::Adds(t(0,1), x(0,1), 0.0f, validB);
        // t_m = x_m * w_m, m=1..7
        for (int m = 1; m <= 7; ++m) {
            AscendC::Mul(u, x(m,0), w(m,0), validB); AscendC::Mul(v, x(m,1), w(m,1), validB);
            AscendC::Sub(t(m,0), u, v, validB);
            AscendC::Mul(u, x(m,0), w(m,1), validB); AscendC::Mul(v, x(m,1), w(m,0), validB);
            AscendC::Add(t(m,1), u, v, validB);
        }
        AscendC::PipeBarrier<PIPE_V>();
        inQ.FreeTensor(in2);
        outQ.EnQue(outT);

        auto out2 = outQ.DeQue<float>();
        // Каст fp32 -> fp16 ОДНИМ вызовом на все 16*CUBE_BATCH элементов
        // (Re(t0..7) и Im(t0..7) лежат подряд, строки 0..15, по CUBE_BATCH
        // каждая — ровно 16*CUBE_BATCH контигуозных float перед u/v-scratch
        // в строках 16/17, которые сюда не входят).
        auto halfT = castQ.AllocTensor<half>();
        AscendC::Cast(halfT, out2[0], AscendC::RoundMode::CAST_RINT, 16 * CUBE_BATCH);
        castQ.EnQue(halfT);
        outQ.FreeTensor(out2);

        auto halfOut = castQ.DeQue<half>();
        for (int m = 0; m < 8; ++m) {
            AscendC::DataCopy(scratchXGm[m       * CUBE_BATCH], halfOut[m       * CUBE_BATCH], validB);
            AscendC::DataCopy(scratchXGm[(8 + m) * CUBE_BATCH], halfOut[(8 + m) * CUBE_BATCH], validB);
        }
        castQ.FreeTensor(halfOut);
    }

private:
    AscendC::TPipe pipe;
    AscendC::GlobalTensor<float> xrGm, xiGm, wGm;
    AscendC::GlobalTensor<half>  scratchXGm;
    AscendC::TQue<AscendC::TPosition::VECIN, 2>  inQ;
    AscendC::TQue<AscendC::TPosition::VECOUT, 2> outQ;
    AscendC::TQue<AscendC::TPosition::VECOUT, 2> castQ;
};

// k0/shift/eighth заменяют i[8]=k0+shift+m*eighth — компактнее протаскивать
// через launch-параметры, чем массив из 8 адресов.
extern "C" __global__ __aicore__ void fft8_cube1_repack(
    GM_ADDR XR, GM_ADDR XI, GM_ADDR W, GM_ADDR SCRATCH_X,
    uint32_t n, uint32_t k0, uint32_t shift, uint32_t eighth,
    uint32_t validB, uint32_t wOffset, uint32_t aligned, uint32_t wTotal)
{
    KernelFFT8Cube1Repack op;
    op.Init(XR, XI, W, SCRATCH_X, n, wTotal);
    op.Process(k0, shift, eighth, validB, wOffset, aligned);
}

#ifndef ASCENDC_CPU_DEBUG
void fft8_cube1_repack_do(uint32_t blockDim, void *stream,
                           void *xr, void *xi, void *w, void *scratchX,
                           uint32_t n, uint32_t k0, uint32_t shift, uint32_t eighth,
                           uint32_t validB, uint32_t wOffset, uint32_t aligned, uint32_t wTotal)
{
    fft8_cube1_repack<<<blockDim, nullptr, stream>>>(xr, xi, w, scratchX,
                                                       n, k0, shift, eighth,
                                                       validB, wOffset, aligned, wTotal);
}
#endif