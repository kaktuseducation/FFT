// kernels/dit_fused_xs.cpp
// R4-DT-FS-XS  (radix4 DIT, fused, X=Split, W=Packed-3)
//
// Та же butterfly-математика, что в dit_perstage_xs.cpp (см. формулу и
// проверку на N=4 в host/fft_host_utils.h). Отличие — цикл по стадиям
// внутри одного kernel launch с meta-буфером [quarter,aligned,wOffset] x
// numStages, и тот же PipeBarrier<PIPE_ALL>-кандидат между стадиями для
// Бага 2, что и в radix2 dit_fused_xs.cpp.

#include "kernel_operator.h"
#include <cstdint>

class KernelFFT4FusedXS {
public:
    __aicore__ inline KernelFFT4FusedXS() {}

    __aicore__ inline void Init(GM_ADDR XR, GM_ADDR XI, GM_ADDR W, GM_ADDR META,
                                 uint32_t full_size, uint32_t num_stages, uint32_t blk_num)
    {
        this->n         = full_size;
        this->tile      = 64;
        this->numStages = num_stages;
        // Шаг раздачи тайлов по блокам в многоядерном режиме. В одноядерном
        // (legacy) режиме не используется вовсе.
        this->blkNum    = blk_num > 0 ? blk_num : 1;

        uint32_t bufS = this->tile * sizeof(float);

        xrGm.SetGlobalBuffer((__gm__ float *)XR, this->n);
        xiGm.SetGlobalBuffer((__gm__ float *)XI, this->n);
        wGm.SetGlobalBuffer((__gm__ float *)W, this->n * 6);   // верхняя оценка, как в radix2
        metaGm.SetGlobalBuffer((__gm__ uint32_t *)META, this->numStages * 3);

        // ВАЖНО: одна входная и одна выходная очередь (а не 14 + 8).
        // Число hardware event ID ограничено (8 на пару конвейеров), при большем
        // числе TQue события пересекаются -> deadlock в EnQue/DeQue (зависание).
        // Все тензоры лежат в одном буфере и нарезаются смещениями по TILE_F float.
        pipe.InitBuffer(inQ,  2, NIN_T  * bufS);
        pipe.InitBuffer(outQ, 2, NOUT_T * bufS);
        pipe.InitBuffer(calcBuf, NCALC_T * bufS);
    }

    // ========================================================================
    // ДВА РЕЖИМА РАБОТЫ
    //
    // Process() — как было: все стадии внутри ОДНОГО launch, одно ядро.
    // Сохранён дословно, чтобы вариант r4/dit_fs_xs остался побитово тем же и
    // его числа можно было сравнивать с уже снятой матрицей.
    //
    // ProcessOneStage(stage) — многоядерный режим: ОДНА стадия за launch,
    // тайлы стадии раздаются по блокам (ядрам) шагом blkNum. Разнести стадии
    // по отдельным launch'ам здесь обязательно: стадия s+1 читает то, что
    // стадия s записала по ВСЕМУ массиву, то есть между ними нужен барьер,
    // видимый всем ядрам. Ручной cross-core барьер внутри кернела — источник
    // зависаний, а порядок между launch'ами на одном stream даётся бесплатно
    // семантикой самого stream. Тот же приём, что в r8_dit_fused_xs.cpp.
    //
    // Почему раздача тайлов по ядрам корректна: тайл (k, shift) читает и
    // пишет ровно свои четыре отрезка i0..i3 длиной valid, а отрезки разных
    // тайлов не пересекаются (shift'ы режут [0, quarter) на куски по tile,
    // k-блоки идут подряд). Межтайловых зависимостей внутри стадии нет.
    // ВНИМАНИЕ на будущее: хвостовой трюк в CopyOut (дочитывание GM при
    // alignedValid > curValid) записывает ЧУЖИЕ элементы обратно. Сейчас он
    // не срабатывает никогда, потому что quarter — степень четвёрки не ниже
    // 16, а 16 и 64 кратны восьми. Если появится батчинг по k-блокам и
    // quarter станет меньше 8, этот трюк превратится в гонку между ядрами.
    // ========================================================================
    __aicore__ inline void Process()
    {
        for (uint32_t s = 0; s < this->numStages; ++s) {
            uint32_t quarter = metaGm.GetValue(s * 3 + 0);
            uint32_t aligned = metaGm.GetValue(s * 3 + 1);
            uint32_t wOffset = metaGm.GetValue(s * 3 + 2);
            uint32_t block   = quarter << 2;

            ProcessStage(quarter, aligned, wOffset, block);

            // --- КАНДИДАТ-ФИКС для Бага 2, тот же принцип что в radix2 ---
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }

    __aicore__ inline void ProcessOneStage(uint32_t stage)
    {
        uint32_t quarter = metaGm.GetValue(stage * 3 + 0);
        uint32_t aligned = metaGm.GetValue(stage * 3 + 1);
        uint32_t wOffset = metaGm.GetValue(stage * 3 + 2);
        uint32_t block   = quarter << 2;

        uint32_t shiftsPerBlk = (quarter + this->tile - 1) / this->tile;
        uint32_t nTiles       = (this->n / block) * shiftsPerBlk;
        uint32_t bid          = (uint32_t)AscendC::GetBlockIdx();

        for (uint32_t tid = bid; tid < nTiles; tid += this->blkNum) {
            uint32_t k     = (tid / shiftsPerBlk) * block;
            uint32_t shift = (tid % shiftsPerBlk) * this->tile;
            ProcessTile(quarter, aligned, wOffset, k, shift);
        }
    }

private:
    __aicore__ inline void ProcessStage(uint32_t quarter, uint32_t aligned, uint32_t wOffset, uint32_t block)
    {
        for (uint32_t k = 0; k < this->n; k += block) {
            for (uint32_t shift = 0; shift < quarter; shift += this->tile) {
                ProcessTile(quarter, aligned, wOffset, k, shift);
            }
        }
    }

    // Тело бывшего внутреннего цикла ProcessStage, вынесенное как есть —
    // общее для обоих режимов, чтобы математика была заведомо одна и та же.
    __aicore__ inline void ProcessTile(uint32_t quarter, uint32_t aligned, uint32_t wOffset,
                                        uint32_t k, uint32_t shift)
    {
        uint32_t i0 = k + shift, i1 = i0 + quarter, i2 = i1 + quarter, i3 = i2 + quarter;
        uint32_t valid = (quarter - shift < this->tile) ? (quarter - shift) : this->tile;

        this->curValid       = valid;
        this->alignedValid   = ((valid + 7)  / 8)  * 8;
        this->computeAligned = ((valid + 63) / 64) * 64;

        CopyIn(i0, i1, i2, i3, shift, aligned, wOffset);
        Compute();
        CopyOut(i0, i1, i2, i3);
    }

    __aicore__ inline void CopyIn(uint32_t i0, uint32_t i1, uint32_t i2, uint32_t i3, uint32_t shift,
                                   uint32_t aligned, uint32_t wOffset)
    {
        auto inT = inQ.AllocTensor<float>();
        auto x0r = inT[0 * TILE_F];
        auto x0i = inT[1 * TILE_F];
        auto x1r = inT[2 * TILE_F];
        auto x1i = inT[3 * TILE_F];
        auto x2r = inT[4 * TILE_F];
        auto x2i = inT[5 * TILE_F];
        auto x3r = inT[6 * TILE_F];
        auto x3i = inT[7 * TILE_F];

        AscendC::DataCopy(x0r, xrGm[i0], this->alignedValid); AscendC::DataCopy(x0i, xiGm[i0], this->alignedValid);
        AscendC::DataCopy(x1r, xrGm[i1], this->alignedValid); AscendC::DataCopy(x1i, xiGm[i1], this->alignedValid);
        AscendC::DataCopy(x2r, xrGm[i2], this->alignedValid); AscendC::DataCopy(x2i, xiGm[i2], this->alignedValid);
        AscendC::DataCopy(x3r, xrGm[i3], this->alignedValid); AscendC::DataCopy(x3i, xiGm[i3], this->alignedValid);

        auto w1r = inT[8 * TILE_F];
        auto w1i = inT[9 * TILE_F];
        auto w2r = inT[10 * TILE_F];
        auto w2i = inT[11 * TILE_F];
        auto w3r = inT[12 * TILE_F];
        auto w3i = inT[13 * TILE_F];

        AscendC::DataCopy(w1r, wGm[wOffset + shift],                     this->alignedValid);
        AscendC::DataCopy(w1i, wGm[wOffset + aligned + shift],           this->alignedValid);
        AscendC::DataCopy(w2r, wGm[wOffset + 2*aligned + shift],         this->alignedValid);
        AscendC::DataCopy(w2i, wGm[wOffset + 3*aligned + shift],         this->alignedValid);
        AscendC::DataCopy(w3r, wGm[wOffset + 4*aligned + shift],         this->alignedValid);
        AscendC::DataCopy(w3i, wGm[wOffset + 5*aligned + shift],         this->alignedValid);


        inQ.EnQue(inT);
    }

    __aicore__ inline void Compute()
    {
        auto inT = inQ.DeQue<float>();
        auto x0r = inT[0 * TILE_F];
        auto x0i = inT[1 * TILE_F];
        auto x1r = inT[2 * TILE_F];
        auto x1i = inT[3 * TILE_F];
        auto x2r = inT[4 * TILE_F];
        auto x2i = inT[5 * TILE_F];
        auto x3r = inT[6 * TILE_F];
        auto x3i = inT[7 * TILE_F];
        auto w1r = inT[8 * TILE_F];
        auto w1i = inT[9 * TILE_F];
        auto w2r = inT[10 * TILE_F];
        auto w2i = inT[11 * TILE_F];
        auto w3r = inT[12 * TILE_F];
        auto w3i = inT[13 * TILE_F];

        auto calcT = calcBuf.Get<float>();
        auto t1r = calcT[0 * TILE_F];
        auto t1i = calcT[1 * TILE_F];
        auto t2r = calcT[2 * TILE_F];
        auto t2i = calcT[3 * TILE_F];
        auto t3r = calcT[4 * TILE_F];
        auto t3i = calcT[5 * TILE_F];
        auto u = calcT[6 * TILE_F];
        auto v = calcT[7 * TILE_F];

        if (this->computeAligned > this->alignedValid) {
            uint32_t tail = this->computeAligned - this->alignedValid;
            AscendC::Duplicate(w1r[this->alignedValid], (float)0.0f, tail);
            AscendC::Duplicate(w1i[this->alignedValid], (float)0.0f, tail);
            AscendC::Duplicate(w2r[this->alignedValid], (float)0.0f, tail);
            AscendC::Duplicate(w2i[this->alignedValid], (float)0.0f, tail);
            AscendC::Duplicate(w3r[this->alignedValid], (float)0.0f, tail);
            AscendC::Duplicate(w3i[this->alignedValid], (float)0.0f, tail);
            AscendC::PipeBarrier<PIPE_V>();
        }
        uint32_t len = this->computeAligned;

        AscendC::Mul(u, x1r, w1r, len); AscendC::Mul(v, x1i, w1i, len); AscendC::Sub(t1r, u, v, len);
        AscendC::Mul(u, x1r, w1i, len); AscendC::Mul(v, x1i, w1r, len); AscendC::Add(t1i, u, v, len);
        AscendC::Mul(u, x2r, w2r, len); AscendC::Mul(v, x2i, w2i, len); AscendC::Sub(t2r, u, v, len);
        AscendC::Mul(u, x2r, w2i, len); AscendC::Mul(v, x2i, w2r, len); AscendC::Add(t2i, u, v, len);
        AscendC::Mul(u, x3r, w3r, len); AscendC::Mul(v, x3i, w3i, len); AscendC::Sub(t3r, u, v, len);
        AscendC::Mul(u, x3r, w3i, len); AscendC::Mul(v, x3i, w3r, len); AscendC::Add(t3i, u, v, len);
        AscendC::PipeBarrier<PIPE_V>();

        auto outT = outQ.AllocTensor<float>();
        auto y0r = outT[0 * TILE_F];
        auto y0i = outT[1 * TILE_F];
        auto y1r = outT[2 * TILE_F];
        auto y1i = outT[3 * TILE_F];
        auto y2r = outT[4 * TILE_F];
        auto y2i = outT[5 * TILE_F];
        auto y3r = outT[6 * TILE_F];
        auto y3i = outT[7 * TILE_F];

        AscendC::Add(y0r, x0r, t1r, len); AscendC::Add(y0r, y0r, t2r, len); AscendC::Add(y0r, y0r, t3r, len);
        AscendC::Add(y0i, x0i, t1i, len); AscendC::Add(y0i, y0i, t2i, len); AscendC::Add(y0i, y0i, t3i, len);
        AscendC::Add(y1r, x0r, t1i, len); AscendC::Sub(y1r, y1r, t2r, len); AscendC::Sub(y1r, y1r, t3i, len);
        AscendC::Sub(y1i, x0i, t1r, len); AscendC::Sub(y1i, y1i, t2i, len); AscendC::Add(y1i, y1i, t3r, len);
        AscendC::Sub(y2r, x0r, t1r, len); AscendC::Add(y2r, y2r, t2r, len); AscendC::Sub(y2r, y2r, t3r, len);
        AscendC::Sub(y2i, x0i, t1i, len); AscendC::Add(y2i, y2i, t2i, len); AscendC::Sub(y2i, y2i, t3i, len);
        AscendC::Sub(y3r, x0r, t1i, len); AscendC::Sub(y3r, y3r, t2r, len); AscendC::Add(y3r, y3r, t3i, len);
        AscendC::Add(y3i, x0i, t1r, len); AscendC::Sub(y3i, y3i, t2i, len); AscendC::Sub(y3i, y3i, t3r, len);
        AscendC::PipeBarrier<PIPE_V>();

        outQ.EnQue(outT);
        inQ.FreeTensor(inT);
    }

    __aicore__ inline void CopyOut(uint32_t i0, uint32_t i1, uint32_t i2, uint32_t i3)
    {
        auto outT = outQ.DeQue<float>();
        auto y0r = outT[0 * TILE_F];
        auto y0i = outT[1 * TILE_F];
        auto y1r = outT[2 * TILE_F];
        auto y1i = outT[3 * TILE_F];
        auto y2r = outT[4 * TILE_F];
        auto y2i = outT[5 * TILE_F];
        auto y3r = outT[6 * TILE_F];
        auto y3i = outT[7 * TILE_F];

        if (this->alignedValid > this->curValid) {
            for (uint32_t i = this->curValid; i < this->alignedValid; i++) {
                y0r.SetValue(i, xrGm.GetValue(i0 + i)); y0i.SetValue(i, xiGm.GetValue(i0 + i));
                y1r.SetValue(i, xrGm.GetValue(i1 + i)); y1i.SetValue(i, xiGm.GetValue(i1 + i));
                y2r.SetValue(i, xrGm.GetValue(i2 + i)); y2i.SetValue(i, xiGm.GetValue(i2 + i));
                y3r.SetValue(i, xrGm.GetValue(i3 + i)); y3i.SetValue(i, xiGm.GetValue(i3 + i));
            }
        }

        AscendC::DataCopy(xrGm[i0], y0r, this->alignedValid); AscendC::DataCopy(xiGm[i0], y0i, this->alignedValid);
        AscendC::DataCopy(xrGm[i1], y1r, this->alignedValid); AscendC::DataCopy(xiGm[i1], y1i, this->alignedValid);
        AscendC::DataCopy(xrGm[i2], y2r, this->alignedValid); AscendC::DataCopy(xiGm[i2], y2i, this->alignedValid);
        AscendC::DataCopy(xrGm[i3], y3r, this->alignedValid); AscendC::DataCopy(xiGm[i3], y3i, this->alignedValid);

        outQ.FreeTensor(outT);
    }

private:
    AscendC::TPipe pipe;
    AscendC::GlobalTensor<float>    xrGm, xiGm, wGm;
    AscendC::GlobalTensor<uint32_t> metaGm;

    static constexpr uint32_t TILE_F  = 64;   // float на один тензор-срез (= tile)
    static constexpr uint32_t NIN_T   = 14;   // x0..x3 (re,im) + w1..w3 (re,im)
    static constexpr uint32_t NOUT_T  = 8;    // y0..y3 (re,im)
    static constexpr uint32_t NCALC_T = 8;    // t1..t3 (re,im) + u + v

    AscendC::TQue<AscendC::TPosition::VECIN,  2> inQ;
    AscendC::TQue<AscendC::TPosition::VECOUT, 2> outQ;
    AscendC::TBuf<AscendC::TPosition::VECCALC>   calcBuf;

    uint32_t n, tile, numStages;
    uint32_t blkNum;                 // шаг раздачи тайлов по блокам (многояд. режим)
    uint32_t curValid, alignedValid, computeAligned;
};

// stage == FFT4_ALL_STAGES — одноядерный legacy-режим: кернел сам проходит все
// стадии за один launch (вариант r4/dit_fs_xs, как было до многоядерности).
// Любое другое значение — многоядерный режим: считается ровно эта стадия,
// тайлы раздаются по блокам (вариант r4/dit_fsm_xs).
//
// Сделано одним кернелом с двумя режимами, а не копией файла, намеренно:
// математика (CopyIn/Compute/CopyOut) обязана быть у обоих вариантов
// буквально одной и той же, иначе сравнение "одно ядро против сорока"
// перестаёт быть сравнением числа ядер.
#define FFT4_ALL_STAGES 0xFFFFFFFFu

extern "C" __global__ __aicore__ void fft4_dit_fused_xs(
    GM_ADDR XR, GM_ADDR XI, GM_ADDR W, GM_ADDR META,
    uint32_t full_size, uint32_t num_stages, uint32_t stage, uint32_t blk_num)
{
    KernelFFT4FusedXS op;
    op.Init(XR, XI, W, META, full_size, num_stages, blk_num);
    if (stage == FFT4_ALL_STAGES) {
        op.Process();
    } else {
        op.ProcessOneStage(stage);
    }
}

#ifndef ASCENDC_CPU_DEBUG
void fft4_dit_fused_xs_do(uint32_t blockDim, void *stream,
                           void *xr, void *xi, void *w, void *meta,
                           uint32_t full_size, uint32_t num_stages,
                           uint32_t stage, uint32_t blk_num)
{
    fft4_dit_fused_xs<<<blockDim, nullptr, stream>>>(xr, xi, w, meta,
                                                      full_size, num_stages, stage, blk_num);
}
#endif