// host/kernel_decls.h  (FFTBench)
//
// Объявления ВСЕХ кернелов проекта. Собрано из трёх kernel_decls.h исходных
// проектов простой склейкой: имена кернелов изначально не пересекались
// (префиксы fft_ / fft4_ / fft8_), поэтому конфликтов нет и править ничего
// не потребовалось. Исторические комментарии сохранены — в них описаны
// грабли, на которые мы уже наступали.
#pragma once
#include <cstdint>

#ifdef ASCENDC_CPU_DEBUG
#include "kernel_operator.h"
#include "tikicpulib.h"
#endif

// ###########################################################################
// ##  radix-2  (r2/Radix2/host/kernel_decls.h)
// ###########################################################################
// ---- dit_perstage_xs.cpp ----
#ifdef ASCENDC_CPU_DEBUG
extern "C" __global__ __aicore__ void fft_hybrid(
    GM_ADDR XR, GM_ADDR XI, GM_ADDR W,
    uint32_t full_size, uint32_t half_size, uint32_t block_size, uint32_t aligned);
#endif
#ifndef ASCENDC_CPU_DEBUG
void fft_hybrid_do(uint32_t blockDim, void *stream,
    void *xr, void *xi, void *w,
    uint32_t full_size, uint32_t half_size, uint32_t block_size, uint32_t aligned);
#endif

// ---- dit_perstage_xi.cpp ----
#ifdef ASCENDC_CPU_DEBUG
extern "C" __global__ __aicore__ void fft_interleaved(
    GM_ADDR X, GM_ADDR W,
    uint32_t full_size, uint32_t half_size, uint32_t block_size, uint32_t aligned);
#endif
#ifndef ASCENDC_CPU_DEBUG
void fft_interleaved_do(uint32_t blockDim, void *stream,
    void *x, void *w,
    uint32_t full_size, uint32_t half_size, uint32_t block_size, uint32_t aligned);
#endif

// ---- dit_fused_xs.cpp ----
#ifdef ASCENDC_CPU_DEBUG
extern "C" __global__ __aicore__ void fft_dit_fused_xs(
    GM_ADDR XR, GM_ADDR XI, GM_ADDR W, GM_ADDR META,
    uint32_t full_size, uint32_t num_stages, uint32_t stage, uint32_t blk_num);
#endif
// ПРАВКА (многоядерность radix-2): stage и blk_num, как у radix-4 и radix-8.
//   stage == 0xFFFFFFFFu — одноядерный legacy-режим (вариант r2/dit_fs_xs),
//     числа остались побитово теми же;
//   иначе — ровно эта стадия, тайлы по блокам (вариант r2/dit_fsm_xs).
#ifndef ASCENDC_CPU_DEBUG
void fft_dit_fused_xs_do(uint32_t blockDim, void *stream,
    void *xr, void *xi, void *w, void *meta,
    uint32_t full_size, uint32_t num_stages, uint32_t stage, uint32_t blk_num);
#endif

// ---- dit_fused_xi.cpp ----
#ifdef ASCENDC_CPU_DEBUG
extern "C" __global__ __aicore__ void fft_dit_fused_xi(
    GM_ADDR X, GM_ADDR W, GM_ADDR META,
    uint32_t full_size, uint32_t num_stages);
#endif
#ifndef ASCENDC_CPU_DEBUG
void fft_dit_fused_xi_do(uint32_t blockDim, void *stream,
    void *x, void *w, void *meta,
    uint32_t full_size, uint32_t num_stages);
#endif

// ---- stockham_perstage_xs.cpp ----
#ifdef ASCENDC_CPU_DEBUG
extern "C" __global__ __aicore__ void fft_stockham_xs(
    GM_ADDR XR_IN, GM_ADDR XI_IN, GM_ADDR XR_OUT, GM_ADDR XI_OUT, GM_ADDR W,
    uint32_t full_size, uint32_t half_size, uint32_t aligned);
#endif
#ifndef ASCENDC_CPU_DEBUG
void fft_stockham_xs_do(uint32_t blockDim, void *stream,
    void *xr_in, void *xi_in, void *xr_out, void *xi_out, void *w,
    uint32_t full_size, uint32_t half_size, uint32_t aligned);
#endif

// ---- stockham_perstage_xi.cpp ----
#ifdef ASCENDC_CPU_DEBUG
extern "C" __global__ __aicore__ void fft_stockham_xi(
    GM_ADDR X_IN, GM_ADDR X_OUT, GM_ADDR W,
    uint32_t full_size, uint32_t half_size, uint32_t aligned);
#endif
#ifndef ASCENDC_CPU_DEBUG
void fft_stockham_xi_do(uint32_t blockDim, void *stream,
    void *x_in, void *x_out, void *w,
    uint32_t full_size, uint32_t half_size, uint32_t aligned);
#endif

// ---- stockham_fused_xs.cpp ----
#ifdef ASCENDC_CPU_DEBUG
extern "C" __global__ __aicore__ void fft_stockham_fused_xs(
    GM_ADDR XR0, GM_ADDR XI0, GM_ADDR XR1, GM_ADDR XI1, GM_ADDR W, GM_ADDR META,
    uint32_t full_size, uint32_t num_stages);
#endif
#ifndef ASCENDC_CPU_DEBUG
void fft_stockham_fused_xs_do(uint32_t blockDim, void *stream,
    void *xr0, void *xi0, void *xr1, void *xi1, void *w, void *meta,
    uint32_t full_size, uint32_t num_stages);
#endif

// ---- stockham_fused_xi.cpp ----
#ifdef ASCENDC_CPU_DEBUG
extern "C" __global__ __aicore__ void fft_stockham_fused_xi(
    GM_ADDR X0, GM_ADDR X1, GM_ADDR W, GM_ADDR META,
    uint32_t full_size, uint32_t num_stages);
#endif
#ifndef ASCENDC_CPU_DEBUG
void fft_stockham_fused_xi_do(uint32_t blockDim, void *stream,
    void *x0, void *x1, void *w, void *meta,
    uint32_t full_size, uint32_t num_stages);
#endif

// ###########################################################################
// ##  radix-4  (r4/Radix4/host/kernel_decls.h)
// ###########################################################################
// ---- dit_perstage_xs.cpp ----
#ifdef ASCENDC_CPU_DEBUG
extern "C" __global__ __aicore__ void fft4_hybrid(
    GM_ADDR XR, GM_ADDR XI, GM_ADDR W,
    uint32_t full_size, uint32_t quarter_size, uint32_t block_size, uint32_t aligned);
#endif
#ifndef ASCENDC_CPU_DEBUG
void fft4_hybrid_do(uint32_t blockDim, void *stream,
    void *xr, void *xi, void *w,
    uint32_t full_size, uint32_t quarter_size, uint32_t block_size, uint32_t aligned);
#endif

// ---- dit_fused_xs.cpp ----
#ifdef ASCENDC_CPU_DEBUG
extern "C" __global__ __aicore__ void fft4_dit_fused_xs(
    GM_ADDR XR, GM_ADDR XI, GM_ADDR W, GM_ADDR META,
    uint32_t full_size, uint32_t num_stages, uint32_t stage, uint32_t blk_num);
#endif
// ПРАВКА (многоядерность radix-4): добавлены stage и blk_num.
//   stage == 0xFFFFFFFFu — одноядерный legacy-режим, кернел сам проходит все
//     стадии за один launch. Это вариант r4/dit_fs_xs, его числа остались
//     побитово теми же, что до правки.
//   иначе — многоядерный режим: ровно эта стадия, тайлы по блокам шагом
//     blk_num (= blockDim). Это вариант r4/dit_fsm_xs.
// Подробное обоснование, почему стадии разнесены по launch'ам, — в шапке
// двух Process-методов в kernels/r4_dit_fused_xs.cpp.
#ifndef ASCENDC_CPU_DEBUG
void fft4_dit_fused_xs_do(uint32_t blockDim, void *stream,
    void *xr, void *xi, void *w, void *meta,
    uint32_t full_size, uint32_t num_stages, uint32_t stage, uint32_t blk_num);
#endif

// ---- stockham_perstage_xs.cpp ----
#ifdef ASCENDC_CPU_DEBUG
extern "C" __global__ __aicore__ void fft4_stockham_xs(
    GM_ADDR XR_IN, GM_ADDR XI_IN, GM_ADDR XR_OUT, GM_ADDR XI_OUT, GM_ADDR W,
    uint32_t full_size, uint32_t quarter_size, uint32_t aligned);
#endif
#ifndef ASCENDC_CPU_DEBUG
void fft4_stockham_xs_do(uint32_t blockDim, void *stream,
    void *xr_in, void *xi_in, void *xr_out, void *xi_out, void *w,
    uint32_t full_size, uint32_t quarter_size, uint32_t aligned);
#endif

// ---- stockham_fused_xs.cpp ----
#ifdef ASCENDC_CPU_DEBUG
extern "C" __global__ __aicore__ void fft4_stockham_fused_xs(
    GM_ADDR XR0, GM_ADDR XI0, GM_ADDR XR1, GM_ADDR XI1, GM_ADDR W, GM_ADDR META,
    uint32_t full_size, uint32_t num_stages);
#endif
#ifndef ASCENDC_CPU_DEBUG
void fft4_stockham_fused_xs_do(uint32_t blockDim, void *stream,
    void *xr0, void *xi0, void *xr1, void *xi1, void *w, void *meta,
    uint32_t full_size, uint32_t num_stages);
#endif

// ###########################################################################
// ##  radix-8  (live/Radix8/host/kernel_decls.h)
// ###########################################################################
// ---- kernels/dit_fused_xs.cpp : dit_fs_xs (Vector/AIV, real radix-8 DIT) ----
#ifdef ASCENDC_CPU_DEBUG
extern "C" __global__ __aicore__ void fft8_dit_fused_xs(
    GM_ADDR XR, GM_ADDR XI, GM_ADDR W, GM_ADDR META,
    uint32_t full_size, uint32_t num_stages, uint32_t stage, uint32_t blk_num);
#endif
// ПРАВКА (многоядерность): один вызов = ОДНА стадия (аргумент stage), а не
// весь FFT. blk_num — то же число, что передано в blockDim: кернел раздаёт
// тайлы стадии по ядрам с этим шагом. Подробности и обоснование, почему
// стадии разнесены по launch'ам, — в шапке Process() в
// kernels/dit_fused_xs.cpp.
#ifndef ASCENDC_CPU_DEBUG
void fft8_dit_fused_xs_do(uint32_t blockDim, void *stream,
    void *xr, void *xi, void *w, void *meta,
    uint32_t full_size, uint32_t num_stages,
    uint32_t stage, uint32_t blk_num);
#endif

// ---- kernels/cube1_repack.cpp + kernels/cube1_unpack.cpp : cube1_fs_xs,
// Vector/AIV половина (repack/twiddle + unpack) ----
// (прежний общий файл kernels/cube1_repack_unpack.cpp разделён на два — по
// одному __global__-кернелу на файл; если он всё ещё лежит в kernels/, его
// надо удалить: это мёртвый дубль с теми же именами кернелов.)
// ПРАВКА: раньше были объявлены через ACLRT_LAUNCH_KERNEL + автогенерируемые
// aclrtlaunch_fft8_cube1_repack.h/unpack.h — тот же механизм, которым
// ОБЯЗАН запускаться fft8_cube1_matmul (см. ниже). На практике именно этот
// путь запуска для чистых Vector-кернелов давал детерминированный aicore
// exception (0x7150026, "unaligned UUB"), не зависящий от содержимого тела
// кернела (проверено тремя разными переписываниями). Единственный во всём
// проекте проверенно рабочий способ запуска AIV-кернела — ручной
// _do()-wrapper с <<<blockDim,nullptr,stream>>> (как у fft8_dit_fused_xs_do
// выше) — переведены на него же, см. kernels/cube1_repack_unpack.cpp.
// ПРАВКА (батчинг по k-блокам): вместо (shift, validB) теперь
// (shift0, gCount, perBlock) — батч собирается из gCount подряд идущих
// k-блоков по perBlock бабочек из каждого, validB = gCount*perBlock.
// Твиддлы берутся из отдельного cube-буфера (wc) с раскладкой под этот
// батчинг: wcOffset/wcRowLen — смещение и длина строки стадии в нём
// (build_cube_twiddle_plan8 в host/fft_host_utils.h).
// terms — число слагаемых схемы восстановления точности (1/2/3), см.
// kernels/cube_fused_xs.cpp: repack пишет X_lo только при terms>=2, unpack
// складывает соответствующее число Y-блоков.
#ifdef ASCENDC_CPU_DEBUG
extern "C" __global__ __aicore__ void fft8_cube1_repack(
    GM_ADDR XR, GM_ADDR XI, GM_ADDR WC, GM_ADDR SCRATCH_X,
    uint32_t n, uint32_t tileBase, uint32_t kStride, uint32_t shiftStride,
    uint32_t shiftsPerBlk, uint32_t eighth, uint32_t gCount, uint32_t perBlock,
    uint32_t wcOffset, uint32_t wcRowLen, uint32_t wcTotal, uint32_t terms);
extern "C" __global__ __aicore__ void fft8_cube1_unpack(
    GM_ADDR XR, GM_ADDR XI, GM_ADDR SCRATCH_Y,
    uint32_t n, uint32_t tileBase, uint32_t kStride, uint32_t shiftStride,
    uint32_t shiftsPerBlk, uint32_t eighth, uint32_t gCount, uint32_t perBlock,
    uint32_t terms);
#endif
#ifndef ASCENDC_CPU_DEBUG
// ПРАВКА (многоядерность Cube): один вызов = ГРУППА тайлов стадии, по одному
// тайлу на блок, blockDim = число тайлов в группе. Вместо готовых (k0, shift0)
// передаётся правило перечисления (tileBase + strides), по которому блок
// находит свой тайл из blockIdx; формула и инварианты — в шапке Process()
// в kernels/cube1_repack.cpp. Слот в scratch-буферах = blockIdx.
void fft8_cube1_repack_do(uint32_t blockDim, void *stream,
    void *xr, void *xi, void *wc, void *scratchX,
    uint32_t n, uint32_t tileBase, uint32_t kStride, uint32_t shiftStride,
    uint32_t shiftsPerBlk, uint32_t eighth, uint32_t gCount, uint32_t perBlock,
    uint32_t wcOffset, uint32_t wcRowLen, uint32_t wcTotal, uint32_t terms);
void fft8_cube1_unpack_do(uint32_t blockDim, void *stream,
    void *xr, void *xi, void *scratchY,
    uint32_t n, uint32_t tileBase, uint32_t kStride, uint32_t shiftStride,
    uint32_t shiftsPerBlk, uint32_t eighth, uint32_t gCount, uint32_t perBlock,
    uint32_t terms);
#endif

// ---- kernels/cube_fused_xs.cpp : cube1_fs_xs (Cube/AIC, единый 16x16
// блочно-вещественный matmul на группу) ----
// META — тот же формат (eighth, aligned, wOffset), что и у dit_fs_xs;
// per-group твиддлы читаются из того же W-буфера, что и в Vector-варианте
// (см. комментарий в шапке kernels/cube_fused_xs.cpp). NPU-only: CPU-debug
// (tikicpulib) симуляция Cube-юнита здесь не реализована.
//
// fft8_cube1_matmul — единственный кернел, для которого ОБЯЗАТЕЛЕН
// ACLRT_LAUNCH_KERNEL(...) + автогенерируемый aclrtlaunch_fft8_cube1_matmul.h
// (по образцу официального 11_matmul_kernellaunch): ручной _do()-wrapper для
// него ломает сборку, когда внутри участвует Matmul high-level API (см.
// переписку про MatmulChipCap::GetFeatures() в CMakeLists.txt). Repack/unpack
// выше на этот механизм больше НЕ полагаются — см. правку выше.
#ifndef ASCENDC_CPU_DEBUG
#include "aclrtlaunch_fft8_cube1_matmul.h"
#endif

// ---- kernels/cube_fused_xs.cpp : cube2_fs_xs (Cube/AIC, два отдельных
// real-matmul: Yr = Wr*Xr - Wi*Xi, Yi = Wr*Xi + Wi*Xr, каждый как свой
// вызов Matmul на 8x8 real-блоках, без склейки в 16x16) ----
// ПРАВКА: последние два параметра — GM_ADDR workspace, GM_ADDR tilingGm
// (обязательное соглашение cube-библиотеки, собираемой с
// -DHAVE_WORKSPACE/-DHAVE_TILING, см. common/cmake/npu_lib.cmake).
// Ручной _do()-wrapper у cube2 УБРАН (он жил в одном файле с Matmul-кернелом
// и ниоткуда не вызывался): когда cube2 будет реализован, запускать его
// нужно через ACLRT_LAUNCH_KERNEL(fft8_cube2_fused_xs)(...), как cube1.
#ifdef ASCENDC_CPU_DEBUG
extern "C" __global__ __aicore__ void fft8_cube2_fused_xs(
    GM_ADDR XR, GM_ADDR XI, GM_ADDR WR8, GM_ADDR WI8, GM_ADDR META, GM_ADDR SCRATCH,
    uint32_t full_size, uint32_t num_stages, GM_ADDR workspace, GM_ADDR tilingGm);
#endif