// kernels/cube1_common.h  (radix8, cube1_fs_xs)
//
// Общая константа между kernels/cube1_repack_unpack.cpp (чисто Vector) и
// kernels/cube_fused_xs.cpp (Matmul/Cube) — вынесена в отдельный заголовок,
// чтобы не дублировать литерал и не завязываться на #include друг друга:
// файлы НАМЕРЕННО разделены (см. CMakeLists.txt — KERNEL_FILES_CUBE
// компилируется через отдельный ascendc_library()-конвейер, иначе
// host_bisheng_obj падает на MatmulChipCap::GetFeatures(); ровно та же
// причина, по которой repack/unpack пришлось вынести из cube_fused_xs.cpp —
// будучи физически в одном файле с Matmul-кернелом, они собирались через
// тот же "cube"-конвейер и ловили aicore exception на ровном месте).
#pragma once
#include <cstdint>

// Размер батча групп на один Cube-вызов. B=128 — кратно типичному второму
// измерению Cube-тайла (32/64/128 в зависимости от dtype).
static constexpr uint32_t CUBE_BATCH = 128;
