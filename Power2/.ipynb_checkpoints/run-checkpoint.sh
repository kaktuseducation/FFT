#!/bin/bash
# FFTBench — сборка + прогон одного или нескольких вариантов + сводка.
#
# Для полной матрицы (все радиксы × варианты × N × tile × батч) есть отдельный
# скрипт: bash run_matrix.sh
#
# Имена вариантов — ПОЛНЫЕ, "impl/variant": dit_fs_xs есть у всех трёх
# радиксов, короткое имя для него неоднозначно. Полный список:
#   ./build/ascendc_kernels_bbit --list
#
# ВАЖНО про N: оно берётся из размера input/input_x.bin и должно быть степенью
# радикса варианта. radix-2 — любое 2^k, radix-4 — только 4^k, radix-8 —
# только 8^k. Сравнивать радиксы между собой можно на N = 2^(6k), то есть
# 64, 4096, 262144.
#
# Использование:
#   bash run.sh -c r4/dit_fs_xs -r npu
#   bash run.sh -c "r2/dit_fs_xs r4/dit_fs_xs r8/dit_fs_xs" -n 4096 -t 8
#   bash run.sh -c all -r npu -v Ascend910B4 -n 4096 -t 8 -w 2 -b 5
#   bash run.sh -c r8/cube1x3_fs_xs -B 256        # ширина Cube-батча (ЗАГЛАВНАЯ -B)
#
# Про флаги -b и -B: строчная -b это число замеров, ЗАГЛАВНАЯ -B — ширина
# Cube-батча (требует пересборки, поэтому флаг сборки).

set -e
CURRENT_DIR=$(cd $(dirname ${BASH_SOURCE:-$0}) && pwd)
cd "$CURRENT_DIR"

exec > >(tee run.log) 2>&1
echo "[INFO] run.sh started, cwd=$(pwd)"

VARIANT=""
RUN_MODE="npu"
SOC_VERSION="Ascend910B4"
N=4096
TILE_SIZE=8
WARMUP=2
BENCH_RUNS=5
CUBE_BATCH=128

while getopts "c:r:v:n:t:w:b:B:" opt; do
    case "$opt" in
        c) VARIANT="$OPTARG" ;;
        r) RUN_MODE="$OPTARG" ;;
        v) SOC_VERSION="$OPTARG" ;;
        n) N="$OPTARG" ;;
        t) TILE_SIZE="$OPTARG" ;;
        w) WARMUP="$OPTARG" ;;
        b) BENCH_RUNS="$OPTARG" ;;
        B) CUBE_BATCH="$OPTARG" ;;
        *) echo "[ERROR] unknown option: -$OPTARG"; exit 1 ;;
    esac
done

if [ -z "$VARIANT" ]; then
    echo "[ERROR] -c <variant> обязателен. Полный список — после сборки:"
    echo "        ./build/ascendc_kernels_bbit --list"
    echo "        либо -c all, чтобы прогнать всё, что подходит под N."
    exit 1
fi

if [ -n "$ASCEND_INSTALL_PATH" ]; then
    _ASCEND_INSTALL_PATH=$ASCEND_INSTALL_PATH
elif [ -n "$ASCEND_HOME_PATH" ]; then
    _ASCEND_INSTALL_PATH=$ASCEND_HOME_PATH
elif [ -d "$HOME/Ascend/ascend-toolkit/latest" ]; then
    _ASCEND_INSTALL_PATH=$HOME/Ascend/ascend-toolkit/latest
else
    _ASCEND_INSTALL_PATH=/usr/local/Ascend/ascend-toolkit/latest
fi
echo "[INFO] using ASCEND install path: ${_ASCEND_INSTALL_PATH}"
if [ ! -f "${_ASCEND_INSTALL_PATH}/bin/setenv.bash" ]; then
    echo "[ERROR] ${_ASCEND_INSTALL_PATH}/bin/setenv.bash not found."
    exit 1
fi
set +e
source "${_ASCEND_INSTALL_PATH}/bin/setenv.bash"
set -e

echo "[INFO] RUN_MODE=${RUN_MODE}  SOC=${SOC_VERSION}  N=${N}  TILE=${TILE_SIZE}  CUBE_BATCH=${CUBE_BATCH}"

echo "[INFO] 1/4 generating input + golden data (N=${N})"
python3 scripts/gen_data.py -N "${N}" --input-dir ./input --output-dir ./output

echo "[INFO] 2/4 building (RUN_MODE=${RUN_MODE}, CUBE_BATCH=${CUBE_BATCH})"
rm -rf build
cmake -B build \
    -DRUN_MODE=${RUN_MODE} \
    -DSOC_VERSION=${SOC_VERSION} \
    -DCUBE_BATCH=${CUBE_BATCH} \
    -DASCEND_CANN_PACKAGE_PATH=${_ASCEND_INSTALL_PATH}
cmake --build build -j

export LD_LIBRARY_PATH="${CURRENT_DIR}/build/lib:${LD_LIBRARY_PATH}"
BIN=./build/ascendc_kernels_bbit

# Один запуск run.sh = одна чистая таблица. CSV накопительные, и в прошлом из
# этого уже выходили ложные выводы: в сводке месяцами жили строки времён
# отладки (с дедлоками и мусорными числами), по которым казалось, что рабочий
# вариант сломан. Поэтому старые таблицы уезжают в архив с таймстампом.
mkdir -p ./output ./logs
_ts="$(date +%Y%m%d_%H%M%S)"
for _csv in matrix results validation numpy_baseline; do
    if [ -s "./output/${_csv}.csv" ]; then
        mv "./output/${_csv}.csv" "./output/${_csv}.csv.bak_${_ts}"
    fi
done
echo "[INFO] прежние CSV (если были) сохранены как output/*.csv.bak_${_ts}"

# -c all = всё, что подходит под текущее N (бинарник сам знает радиксы)
if [ "$VARIANT" = "all" ]; then
    VARIANT=$($BIN --list | tail -n +2 | awk -F, '{print $1"/"$2}' | tr '\n' ' ')
    echo "[INFO] -c all развёрнуто в: $VARIANT"
fi

echo ""
echo "[INFO] 3/4 running"
for v in $VARIANT; do
    echo ""
    echo "[INFO] вариант ${v}"
    # Сверка с golden теперь внутри бинарника, отдельный шаг не нужен:
    #   0 = PASS, 2 = вариант не дал результата, 3 = FAIL, 1 = ошибка
    # "|| rc=$?" здесь обязательно: у скрипта включён set -e, и без этого
    # первая же не прошедшая сверку клетка оборвала бы весь прогон,
    # не добравшись ни до остальных вариантов, ни до сводки.
    rc=0
    $BIN "$v" "${TILE_SIZE}" "${WARMUP}" "${BENCH_RUNS}" || rc=$?
    case $rc in
        0) echo "[PASS] $v" ;;
        2) echo "[SKIP] $v — вариант не дал результата (заглушка или ошибка кернела)" ;;
        3) echo "[FAIL] $v — сверка с golden не прошла" ;;
        *) echo "[ERROR] $v — код $rc (несовместимое N или неизвестный вариант)" ;;
    esac
done

echo ""
echo "[INFO] NumPy baseline на том же input_x.bin"
python3 scripts/bench_numpy.py --input ./input/input_x.bin --csv ./output/numpy_baseline.csv

echo ""
echo "[INFO] 4/4 ===== сводка ====="
python3 scripts/analyze_matrix.py