#!/bin/bash
# FFTBench — ПОЛНАЯ МАТРИЦА: все варианты × размеры N × пороги tile ×
# ширины Cube-батча, одна таблица output/matrix.csv на выходе.
#
# Это тот самый "один большой тест". Запускается так:
#   bash run_matrix.sh                       # дефолтная сетка
#   bash run_matrix.sh -n "64 4096 262144" -t "8 64 512" -B "64 128 256"
#   bash run_matrix.sh -c "r4/dit_fs_xs r8/dit_fs_xs"   # только эти варианты
#   bash run_matrix.sh -q                    # быстрый прогон (1 прогрев, 3 замера)
#
# ЧТО УЧИТЫВАЕТ СКРИПТ, ЧТОБЫ НЕ ГОНЯТЬ БЕССМЫСЛЕННЫЕ КЛЕТКИ
#
# 1. N обязано быть степенью радикса: radix-2 работает на любом 2^k, radix-4
#    только на 4^k, radix-8 только на 8^k. Общие для всех трёх — N = 2^(6k):
#    64, 4096, 262144. Несовместимые пары просто не ставятся в очередь.
# 2. tile (порог min_npu_*) имеет смысл только при tile <= N/radix: при
#    большем пороге на NPU не остаётся ни одной стадии.
# 3. CUBE_BATCH — КОМПАЙЛ-ТАЙМ константа, то есть каждое её значение требует
#    полной пересборки. Поэтому внешний цикл именно по ней. И на батчах,
#    кроме первого, гоняются ТОЛЬКО cube-варианты: Vector-варианты от батча
#    не зависят вообще, и повторять их — чистая потеря времени.
# 4. Список вариантов и их радиксы берутся у самого бинарника (`--list`), а не
#    дублируются здесь: добавил вариант в реестр — он сразу попал в матрицу.
#
# Прогресс пишется в stdout, полные логи каждой клетки — в logs/.
# Таблица накапливается в output/matrix.csv (одна строка на клетку, со всеми
# осями и результатом сверки). Разбор: python3 scripts/analyze_matrix.py

set -u
CURRENT_DIR=$(cd $(dirname ${BASH_SOURCE:-$0}) && pwd)
cd "$CURRENT_DIR"

N_LIST="4096 262144"
TILE_LIST="8 64 512"
B_LIST="128"
VARIANTS=""
RUN_MODE="npu"
SOC_VERSION="Ascend910B4"
WARMUP=2
BENCH_RUNS=5
TIMEOUT=600
FRESH=1

while getopts "n:t:B:c:r:v:w:b:T:qk" opt; do
    case "$opt" in
        n) N_LIST="$OPTARG" ;;
        t) TILE_LIST="$OPTARG" ;;
        B) B_LIST="$OPTARG" ;;
        c) VARIANTS="$OPTARG" ;;
        r) RUN_MODE="$OPTARG" ;;
        v) SOC_VERSION="$OPTARG" ;;
        w) WARMUP="$OPTARG" ;;
        b) BENCH_RUNS="$OPTARG" ;;
        T) TIMEOUT="$OPTARG" ;;
        q) WARMUP=1; BENCH_RUNS=3 ;;
        k) FRESH=0 ;;                 # дописывать в существующую таблицу
        *) echo "[ERROR] unknown option: -$OPTARG"; exit 1 ;;
    esac
done

# ВНИМАНИЕ: у скрипта включён set -u, поэтому все переменные окружения,
# которых может не быть, читаются только через ${VAR:-} — иначе скрипт
# падает с "unbound variable" ещё до первой полезной строки.
if [ -n "${ASCEND_INSTALL_PATH:-}" ]; then
    _ASCEND=$ASCEND_INSTALL_PATH
elif [ -n "${ASCEND_HOME_PATH:-}" ]; then
    _ASCEND=$ASCEND_HOME_PATH
elif [ -d "$HOME/Ascend/ascend-toolkit/latest" ]; then
    _ASCEND=$HOME/Ascend/ascend-toolkit/latest
else
    _ASCEND=/usr/local/Ascend/ascend-toolkit/latest
fi
if [ ! -f "${_ASCEND}/bin/setenv.bash" ]; then
    echo "[ERROR] ${_ASCEND}/bin/setenv.bash не найден"; exit 1
fi
# set -e здесь НЕ включаем сознательно: бинарник штатно возвращает ненулевые
# коды (3 = сверка не прошла, 2 = вариант без результата), и при set -e первая
# же FAIL-клетка оборвала бы весь прогон матрицы. Коды мы разбираем сами.
source "${_ASCEND}/bin/setenv.bash" || true

mkdir -p ./output ./logs
BIN=./build/ascendc_kernels_bbit
TABLE=./output/matrix.csv

if [ "$FRESH" -eq 1 ]; then
    _ts="$(date +%Y%m%d_%H%M%S)"
    for _csv in matrix results numpy_baseline validation; do
        [ -s "./output/${_csv}.csv" ] && mv "./output/${_csv}.csv" "./output/${_csv}.csv.bak_${_ts}"
    done
    echo "[INFO] прежние таблицы сохранены как output/*.csv.bak_${_ts}"
fi

# N является степенью radix?
is_pow() {
    local n=$1 r=$2
    [ "$n" -ge "$r" ] || return 1
    while [ "$n" -gt 1 ]; do
        if [ $((n % r)) -ne 0 ]; then return 1; fi
        n=$((n / r))
    done
    return 0
}

echo "[INFO] сетка: N={$N_LIST}  tile={$TILE_LIST}  CUBE_BATCH={$B_LIST}"
echo "[INFO] warmup=$WARMUP runs=$BENCH_RUNS timeout=${TIMEOUT}s"

total=0; ok=0; fail=0; skip=0; err=0; firstB=1

for B in $B_LIST; do
    echo ""
    echo "############################################################"
    echo "[INFO] сборка: RUN_MODE=$RUN_MODE CUBE_BATCH=$B"
    echo "############################################################"
    rm -rf build
    if ! cmake -B build -DRUN_MODE=${RUN_MODE} -DSOC_VERSION=${SOC_VERSION} \
               -DCUBE_BATCH=${B} -DASCEND_CANN_PACKAGE_PATH=${_ASCEND} \
               > ./logs/cmake_B${B}.log 2>&1; then
        echo "[FAIL] cmake при CUBE_BATCH=$B — см. logs/cmake_B${B}.log"; firstB=0; continue
    fi
    if ! cmake --build build -j > ./logs/build_B${B}.log 2>&1; then
        echo "[FAIL] сборка при CUBE_BATCH=$B — см. logs/build_B${B}.log"
        echo "       (на больших B это может быть нехватка UB — это тоже результат)"
        firstB=0; continue
    fi
    export LD_LIBRARY_PATH="${CURRENT_DIR}/build/lib:${LD_LIBRARY_PATH:-}"

    # Реестр вариантов берём у бинарника: impl,variant,radix,...,unit,...
    LIST=$($BIN --list | tail -n +2)
    if [ -z "$LIST" ]; then echo "[FAIL] --list пуст"; exit 1; fi

    for N in $N_LIST; do
        echo ""
        echo "[INFO] ===== N=$N: генерация входа и golden ====="
        python3 scripts/gen_data.py -N "$N" --input-dir ./input --output-dir ./output > /dev/null
        python3 scripts/bench_numpy.py --input ./input/input_x.bin \
                --csv ./output/numpy_baseline.csv > ./logs/numpy_N${N}.log 2>&1

        while IFS=, read -r impl variant radix algo gran layout unit terms cores; do
            full="${impl}/${variant}"
            # фильтр по -c, если задан
            if [ -n "$VARIANTS" ]; then
                case " $VARIANTS " in *" $full "*) ;; *) continue ;; esac
            fi
            # Vector-варианты от CUBE_BATCH не зависят — гоняем только на первом батче
            if [ "$unit" != "cube" ] && [ "$firstB" -eq 0 ]; then continue; fi
            # N обязано быть степенью радикса
            if ! is_pow "$N" "$radix"; then
                echo "[SKIP] $full N=$N — N не степень $radix"
                skip=$((skip + 1)); continue
            fi
            for tile in $TILE_LIST; do
                # при tile > N/radix на NPU не остаётся ни одной стадии
                if [ "$tile" -gt $((N / radix)) ]; then continue; fi
                total=$((total + 1))
                log="./logs/${impl}_${variant}_N${N}_T${tile}_B${B}.log"
                printf "[RUN ] %-18s N=%-7s tile=%-5s B=%-4s ... " "$full" "$N" "$tile" "$B"
                rc=0
                timeout "$TIMEOUT" $BIN "$full" "$tile" "$WARMUP" "$BENCH_RUNS" \
                    > "$log" 2>&1 || rc=$?
                case $rc in
                    0) t=$(grep -m1 "^Total:" "$log" | sed -E 's/.*avg=([0-9.]+).*/\1/')
                       echo "PASS  total=${t} us"; ok=$((ok + 1)) ;;
                    3) t=$(grep -m1 "^Total:" "$log" | sed -E 's/.*avg=([0-9.]+).*/\1/')
                       ma=$(grep -m1 "\[VERIFY\]" "$log" | sed -E 's/.*max_abs=([0-9.e+-]+).*/\1/')
                       echo "FAIL  total=${t} us  max_abs=${ma}"; fail=$((fail + 1)) ;;
                    2) echo "SKIP  (нет результата: заглушка или ошибка кернела)"; skip=$((skip + 1)) ;;
                    124) echo "TIMEOUT ${TIMEOUT}s — см. $log"; err=$((err + 1)) ;;
                    *) echo "ERROR код $rc — см. $log"; err=$((err + 1)) ;;
                esac
            done
        done <<< "$LIST"
    done
    firstB=0
done

echo ""
echo "############################################################"
echo "[INFO] матрица пройдена: клеток=$total  PASS=$ok  FAIL=$fail  SKIP=$skip  ERROR=$err"
echo "[INFO] таблица: $TABLE"
echo "############################################################"
echo ""
python3 scripts/analyze_matrix.py