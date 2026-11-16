#!/bin/bash
# FFTBench — THE FULL MATRIX: all the variants × sizes N × tile thresholds
# × Cube batch widths, one table output/matrix.csv on the way out.
#
# This is the "one big test" itself. It is run like this:
#   bash run_matrix.sh                       # the default grid
#   bash run_matrix.sh -n "64 4096 262144" -t "8 64 512" -B "64 128 256"
#   bash run_matrix.sh -c "r4/dit_fs_xs r8/dit_fs_xs"   # only these variants
#   bash run_matrix.sh -q                    # a quick run (1 warmup, 3 measurements)
#
# WHAT THE SCRIPT TAKES INTO ACCOUNT SO AS NOT TO RUN POINTLESS CELLS
#
# 1. N must be a power of the radix: radix-2 works on any 2^k, radix-4 only on
#    4^k, radix-8 only on 8^k. The ones common to all three are N = 2^(6k):
#    64, 4096, 262144. Incompatible pairs simply never get queued.
# 2. tile (the min_npu_* threshold) means something only at tile <= N/radix: at
#    a larger threshold not a single stage is left on the NPU.
# 3. CUBE_BATCH is a COMPILE-TIME constant, that is, every value of it needs a
#    full rebuild. Hence the outer loop is over exactly that. And on the
#    batches other than the first, ONLY the cube variants are run: the Vector
#    variants do not depend on the batch at all, and repeating them is pure
#    waste of time.
# 4. The variant list and their radices are taken from the binary itself
#    (`--list`) rather than duplicated here: add a variant to the registry and
#    it lands in the matrix straight away.
#
# The progress is written to stdout, the full logs of every cell — into logs/.
# The table accumulates in output/matrix.csv (one row per cell, with all the
# axes and the verification result). Analysis:
# python3 scripts/analyze_matrix.py

set -u
CURRENT_DIR=$(cd $(dirname ${BASH_SOURCE:-$0}) && pwd)
cd "$CURRENT_DIR"

# The default grid is deliberately the FULL one: 3 sizes x 3 thresholds x
# 3 batch widths. Two of those three axes only yield paired comparisons when
# they have at least two values in the table, so a narrower default silently
# costs conclusions: with B_LIST="128" the "Cube batch" axis reports "no
# paired comparisons", and without N=64 there is no small-N point at which to
# see that the ranking of the variants changes with size.
# N: only 64, 4096 and 262144 are powers of 2, 4 AND 8 at once, so these are
# the only sizes on which radices can be compared with each other at all.
N_LIST="64 4096 262144"
TILE_LIST="8 64 512"
B_LIST="64 128 256"
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
        k) FRESH=0 ;;                 # append to the existing table
        *) echo "[ERROR] unknown option: -$OPTARG"; exit 1 ;;
    esac
done

# ATTENTION: the script has set -u on, so every environment variable that may
# not be there is read only through ${VAR:-} — otherwise the script falls over
# with "unbound variable" before the first useful line.
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
    echo "[ERROR] ${_ASCEND}/bin/setenv.bash not found"; exit 1
fi
# We deliberately do NOT turn set -e on here: the binary returns non-zero codes
# as a matter of course (3 = verification did not pass, 2 = a variant with no
# result), and under set -e the very first FAIL cell would cut the whole matrix
# run short. We take the codes apart ourselves.
source "${_ASCEND}/bin/setenv.bash" || true

mkdir -p ./output ./logs
BIN=./build/ascendc_kernels_bbit
TABLE=./output/matrix.csv

if [ "$FRESH" -eq 1 ]; then
    _ts="$(date +%Y%m%d_%H%M%S)"
    for _csv in matrix results numpy_baseline mindspore_baseline mindspore_split_all_runs validation; do
        [ -s "./output/${_csv}.csv" ] && mv "./output/${_csv}.csv" "./output/${_csv}.csv.bak_${_ts}"
    done
    echo "[INFO] the previous tables kept as output/*.csv.bak_${_ts}"
fi

# Is N a power of radix?
is_pow() {
    local n=$1 r=$2
    [ "$n" -ge "$r" ] || return 1
    while [ "$n" -gt 1 ]; do
        if [ $((n % r)) -ne 0 ]; then return 1; fi
        n=$((n / r))
    done
    return 0
}

echo "[INFO] grid: N={$N_LIST}  tile={$TILE_LIST}  CUBE_BATCH={$B_LIST}"
echo "[INFO] warmup=$WARMUP runs=$BENCH_RUNS timeout=${TIMEOUT}s"

total=0; ok=0; fail=0; skip=0; err=0; firstB=1

for B in $B_LIST; do
    echo ""
    echo "############################################################"
    echo "[INFO] build: RUN_MODE=$RUN_MODE CUBE_BATCH=$B"
    echo "############################################################"
    rm -rf build
    if ! cmake -B build -DRUN_MODE=${RUN_MODE} -DSOC_VERSION=${SOC_VERSION} \
               -DCUBE_BATCH=${B} -DASCEND_CANN_PACKAGE_PATH=${_ASCEND} \
               > ./logs/cmake_B${B}.log 2>&1; then
        echo "[FAIL] cmake at CUBE_BATCH=$B — see logs/cmake_B${B}.log"; firstB=0; continue
    fi
    if ! cmake --build build -j > ./logs/build_B${B}.log 2>&1; then
        echo "[FAIL] build at CUBE_BATCH=$B — see logs/build_B${B}.log"
        echo "       (at large B this may be a shortage of UB — that is a result too)"
        firstB=0; continue
    fi
    export LD_LIBRARY_PATH="${CURRENT_DIR}/build/lib:${LD_LIBRARY_PATH:-}"

    # The variant registry comes from the binary: impl,variant,radix,...,unit,...
    LIST=$($BIN --list | tail -n +2)
    if [ -z "$LIST" ]; then echo "[FAIL] --list is empty"; exit 1; fi

    for N in $N_LIST; do
        echo ""
        echo "[INFO] ===== N=$N: generating the input and golden ====="
        python3 scripts/gen_data.py -N "$N" --input-dir ./input --output-dir ./output > /dev/null
        python3 scripts/bench_numpy.py --input ./input/input_x.bin \
                --csv ./output/numpy_baseline.csv > ./logs/numpy_N${N}.log 2>&1
        # The vendor NPU baseline: the same transform, on the same
        # device, through MindSpore. This is the comparison that holds
        # the hardware fixed, so it is the one that isolates what these
        # kernels contribute. A missing MindSpore is not fatal to the
        # sweep -- the cell simply goes unmeasured and says so.
        python3 scripts/bench_mindspore.py --input ./input/input_x.bin \
                --csv ./output/mindspore_baseline.csv \
                > ./logs/mindspore_N${N}.log 2>&1 \
            || echo "[WARN] MindSpore baseline failed at N=$N -- see logs/mindspore_N${N}.log"

        # The column order is set by print_list() in host/main.cpp — it is
        # read positionally, so a new field there needs an edit here too.
        # Right now:
        # impl,variant,radix,algo,granularity,layout,unit,terms,mm,cores
        while IFS=, read -r impl variant radix algo gran layout unit terms mm cores; do
            full="${impl}/${variant}"
            # the -c filter, if one was given
            if [ -n "$VARIANTS" ]; then
                case " $VARIANTS " in *" $full "*) ;; *) continue ;; esac
            fi
            # Vector variants do not depend on CUBE_BATCH — first batch only
            if [ "$unit" != "cube" ] && [ "$firstB" -eq 0 ]; then continue; fi
            # N must be a power of the radix
            if ! is_pow "$N" "$radix"; then
                echo "[SKIP] $full N=$N — N is not a power of $radix"
                skip=$((skip + 1)); continue
            fi
            for tile in $TILE_LIST; do
                # at tile > N/radix not a single stage is left on the NPU
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
                    2) echo "SKIP  (no result: a stub or a kernel error)"; skip=$((skip + 1)) ;;
                    124) echo "TIMEOUT ${TIMEOUT}s — see $log"; err=$((err + 1)) ;;
                    *) echo "ERROR code $rc — see $log"; err=$((err + 1)) ;;
                esac
            done
        done <<< "$LIST"
    done
    firstB=0
done

echo ""
echo "############################################################"
echo "[INFO] matrix walked: cells=$total  PASS=$ok  FAIL=$fail  SKIP=$skip  ERROR=$err"
echo "[INFO] table: $TABLE"
echo "############################################################"
echo ""
python3 scripts/analyze_matrix.py
