#!/bin/bash
# FFTBench — build + run of one or several variants + a summary.
#
# For the full matrix (all the radices × variants × N × tile × batch) there
# is a script of its own: bash run_matrix.sh
#
# Variant names are the FULL ones, "impl/variant": all three radices have a
# dit_fs_xs, so the short name for it is ambiguous. The full list:
#   ./build/ascendc_kernels_bbit --list
#
# IMPORTANT about N: it is taken from the size of input/input_x.bin and must be
# a power of the variant's radix. radix-2 — any 2^k, radix-4 — only 4^k,
# radix-8 — only 8^k. The radices can be compared against each other on
# N = 2^(6k), that is, 64, 4096, 262144.
#
# Usage:
#   bash run.sh -c r4/dit_fs_xs -r npu
#   bash run.sh -c "r2/dit_fs_xs r4/dit_fs_xs r8/dit_fs_xs" -n 4096 -t 8
#   bash run.sh -c all -r npu -v Ascend910B4 -n 4096 -t 8 -w 2 -b 5
#   bash run.sh -c r8/cube1x3_fs_xs -B 256        # Cube batch width (CAPITAL -B)
#
# About the flags -b and -B: lowercase -b is the number of measurements,
# CAPITAL -B is the Cube batch width (it needs a rebuild, hence a build flag).

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
    echo "[ERROR] -c <variant> is mandatory. The full list — after the build:"
    echo "        ./build/ascendc_kernels_bbit --list"
    echo "        or -c all, to run everything that fits the N."
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

# One invocation of run.sh = one clean table. The CSVs are cumulative, and in
# the past that has already produced false conclusions: rows from the debugging
# era (with deadlocks and garbage numbers) lived on in the summary for months,
# and by them it looked as though a working variant were broken. So the old
# tables go off into the archive with a timestamp.
mkdir -p ./output ./logs
_ts="$(date +%Y%m%d_%H%M%S)"
for _csv in matrix results validation numpy_baseline; do
    if [ -s "./output/${_csv}.csv" ]; then
        mv "./output/${_csv}.csv" "./output/${_csv}.csv.bak_${_ts}"
    fi
done
echo "[INFO] the previous CSVs (if there were any) kept as output/*.csv.bak_${_ts}"

# -c all = everything that fits the current N (the binary knows the radices)
if [ "$VARIANT" = "all" ]; then
    VARIANT=$($BIN --list | tail -n +2 | awk -F, '{print $1"/"$2}' | tr '\n' ' ')
    echo "[INFO] -c all expanded into: $VARIANT"
fi

echo ""
echo "[INFO] 3/4 running"
for v in $VARIANT; do
    echo ""
    echo "[INFO] variant ${v}"
    # Verification against golden is inside the binary now, a step of its own
    # is not needed:
    #   0 = PASS, 2 = the variant gave no result, 3 = FAIL, 1 = an error
    # The "|| rc=$?" here is mandatory: the script has set -e on, and without
    # it the very first cell that failed verification would cut the whole run
    # short, never reaching either the remaining variants or the summary.
    rc=0
    $BIN "$v" "${TILE_SIZE}" "${WARMUP}" "${BENCH_RUNS}" || rc=$?
    case $rc in
        0) echo "[PASS] $v" ;;
        2) echo "[SKIP] $v — the variant gave no result (a stub or a kernel error)" ;;
        3) echo "[FAIL] $v — verification against golden did not pass" ;;
        *) echo "[ERROR] $v — code $rc (an incompatible N or an unknown variant)" ;;
    esac
done

echo ""
echo "[INFO] NumPy baseline on that same input_x.bin"
python3 scripts/bench_numpy.py --input ./input/input_x.bin --csv ./output/numpy_baseline.csv

echo ""
echo "[INFO] 4/4 ===== summary ====="
python3 scripts/analyze_matrix.py
