#!/bin/bash
# OddFFT — build + run of one or several variants + a summary.
#
# For the full matrix (radix x butterfly x scheme x N x tile) there is a script
# of its own: bash run_matrix.sh
#
# Variant names are the FULL ones, "impl/variant": all three radices will have
# a sym_psm_xs, so the short name is ambiguous. The full list:
#   ./build/oddfft --list
#
# IMPORTANT about N: it is taken from the size of input/input_x.bin and must be
# a power of the variant's radix. There is NO N shared by radix 3, 5 and 7 —
# compare at the nearest sizes (2187/3125/2401 or 19683/15625/16807) and read
# the ns_per_pt column of output/matrix.csv, which normalises by N*log2(N).
#
# Usage:
#   bash run.sh -c r3/sym_psm_xs -n 2187
#   bash run.sh -c "r3/sym_ps_xs r3/sym_psm_xs" -n 2187 -t 64
#   bash run.sh -c all -n 6561 -t 64 -w 2 -b 5

set -e
CURRENT_DIR=$(cd "$(dirname "${BASH_SOURCE:-$0}")" && pwd)
cd "$CURRENT_DIR"

exec > >(tee run.log) 2>&1
echo "[INFO] run.sh started, cwd=$(pwd)"

VARIANT=""
RUN_MODE="npu"
SOC_VERSION="Ascend910B4"
N=2187
TILE_SIZE=64
MIN_T=8
WARMUP=2
BENCH_RUNS=5

while getopts "c:r:v:n:t:w:b:M:" opt; do
    case "$opt" in
        c) VARIANT="$OPTARG" ;;
        r) RUN_MODE="$OPTARG" ;;
        v) SOC_VERSION="$OPTARG" ;;
        n) N="$OPTARG" ;;
        t) TILE_SIZE="$OPTARG" ;;
        w) WARMUP="$OPTARG" ;;
        b) BENCH_RUNS="$OPTARG" ;;
        M) MIN_T="$OPTARG" ;;
        *) echo "[ERROR] unknown option: -$OPTARG"; exit 1 ;;
    esac
done

if [ -z "$VARIANT" ]; then
    echo "[ERROR] -c <variant> is mandatory. The full list — after the build:"
    echo "        ./build/oddfft --list"
    echo "        or -c all, to run everything that fits the N."
    exit 1
fi

if [ -n "${ASCEND_INSTALL_PATH:-}" ]; then
    _ASCEND_INSTALL_PATH=$ASCEND_INSTALL_PATH
elif [ -n "${ASCEND_HOME_PATH:-}" ]; then
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

echo "[INFO] RUN_MODE=${RUN_MODE}  SOC=${SOC_VERSION}  N=${N}  TILE=${TILE_SIZE}  MIN_T=${MIN_T}"

echo "[INFO] 1/4 generating input + golden data (N=${N})"
python3 scripts/gen_data.py -N "${N}" --input-dir ./input --output-dir ./output

echo "[INFO] 2/4 building (RUN_MODE=${RUN_MODE})"
rm -rf build
cmake -B build \
    -DRUN_MODE=${RUN_MODE} \
    -DSOC_VERSION=${SOC_VERSION} \
    -DASCEND_CANN_PACKAGE_PATH=${_ASCEND_INSTALL_PATH}
cmake --build build -j

export LD_LIBRARY_PATH="${CURRENT_DIR}/build/lib:${LD_LIBRARY_PATH}"
BIN=./build/oddfft

# One invocation of run.sh = one clean table. The CSVs are cumulative, and in
# the power-of-two project that cumulation already produced false conclusions:
# rows from the debugging era lived on in the summary and made a working
# variant look broken. So the old tables are archived with a timestamp.
mkdir -p ./output ./logs
_ts="$(date +%Y%m%d_%H%M%S)"
# numpy_baseline.csv is deliberately NOT archived. It is per-N reference data,
# not per-run results: bench_numpy.py appends and the analyser takes the last
# row per N. Archiving it on every invocation left the baseline holding only
# the most recent N and blanked the vs-NumPy column for every other size.
for _csv in matrix results validation; do
    if [ -s "./output/${_csv}.csv" ]; then
        mv "./output/${_csv}.csv" "./output/${_csv}.csv.bak_${_ts}"
    fi
done
echo "[INFO] the previous CSVs (if there were any) kept as output/*.csv.bak_${_ts}"

if [ "$VARIANT" = "all" ]; then
    VARIANT=$($BIN --list | tail -n +2 | awk -F, '{print $1"/"$2}' | tr '\n' ' ')
    echo "[INFO] -c all expanded into: $VARIANT"
fi

echo ""
echo "[INFO] 3/4 running"
for v in $VARIANT; do
    echo ""
    echo "[INFO] variant ${v}"
    # Verification against golden happens inside the binary:
    #   0 = PASS, 2 = the variant gave no result, 3 = FAIL, 1 = an error.
    # The "|| rc=$?" is mandatory: set -e is on, and without it the first cell
    # that failed verification would cut the run short before the summary.
    rc=0
    $BIN "$v" "${TILE_SIZE}" "${WARMUP}" "${BENCH_RUNS}" "${MIN_T}" || rc=$?
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
