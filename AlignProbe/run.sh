#!/bin/bash
# AlignProbe — answers the one hardware question that gates the odd-radix
# kernel design: which GM<->UB primitive can move data at an ODD stride?
#
#   bash run.sh                 # build, then run every mode
#   bash run.sh -P 0            # rebuild without the DataCopyPad mode
#
# Each mode runs as its own process, so a mode that faults the AI core does not
# take the others with it; the script reports per-mode and keeps going.
set -u
CURRENT_DIR=$(cd "$(dirname "${BASH_SOURCE:-$0}")" && pwd)
cd "$CURRENT_DIR"

PAD=1
SOC=Ascend910B4
while getopts "P:v:" opt; do
    case "$opt" in
        P) PAD="$OPTARG" ;;
        v) SOC="$OPTARG" ;;
        *) echo "[ERROR] unknown option"; exit 1 ;;
    esac
done

if [ -n "${ASCEND_INSTALL_PATH:-}" ]; then _A=$ASCEND_INSTALL_PATH
elif [ -n "${ASCEND_HOME_PATH:-}" ];   then _A=$ASCEND_HOME_PATH
elif [ -d "$HOME/Ascend/ascend-toolkit/latest" ]; then _A=$HOME/Ascend/ascend-toolkit/latest
else _A=/usr/local/Ascend/ascend-toolkit/latest; fi
[ -f "${_A}/bin/setenv.bash" ] && source "${_A}/bin/setenv.bash" || true

mkdir -p logs
echo "[INFO] building (SOC=$SOC, HAVE_DATACOPYPAD=$PAD)"
rm -rf build
if ! cmake -B build -DRUN_MODE=npu -DSOC_VERSION=$SOC \
        -DHAVE_DATACOPYPAD=$PAD -DASCEND_CANN_PACKAGE_PATH=${_A} \
        > logs/cmake.log 2>&1; then
    echo "[FAIL] cmake — see logs/cmake.log"; tail -20 logs/cmake.log; exit 1
fi
if ! cmake --build build -j > logs/build.log 2>&1; then
    echo "[FAIL] build — see logs/build.log"
    echo "       if it is DataCopyPad / DataCopyExtParams that is missing,"
    echo "       rerun as:  bash run.sh -P 0"
    grep -iE "error" logs/build.log | head -20
    exit 1
fi
export LD_LIBRARY_PATH="${CURRENT_DIR}/build/lib:${LD_LIBRARY_PATH:-}"
BIN=./build/align_probe

for m in aligned unaligned pad padodd vecunalign scalar trunc bench; do
    echo ""
    echo "############################################################"
    # ${PIPESTATUS[0]}, not $? : in a pipeline $? is the exit status of the
    # LAST command, which here is tee and which always succeeds. With `|| rc=$?`
    # every mode was reported "[ OK ]" no matter what -- including one that
    # died on an aicore exception halfway through its table. A harness that
    # cannot report failure is worse than no harness.
    timeout 120 $BIN "$m" 2>&1 | tee "logs/${m}.log"
    rc=${PIPESTATUS[0]}
    # A mode that faults mid-table exits non-zero AND never prints its own
    # [RESULT] line, so both are checked: the exit code can be masked by the
    # runtime, the missing summary line cannot.
    if ! grep -q "^\[RESULT\] mode=" "logs/${m}.log"; then
        echo "[CRASH] mode=$m stopped before finishing (no [RESULT] line) —"
        echo "        the last table row printed is where it died."
        continue
    fi
    case $rc in
        0)   echo "[ OK ] mode=$m" ;;
        3)   echo "[FAIL] mode=$m produced wrong data — see logs/${m}.log" ;;
        124) echo "[HANG] mode=$m timed out" ;;
        *)   echo "[CRASH] mode=$m exit=$rc — this mode is not usable on this device" ;;
    esac
done
echo ""
echo "############################################################"
echo "[INFO] per-mode logs in logs/"
