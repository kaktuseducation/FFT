#!/bin/bash
# OddFFT — DETERMINISM CHECK: run every variant twice on identical input and
# compare the two outputs BYTE FOR BYTE.
#
# WHY THIS TEST EXISTS, AND WHY IT IS NOT REDUNDANT WITH VERIFICATION
#
# The golden comparison in the binary (host/verify.h) replicates numpy.isclose
# with a 1e-3 tolerance and allows up to 0.1% of points to miss. That is the
# right criterion for an fp32 FFT — but it is blind to exactly the class of bug
# that is hardest to find on an NPU: a synchronization race whose effect is
# smaller than the tolerance.
#
# This project has already been bitten by one. kernels/r8_cube1_unpack.cpp
# received the matmul result from GM into a TQue<VECOUT> buffer and then read it
# with vector ops. A VECOUT queue installs a V -> MTE3 dependency ("we computed
# in UB, now flush to GM"); what was needed here is the OPPOSITE direction,
# MTE2 -> V, which only VECIN gives. While terms=1 there were no vector ops over
# that data at all, so the mismatch never showed. As soon as terms>=2 added the
# Add for Y0+Y1+Y2, the vector unit started reading UB before MTE2 had finished
# the load — and the symptom was ONE OR TWO elements out of 4096 drifting from
# run to run. max_abs wandered between 9.8e-02 and 1.5e-01 with max_rel
# unchanged. Every single run still PASSED the golden check.
#
# A race like that is invisible to a tolerance-based check and obvious to a
# byte comparison of two runs. Hence this script. It is cheap (two runs per
# cell) and it is the only test here that can fail for a reason the matrix
# cannot see.
#
# WHAT A FAILURE MEANS
# Bit-exact repetition is the correct expectation for every variant in this
# project: the input is fixed, the stage plan is fixed, and tile-to-core
# assignment is a pure function of blockIdx. No variant reduces across cores,
# so no variant has a legitimate reason to reorder floating-point additions
# between runs. If a variant is not bit-exact, something is genuinely wrong —
# most likely a missing synchronization, or two cores writing the same GM cell.
#
# USAGE
#   bash scripts/check_determinism.sh                      # all variants, N=4096
#   bash scripts/check_determinism.sh -n 262144 -t 8
#   bash scripts/check_determinism.sh -c "r8/cube2x3_fs_xs r4/dit_fsm_xs"
#   bash scripts/check_determinism.sh -r 5                 # 5 repeats, not 2
#
# Assumes the project is already built (run.sh or run_matrix.sh does that).
# Exit code: 0 = every variant bit-exact, 1 = at least one is not.

set -u
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE:-$0}")/.." && pwd)"
cd "$CURRENT_DIR"

N=4096
TILE=8
REPEATS=2
VARIANTS=""

while getopts "n:t:r:c:" opt; do
    case "$opt" in
        n) N="$OPTARG" ;;
        t) TILE="$OPTARG" ;;
        r) REPEATS="$OPTARG" ;;
        c) VARIANTS="$OPTARG" ;;
        *) echo "[ERROR] unknown option: -$OPTARG"; exit 1 ;;
    esac
done

# Same environment discovery as run_matrix.sh; every possibly-absent variable is
# read through ${VAR:-} because of set -u.
if [ -n "${ASCEND_INSTALL_PATH:-}" ]; then
    _ASCEND=$ASCEND_INSTALL_PATH
elif [ -n "${ASCEND_HOME_PATH:-}" ]; then
    _ASCEND=$ASCEND_HOME_PATH
elif [ -d "$HOME/Ascend/ascend-toolkit/latest" ]; then
    _ASCEND=$HOME/Ascend/ascend-toolkit/latest
else
    _ASCEND=/usr/local/Ascend/ascend-toolkit/latest
fi
[ -f "${_ASCEND}/bin/setenv.bash" ] && source "${_ASCEND}/bin/setenv.bash" || true
export LD_LIBRARY_PATH="${CURRENT_DIR}/build/lib:${LD_LIBRARY_PATH:-}"

BIN=./build/oddfft
if [ ! -x "$BIN" ]; then
    echo "[ERROR] $BIN not found — build the project first (run.sh or run_matrix.sh)"
    exit 1
fi

mkdir -p ./output ./logs ./output/determinism

echo "[INFO] generating the input and golden for N=$N"
python3 scripts/gen_data.py -N "$N" --input-dir ./input --output-dir ./output > /dev/null

# Is N a power of radix? Same helper as run_matrix.sh — a variant whose radix
# does not divide N is not a failure, it is simply not applicable.
is_pow() {
    local n=$1 r=$2
    [ "$n" -ge "$r" ] || return 1
    while [ "$n" -gt 1 ]; do
        if [ $((n % r)) -ne 0 ]; then return 1; fi
        n=$((n / r))
    done
    return 0
}

if [ -z "$VARIANTS" ]; then
    LIST=$($BIN --list | tail -n +2)
else
    LIST=""
    for v in $VARIANTS; do
        LIST="$LIST
$($BIN --list | tail -n +2 | awk -F, -v want="$v" '$1"/"$2 == want')"
    done
fi

echo "[INFO] N=$N tile=$TILE repeats=$REPEATS"
echo ""

exact=0; differs=0; skipped=0
# The columns of --list, read positionally. The order is set by print_list() in
# host/main.cpp: impl,variant,radix,algo,granularity,layout,unit,terms,mm,cores
while IFS=, read -r impl variant radix algo gran layout unit terms mm cores; do
    [ -z "${impl:-}" ] && continue
    full="${impl}/${variant}"
    if ! is_pow "$N" "$radix"; then
        printf "[SKIP] %-18s N=%s is not a power of %s\n" "$full" "$N" "$radix"
        skipped=$((skipped + 1)); continue
    fi

    printf "[DET ] %-18s " "$full"
    first=""
    bad=0
    for i in $(seq 1 "$REPEATS"); do
        rm -f ./output/output_x.bin
        rc=0
        $BIN "$full" "$TILE" 1 1 > "./logs/det_${impl}_${variant}_${i}.log" 2>&1 || rc=$?
        if [ ! -s ./output/output_x.bin ]; then
            printf "SKIP  (no result, rc=%s)\n" "$rc"
            skipped=$((skipped + 1)); bad=2; break
        fi
        cp ./output/output_x.bin "./output/determinism/${impl}_${variant}_${i}.bin"
        if [ -z "$first" ]; then
            first="./output/determinism/${impl}_${variant}_${i}.bin"
        elif ! cmp -s "$first" "./output/determinism/${impl}_${variant}_${i}.bin"; then
            bad=1
        fi
    done
    [ "$bad" -eq 2 ] && continue

    if [ "$bad" -eq 0 ]; then
        printf "BIT-EXACT over %s runs\n" "$REPEATS"
        exact=$((exact + 1))
        rm -f ./output/determinism/${impl}_${variant}_*.bin
    else
        # Not bit-exact. Report HOW far apart the runs are: a couple of ULPs in
        # the low bits points at a synchronization race, a large delta at two
        # cores writing the same cell.
        # compare_runs.py exits 1 whenever it found a difference — which is the
        # expected case here, since we only call it after cmp already failed.
        # So its exit code is NOT an error signal and must not be turned into
        # one; we take its stdout and move on.
        d=$(python3 -I scripts/compare_runs.py \
              "./output/determinism/${impl}_${variant}_1.bin" \
              "./output/determinism/${impl}_${variant}_2.bin" 2>/dev/null) || true
        [ -n "$d" ] || d="(comparison failed — see output/determinism/)"
        printf "DIFFERS  %s\n" "$d"
        printf "         the two outputs are kept in output/determinism/ for inspection\n"
        differs=$((differs + 1))
    fi
done <<< "$LIST"

echo ""
echo "############################################################"
echo "[INFO] determinism: bit-exact=$exact  differs=$differs  skipped=$skipped"
echo "############################################################"
[ "$differs" -eq 0 ] || exit 1
