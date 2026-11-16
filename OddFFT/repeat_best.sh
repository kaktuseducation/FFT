#!/usr/bin/env bash
# repeat_best.sh — repeat the FASTEST measured cell at one N, several times,
# and report the mean, the spread and the ratio against NumPy.
#
# WHY THIS EXISTS. The matrix reports one measurement per cell, and the fastest
# cell of a table is, by construction, the one that drew the kindest sample. A
# headline taken straight from it is biased upward: earlier in this project a
# matrix cell said 6.50x and five repeats of that same configuration put the
# mean at 4.77x -- the cell sat below all five. A number going into the thesis
# has to be a mean over repeats with its spread, not the best single draw.
#
# Usage:  bash repeat_best.sh [N] [REPEATS]
#         bash repeat_best.sh 100800 5
#
# Run it from the OddFFT directory, AFTER run_matrix.sh, so output/matrix.csv
# holds the table whose best cell is to be repeated.

set +e
N=${1:-100800}
REPEATS=${2:-5}
CSV=./output/matrix.csv

if [ ! -s "$CSV" ]; then
    echo "[ERROR] $CSV not found or empty -- run run_matrix.sh first"
    exit 1
fi
if [ ! -x ./build/oddfft ]; then
    echo "[ERROR] ./build/oddfft not found -- run run_matrix.sh first"
    exit 1
fi

# THE ENVIRONMENT. The binary needs the CANN runtime and its own kernel
# library, exactly as run_matrix.sh sets them up. v1 of this script did
# neither, so every run died before printing anything and all five repeats
# came back "no [RESULT] line" -- with the reason thrown away, because the
# runs were piped with 2>/dev/null. Both halves of that are fixed here: the
# environment is set, and stderr is kept.
if [ -n "${ASCEND_INSTALL_PATH:-}" ]; then
    _ASCEND=$ASCEND_INSTALL_PATH
elif [ -n "${ASCEND_HOME_PATH:-}" ]; then
    _ASCEND=$ASCEND_HOME_PATH
elif [ -d "$HOME/Ascend/ascend-toolkit/latest" ]; then
    _ASCEND=$HOME/Ascend/ascend-toolkit/latest
else
    _ASCEND=/usr/local/Ascend/ascend-toolkit/latest
fi
if [ -f "${_ASCEND}/bin/setenv.bash" ]; then
    source "${_ASCEND}/bin/setenv.bash"
else
    echo "[WARN ] ${_ASCEND}/bin/setenv.bash not found -- the runs may fail to start"
fi
export LD_LIBRARY_PATH="$(pwd)/build/lib:${LD_LIBRARY_PATH:-}"

# Column order is fixed by MATRIX_HEADER in host/main.cpp:
#   1 impl  2 variant ... 11 N  12 tile  13 min_t ... 19 total_us ... 25 status
BEST=$(awk -F, -v n="$N" '
    $11 == n && $25 == "PASS" {
        if (best == "" || $19 + 0 < best + 0) { best = $19; v = $1 "/" $2; t = $12; m = $13 }
    }
    END { if (v != "") print v, t, m, best }' "$CSV")

if [ -z "$BEST" ]; then
    echo "[ERROR] no PASSing cell at N=$N in $CSV"
    echo "        N values present:"
    awk -F, 'NR > 1 && $25 == "PASS" { print "          " $11 }' "$CSV" | sort -un
    exit 1
fi

set -- $BEST
V=$1; T=$2; M=$3; TABLE_US=$4
echo "[BEST ] from the table: $V  tile=$T  minT=$M  total_us=$TABLE_US"
echo "[INFO ] repeating it $REPEATS times with the SAME settings the table used"
echo "[INFO ] OMP_NUM_THREADS=${OMP_NUM_THREADS:-<unset>}  OMP_PROC_BIND=${OMP_PROC_BIND:-<unset>}"
echo ""

rm -f ./input/input_x.bin ./output/golden.bin
if ! python3 scripts/gen_data.py -N "$N" --any --input-dir ./input --output-dir ./output > /dev/null 2>&1; then
    echo "[ERROR] gen_data failed for N=$N"
    exit 1
fi

# NumPy on the very same input, for the ratio.
#
# The SAME anchored parse as the repeats below. bench_numpy.py prints its
# [RESULT] line with total_us, min_us, max_us, stages_npu and stages_cpu on it,
# so "the last number on the line" would quietly return stages_cpu. That class
# of mistake -- a pattern written against output nobody looked at -- is what
# made the v9 calibration fail silently, and it is not worth repeating here.
mkdir -p ./logs
python3 scripts/bench_numpy.py --input ./input/input_x.bin \
        --csv ./output/numpy_baseline.csv > ./logs/repeat_numpy.log 2>&1
NP=$(awk '/^\[RESULT\]/ {
              for (j = 1; j <= NF; j++)
                  if ($j ~ /^total_us=[0-9]/) { sub(/^total_us=/, "", $j); print $j; exit }
          }' ./logs/repeat_numpy.log)
if [ -z "$NP" ]; then
    echo "[WARN ] no NumPy baseline. What bench_numpy.py said:"
    tail -6 ./logs/repeat_numpy.log | sed 's/^/[WARN ]   /'
fi

TMP=$(mktemp)
RUNLOG=./logs/repeat_best.log
mkdir -p ./logs
for i in $(seq 1 "$REPEATS"); do
    # stdout AND stderr both to the log, then parse the log. Never 2>/dev/null:
    # that is what made the first version of this script fail in silence.
    ./build/oddfft "$V" "$T" 2 5 "$M" > "$RUNLOG" 2>&1
    us=$(awk '/^\[RESULT\]/ {
                  for (j = 1; j <= NF; j++)
                      if ($j ~ /^total_us=[0-9]/) { sub(/^total_us=/, "", $j); print $j; exit }
              }' "$RUNLOG")
    if [ -z "$us" ]; then
        echo "[WARN ] repeat $i produced no [RESULT] line. What the run actually said:"
        tail -8 "$RUNLOG" | sed 's/^/[WARN ]   /'
        continue
    fi
    printf "[REPEAT] %d/%d  total_us=%s\n" "$i" "$REPEATS" "$us"
    echo "$us" >> "$TMP"
done

echo ""
awk -v tbl="$TABLE_US" -v np="$NP" -v n="$N" '
    { x[NR] = $1; s += $1 }
    END {
        if (NR == 0) { print "[ERROR] no repeat produced a result"; exit 1 }
        mean = s / NR
        for (i = 1; i <= NR; i++) { d = x[i] - mean; ss += d * d }
        sd = (NR > 1) ? sqrt(ss / (NR - 1)) : 0
        lo = x[1]; hi = x[1]
        for (i = 2; i <= NR; i++) { if (x[i] < lo) lo = x[i]; if (x[i] > hi) hi = x[i] }
        printf "[STAT ] N=%s  repeats=%d\n", n, NR
        printf "[STAT ] mean   = %.2f us   sd = %.2f us  (cv %.1f%%)\n", mean, sd, 100 * sd / mean
        printf "[STAT ] min    = %.2f us   max = %.2f us\n", lo, hi
        printf "[STAT ] the table cell said %.2f us", tbl + 0
        if (tbl + 0 < lo) printf "  <-- BELOW every repeat: a favourable draw, do not quote it"
        else if (tbl + 0 > hi) printf "  <-- above every repeat"
        else printf "  (inside the repeat range)"
        printf "\n"
        if (np + 0 > 0) {
            printf "[STAT ] NumPy  = %.2f us\n", np + 0
            printf "[QUOTE] %.2fx faster than NumPy, mean of %d repeats (range %.2fx-%.2fx)\n", \
                   np / mean, NR, np / hi, np / lo
        } else {
            print  "[STAT ] NumPy baseline could not be read -- compute the ratio by hand"
        }
    }' "$TMP"
rm -f "$TMP"
