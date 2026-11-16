#!/bin/bash
# Tidy ~/work without losing anything.
#
# Nothing is deleted. Everything judged unnecessary is moved into
# ~/work/_trash/, so a mistake costs one `mv` to undo. Delete that folder
# yourself once you are satisfied.
#
#   bash cleanup.sh          # show what would move, change nothing
#   bash cleanup.sh --do     # actually move it
#
# What is kept:
#   Power2/ OddFFT/ AlignProbe/   the projects
#   samples_matmul_ref/           Huawei's reference sample, cited in Ch.2
#   sweep_baselines.sh            the sweep driver
#   *.json                        CANN runtime config, harmless and small
#   lost+found                    filesystem's own, never touch
#
# What is moved aside:
#   *.tar.gz        already unpacked; the projects are the live copies
#   npu-baseline/   staging folder; its scripts are in the projects now
#   *.log *.txt     old terminal captures
#   loose *.csv     superseded by the ones inside output/

set -u
WORK="${1:-$HOME/work}"
if [ "$WORK" = "--do" ]; then WORK="$HOME/work"; fi

DO=0
for a in "$@"; do [ "$a" = "--do" ] && DO=1; done

cd "$WORK" || { echo "[ERROR] no $WORK"; exit 1; }
TRASH="$WORK/_trash"

echo "=============================================="
echo " Tidying $WORK"
[ $DO -eq 0 ] && echo " DRY RUN — nothing will move. Add --do to apply."
echo "=============================================="
echo ""

# ---- 1. check the projects are intact before touching anything ----
echo "Projects:"
ok=1
for p in Power2 OddFFT AlignProbe; do
    if [ -d "$p" ]; then
        ns=$(ls "$p/scripts" 2>/dev/null | wc -l)
        no=$(ls "$p/output" 2>/dev/null | wc -l)
        nk=$(ls "$p/kernels" 2>/dev/null | wc -l)
        printf "  %-12s scripts=%-3s output=%-3s kernels=%-3s %s\n" \
               "$p" "$ns" "$no" "$nk" "$(du -sh "$p" 2>/dev/null | cut -f1)"
    else
        printf "  %-12s MISSING\n" "$p"
        [ "$p" != "AlignProbe" ] && ok=0
    fi
done
echo ""

if [ $ok -eq 0 ]; then
    echo "[STOP] A project directory is missing. Restore it from the archive"
    echo "       before tidying:"
    echo "         tar xzf 24SF03451-FFT-NPU-evidence.tar.gz"
    exit 1
fi

# ---- 2. the measurement files that must survive ----
echo "Measurement files (these must not be lost):"
for p in Power2 OddFFT; do
    for f in matrix.csv matrix_final.csv numpy_baseline.csv \
             mindspore_baseline.csv vendor_probe.csv; do
        [ -f "$p/output/$f" ] && \
            printf "  %-10s %-26s %s\n" "$p" "$f" \
                   "$(wc -l < "$p/output/$f") rows"
    done
done
echo ""

# ---- 3. what will be moved aside ----
echo "To be moved into _trash/:"
MOVE=()
for f in *.tar.gz; do [ -e "$f" ] && MOVE+=("$f"); done
for f in *.log; do [ -e "$f" ] && MOVE+=("$f"); done
for f in commads.txt; do [ -e "$f" ] && MOVE+=("$f"); done
for f in matrix_B64_B128.csv csv_for_fit.tar.gz; do
    [ -e "$f" ] && MOVE+=("$f")
done
[ -d npu-baseline ] && MOVE+=("npu-baseline")

if [ ${#MOVE[@]} -eq 0 ]; then
    echo "  (nothing — already tidy)"
else
    total=0
    for f in "${MOVE[@]}"; do
        sz=$(du -sk "$f" 2>/dev/null | cut -f1)
        total=$((total + sz))
        printf "  %-34s %s\n" "$f" "$(du -sh "$f" 2>/dev/null | cut -f1)"
    done
    echo ""
    printf "  total: %d MB\n" $((total / 1024))
fi
echo ""

echo "Kept as is:"
for f in samples_matmul_ref sweep_baselines.sh aclinit.json \
         fusion_result.json lost+found; do
    [ -e "$f" ] && printf "  %s\n" "$f"
done
echo ""

if [ $DO -eq 0 ]; then
    echo "=============================================="
    echo "Nothing changed. To apply:  bash cleanup.sh --do"
    echo "=============================================="
    exit 0
fi

# ---- 4. apply ----
mkdir -p "$TRASH"
n=0
for f in "${MOVE[@]}"; do
    mv -- "$f" "$TRASH/" 2>/dev/null && n=$((n + 1))
done
echo "=============================================="
echo "Moved $n item(s) into $TRASH"
echo ""
echo "Check that everything still works, then delete it:"
echo "  rm -rf $TRASH"
echo ""
echo "To undo instead:"
echo "  mv $TRASH/* $WORK/"
echo "=============================================="
ls -la "$WORK"
