#!/bin/bash
# Build the aclRfft1D benchmark against the installed CANN toolkit.
#
# The library names differ slightly between CANN releases, so this probes for
# what is actually present rather than assuming a fixed link line, and prints
# what it found. If the link fails, the message says which library was
# missing rather than leaving a raw linker error.

set -u

ASCEND="${ASCEND_HOME_PATH:-${ASCEND_TOOLKIT_HOME:-/usr/local/Ascend/ascend-toolkit/latest}}"
if [ ! -d "$ASCEND" ]; then
    echo "[ERROR] CANN toolkit not found at $ASCEND"
    echo "        source set_env.sh first, or set ASCEND_HOME_PATH"
    exit 1
fi

ARCH=$(uname -m)
case "$ARCH" in
    aarch64) SUB="aarch64-linux" ;;
    x86_64)  SUB="x86_64-linux"  ;;
    *)       echo "[ERROR] unexpected architecture $ARCH"; exit 1 ;;
esac

INC="$ASCEND/$SUB/include"
LIB="$ASCEND/$SUB/lib64"
[ -d "$INC" ] || INC="$ASCEND/include"
[ -d "$LIB" ] || LIB="$ASCEND/lib64"

echo "[INFO] toolkit : $ASCEND"
echo "[INFO] include : $INC"
echo "[INFO] lib64   : $LIB"

HDR="$INC/aclnnop/acl_rfft1d.h"
if [ ! -f "$HDR" ]; then
    echo "[ERROR] $HDR not found."
    echo "        This CANN release has no rfft1d operator; nothing to measure."
    exit 1
fi
echo "[OK  ] header  : aclnnop/acl_rfft1d.h"

LINK=""
for want in ascendcl opapi nnopbase; do
    if ls "$LIB"/lib${want}.so >/dev/null 2>&1; then
        LINK="$LINK -l$want"
        echo "[OK  ] library : lib${want}.so"
    else
        echo "[WARN] library : lib${want}.so not found, omitting from link"
    fi
done

if [ -z "$LINK" ]; then
    echo "[ERROR] none of the expected CANN libraries are present in $LIB"
    exit 1
fi

SRC="$(dirname "$0")/bench_aclrfft.cpp"
OUT="$(dirname "$0")/bench_aclrfft"

set -x
g++ -std=c++17 -O2 "$SRC" -o "$OUT" \
    -I"$INC" -I"$INC/aclnn" -I"$INC/aclnnop" \
    -L"$LIB" $LINK \
    -Wl,-rpath,"$LIB"
rc=$?
set +x

if [ $rc -ne 0 ]; then
    echo ""
    echo "[FAIL] compilation failed."
    echo "       If the error is an undefined reference to aclCreateTensor,"
    echo "       this release keeps it in a different library; try adding"
    echo "       -lnnopbase or -lopapi explicitly, and check:"
    echo "         nm -D $LIB/libnnopbase.so | grep aclCreateTensor"
    exit $rc
fi

echo ""
echo "[OK  ] built $OUT"
echo ""
echo "Run it:"
echo "  $OUT --n 4096 --runs 20"
echo "  $OUT --n 64 --runs 20      # short length, where the plateau shows"
echo "  for N in 8 16 32 64 128 256 512 1024 4096 65536; do"
echo "      $OUT --n \$N --runs 10 --csv output/aclrfft_baseline.csv"
echo "  done"
