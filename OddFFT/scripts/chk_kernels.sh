#!/bin/bash
# Local syntax check for AscendC kernels against the fake kernel_operator.h.
# Usage: ./chk_kernels.sh <project-dir>
#
# IMPORTANT: the include set here MIRRORS THE REAL KERNEL BUILD, which is
# ascendc_library() with no extra include directories. So the only -I given is
# the stub directory standing in for the toolkit; a kernel may include headers
# from its own directory and nothing else.
#
# This was learnt the hard way. An earlier version passed -I <proj>/common as
# well, every kernel compiled locally, and every one of them then failed on the
# device with "'tiling.h' file not found". A check more permissive than the
# build it stands in for will certify exactly the bugs it exists to catch.
set -u
PROJ="${1:?usage: chk_kernels.sh <project-dir>}"
STUBS="${ODDFFT_STUBS:-$(cd "$(dirname "$0")/.." && pwd)/stubs}"
ok=0; bad=0; skip=0; badlist=""
for f in "$PROJ"/kernels/*.cpp; do
    [ -e "$f" ] || continue
    if grep -q 'matmul_intf\.h' "$f"; then
        skip=$((skip+1)); echo "SKIP (matmul API, device-only): $(basename "$f")"; continue
    fi
    if g++ -std=c++17 -fsyntax-only -Wall \
           -include "$STUBS/gm_fix.h" -D__aicore__= -D__global__= -DASCENDC_CPU_DEBUG \
           -I "$STUBS" "$f" 2>/tmp/cl_k_err.txt; then
        ok=$((ok+1)); echo "ok   $(basename "$f")"
    else
        bad=$((bad+1)); badlist="$badlist $(basename "$f")"
        echo "FAIL $(basename "$f")"; sed -n '1,8p' /tmp/cl_k_err.txt
    fi
done
# Structural check, independent of the compiler: every quoted include in a
# kernel must name a file that sits in kernels/. The toolkit's own headers are
# angle-bracketed or well-known; anything else quoted has to be local, because
# the kernel target has no other include path. This catches the mistake at the
# level it actually happens -- a header placed in the wrong directory -- rather
# than relying on the stub set happening to be right.
echo "------------------------------------------"
echo "include-location check (every quoted include must sit in kernels/):"
TOOLKIT="kernel_operator.h|lib/matmul_intf.h|acl/|tiling/"
incbad=0
for f in "$PROJ"/kernels/*.cpp "$PROJ"/kernels/*.h; do
    [ -e "$f" ] || continue
    while read -r inc; do
        [ -n "$inc" ] || continue
        if echo "$inc" | grep -qE "^($TOOLKIT)"; then continue; fi
        if [ ! -e "$PROJ/kernels/$inc" ]; then
            echo "  BAD  $(basename "$f") includes \"$inc\" which is not in kernels/"
            incbad=$((incbad+1))
        fi
    done < <(grep -oE '#include[[:space:]]*"[^"]+"' "$f" | sed 's/.*"\(.*\)"/\1/')
done
if [ $incbad -ne 0 ]; then
    echo "  => $incbad bad include(s). The kernel build has NO extra -I, so these"
    echo "     compile here and fail on the device."
    exit 1
fi
echo "  all quoted includes are local to kernels/ -- ok"

echo "------------------------------------------"
echo "clean=$ok  failed=$bad  skipped=$skip"
if [ $bad -ne 0 ]; then echo "FAILED:$badlist"; exit 1; fi
echo "ALL CHECKED KERNELS CLEAN"
