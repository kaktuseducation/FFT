#!/bin/bash
# OddFFT host-side tests. No device and no numpy needed -- this is what a
# reviewer can run anywhere, and what must pass before any kernel is believed.
#
#   host_selftest       the host arithmetic against a naive O(N^2) DFT in
#                       double, including MIXED radix lists and the
#                       digit-reverse bijection
#   gen_butterfly_test  the EMITTED body of all 12 generated kernels, run
#                       scalar against a reference DFT. This is what checks
#                       the register allocation -- a calc slot freed one step
#                       too early compiles perfectly and computes a plausible
#                       wrong answer
#   arb_selftest        Bluestein (any N) and Rader (prime N) against a naive
#                       O(N^2) DFT, driven through the registry exactly as
#                       main() drives them -- so it tests the shipped path and
#                       not a second copy of the same index algebra
#   sim_odd             the generated kernels' exact index arithmetic for every
#                       radix: the result AND that every element of every stage
#                       is written exactly once across all simulated cores
#
# Between them: gen_butterfly_test covers the arithmetic, sim_odd the
# addressing, host_selftest the plan and the permutation.
set -e
cd "$(dirname "${BASH_SOURCE:-$0}")"
mkdir -p ../build_tests
CXX=${CXX:-g++}
FLAGS="-std=c++17 -O2 -Wall -Wextra -I ../host -I ../common -I ../kernels"
# OpenMP if the compiler has it, so the tests exercise the SAME code the
# benchmark runs. Without it the pragmas vanish and the tests still pass --
# that is deliberate, a reviewer should not need OpenMP to check the project --
# but then they are not checking the parallel path, so say which one ran.
if $CXX -fopenmp -x c++ -fsyntax-only /dev/null 2>/dev/null; then
    FLAGS="$FLAGS -fopenmp"
    echo "[TEST] OpenMP: ON  (the parallel host prefix is under test)"
else
    echo "[TEST] OpenMP: off (the host prefix will be checked serially only)"
fi

echo "[TEST] building"
$CXX $FLAGS -o ../build_tests/host_selftest      host_selftest.cpp
$CXX $FLAGS -o ../build_tests/gen_butterfly_test gen_butterfly_test.cpp
$CXX $FLAGS -o ../build_tests/sim_odd            sim_odd.cpp
# arb_selftest links host/impl_arblen.cpp itself, in its CPU-debug form.
# No extra include path and no CANN: neither file touches common/data_utils.h,
# which includes acl/acl.h unconditionally. An earlier version of this line
# carried a -I pointing at a scratch directory that exists only on the machine
# the patch was built on, so the test compiled there and nowhere else.
$CXX $FLAGS -DASCENDC_CPU_DEBUG \
    -o ../build_tests/arb_selftest arb_selftest.cpp ../host/impl_arblen.cpp

rc=0
echo ""
echo "=================== host_selftest ==================="
../build_tests/host_selftest | tail -32 || rc=$?
echo ""
echo "=================== gen_butterfly_test (arithmetic) ==================="
../build_tests/gen_butterfly_test || rc=$?
echo ""
echo "=================== sim_odd (addressing + coverage) ==================="
../build_tests/sim_odd || rc=$?
echo ""
echo "=================== arb_selftest (Bluestein / Rader) ==================="
# Runs from the project root: the run-functions write ./output/output_x.bin.
( cd .. && mkdir -p output && ./build_tests/arb_selftest \
    | grep -Ev '^\[(BENCH|RUN|WARMUP|RESULT)|^(NPU|CPU|Total):|^=====|^-----' ) || rc=$?

echo ""
if [ "$rc" = "0" ]; then echo "[TEST] ALL TESTS PASS"; else echo "[TEST] FAILURES (rc=$rc)"; fi
exit $rc
