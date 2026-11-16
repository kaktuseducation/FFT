#!/usr/bin/python3
# coding=utf-8
#
# A benchmark of the SciPy FFT (pocketfft) on the same input_x.bin the NPU
# variants run, written to mirror bench_numpy.py exactly: same input parsing,
# same verification against golden.bin, same CSV schema. Only the library
# differs, so the two rows form a paired comparison.
#
# WHY THIS EXISTS, beyond "a second host baseline":
#
#   1. numpy.fft.fft always computes in double precision.  Feeding it the
#      complex64 array built from input_x.bin does not change that: NumPy
#      promotes to complex128 internally and returns complex128.  The AscendC
#      kernels under study are single precision.  The existing NumPy row is
#      therefore a double-precision baseline being compared against
#      single-precision kernels, which differs on two axes at once.
#
#   2. scipy.fft preserves the input precision.  Run on complex64 it is a
#      genuine single-precision baseline, and that is the row the kernels
#      should be compared against.  Run on complex128 it is the
#      precision-matched counterpart of the NumPy row, isolating the library.
#
# So this script records up to two rows per length, tagged distinctly:
#
#   scipy_c64    single precision — compare against the kernels
#   scipy_c128   double precision — compare against numpy
#
# Threading is a third axis and is kept out of the default: scipy.fft accepts
# a `workers` argument, and using it would make the row differ from NumPy in
# both library and parallelism.  --workers is available for a deliberate,
# separately tagged measurement.

import argparse
import os
import sys
import time

import numpy as np

try:
    import scipy
    import scipy.fft as sfft
except ImportError:
    print("[ERROR ] scipy is not installed in this environment")
    sys.exit(1)

RESULTS_HEADER = ("variant,N,tile_size,npu_avg_us,cpu_avg_us,"
                  "total_avg_us,total_min_us,total_max_us\n")


def verify(out, golden_path, label, N, validation_path):
    """Same tolerance and same reporting as bench_numpy.py, so a baseline that
    would not pass the project's own check cannot be quoted as a fast one."""
    if not (golden_path and os.path.exists(golden_path)):
        return
    g = np.fromfile(golden_path, dtype=np.float32)
    golden = g[0::2] + 1j * g[1::2]
    o64 = out.astype(np.complex64)
    g64 = golden.astype(np.complex64)
    bad = int((~np.isclose(o64, g64, rtol=1e-3, atol=1e-3, equal_nan=True)).sum())
    ratio = bad / float(golden.size)
    max_abs = float(np.abs(out - golden).max())
    max_rel = float((np.abs(out - golden) /
                     np.maximum(np.abs(golden), 1e-30)).max())
    status = "PASS" if ratio <= 1e-3 else "FAIL"
    print("[VERIFY] impl=%s N=%d tile=0 status=%s mismatches=%d/%d "
          "err_ratio=%.6f tol=1.0e-03 max_abs=%.3e max_rel=%.3e"
          % (label, N, status, bad, golden.size, ratio, max_abs, max_rel))
    if validation_path:
        need_header = not os.path.exists(validation_path)
        with open(validation_path, "a") as f:
            if need_header:
                f.write("impl,variant,N,tile_size,status,mismatches,points,"
                        "err_ratio,max_abs_err,max_rel_err\n")
            f.write("scipy,%s,%d,0,%s,%d,%d,%.6f,%.6e,%.6e\n"
                    % (label, N, status, bad, golden.size, ratio,
                       max_abs, max_rel))


def bench_one(x, variant, label, warmup, runs, workers, golden_path,
              csv_path, validation_path):
    N = x.size

    print("================================================================")
    print("[BENCH ] impl=%s  N=%d  tile=0  warmup=%d  runs=%d"
          % (label, N, warmup, runs))
    print("[BENCH ] scipy %s, pocketfft, dtype=%s, workers=%d"
          % (scipy.__version__, x.dtype, workers))
    print("[BENCH ] NOTE: numpy.fft always computes in double; this row is %s"
          % ("single precision, matching the kernels"
             if x.dtype == np.complex64 else
             "double precision, matching the numpy row"))
    print("----------------------------------------------------------------")

    kw = {} if workers == 1 else {"workers": workers}
    for _ in range(warmup):
        sfft.fft(x, **kw)
    print("[WARMUP] %d/%d done" % (warmup, warmup))

    times_us = []
    out = None
    for i in range(runs):
        t0 = time.perf_counter()
        out = sfft.fft(x, **kw)
        t1 = time.perf_counter()
        dt = (t1 - t0) * 1e6
        times_us.append(dt)
        print("[RUN   ] %d/%d  npu_us=%10.2f  cpu_us=%10.2f  total_us=%10.2f"
              % (i + 1, runs, 0.0, dt, dt))

    t = np.array(times_us)
    avg, mn, mx = float(t.mean()), float(t.min()), float(t.max())
    med = float(np.median(t))

    print("----------------------------------------------------------------")
    print("CPU:   avg=%.2f us  median=%.2f us  min=%.2f us  max=%.2f us"
          % (avg, med, mn, mx))
    if avg > 1.15 * med:
        print("[NOTE  ] the mean exceeds its median by %.0f%%, so a few stalled "
              "runs are pulling it up" % (100.0 * (avg / med - 1.0)))
    print("[RESULT] impl=%s N=%d tile=0 npu_us=%.2f cpu_us=%.2f total_us=%.2f "
          "min_us=%.2f max_us=%.2f stages_npu=%d stages_cpu=%d"
          % (label, N, 0.0, avg, avg, mn, mx, 0, 0))

    verify(out, golden_path, label, N, validation_path)
    print("================================================================")

    if csv_path:
        header_needed = not os.path.exists(csv_path)
        with open(csv_path, "a") as f:
            if header_needed:
                f.write(RESULTS_HEADER)
            # npu_avg = 0: this is a host baseline, as in bench_numpy.py.
            f.write("%s,%d,0,0.00,%.2f,%.2f,%.2f,%.2f\n"
                    % (variant, N, avg, avg, mn, mx))

    return t


def main():
    p = argparse.ArgumentParser(
        description="Benchmark scipy.fft.fft on input_x.bin, as a host "
                    "baseline that can be precision-matched to the kernels.")
    p.add_argument("--input", type=str, default="./input/input_x.bin")
    p.add_argument("--golden", type=str, default="./output/golden.bin")
    p.add_argument("-w", "--warmup", type=int, default=5)
    p.add_argument("-b", "--runs", type=int, default=20)
    p.add_argument("--csv", type=str, default="./output/results.csv")
    p.add_argument("--validation", type=str, default="./output/validation.csv")
    p.add_argument("--no-csv", action="store_true")
    p.add_argument("--dtype", choices=["single", "double", "both"],
                   default="both",
                   help="single = complex64 (matches the kernels), "
                        "double = complex128 (matches numpy), both = two rows")
    p.add_argument("--workers", type=int, default=1,
                   help="scipy threads; leave at 1 to keep the comparison "
                        "with numpy on one axis")
    args = p.parse_args()

    if not os.path.exists(args.input):
        print("[ERROR ] %s not found — run scripts/gen_data.py first" % args.input)
        sys.exit(1)

    raw = np.fromfile(args.input, dtype=np.float32)
    x64 = (raw[0::2] + 1j * raw[1::2]).astype(np.complex64)

    csv_path = None if args.no_csv else args.csv
    val_path = None if args.no_csv else args.validation
    suffix = "" if args.workers == 1 else "_w%d" % args.workers

    wanted = (["single", "double"] if args.dtype == "both" else [args.dtype])
    for which in wanted:
        if which == "single":
            bench_one(x64, "scipy_c64" + suffix, "scipy/fft:c64",
                      args.warmup, args.runs, args.workers,
                      args.golden, csv_path, val_path)
        else:
            bench_one(x64.astype(np.complex128), "scipy_c128" + suffix,
                      "scipy/fft:c128", args.warmup, args.runs, args.workers,
                      args.golden, csv_path, val_path)


if __name__ == "__main__":
    main()
