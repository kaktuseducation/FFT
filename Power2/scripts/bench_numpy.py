#!/usr/bin/python3
# coding=utf-8
#
# A benchmark of the NumPy FFT on that same input_x.bin the NPU variants run.
# The output is in the same [BENCH ]/[RESULT] format as the C++ binary's, so
# that the comparison reads uniformly. NumPy is additionally checked against
# golden.bin (the [VERIFY] line) — the baseline has to pass the test too.

import argparse
import os
import sys
import time

import numpy as np

RESULTS_HEADER = ("variant,N,tile_size,npu_avg_us,cpu_avg_us,"
                  "total_avg_us,total_min_us,total_max_us\n")


def bench_numpy(input_path, warmup, runs, csv_path=None, golden_path=None,
                validation_path=None):
    raw = np.fromfile(input_path, dtype=np.float32)
    N = raw.size // 2
    x = raw[0::2] + 1j * raw[1::2]

    print("================================================================")
    print("[BENCH ] impl=numpy/fft  N=%d  tile=0  warmup=%d  runs=%d" % (N, warmup, runs))
    print("[BENCH ] twiddles=internal (numpy) — baseline")
    print("----------------------------------------------------------------")

    for _ in range(warmup):
        np.fft.fft(x)
    print("[WARMUP] %d/%d done" % (warmup, warmup))

    times_us = []
    out = None
    for i in range(runs):
        t0 = time.perf_counter()
        out = np.fft.fft(x)
        t1 = time.perf_counter()
        dt = (t1 - t0) * 1e6
        times_us.append(dt)
        print("[RUN   ] %d/%d  npu_us=%10.2f  cpu_us=%10.2f  total_us=%10.2f"
              % (i + 1, runs, 0.0, dt, dt))

    times_us = np.array(times_us)
    avg, mn, mx = times_us.mean(), times_us.min(), times_us.max()

    print("----------------------------------------------------------------")
    print("NPU:   avg=%.2f us  min=%.2f us  max=%.2f us" % (0.0, 0.0, 0.0))
    print("CPU:   avg=%.2f us  min=%.2f us  max=%.2f us" % (avg, mn, mx))
    print("Total: avg=%.2f us  min=%.2f us  max=%.2f us" % (avg, mn, mx))
    print("[RESULT] impl=numpy/fft N=%d tile=0 npu_us=%.2f cpu_us=%.2f total_us=%.2f "
          "min_us=%.2f max_us=%.2f stages_npu=%d stages_cpu=%d"
          % (N, 0.0, avg, avg, mn, mx, 0, 0))

    if golden_path and os.path.exists(golden_path):
        g = np.fromfile(golden_path, dtype=np.float32)
        golden = g[0::2] + 1j * g[1::2]
        bad = int((~np.isclose(out.astype(np.complex64), golden.astype(np.complex64),
                               rtol=1e-3, atol=1e-3, equal_nan=True)).sum())
        ratio = bad / float(golden.size)
        max_abs = float(np.abs(out - golden).max())
        status = "PASS" if ratio <= 1e-3 else "FAIL"
        print("[VERIFY] impl=numpy/fft N=%d tile=0 status=%s mismatches=%d/%d "
              "err_ratio=%.6f tol=1.0e-03 max_abs=%.3e max_rel=%.3e"
              % (N, status, bad, golden.size, ratio, max_abs,
                 float((np.abs(out - golden) / np.maximum(np.abs(golden), 1e-30)).max())))
        if validation_path:
            need_header = not os.path.exists(validation_path)
            with open(validation_path, "a") as f:
                if need_header:
                    f.write("impl,variant,N,tile_size,status,mismatches,points,"
                            "err_ratio,max_abs_err,max_rel_err\n")
                f.write("numpy,fft,%d,0,%s,%d,%d,%.6f,%.6e,%.6e\n"
                        % (N, status, bad, golden.size, ratio, max_abs,
                           float((np.abs(out - golden) /
                                  np.maximum(np.abs(golden), 1e-30)).max())))

    print("================================================================")

    if csv_path:
        header_needed = not os.path.exists(csv_path)
        with open(csv_path, "a") as f:
            if header_needed:
                f.write(RESULTS_HEADER)
            # npu_avg=0 (not relevant), total = the numpy latency itself,
            # so that it lands in the same table as append_csv_row()'s rows
            f.write("numpy,%d,0,0.00,%.2f,%.2f,%.2f,%.2f\n" % (N, avg, avg, mn, mx))

    return times_us


def parse_args():
    p = argparse.ArgumentParser(
        description="Benchmark numpy.fft.fft on input_x.bin for comparison with NPU variants.")
    p.add_argument("--input", type=str, default="./input/input_x.bin")
    p.add_argument("--golden", type=str, default="./output/golden.bin")
    p.add_argument("-w", "--warmup", type=int, default=5)
    p.add_argument("-b", "--runs", type=int, default=20)
    p.add_argument("--csv", type=str, default="./output/results.csv")
    p.add_argument("--validation", type=str, default="./output/validation.csv")
    p.add_argument("--no-csv", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not os.path.exists(args.input):
        print("[ERROR ] %s not found — run scripts/gen_data.py first" % args.input)
        sys.exit(1)
    bench_numpy(args.input, args.warmup, args.runs,
                None if args.no_csv else args.csv,
                args.golden,
                None if args.no_csv else args.validation)