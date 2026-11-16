#!/usr/bin/python3
# coding=utf-8
#
# A benchmark of the Ascend signal-processing library's transform, reached
# through mindscience.sciops.fft.ASD_FFT, on the same input_x.bin the custom
# kernels run.
#
# WHY THIS ONE MATTERS MORE THAN THE OTHERS
#
# aclRfft1D, already measured, is the CANN operator: one real-to-complex
# transform, no complex-to-complex path at the operator level at all.  That
# absence is one of this dissertation's premises.  ASD_FFT comes from a
# different library -- the signal-processing library, which ships its own FFT
# kernel and its own plan framework -- and it IS complex-to-complex.  So it is
# not a second wrapper over the same code.  It is a second vendor
# implementation of the computation this work implements, on the same silicon.
#
# Three outcomes are possible and all three are worth having:
#
#   it is slower than these kernels   the claim stands against a second
#                                     vendor implementation, not just one
#   it is comparable                  the honest finding is parity with a
#                                     library most users never find
#   it is faster                      then the thesis reports that, says by
#                                     how much, and the method still stands:
#                                     the method is how the design was
#                                     chosen, not the speed it reached
#
# The third is the reason to run this now rather than after the text is
# finished.
#
# WHAT IT DOES NOT DO
#
# It does not claim ASD_FFT is the same algorithm, or tuned for these lengths,
# or that the comparison is like for like in any sense beyond running the same
# transform of the same input on the same device.  That is the only sense a
# user cares about, and it is the sense the rest of the comparisons use.
#
# The output format matches bench_mindspore.py and bench_numpy.py exactly, so
# these rows land in the same tables and make_comparison.py consumes them
# without a special case.
#
# Usage:
#   python3 scripts/bench_asdsip.py                      # N from input_x.bin
#   python3 scripts/bench_asdsip.py -b 20 -w 5
#   python3 scripts/bench_asdsip.py --csv ./output/asdsip_baseline.csv

import argparse
import os
import sys
import time

import numpy as np

RESULTS_HEADER = ("variant,N,tile_size,npu_avg_us,cpu_avg_us,"
                  "total_avg_us,total_min_us,total_max_us\n")
SPLIT_HEADER = ("impl,N,run,h2d_us,compute_us,d2h_us,total_us\n")
VALIDATION_HEADER = "variant,N,max_abs_err,max_rel_err,verdict\n"


def resolve_asd_fft():
    """Return (callable(xr, xi) -> (yr, yi), name).

    The library is reachable under more than one spelling depending on the
    release: a class ASD_FFT that is instantiated, or a function asd_fft.
    Both are tried rather than assuming one, because guessing wrong here
    reads as "the library is absent" when it is merely spelled differently.
    """
    from mindscience.sciops import fft as sfft

    cls = getattr(sfft, "ASD_FFT", None)
    if cls is not None:
        op = cls()
        return (lambda xr, xi: op(xr, xi)), "mindscience.sciops.fft.ASD_FFT"

    fn = getattr(sfft, "asd_fft", None)
    if fn is not None:
        return (lambda xr, xi: fn(xr, xi)), "mindscience.sciops.fft.asd_fft"

    raise RuntimeError("mindscience.sciops.fft exposes neither ASD_FFT nor "
                       "asd_fft; run probe_impls.py to see what it does have")


def resolve_sync(ms):
    """Block until the device queue is empty.

    Same barrier problem as bench_mindspore.py: the call moved between
    releases.  Without it the compute figure measures the time to queue the
    op rather than to run it, and comes out absurdly small.
    """
    runtime = getattr(ms, "runtime", None)
    if runtime is not None and hasattr(runtime, "synchronize"):
        return runtime.synchronize, "mindspore.runtime.synchronize"
    hal = getattr(ms, "hal", None)
    if hal is not None and hasattr(hal, "synchronize"):
        return hal.synchronize, "mindspore.hal.synchronize"
    return None, "none (falling back to asnumpy as the barrier)"


def bench(input_path, warmup, runs, device_id, csv_path, golden_path,
          validation_path, split_path):
    import mindspore as ms

    raw = np.fromfile(input_path, dtype=np.float32)
    N = raw.size // 2
    # ASD_FFT takes the real and imaginary parts as two float32 tensors
    # rather than one complex tensor, and the example gives them a leading
    # batch dimension, so the input is shaped (1, N).
    xr_np = np.ascontiguousarray(raw[0::2]).reshape(1, N)
    xi_np = np.ascontiguousarray(raw[1::2]).reshape(1, N)

    ms.set_context(device_target="Ascend", device_id=device_id)
    fft, fft_name = resolve_asd_fft()
    sync, sync_name = resolve_sync(ms)

    print("================================================================")
    print("[BENCH ] impl=asdsip  N=%d  tile=0  warmup=%d  runs=%d"
          % (N, warmup, runs))
    print("[BENCH ] op=%s" % fft_name)
    print("[BENCH ] barrier=%s" % sync_name)
    print("================================================================")

    def barrier(t):
        if sync is not None:
            sync()
            return None
        return t.asnumpy()

    for _ in range(max(warmup, 1)):
        yr, yi = fft(ms.Tensor(xr_np), ms.Tensor(xi_np))
        barrier(yr)
    print("[WARMUP] %d/%d done" % (max(warmup, 1), max(warmup, 1)))

    h2d_us, compute_us, d2h_us, total_us = [], [], [], []
    out = None

    for i in range(runs):
        t0 = time.perf_counter()

        tr = ms.Tensor(xr_np)
        ti = ms.Tensor(xi_np)
        if sync is not None:
            sync()
        t1 = time.perf_counter()

        yr, yi = fft(tr, ti)
        if sync is not None:
            sync()
        t2 = time.perf_counter()

        out = yr.asnumpy().ravel() + 1j * yi.asnumpy().ravel()
        t3 = time.perf_counter()

        h2d = (t1 - t0) * 1e6
        comp = (t2 - t1) * 1e6
        d2h = (t3 - t2) * 1e6
        tot = (t3 - t0) * 1e6
        h2d_us.append(h2d)
        compute_us.append(comp)
        d2h_us.append(d2h)
        total_us.append(tot)

        print("[RUN   ] %d/%d  npu_us=%10.2f  cpu_us=%10.2f  total_us=%10.2f"
              % (i + 1, runs, comp, h2d + d2h, tot))

        if split_path:
            need = not os.path.exists(split_path)
            with open(split_path, "a") as f:
                if need:
                    f.write(SPLIT_HEADER)
                f.write("asdsip,%d,%d,%.2f,%.2f,%.2f,%.2f\n"
                        % (N, i + 1, h2d, comp, d2h, tot))

    comp_a = np.array(compute_us)
    tot_a = np.array(total_us)
    npu_avg = comp_a.mean()
    cpu_avg = np.array(h2d_us).mean() + np.array(d2h_us).mean()
    avg, mn, mx = tot_a.mean(), tot_a.min(), tot_a.max()

    print("----------------------------------------------------------------")
    print("NPU:   avg=%.2f us" % npu_avg)
    print("CPU:   avg=%.2f us  (h2d %.2f + d2h %.2f)"
          % (cpu_avg, np.array(h2d_us).mean(), np.array(d2h_us).mean()))
    print("TOTAL: avg=%.2f us  min=%.2f us  max=%.2f us" % (avg, mn, mx))

    # Verification against the same golden file every other implementation is
    # checked against.  An implementation that is fast and wrong is not a
    # comparison point, and this is the only thing that tells them apart.
    if golden_path and os.path.exists(golden_path) and out is not None:
        g = np.fromfile(golden_path, dtype=np.float32)
        gold = g[0::2] + 1j * g[1::2]
        n = min(gold.size, out.size)
        abs_err = np.abs(out[:n] - gold[:n])
        denom = np.maximum(np.abs(gold[:n]), 1e-6)
        max_abs = float(abs_err.max())
        max_rel = float((abs_err / denom).max())
        verdict = "PASS" if max_rel < 1e-3 else "FAIL"
        print("VERIFY: max_abs=%.3e  max_rel=%.3e  %s"
              % (max_abs, max_rel, verdict))
        if validation_path:
            need = not os.path.exists(validation_path)
            with open(validation_path, "a") as f:
                if need:
                    f.write(VALIDATION_HEADER)
                f.write("asdsip,%d,%.6e,%.6e,%s\n"
                        % (N, max_abs, max_rel, verdict))
    else:
        print("VERIFY: no golden file at %s, skipped" % golden_path)

    if csv_path:
        need = not os.path.exists(csv_path)
        with open(csv_path, "a") as f:
            if need:
                f.write(RESULTS_HEADER)
            f.write("asdsip,%d,0,%.2f,%.2f,%.2f,%.2f,%.2f\n"
                    % (N, npu_avg, cpu_avg, avg, mn, mx))
        print("[CSV   ] appended to %s" % csv_path)
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, default="./input/input_x.bin")
    p.add_argument("--golden", type=str, default="./output/golden.bin")
    p.add_argument("-w", "--warmup", type=int, default=5)
    p.add_argument("-b", "--runs", type=int, default=20)
    p.add_argument("-d", "--device-id", type=int, default=0)
    p.add_argument("--csv", type=str, default="./output/asdsip_baseline.csv")
    p.add_argument("--split", type=str, default="./output/asdsip_split.csv")
    p.add_argument("--validation", type=str, default="./output/validation.csv")
    p.add_argument("--no-csv", action="store_true")
    a = p.parse_args()

    if not os.path.exists(a.input):
        print("no input at %s -- generate it the way the kernel harness does"
              % a.input)
        return 2
    try:
        return bench(a.input, a.warmup, a.runs, a.device_id,
                     None if a.no_csv else a.csv, a.golden,
                     None if a.no_csv else a.validation,
                     None if a.no_csv else a.split)
    except Exception as exc:
        # A missing library is a result, not a crash: it goes in the text as
        # "sought, not available on this image".
        print("[FAIL  ] %s: %s" % (type(exc).__name__, exc))
        return 1


if __name__ == "__main__":
    sys.exit(main())
