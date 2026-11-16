#!/usr/bin/python3
# coding=utf-8
#
# A benchmark of the MindSpore FFT on that same input_x.bin the custom kernels
# run, executed on the same Ascend device.  This is the *vendor baseline*: the
# transform a user of this hardware would reach for if they did not write a
# kernel themselves.  It is the comparison that matters for the claim this
# project makes, because it holds the device fixed and varies only the
# implementation -- unlike the NumPy baseline, which varies the device as well.
#
# The output format is deliberately identical to bench_numpy.py's, so that both
# baselines land in the same tables and the same paired-comparison machinery
# applies to them without a special case.
#
# The timing is split three ways, because for an accelerator baseline the split
# is the interesting part:
#
#   h2d_us      host -> device copy of the input
#   compute_us  the transform itself, with the device synchronised on both ends
#   d2h_us      device -> host copy of the result
#   total_us    all three, which is what a caller actually waits for
#
# compute_us maps onto the npu_us column of the kernel harness and the two
# copies onto cpu_us, so a row from here is directly comparable with a row from
# the C++ binary.  Synchronisation is explicit: without it the compute figure
# would measure the time to *queue* the op, not to run it, and would come out
# absurdly small.  See sync() below for how the barrier is located across
# MindSpore versions.

import argparse
import os
import sys
import time

import numpy as np

RESULTS_HEADER = ("variant,N,tile_size,npu_avg_us,cpu_avg_us,"
                  "total_avg_us,total_min_us,total_max_us\n")

# A wider CSV kept alongside the compatible one: same rows, but with the
# three-way split preserved rather than folded into npu/cpu.  The dissertation
# needs the split to explain where the vendor baseline spends its time.
SPLIT_HEADER = ("impl,N,run,h2d_us,compute_us,d2h_us,total_us\n")


def resolve_fft(ms, prefer="auto"):
    """Return (fft_callable, name) for whatever this MindSpore exposes.

    mindspore.ops.fft arrived in 2.2 and is the one to prefer: it takes a
    complex tensor and returns a complex tensor, matching numpy.fft.fft.
    Older builds only have the FFTWithSize primitive, whose signature differs
    enough that it is wrapped rather than called directly.
    """
    ops = ms.ops

    def make_prim():
        # signal_ndim=1, inverse=False, real=False is the complex-to-complex
        # forward transform; norm="backward" matches NumPy, which applies no
        # scaling on the forward pass.
        FFTWithSize = getattr(ops, "FFTWithSize", None)
        if FFTWithSize is None:
            return None
        return FFTWithSize(signal_ndim=1, inverse=False, real=False,
                           norm="backward")

    # The two are separate code paths inside MindSpore and need not perform
    # alike, which is why they can be selected explicitly rather than only
    # falling back one to the other.
    if prefer == "fftwithsize":
        prim = make_prim()
        if prim is None:
            raise RuntimeError("this MindSpore build has no ops.FFTWithSize")
        return prim, "mindspore.ops.FFTWithSize"

    if prefer in ("auto", "fft"):
        fn = getattr(ops, "fft", None)
        if callable(fn):
            return fn, "mindspore.ops.fft"
        if prefer == "fft":
            raise RuntimeError("this MindSpore build has no ops.fft")

    prim = make_prim()
    if prim is not None:
        return prim, "mindspore.ops.FFTWithSize"

    raise RuntimeError(
        "this MindSpore build exposes neither ops.fft nor ops.FFTWithSize; "
        "run `python3 -c \"import mindspore; print(mindspore.__version__)\"` "
        "and report the version")


def resolve_sync(ms):
    """Return a callable that blocks until the device queue is empty.

    The barrier moved between releases: mindspore.runtime.synchronize() in
    2.4+, mindspore.hal.synchronize() in 2.2-2.3.  If neither is present the
    caller must fall back to forcing a copy back, which is what the None
    return signals.
    """
    runtime = getattr(ms, "runtime", None)
    if runtime is not None and hasattr(runtime, "synchronize"):
        return runtime.synchronize, "mindspore.runtime.synchronize"

    hal = getattr(ms, "hal", None)
    if hal is not None and hasattr(hal, "synchronize"):
        return hal.synchronize, "mindspore.hal.synchronize"

    return None, "none (falling back to asnumpy as the barrier)"


def bench_mindspore(input_path, warmup, runs, device_id, csv_path=None,
                    golden_path=None, validation_path=None, split_path=None,
                    prefer="auto"):
    import mindspore as ms

    raw = np.fromfile(input_path, dtype=np.float32)
    N = raw.size // 2
    x = (raw[0::2] + 1j * raw[1::2]).astype(np.complex64)

    ms.set_context(device_target="Ascend", device_id=device_id)
    fft, fft_name = resolve_fft(ms, prefer)
    sync, sync_name = resolve_sync(ms)

    def barrier(t):
        """Block until t is actually computed on the device."""
        if sync is not None:
            sync()
            return None
        # No explicit barrier available: pulling the data back is itself one,
        # but it also costs a d2h copy, so the caller is told what happened.
        return t.asnumpy()

    print("================================================================")
    print("[BENCH ] impl=mindspore/fft  N=%d  tile=0  warmup=%d  runs=%d"
          % (N, warmup, runs))
    print("[BENCH ] op=%s  device=Ascend:%d  dtype=complex64"
          % (fft_name, device_id))
    print("[BENCH ] barrier=%s" % sync_name)
    print("[BENCH ] twiddles=internal (mindspore) - vendor baseline")
    print("----------------------------------------------------------------")

    # The warmup is not optional here and is not the same thing as the NumPy
    # warmup.  The first call on a fresh context pays for device init, graph
    # compilation and the op's own tiling lookup; on Ascend that first call can
    # be two orders of magnitude slower than the steady state.  Measuring it
    # would not describe the baseline a user experiences on a warm process.
    for _ in range(max(warmup, 1)):
        t = ms.Tensor(x)
        y = fft(t)
        barrier(y)
    print("[WARMUP] %d/%d done" % (max(warmup, 1), max(warmup, 1)))

    h2d_us, compute_us, d2h_us, total_us = [], [], [], []
    out = None

    for i in range(runs):
        t0 = time.perf_counter()

        t = ms.Tensor(x)
        if sync is not None:
            sync()
        t1 = time.perf_counter()

        y = fft(t)
        if sync is not None:
            sync()
        t2 = time.perf_counter()

        out = y.asnumpy()
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
            need_header = not os.path.exists(split_path)
            with open(split_path, "a") as f:
                if need_header:
                    f.write(SPLIT_HEADER)
                f.write("mindspore,%d,%d,%.2f,%.2f,%.2f,%.2f\n"
                        % (N, i + 1, h2d, comp, d2h, tot))

    h2d_a = np.array(h2d_us)
    comp_a = np.array(compute_us)
    d2h_a = np.array(d2h_us)
    tot_a = np.array(total_us)

    npu_avg = comp_a.mean()
    cpu_avg = h2d_a.mean() + d2h_a.mean()
    avg, mn, mx = tot_a.mean(), tot_a.min(), tot_a.max()

    print("----------------------------------------------------------------")
    print("H2D:   avg=%.2f us  min=%.2f us  max=%.2f us"
          % (h2d_a.mean(), h2d_a.min(), h2d_a.max()))
    print("NPU:   avg=%.2f us  min=%.2f us  max=%.2f us"
          % (npu_avg, comp_a.min(), comp_a.max()))
    print("D2H:   avg=%.2f us  min=%.2f us  max=%.2f us"
          % (d2h_a.mean(), d2h_a.min(), d2h_a.max()))
    print("CPU:   avg=%.2f us  (h2d+d2h)" % cpu_avg)
    print("Total: avg=%.2f us  min=%.2f us  max=%.2f us  sd=%.2f"
          % (avg, mn, mx, tot_a.std(ddof=1) if runs > 1 else 0.0))
    print("[RESULT] impl=mindspore/fft N=%d tile=0 npu_us=%.2f cpu_us=%.2f "
          "total_us=%.2f min_us=%.2f max_us=%.2f stages_npu=%d stages_cpu=%d"
          % (N, npu_avg, cpu_avg, avg, mn, mx, 0, 0))

    # The baseline is verified against the same golden.bin the kernels are
    # checked against, on the same tolerance.  A baseline that does not pass
    # its own correctness test cannot be used to judge anything, and a vendor
    # op quietly computing in reduced precision would show up right here.
    if golden_path and os.path.exists(golden_path) and out is not None:
        g = np.fromfile(golden_path, dtype=np.float32)
        golden = (g[0::2] + 1j * g[1::2]).astype(np.complex64)
        o = out.astype(np.complex64)
        bad = int((~np.isclose(o, golden, rtol=1e-3, atol=1e-3,
                               equal_nan=True)).sum())
        ratio = bad / float(golden.size)
        max_abs = float(np.abs(o - golden).max())
        max_rel = float((np.abs(o - golden) /
                         np.maximum(np.abs(golden), 1e-30)).max())
        status = "PASS" if ratio <= 1e-3 else "FAIL"
        print("[VERIFY] impl=mindspore/fft N=%d tile=0 status=%s "
              "mismatches=%d/%d err_ratio=%.6f tol=1.0e-03 max_abs=%.3e "
              "max_rel=%.3e"
              % (N, status, bad, golden.size, ratio, max_abs, max_rel))
        if validation_path:
            need_header = not os.path.exists(validation_path)
            with open(validation_path, "a") as f:
                if need_header:
                    f.write("impl,variant,N,tile_size,status,mismatches,points,"
                            "err_ratio,max_abs_err,max_rel_err\n")
                f.write("mindspore,fft,%d,0,%s,%d,%d,%.6f,%.6e,%.6e\n"
                        % (N, status, bad, golden.size, ratio, max_abs, max_rel))

    print("================================================================")

    if csv_path:
        header_needed = not os.path.exists(csv_path)
        with open(csv_path, "a") as f:
            if header_needed:
                f.write(RESULTS_HEADER)
            tag = ("mindspore_fftwithsize"
                   if "FFTWithSize" in fft_name else "mindspore")
            f.write("%s,%d,0,%.2f,%.2f,%.2f,%.2f,%.2f\n"
                    % (tag, N, npu_avg, cpu_avg, avg, mn, mx))

    return tot_a


def parse_args():
    p = argparse.ArgumentParser(
        description="Benchmark MindSpore's FFT on input_x.bin, on the same "
                    "Ascend device as the custom kernels, for comparison "
                    "against them.")
    p.add_argument("--input", type=str, default="./input/input_x.bin")
    p.add_argument("--golden", type=str, default="./output/golden.bin")
    p.add_argument("-w", "--warmup", type=int, default=5,
                   help="warmup iterations; the first call pays for device "
                        "init and op compilation, so this must be >= 1")
    p.add_argument("-b", "--runs", type=int, default=20)
    p.add_argument("-d", "--device-id", type=int, default=0)
    p.add_argument("--csv", type=str, default="./output/mindspore_baseline.csv")
    p.add_argument("--split", type=str,
                   default="./output/mindspore_split_all_runs.csv",
                   help="per-run h2d/compute/d2h breakdown, every run kept")
    p.add_argument("--validation", type=str, default="./output/validation.csv")
    p.add_argument("--op", choices=["auto", "fft", "fftwithsize"],
                   default="auto",
                   help="which MindSpore entry point to measure: ops.fft, "
                        "the FFTWithSize primitive, or whichever is present")
    p.add_argument("--no-csv", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not os.path.exists(args.input):
        print("[ERROR ] %s not found - run scripts/gen_data.py first"
              % args.input)
        sys.exit(1)
    try:
        bench_mindspore(args.input, args.warmup, args.runs, args.device_id,
                        None if args.no_csv else args.csv,
                        args.golden,
                        None if args.no_csv else args.validation,
                        None if args.no_csv else args.split,
                        args.op)
    except ImportError as e:
        print("[ERROR ] MindSpore is not importable in this environment: %s" % e)
        print("[ERROR ] the NPU baseline cannot be measured without it; the "
              "run is left with no row rather than a guessed one")
        sys.exit(1)
