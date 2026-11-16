#!/usr/bin/python3
# coding=utf-8
#
# A benchmark of PyTorch's transform on the Ascend device, through the
# torch_npu backend.  This is the second vendor-supported path to the same
# silicon: MindSpore and PyTorch are both maintained by or with Huawei for
# this hardware, and they do not necessarily share an implementation.
#
# Why a second vendor path matters.  The fixed cost established in Section 6.5
# was measured through MindSpore alone, and a reader may reasonably ask
# whether it belongs to the device or to that one framework.  If PyTorch on
# the same chip carries a cost of the same order, the finding is about the
# platform.  If it does not, the finding is about MindSpore, which is a
# narrower and differently interesting claim.  Either outcome is worth having,
# and the measurement is the only way to tell them apart.
#
# The output format matches bench_numpy.py and bench_mindspore.py exactly, so
# rows from all three land in the same tables and the same comparison scripts
# consume them without a special case.
#
# Usage:
#   python3 scripts/bench_torch_npu.py                       # N from input_x.bin
#   python3 scripts/bench_torch_npu.py -b 20 -w 5
#   python3 scripts/bench_torch_npu.py --device-id 0

import argparse
import os
import sys
import time

import numpy as np

RESULTS_HEADER = ("variant,N,tile_size,npu_avg_us,cpu_avg_us,"
                  "total_avg_us,total_min_us,total_max_us\n")

SPLIT_HEADER = "impl,N,run,h2d_us,compute_us,d2h_us,total_us\n"


def resolve_device(torch, torch_npu, device_id):
    """Return the device string for the Ascend backend.

    torch_npu registers an "npu" device type on import.  Older builds expose
    it only after an explicit set_device, which is why that call is made here
    rather than relied upon implicitly.
    """
    if not hasattr(torch, "npu"):
        raise RuntimeError(
            "torch.npu is absent: torch_npu did not register its backend. "
            "Check that the torch_npu version matches the installed torch.")
    if not torch.npu.is_available():
        raise RuntimeError(
            "torch.npu.is_available() is False: no Ascend device visible to "
            "PyTorch. Check the CANN environment and ASCEND_RT_VISIBLE_DEVICES.")
    torch.npu.set_device(device_id)
    return "npu:%d" % device_id


def main():
    p = argparse.ArgumentParser(
        description="Benchmark PyTorch's FFT on the Ascend device through "
                    "torch_npu, for comparison with the custom kernels and "
                    "with MindSpore on the same silicon.")
    p.add_argument("--input", default="./input/input_x.bin")
    p.add_argument("--golden", default="./output/golden.bin")
    p.add_argument("-w", "--warmup", type=int, default=5,
                   help="warmup iterations; the first call pays for device "
                        "init and operator compilation and must be excluded")
    p.add_argument("-b", "--runs", type=int, default=20)
    p.add_argument("-d", "--device-id", type=int, default=0)
    p.add_argument("--csv", default="./output/torchnpu_baseline.csv")
    p.add_argument("--split", default="./output/torchnpu_split_all_runs.csv")
    p.add_argument("--validation", default="./output/validation.csv")
    p.add_argument("--no-csv", action="store_true")
    args = p.parse_args()

    if not os.path.exists(args.input):
        print("[ERROR ] %s not found - run scripts/gen_data.py first"
              % args.input)
        return 1

    try:
        import torch
    except ImportError as e:
        print("[ERROR ] PyTorch is not importable: %s" % e)
        return 1
    try:
        import torch_npu  # noqa: F401  (registers the npu backend on import)
    except ImportError as e:
        print("[ERROR ] torch_npu is not importable: %s" % e)
        print("[ERROR ] without it PyTorch cannot reach the Ascend device; "
              "the run is left with no row rather than a guessed one")
        return 1

    raw = np.fromfile(args.input, dtype=np.float32)
    N = raw.size // 2
    x_np = (raw[0::2] + 1j * raw[1::2]).astype(np.complex64)

    try:
        dev = resolve_device(torch, torch_npu, args.device_id)
    except RuntimeError as e:
        print("[ERROR ] %s" % e)
        return 1

    x_cpu = torch.from_numpy(x_np)

    print("================================================================")
    print("[BENCH ] impl=torch_npu/fft  N=%d  tile=0  warmup=%d  runs=%d"
          % (N, args.warmup, args.runs))
    print("[BENCH ] torch=%s  device=%s  dtype=%s"
          % (torch.__version__, dev, x_cpu.dtype))
    print("[BENCH ] twiddles=internal (torch) - vendor baseline")
    print("----------------------------------------------------------------")

    # The warmup is not optional. The first call on a fresh process pays for
    # device initialisation and operator compilation, which on this platform
    # can exceed the steady state by orders of magnitude.
    for _ in range(max(args.warmup, 1)):
        t = x_cpu.to(dev)
        y = torch.fft.fft(t)
        torch.npu.synchronize()
    print("[WARMUP] %d/%d done" % (max(args.warmup, 1), max(args.warmup, 1)))

    h2d_us, comp_us, d2h_us, tot_us = [], [], [], []
    out = None

    for i in range(args.runs):
        t0 = time.perf_counter()

        t = x_cpu.to(dev)
        torch.npu.synchronize()
        t1 = time.perf_counter()

        y = torch.fft.fft(t)
        torch.npu.synchronize()
        t2 = time.perf_counter()

        out = y.cpu().numpy()
        t3 = time.perf_counter()

        h2d = (t1 - t0) * 1e6
        comp = (t2 - t1) * 1e6
        d2h = (t3 - t2) * 1e6
        tot = (t3 - t0) * 1e6
        h2d_us.append(h2d); comp_us.append(comp)
        d2h_us.append(d2h); tot_us.append(tot)

        print("[RUN   ] %d/%d  npu_us=%10.2f  cpu_us=%10.2f  total_us=%10.2f"
              % (i + 1, args.runs, comp, h2d + d2h, tot))

        if args.split and not args.no_csv:
            need = not os.path.exists(args.split)
            with open(args.split, "a") as f:
                if need:
                    f.write(SPLIT_HEADER)
                f.write("torch_npu,%d,%d,%.2f,%.2f,%.2f,%.2f\n"
                        % (N, i + 1, h2d, comp, d2h, tot))

    h2d_a = np.array(h2d_us); comp_a = np.array(comp_us)
    d2h_a = np.array(d2h_us); tot_a = np.array(tot_us)
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
          % (avg, mn, mx, tot_a.std(ddof=1) if args.runs > 1 else 0.0))
    print("[RESULT] impl=torch_npu/fft N=%d tile=0 npu_us=%.2f cpu_us=%.2f "
          "total_us=%.2f min_us=%.2f max_us=%.2f stages_npu=%d stages_cpu=%d"
          % (N, npu_avg, cpu_avg, avg, mn, mx, 0, 0))

    # Verified against the same golden file, under the same tolerance, as
    # every kernel and every other baseline.  A baseline that fails its own
    # correctness check cannot be used to judge anything.
    if args.golden and os.path.exists(args.golden) and out is not None:
        g = np.fromfile(args.golden, dtype=np.float32)
        golden = (g[0::2] + 1j * g[1::2]).astype(np.complex64)
        o = out.astype(np.complex64)
        bad = int((~np.isclose(o, golden, rtol=1e-3, atol=1e-3,
                               equal_nan=True)).sum())
        ratio = bad / float(golden.size)
        max_abs = float(np.abs(o - golden).max())
        max_rel = float((np.abs(o - golden) /
                         np.maximum(np.abs(golden), 1e-30)).max())
        status = "PASS" if ratio <= 1e-3 else "FAIL"
        print("[VERIFY] impl=torch_npu/fft N=%d tile=0 status=%s "
              "mismatches=%d/%d err_ratio=%.6f tol=1.0e-03 max_abs=%.3e "
              "max_rel=%.3e"
              % (N, status, bad, golden.size, ratio, max_abs, max_rel))
        if args.validation and not args.no_csv:
            need = not os.path.exists(args.validation)
            with open(args.validation, "a") as f:
                if need:
                    f.write("impl,variant,N,tile_size,status,mismatches,"
                            "points,err_ratio,max_abs_err,max_rel_err\n")
                f.write("torch_npu,fft,%d,0,%s,%d,%d,%.6f,%.6e,%.6e\n"
                        % (N, status, bad, golden.size, ratio, max_abs,
                           max_rel))

    print("================================================================")

    if args.csv and not args.no_csv:
        need = not os.path.exists(args.csv)
        with open(args.csv, "a") as f:
            if need:
                f.write(RESULTS_HEADER)
            f.write("torch_npu,%d,0,%.2f,%.2f,%.2f,%.2f,%.2f\n"
                    % (N, npu_avg, cpu_avg, avg, mn, mx))
    return 0


if __name__ == "__main__":
    sys.exit(main())
