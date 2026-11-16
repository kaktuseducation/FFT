#!/usr/bin/python3
# coding=utf-8
#
# How far a batched invocation amortises the vendor's fixed cost.
#
# WHY THIS EXISTS
#
# Section 6.5 establishes that the vendor transform carries a cost of roughly
# 845 us that does not depend on transform length.  The obvious objection is
# that ops.fft is meant to be called inside a graph on many transforms at
# once, where such a cost is paid once and divided among them.  The objection
# is fair, and this script answers it with a measurement rather than an
# argument: it runs the same transform on batches of increasing size and
# reports the time per transform.
#
# The result bounds the claim rather than defending it.  If the per-transform
# time falls towards the arithmetic as the batch grows, then the fixed cost is
# a property of single-call use and the thesis must say so.  If it plateaus
# well above, the cost is paid per transform and the finding stands in
# general.  Either way the paper is stronger for having the number.
#
# The comparison it enables is also the honest one for a reader deciding
# whether to write a kernel: a custom kernel called once against a vendor
# transform called on a batch of B is the real choice when B transforms are
# needed, and this script supplies the vendor half of it.
#
# Usage:
#   python3 scripts/bench_batch.py -n 4096
#   python3 scripts/bench_batch.py -n 4096 --batches 1 2 4 8 16 32 64 128 256
#   python3 scripts/bench_batch.py -n 262144 --batches 1 4 16 64

import argparse
import csv
import math
import os
import sys
import time

import numpy as np


def summarise(v):
    n = len(v)
    if not n:
        return None
    m = sum(v) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in v) / (n - 1)) if n > 1 else 0.0
    return {"mean": m, "sd": sd, "min": min(v), "max": max(v), "n": n}


def get_sync(ms):
    rt = getattr(ms, "runtime", None)
    if rt is not None and hasattr(rt, "synchronize"):
        return rt.synchronize
    hal = getattr(ms, "hal", None)
    if hal is not None and hasattr(hal, "synchronize"):
        return hal.synchronize
    return None


def main():
    p = argparse.ArgumentParser(
        description="Measure how a batched invocation amortises the vendor "
                    "transform's fixed cost.")
    p.add_argument("-n", "--n", type=int, default=4096,
                   help="transform length, held fixed across batch sizes")
    p.add_argument("--batches", type=int, nargs="*",
                   default=[1, 2, 4, 8, 16, 32, 64, 128, 256])
    p.add_argument("-w", "--warmup", type=int, default=5)
    p.add_argument("-b", "--runs", type=int, default=10)
    p.add_argument("-d", "--device-id", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--csv", default="./output/batch_sweep.csv")
    p.add_argument("--device-only", action="store_true",
                   help="time the operator on a device-resident tensor, "
                        "excluding host transfers; default is end to end")
    args = p.parse_args()

    try:
        import mindspore as ms
    except ImportError as e:
        print("[ERROR ] MindSpore not importable: %s" % e)
        return 1

    ms.set_context(device_target="Ascend", device_id=args.device_id)
    sync = get_sync(ms)
    fft = getattr(ms.ops, "fft", None)
    if fft is None:
        print("[ERROR ] this MindSpore build has no ops.fft")
        return 1

    rng = np.random.default_rng(args.seed)
    N = args.n

    print("=" * 78)
    print("Batched invocation: how far the fixed cost amortises")
    print("=" * 78)
    print("MindSpore %s   N = %d   timing = %s"
          % (ms.__version__, N,
             "device-resident" if args.device_only else "end to end"))
    print("")
    print("%8s %14s %14s %12s %10s"
          % ("batch", "total_us", "per_transform", "sd_us", "runs"))
    print("-" * 62)

    rows = []
    first_per = None

    for B in args.batches:
        x = (rng.uniform(-10, 10, (B, N)) +
             1j * rng.uniform(-10, 10, (B, N))).astype(np.complex64)

        try:
            if args.device_only:
                t = ms.Tensor(x)
                if sync:
                    sync()
                for _ in range(max(args.warmup, 1)):
                    y = fft(t)
                    if sync:
                        sync()
                    else:
                        y.asnumpy()
                times = []
                for _ in range(args.runs):
                    t0 = time.perf_counter()
                    y = fft(t)
                    if sync:
                        sync()
                    else:
                        y.asnumpy()
                    times.append((time.perf_counter() - t0) * 1e6)
            else:
                for _ in range(max(args.warmup, 1)):
                    t = ms.Tensor(x)
                    y = fft(t)
                    y.asnumpy()
                times = []
                for _ in range(args.runs):
                    t0 = time.perf_counter()
                    t = ms.Tensor(x)
                    y = fft(t)
                    y.asnumpy()
                    times.append((time.perf_counter() - t0) * 1e6)
        except Exception as e:
            print("%8d  FAILED: %s" % (B, str(e)[:50]))
            continue

        s = summarise(times)
        per = s["mean"] / B
        if first_per is None:
            first_per = per
        print("%8d %14.2f %14.2f %12.2f %10d"
              % (B, s["mean"], per, s["sd"], s["n"]))
        rows.append({"N": N, "batch": B,
                     "total_us": "%.2f" % s["mean"],
                     "per_transform_us": "%.2f" % per,
                     "sd_us": "%.2f" % s["sd"],
                     "min_us": "%.2f" % s["min"],
                     "max_us": "%.2f" % s["max"],
                     "runs": s["n"],
                     "timing": "device" if args.device_only else "end_to_end"})

    if not rows:
        print("[ERROR ] nothing measured")
        return 1

    print("")
    last_per = float(rows[-1]["per_transform_us"])
    print("Per-transform time falls from %.2f us at batch 1 to %.2f us at "
          "batch %s" % (first_per, last_per, rows[-1]["batch"]))
    print("a factor of %.2f. The remaining per-transform time at the largest"
          % (first_per / last_per if last_per else 0))
    print("batch is what the fixed cost does NOT explain.")
    print("")

    # Where does amortisation stop paying?  Report the batch at which the
    # per-transform time is within 10% of the smallest observed, which is the
    # practical answer to "how large a batch do I need".
    best = min(float(r["per_transform_us"]) for r in rows)
    for r in rows:
        if float(r["per_transform_us"]) <= best * 1.10:
            print("Per-transform time reaches within 10%% of its floor "
                  "(%.2f us) at batch %s." % (best, r["batch"]))
            break

    os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
    need = not os.path.exists(args.csv)
    with open(args.csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if need:
            w.writeheader()
        w.writerows(rows)
    print("")
    print("[WROTE ] %s" % args.csv)
    print("")
    print("How to read this against the thesis: the single-call figure in")
    print("Section 6.4 is the batch-1 row here. If the per-transform time at")
    print("large batch approaches the kernel's own time, the advantage")
    print("reported in Section 6.6 is specific to single-call use and the")
    print("text must say so. If it stays well above, the advantage holds")
    print("for batched use too.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
