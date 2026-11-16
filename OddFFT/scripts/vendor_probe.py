#!/usr/bin/python3
# coding=utf-8
#
# Control experiments for the fixed-cost finding.
#
# The headline measurement -- that the vendor transform costs about 815 us
# regardless of length -- invites three objections, and each is answered by an
# experiment here rather than by argument.
#
#   1. "You measured PyNative; graph mode compiles the op and the overhead
#      goes away."
#      -> --modes runs the same transform under both execution modes and
#         reports them side by side.
#
#   2. "That is the cost of FFT specifically, not of the framework."
#      -> --trivial times an elementwise op on the same tensor through the
#         same path. If a one-pass operation on the same data also costs
#         hundreds of microseconds, the cost is not the transform's.
#
#   3. "The plateau is an artefact of where you sampled."
#      -> --plateau sweeps very short lengths (8 upward), where the work is
#         negligible and anything that remains is overhead.
#
# Nothing here replaces bench_mindspore.py: that one produces the baseline
# rows the comparison tables consume. This one produces the evidence that the
# interpretation of those rows is the right one.
#
# Usage:
#   python3 scripts/vendor_probe.py --all
#   python3 scripts/vendor_probe.py --modes --n 4096
#   python3 scripts/vendor_probe.py --plateau
#   python3 scripts/vendor_probe.py --trivial

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


def make_input(N, seed=42):
    rng = np.random.default_rng(seed)
    return (rng.uniform(-10, 10, N) + 1j * rng.uniform(-10, 10, N)).astype(np.complex64)


def get_sync(ms):
    rt = getattr(ms, "runtime", None)
    if rt is not None and hasattr(rt, "synchronize"):
        return rt.synchronize
    hal = getattr(ms, "hal", None)
    if hal is not None and hasattr(hal, "synchronize"):
        return hal.synchronize
    return None


def time_op(ms, sync, op, tensor, warmup, runs):
    """Time one op on an already-resident tensor, device-synchronised."""
    for _ in range(max(warmup, 1)):
        y = op(tensor)
        if sync:
            sync()
        else:
            y.asnumpy()
    out = []
    for _ in range(runs):
        t0 = time.perf_counter()
        y = op(tensor)
        if sync:
            sync()
        else:
            y.asnumpy()
        out.append((time.perf_counter() - t0) * 1e6)
    return out


def run_modes(args, rows):
    """The same transform under both execution modes.

    Each mode needs its own process in principle, because the mode is set on
    the context before any op runs. Within one process the second set_context
    may be ignored, so this reports which mode the runtime says it is in and
    the caller is told to cross-check with --single-mode runs if the two
    come out identical.
    """
    import mindspore as ms

    print("=" * 70)
    print("Execution mode: the same transform, PyNative against Graph")
    print("=" * 70)
    print("NOTE: mode is a context setting applied before the first op. If the")
    print("      two rows below are suspiciously identical, re-run each mode")
    print("      in its own process with --single-mode.")
    print("")

    for mode_name, mode in (("PYNATIVE", ms.PYNATIVE_MODE),
                            ("GRAPH", ms.GRAPH_MODE)):
        try:
            ms.set_context(device_target="Ascend", device_id=args.device_id,
                           mode=mode)
            sync = get_sync(ms)
            x = make_input(args.n)
            t = ms.Tensor(x)
            ts = time_op(ms, sync, ms.ops.fft, t, args.warmup, args.runs)
            s = summarise(ts)
            actual = ms.get_context("mode")
            print("%-9s requested=%s actual=%s  mean=%.2f us  sd=%.2f  "
                  "min=%.2f  max=%.2f  n=%d"
                  % (mode_name, mode, actual, s["mean"], s["sd"], s["min"],
                     s["max"], s["n"]))
            rows.append({"experiment": "mode", "detail": mode_name,
                         "N": args.n, "mean_us": "%.2f" % s["mean"],
                         "sd_us": "%.2f" % s["sd"], "min_us": "%.2f" % s["min"],
                         "max_us": "%.2f" % s["max"], "runs": s["n"],
                         "note": "actual_mode=%s" % actual})
        except Exception as e:
            print("%-9s FAILED: %s" % (mode_name, e))
            rows.append({"experiment": "mode", "detail": mode_name,
                         "N": args.n, "mean_us": "", "sd_us": "", "min_us": "",
                         "max_us": "", "runs": 0, "note": "failed: %s" % e})
    print("")


def run_trivial(args, rows):
    """An elementwise op on the same tensor, for comparison with the transform.

    If a single-pass operation over the same N complex values costs the same
    order of time as the transform does, then what is being measured is not
    the transform.
    """
    import mindspore as ms

    ms.set_context(device_target="Ascend", device_id=args.device_id)
    sync = get_sync(ms)

    print("=" * 70)
    print("Control: a trivial op against the transform, same tensor, same path")
    print("=" * 70)

    for N in args.trivial_lengths:
        x = make_input(N)
        t = ms.Tensor(x)

        candidates = [("abs", getattr(ms.ops, "abs", None)),
                      ("neg", getattr(ms.ops, "neg", None)),
                      ("fft", getattr(ms.ops, "fft", None))]
        line = []
        for name, op in candidates:
            if op is None:
                continue
            try:
                ts = time_op(ms, sync, op, t, args.warmup, args.runs)
                s = summarise(ts)
                line.append("%s=%.1f" % (name, s["mean"]))
                rows.append({"experiment": "trivial", "detail": name, "N": N,
                             "mean_us": "%.2f" % s["mean"],
                             "sd_us": "%.2f" % s["sd"],
                             "min_us": "%.2f" % s["min"],
                             "max_us": "%.2f" % s["max"],
                             "runs": s["n"], "note": ""})
            except Exception as e:
                line.append("%s=FAIL" % name)
                rows.append({"experiment": "trivial", "detail": name, "N": N,
                             "mean_us": "", "sd_us": "", "min_us": "",
                             "max_us": "", "runs": 0, "note": str(e)[:80]})
        print("N=%-8d  %s   (us, device-synchronised, no host copies)"
              % (N, "  ".join(line)))
    print("")
    print("Reading: if the elementwise ops land in the same range as fft at")
    print("the same N, the cost is the framework's dispatch, not the")
    print("transform's arithmetic.")
    print("")


def run_plateau(args, rows):
    """Very short transforms, where arithmetic is negligible."""
    import mindspore as ms

    ms.set_context(device_target="Ascend", device_id=args.device_id)
    sync = get_sync(ms)

    print("=" * 70)
    print("Plateau: transform time at lengths where the work is negligible")
    print("=" * 70)
    print("%-9s %12s %10s %10s %10s %6s" % ("N", "mean_us", "sd", "min",
                                            "max", "runs"))

    for N in args.plateau_lengths:
        try:
            x = make_input(N)
            t = ms.Tensor(x)
            ts = time_op(ms, sync, ms.ops.fft, t, args.warmup, args.runs)
            s = summarise(ts)
            print("%-9d %12.2f %10.2f %10.2f %10.2f %6d"
                  % (N, s["mean"], s["sd"], s["min"], s["max"], s["n"]))
            rows.append({"experiment": "plateau", "detail": "fft", "N": N,
                         "mean_us": "%.2f" % s["mean"],
                         "sd_us": "%.2f" % s["sd"],
                         "min_us": "%.2f" % s["min"],
                         "max_us": "%.2f" % s["max"],
                         "runs": s["n"], "note": ""})
        except Exception as e:
            print("%-9d FAILED: %s" % (N, str(e)[:60]))

    done = [r for r in rows if r["experiment"] == "plateau" and r["mean_us"]]
    if len(done) >= 2:
        lo = float(done[0]["mean_us"])
        hi = float(done[-1]["mean_us"])
        n_lo = done[0]["N"]
        n_hi = done[-1]["N"]
        work_ratio = (n_hi * math.log2(max(n_hi, 2))) / (n_lo * math.log2(max(n_lo, 2)))
        print("")
        print("Work grows by %.0fx from N=%d to N=%d; measured time by %.2fx."
              % (work_ratio, n_lo, n_hi, hi / lo if lo else 0))
        print("A time ratio far below the work ratio is the plateau.")
    print("")


def main():
    p = argparse.ArgumentParser(
        description="Control experiments for the vendor fixed-cost finding.")
    p.add_argument("--all", action="store_true",
                   help="run every experiment")
    p.add_argument("--modes", action="store_true")
    p.add_argument("--trivial", action="store_true")
    p.add_argument("--plateau", action="store_true")
    p.add_argument("--single-mode", choices=["pynative", "graph"], default=None,
                   help="set one mode and run the transform; for a clean "
                        "per-process mode comparison")
    p.add_argument("-n", "--n", type=int, default=4096)
    p.add_argument("-w", "--warmup", type=int, default=5)
    p.add_argument("-b", "--runs", type=int, default=20)
    p.add_argument("-d", "--device-id", type=int, default=0)
    p.add_argument("--plateau-lengths", type=int, nargs="*",
                   default=[8, 16, 32, 64, 128, 256, 512, 1024, 4096])
    p.add_argument("--trivial-lengths", type=int, nargs="*",
                   default=[256, 4096, 65536])
    p.add_argument("--csv", default="./output/vendor_probe.csv")
    args = p.parse_args()

    if not any([args.all, args.modes, args.trivial, args.plateau,
                args.single_mode]):
        args.all = True

    rows = []

    try:
        import mindspore as ms
        print("MindSpore %s" % ms.__version__)
        print("")
    except ImportError as e:
        print("[ERROR ] MindSpore not importable: %s" % e)
        return 1

    if args.single_mode:
        import mindspore as ms
        mode = ms.GRAPH_MODE if args.single_mode == "graph" else ms.PYNATIVE_MODE
        ms.set_context(device_target="Ascend", device_id=args.device_id,
                       mode=mode)
        sync = get_sync(ms)
        x = make_input(args.n)
        t = ms.Tensor(x)
        ts = time_op(ms, sync, ms.ops.fft, t, args.warmup, args.runs)
        s = summarise(ts)
        print("mode=%s (actual=%s)  N=%d  mean=%.2f us  sd=%.2f  min=%.2f  "
              "max=%.2f  n=%d"
              % (args.single_mode, ms.get_context("mode"), args.n, s["mean"],
                 s["sd"], s["min"], s["max"], s["n"]))
        rows.append({"experiment": "single_mode", "detail": args.single_mode,
                     "N": args.n, "mean_us": "%.2f" % s["mean"],
                     "sd_us": "%.2f" % s["sd"], "min_us": "%.2f" % s["min"],
                     "max_us": "%.2f" % s["max"], "runs": s["n"],
                     "note": "actual=%s" % ms.get_context("mode")})
    else:
        if args.all or args.plateau:
            run_plateau(args, rows)
        if args.all or args.trivial:
            run_trivial(args, rows)
        if args.all or args.modes:
            run_modes(args, rows)

    if rows and args.csv:
        os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
        need = not os.path.exists(args.csv)
        with open(args.csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["experiment", "detail", "N",
                                              "mean_us", "sd_us", "min_us",
                                              "max_us", "runs", "note"])
            if need:
                w.writeheader()
            w.writerows(rows)
        print("[WROTE ] %s" % args.csv)

    return 0


if __name__ == "__main__":
    sys.exit(main())
