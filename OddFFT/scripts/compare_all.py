#!/usr/bin/python3
# coding=utf-8
#
# Cross-implementation comparison over an arbitrary number of baselines.
#
# cross_impl.py was written when there were two baselines and names them in
# its arguments.  This script takes any number, so that torch_npu, scipy and
# anything measured later drop in without editing the analysis.  It keeps the
# distinction that matters: baselines that run on the device under study are
# scored separately from those that run on the host, because a ratio against
# the first isolates the implementation and a ratio against the second does
# not.
#
# Every baseline file is a CSV in the shared format that bench_numpy.py,
# bench_mindspore.py, bench_torch_npu.py and bench_scipy.py all write:
#   variant,N,tile_size,npu_avg_us,cpu_avg_us,total_avg_us,total_min_us,total_max_us
# Repeated rows at the same length are aggregated, never taken singly.
#
# Usage:
#   python3 scripts/compare_all.py \
#       --matrix output/matrix_final.csv \
#       --device output/mindspore_baseline.csv output/torchnpu_baseline.csv \
#       --host   output/numpy_baseline.csv output/scipy_baseline.csv

import argparse
import csv
import math
import os
import sys
from collections import defaultdict


def read_rows(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def fnum(row, key):
    v = (row.get(key) or "").strip()
    if not v:
        return None
    try:
        return float(v)
    except ValueError:
        return None


def summarise(vals):
    n = len(vals)
    if not n:
        return None
    mean = sum(vals) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / (n - 1)) if n > 1 else 0.0
    return {"mean": mean, "sd": sd, "min": min(vals), "max": max(vals), "n": n}


def gmean(xs):
    xs = [x for x in xs if x and x > 0]
    if not xs:
        return None
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def load_baseline(path):
    """{N: summary} over every recorded run in one baseline file."""
    by_n = defaultdict(list)
    label = None
    for r in read_rows(path):
        N = fnum(r, "N")
        t = fnum(r, "total_avg_us")
        if N is None or t is None:
            continue
        by_n[int(N)].append(t)
        if label is None:
            label = (r.get("variant") or "").strip()
    name = label or os.path.splitext(os.path.basename(path))[0]
    return name, {n: summarise(v) for n, v in by_n.items()}


def best_kernel(path):
    best = {}
    for r in read_rows(path):
        if (r.get("status") or "").strip().upper() != "PASS":
            continue
        N = fnum(r, "N")
        t = fnum(r, "total_us")
        if N is None or t is None:
            continue
        N = int(N)
        if N not in best or t < best[N]["total_us"]:
            best[N] = {"total_us": t,
                       "impl": (r.get("impl") or "").strip(),
                       "variant": (r.get("variant") or "").strip(),
                       "tile": (r.get("tile") or "").strip(),
                       "cores": (r.get("cores") or "").strip()}
    return best


def main():
    p = argparse.ArgumentParser(
        description="Compare this work against any number of baselines, "
                    "keeping on-device and host baselines apart.")
    p.add_argument("--matrix", default="./output/matrix.csv")
    p.add_argument("--device", nargs="*", default=[],
                   help="baseline CSVs measured on the device under study")
    p.add_argument("--host", nargs="*", default=[],
                   help="baseline CSVs measured on the host CPU")
    p.add_argument("--out", default="./output/compare_all.csv")
    p.add_argument("--figure", action="store_true")
    args = p.parse_args()

    kern = best_kernel(args.matrix)
    if not kern:
        print("[ERROR ] no PASS rows in %s" % args.matrix)
        return 1

    dev = [load_baseline(p_) for p_ in args.device]
    host = [load_baseline(p_) for p_ in args.host]
    dev = [(n, d) for n, d in dev if d]
    host = [(n, d) for n, d in host if d]

    all_n = sorted(set(list(kern) +
                       [n for _, d in dev + host for n in d]))

    print("=" * 96)
    print("Cross-implementation comparison")
    print("=" * 96)
    print("kernels     : %s  (%d verified lengths)" % (args.matrix, len(kern)))
    for name, d in dev:
        print("on-device   : %-14s %d lengths" % (name, len(d)))
    for name, d in host:
        print("host        : %-14s %d lengths" % (name, len(d)))
    print("")

    hdr = "%-9s %11s" % ("N", "this work")
    for name, _ in dev + host:
        hdr += " %13s" % name[:13]
    print(hdr)
    print("-" * len(hdr))

    rows = []
    for n in all_n:
        k = kern.get(n)
        line = "%-9d %11s" % (n, "%.2f" % k["total_us"] if k else "---")
        row = {"N": n,
               "kernel_us": "%.2f" % k["total_us"] if k else "",
               "kernel_variant": ("%s/%s" % (k["impl"], k["variant"])) if k else ""}
        for name, d in dev + host:
            s = d.get(n)
            line += " %13s" % ("%.2f" % s["mean"] if s else "---")
            row["%s_us" % name] = "%.2f" % s["mean"] if s else ""
            row["%s_runs" % name] = str(s["n"]) if s else "0"
            if k and s:
                row["vs_%s" % name] = "%.4f" % (s["mean"] / k["total_us"])
            else:
                row["vs_%s" % name] = ""
        print(line)
        rows.append(row)

    # ---- summary per baseline -----------------------------------------
    print("")
    print("%-16s %-10s %9s %11s %9s %9s" % (
        "baseline", "kind", "wins/n", "geo mean", "min", "max"))
    print("-" * 70)
    for kind, group in (("on-device", dev), ("host", host)):
        for name, d in group:
            ratios = []
            for n in all_n:
                k = kern.get(n)
                s = d.get(n)
                if k and s:
                    ratios.append(s["mean"] / k["total_us"])
            if not ratios:
                continue
            wins = sum(1 for r in ratios if r > 1)
            print("%-16s %-10s %9s %11s %9.2f %9.2f"
                  % (name, kind, "%d/%d" % (wins, len(ratios)),
                     "%.3f" % gmean(ratios), min(ratios), max(ratios)))

    print("")
    print("A ratio above 1 means this work is faster. On-device baselines")
    print("isolate the implementation, because the hardware is held fixed;")
    print("host baselines vary hardware and implementation together and")
    print("answer a different question.")

    if rows:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        fields = []
        for r in rows:
            for k_ in r:
                if k_ not in fields:
                    fields.append(k_)
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        print("")
        print("[WROTE ] %s" % args.out)

    if args.figure:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("[WARN  ] matplotlib missing; figure skipped")
            return 0
        fig, ax = plt.subplots(figsize=(7.6, 4.4))
        markers = "osv^Dp*"
        for i, (name, d) in enumerate(dev + host):
            xs, ys = [], []
            for n in all_n:
                k = kern.get(n)
                s = d.get(n)
                if k and s:
                    xs.append(n)
                    ys.append(s["mean"] / k["total_us"])
            if xs:
                ax.plot(xs, ys, marker=markers[i % len(markers)], label=name)
        ax.axhline(1.0, linestyle="--", color="0.4", linewidth=1)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("Transform length N")
        ax.set_ylabel("Ratio (>1 = this work faster)")
        ax.set_title("This work against every measured baseline")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fp = os.path.join(os.path.dirname(args.out) or ".",
                          "compare_all.png")
        fig.savefig(fp, dpi=200)
        print("[WROTE ] %s" % fp)

    return 0


if __name__ == "__main__":
    sys.exit(main())
