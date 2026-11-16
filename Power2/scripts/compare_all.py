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
# A BASELINE IS A VARIANT, NOT A FILE.  The first version of this script
# keyed on the file and took the group's name from the first row it saw.
# That was harmless while each file held one implementation, and wrong as
# soon as one did not: mindspore_baseline.csv holds both ops.fft and
# FFTWithSize, and scipy_baseline.csv holds both single and double
# precision.  Averaging those pairs into one column produced a number whose
# two halves differ on two axes at once, which is precisely the comparison
# this project argues is meaningless.  Grouping is therefore by the
# `variant` column, and one file may yield several baselines.
#
# Usage:
#   python3 scripts/compare_all.py \
#       --matrix output/matrix_final.csv \
#       --device output/mindspore_baseline.csv \
#       --host   output/numpy_baseline.csv output/scipy_baseline.csv

import argparse
import csv
import math
import os
import sys
from collections import defaultdict

# Validated categorical palette, in fixed order.  Colour follows the
# baseline, never its rank, so adding a baseline never repaints the others.
SERIES_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                  "#e87ba4", "#008300", "#6250d6", "#e34948"]


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


def load_baselines(path):
    """[(variant, {N: summary})] -- one entry per distinct variant in the file.

    Rows carrying the same variant and the same length are aggregated over
    every recorded run; rows carrying different variants are kept apart.
    """
    by_variant = defaultdict(lambda: defaultdict(list))
    fallback = os.path.splitext(os.path.basename(path))[0]
    for r in read_rows(path):
        N = fnum(r, "N")
        t = fnum(r, "total_avg_us")
        if N is None or t is None:
            continue
        v = (r.get("variant") or "").strip() or fallback
        by_variant[v][int(N)].append(t)
    return [(v, {n: summarise(vals) for n, vals in by_variant[v].items()})
            for v in sorted(by_variant)]


def collect(paths):
    """Flatten several files into one list of baselines, names kept unique."""
    out, seen, split = [], {}, []
    for p_ in paths:
        found = [(n, d) for n, d in load_baselines(p_) if d]
        if len(found) > 1:
            split.append((p_, [n for n, _ in found]))
        for name, d in found:
            if name in seen:
                # Two files carrying the same variant name: keep both, and
                # say which file each came from rather than silently merging.
                name = "%s@%s" % (name,
                                  os.path.splitext(os.path.basename(p_))[0])
            seen[name] = True
            out.append((name, d))
    return out, split


def abbrev(name, width):
    if len(name) <= width:
        return name
    keep = width - 2
    return name[:keep - 5] + ".." + name[-5:]


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
                       "npu_us": fnum(r, "npu_us"),
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

    dev, dev_split = collect(args.device)
    host, host_split = collect(args.host)

    all_n = sorted(set(list(kern) +
                       [n for _, d in dev + host for n in d]))

    print("=" * 100)
    print("Cross-implementation comparison")
    print("=" * 100)
    print("kernels     : %s  (%d verified lengths)" % (args.matrix, len(kern)))
    for name, d in dev:
        print("on-device   : %-22s %d lengths" % (name, len(d)))
    for name, d in host:
        print("host        : %-22s %d lengths" % (name, len(d)))
    for path_, names in dev_split + host_split:
        print("[NOTE  ] %s holds %d implementations and is scored as %d "
              "separate baselines: %s"
              % (path_, len(names), len(names), ", ".join(names)))
    print("")

    # Wide enough for the longest baseline name, so nothing is abbreviated
    # unless a name is genuinely unwieldy.
    W = min(24, max(13, *(len(nm) for nm, _ in dev + host))) if (dev + host) else 13
    hdr = "%-9s %11s" % ("N", "this work")
    for name, _ in dev + host:
        hdr += " %*s" % (W, abbrev(name, W))
    print(hdr)
    print("-" * len(hdr))

    rows = []
    for n in all_n:
        k = kern.get(n)
        line = "%-9d %11s" % (n, "%.2f" % k["total_us"] if k else "---")
        row = {"N": n,
               "kernel_us": "%.2f" % k["total_us"] if k else "",
               "kernel_npu_us": ("%.2f" % k["npu_us"]
                                 if k and k["npu_us"] is not None else ""),
               "kernel_variant": ("%s/%s" % (k["impl"], k["variant"])) if k else ""}
        for name, d in dev + host:
            s = d.get(n)
            line += " %*s" % (W, "%.2f" % s["mean"] if s else "---")
            row["%s_us" % name] = "%.2f" % s["mean"] if s else ""
            row["%s_runs" % name] = str(s["n"]) if s else "0"
            row["vs_%s" % name] = ("%.4f" % (s["mean"] / k["total_us"])
                                   if (k and s) else "")
        print(line)
        rows.append(row)

    # ---- summary per baseline -----------------------------------------
    print("")
    print("%-24s %-10s %9s %11s %9s %9s" % (
        "baseline", "kind", "wins/n", "geo mean", "min", "max"))
    print("-" * 78)
    for kind, group in (("on-device", dev), ("host", host)):
        for name, d in group:
            ratios = [d[n]["mean"] / kern[n]["total_us"]
                      for n in all_n if n in kern and n in d]
            if not ratios:
                continue
            wins = sum(1 for r in ratios if r > 1)
            print("%-24s %-10s %9s %11s %9.2f %9.2f"
                  % (name, kind, "%d/%d" % (wins, len(ratios)),
                     "%.3f" % gmean(ratios), min(ratios), max(ratios)))

    print("")
    print("A ratio above 1 means this work is faster. On-device baselines")
    print("isolate the implementation, because the hardware is held fixed;")
    print("host baselines vary hardware and implementation together and")
    print("answer a different question.")
    print("")
    print("Ratios use total_us on both sides, so they include host-device")
    print("transfers. kernel_npu_us is carried into the CSV for the separate")
    print("question of device-resident data, where the transfer term is absent")
    print("from both sides and must be removed from the baseline as well.")

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
        markers = "osv^Dp*h"
        for i, (name, d) in enumerate(dev + host):
            xs, ys = [], []
            for n in all_n:
                if n in kern and n in d:
                    xs.append(n)
                    ys.append(d[n]["mean"] / kern[n]["total_us"])
            if xs:
                ax.plot(xs, ys, marker=markers[i % len(markers)],
                        color=SERIES_COLOURS[i % len(SERIES_COLOURS)],
                        linewidth=2, markersize=6, label=name)
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
