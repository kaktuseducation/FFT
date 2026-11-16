#!/usr/bin/python3
# coding=utf-8
#
# Builds the implementation-comparison tables and the speedup figure from the
# measurement files, for Chapter 6.
#
# The point of this script is that it never states a baseline as a single
# number.  Every baseline file here holds several runs at the same length --
# numpy_baseline.csv already does, because run_matrix.sh appends a row each
# time the sweep is repeated -- and a comparison quoted against one arbitrary
# draw from that set is not reproducible: a reader with the file gets a
# different ratio depending on which row they pick.  So each length's baseline
# is reduced to mean, standard deviation and range over every recorded run,
# the speedup is quoted at the mean, and the spread is carried alongside it.
# Where a table in the dissertation currently gives a bare figure, this is what
# should replace it.
#
# Three comparisons come out:
#
#   1. vs MindSpore  -- same device, different implementation.  This is the
#      comparison that isolates the contribution of this work, because the
#      hardware is held fixed.  It is the one a reviewer will ask for.
#   2. vs NumPy      -- different device entirely.  Still worth reporting, but
#      it answers "is an NPU worth using here", not "is this kernel good".
#   3. best-variant selection per length, so the table says which
#      configuration won and is traceable back to a matrix row.
#
# Usage:
#   python3 scripts/make_comparison.py
#   python3 scripts/make_comparison.py --matrix output/matrix_final.csv
#   python3 scripts/make_comparison.py --out-dir output/ch6

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
    """Parse a numeric field, returning None rather than raising."""
    v = (row.get(key) or "").strip()
    if not v:
        return None
    try:
        return float(v)
    except ValueError:
        return None


def summarise(values):
    """mean, sd, min, max, n over a list of floats."""
    n = len(values)
    if n == 0:
        return None
    mean = sum(values) / n
    if n > 1:
        var = sum((v - mean) ** 2 for v in values) / (n - 1)
        sd = math.sqrt(var)
    else:
        sd = 0.0
    return {"mean": mean, "sd": sd, "min": min(values), "max": max(values),
            "n": n}


def collect_baseline(path, label):
    """Group every recorded run in a baseline file by N.

    The file is append-only, so repeated sweeps leave several rows per length.
    All of them are kept: the spread between them is the honest error bar on
    every ratio computed from this baseline.
    """
    by_n = defaultdict(list)
    for r in read_rows(path):
        N = fnum(r, "N")
        t = fnum(r, "total_avg_us")
        if N is None or t is None:
            continue
        by_n[int(N)].append(t)
    return {n: summarise(v) for n, v in by_n.items()}


def collect_best_variant(path):
    """For each N, the fastest configuration that verified.

    A FAIL row is never a winner, and never contributes to anything: a wrong
    answer produced quickly is not a result.  The returned record keeps enough
    of the axes to locate the row again in the matrix.
    """
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
            best[N] = {
                "total_us": t,
                "impl": (r.get("impl") or "").strip(),
                "variant": (r.get("variant") or "").strip(),
                "tile": (r.get("tile") or "").strip(),
                "cores": (r.get("cores") or "").strip(),
                "npu_us": fnum(r, "npu_us"),
                "cpu_us": fnum(r, "cpu_us"),
                "min_us": fnum(r, "total_min_us"),
                "max_us": fnum(r, "total_max_us"),
            }
    return best


def ratio_block(kernel_us, base):
    """Speedup of kernel_us against a summarised baseline.

    Returns the ratio at the baseline mean plus the range the baseline's own
    spread implies.  A ratio below 1 is a loss and is reported as such rather
    than inverted silently.
    """
    if base is None or kernel_us is None or kernel_us <= 0:
        return None
    return {
        "at_mean": base["mean"] / kernel_us,
        "lo": base["min"] / kernel_us,
        "hi": base["max"] / kernel_us,
        "base_mean": base["mean"],
        "base_sd": base["sd"],
        "base_n": base["n"],
    }


def fmt_ratio(rb):
    if rb is None:
        return "---"
    r = rb["at_mean"]
    word = "faster" if r >= 1.0 else "slower"
    shown = r if r >= 1.0 else 1.0 / r
    if rb["base_n"] > 1:
        return "%.2f/%s (%.2f-%.2f)" % (shown, word, min(rb["lo"], rb["hi"]),
                                        max(rb["lo"], rb["hi"]))
    return "%.2f/%s" % (shown, word)


def main():
    p = argparse.ArgumentParser(
        description="Build the Chapter 6 implementation-comparison tables "
                    "from the measurement files.")
    p.add_argument("--matrix", default="./output/matrix.csv")
    p.add_argument("--numpy", default="./output/numpy_baseline.csv")
    p.add_argument("--mindspore", default="./output/mindspore_baseline.csv")
    p.add_argument("--out-dir", default="./output")
    p.add_argument("--figure", action="store_true",
                   help="also render the speedup figure (needs matplotlib)")
    args = p.parse_args()

    best = collect_best_variant(args.matrix)
    if not best:
        print("[ERROR ] no PASS rows in %s" % args.matrix)
        return 1

    np_base = collect_baseline(args.numpy, "numpy")
    ms_base = collect_baseline(args.mindspore, "mindspore")

    os.makedirs(args.out_dir, exist_ok=True)

    lengths = sorted(best)
    missing_ms = [n for n in lengths if n not in ms_base]

    print("=" * 72)
    print("Implementation comparison, by transform length")
    print("=" * 72)
    print("matrix       : %s  (%d lengths with a verified winner)"
          % (args.matrix, len(lengths)))
    print("numpy base   : %s  (%d lengths)" % (args.numpy, len(np_base)))
    print("mindspore    : %s  (%d lengths)" % (args.mindspore, len(ms_base)))
    if missing_ms:
        print("")
        print("[WARN  ] no MindSpore measurement at N = %s"
              % ", ".join(str(n) for n in missing_ms))
        print("[WARN  ] those rows carry '---' in the vendor column.  Until")
        print("[WARN  ] the sweep is re-run with bench_mindspore.py they are")
        print("[WARN  ] gaps in the comparison, not values to be filled in.")
    print("")

    # ---- the main table ------------------------------------------------
    hdr = ("%-9s %-24s %12s %22s %22s" %
           ("N", "best verified variant", "total_us", "vs MindSpore (NPU)",
            "vs NumPy (CPU)"))
    print(hdr)
    print("-" * len(hdr))

    out_rows = []
    for n in lengths:
        b = best[n]
        name = "%s/%s t%s c%s" % (b["impl"], b["variant"], b["tile"],
                                  b["cores"])
        rb_ms = ratio_block(b["total_us"], ms_base.get(n))
        rb_np = ratio_block(b["total_us"], np_base.get(n))
        print("%-9d %-24s %12.2f %22s %22s"
              % (n, name, b["total_us"], fmt_ratio(rb_ms), fmt_ratio(rb_np)))

        row = {
            "N": n,
            "impl": b["impl"],
            "variant": b["variant"],
            "tile": b["tile"],
            "cores": b["cores"],
            "kernel_total_us": "%.2f" % b["total_us"],
            "kernel_npu_us": ("%.2f" % b["npu_us"]) if b["npu_us"] is not None else "",
            "kernel_cpu_us": ("%.2f" % b["cpu_us"]) if b["cpu_us"] is not None else "",
        }
        for tag, rb, base in (("mindspore", rb_ms, ms_base.get(n)),
                              ("numpy", rb_np, np_base.get(n))):
            if base is None:
                row["%s_mean_us" % tag] = ""
                row["%s_sd_us" % tag] = ""
                row["%s_runs" % tag] = "0"
                row["speedup_vs_%s" % tag] = ""
                row["speedup_vs_%s_lo" % tag] = ""
                row["speedup_vs_%s_hi" % tag] = ""
            else:
                row["%s_mean_us" % tag] = "%.2f" % base["mean"]
                row["%s_sd_us" % tag] = "%.2f" % base["sd"]
                row["%s_runs" % tag] = str(base["n"])
                row["speedup_vs_%s" % tag] = "%.4f" % rb["at_mean"]
                row["speedup_vs_%s_lo" % tag] = "%.4f" % min(rb["lo"], rb["hi"])
                row["speedup_vs_%s_hi" % tag] = "%.4f" % max(rb["lo"], rb["hi"])
        out_rows.append(row)

    table_path = os.path.join(args.out_dir, "comparison_by_length.csv")
    with open(table_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print("")
    print("[WROTE ] %s" % table_path)

    # ---- the baseline spread, stated separately -------------------------
    print("")
    print("Baseline spread (every recorded run, not a single draw)")
    print("-" * 72)
    print("%-9s %-11s %10s %9s %10s %10s %5s"
          % ("N", "baseline", "mean_us", "sd_us", "min_us", "max_us", "runs"))
    spread_rows = []
    for n in sorted(set(list(np_base) + list(ms_base))):
        for tag, d in (("mindspore", ms_base), ("numpy", np_base)):
            s = d.get(n)
            if s is None:
                continue
            print("%-9d %-11s %10.2f %9.2f %10.2f %10.2f %5d"
                  % (n, tag, s["mean"], s["sd"], s["min"], s["max"], s["n"]))
            spread_rows.append({"N": n, "baseline": tag,
                                "mean_us": "%.2f" % s["mean"],
                                "sd_us": "%.2f" % s["sd"],
                                "min_us": "%.2f" % s["min"],
                                "max_us": "%.2f" % s["max"],
                                "runs": s["n"]})
    if spread_rows:
        sp = os.path.join(args.out_dir, "baseline_spread.csv")
        with open(sp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(spread_rows[0].keys()))
            w.writeheader()
            w.writerows(spread_rows)
        print("")
        print("[WROTE ] %s" % sp)

    # ---- a paste-ready table for the text ------------------------------
    md = [
        "Table 6-X  The best verified variant at each length, against both baselines",
        "",
        "| N | Best verified variant | Total (us) | MindSpore, same device (us) | vs MindSpore | NumPy, host (us) | vs NumPy |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in out_rows:
        def cell(tag):
            m = r["%s_mean_us" % tag]
            if not m:
                return "not measured"
            n_runs = r["%s_runs" % tag]
            sd = r["%s_sd_us" % tag]
            if int(n_runs) > 1:
                return "%s +- %s (n=%s)" % (m, sd, n_runs)
            return "%s (n=1)" % m

        def sp(tag):
            v = r["speedup_vs_%s" % tag]
            if not v:
                return "---"
            f = float(v)
            if f >= 1.0:
                return "**%.2fx faster**" % f
            return "%.2fx slower" % (1.0 / f)

        md.append("| %s | %s/%s, tile %s, %s cores | **%s** | %s | %s | %s | %s |"
                  % (r["N"], r["impl"], r["variant"], r["tile"], r["cores"],
                     r["kernel_total_us"], cell("mindspore"), sp("mindspore"),
                     cell("numpy"), sp("numpy")))
    md += [
        "",
        "Every baseline figure is the mean over the runs recorded in the",
        "measurement files, with the standard deviation and the run count",
        "given, and every speedup is computed against that mean.  The ranges",
        "behind each ratio are in output/comparison_by_length.csv.",
    ]
    md_path = os.path.join(args.out_dir, "comparison_table.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md) + "\n")
    print("[WROTE ] %s" % md_path)

    if args.figure:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("[WARN  ] matplotlib not available; figure skipped")
            return 0

        xs = [n for n in lengths]
        ms_y = [float(r["speedup_vs_mindspore"]) if r["speedup_vs_mindspore"]
                else None for r in out_rows]
        np_y = [float(r["speedup_vs_numpy"]) if r["speedup_vs_numpy"]
                else None for r in out_rows]

        fig, ax = plt.subplots(figsize=(7.0, 4.2))
        for ys, lbl, mk in ((ms_y, "vs MindSpore (same device)", "o"),
                            (np_y, "vs NumPy (host CPU)", "s")):
            px = [x for x, y in zip(xs, ys) if y is not None]
            py = [y for y in ys if y is not None]
            if px:
                ax.plot(px, py, marker=mk, label=lbl)
        ax.axhline(1.0, linestyle="--", linewidth=1, color="0.4")
        ax.annotate("parity", xy=(xs[0], 1.0), xytext=(2, 4),
                    textcoords="offset points", fontsize=8, color="0.3")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("Transform length N")
        ax.set_ylabel("Speedup (>1 = this work is faster)")
        ax.set_title("This work against both baselines")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=9)
        fig.tight_layout()
        fp = os.path.join(args.out_dir, "speedup_both_baselines.png")
        fig.savefig(fp, dpi=200)
        print("[WROTE ] %s" % fp)

    return 0


if __name__ == "__main__":
    sys.exit(main())
