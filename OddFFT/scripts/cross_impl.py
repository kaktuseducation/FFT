#!/usr/bin/python3
# coding=utf-8
#
# Cross-implementation analysis: this work, the vendor library and the host
# reference, scored against each other across every measured length.
#
# WHAT THIS IS FOR
#
# A speedup table answers "is this kernel faster".  It does not answer the
# question a reader of a dissertation actually wants answered, which is "what
# does this tell me about the device".  The three implementations measured here
# differ along two axes at once -- hardware and implementation -- and the
# comparison becomes informative only when those are separated:
#
#   this work vs MindSpore   same device, different implementation
#                            -> what writing the kernel bought
#   MindSpore vs NumPy       different device, both vendor-grade code
#                            -> what the device itself bought
#   this work vs NumPy       both differ
#                            -> the end-to-end figure, least diagnostic
#
# The second of those is the one no speedup table reports and the one that
# carries a finding: if a vendor transform on an accelerator does not beat a
# tuned CPU library, then the accelerator is not delivering on this workload,
# and any speedup the custom kernel shows is attributable to the kernel rather
# than to the hardware.  That is a statement about the platform, it is
# measurable, and it is the kind of claim another paper can cite.
#
# Outputs:
#   cross_implementation.csv   one row per length, all three ratios
#   crossover.md               where each implementation overtakes each other
#   vendor_vs_host.md          the vendor-accelerator-vs-host finding, stated
#                              with its evidence
#   time_split.csv             where the vendor baseline spends its time
#
# Usage:
#   python3 scripts/cross_impl.py
#   python3 scripts/cross_impl.py --figure

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


def baseline_by_n(path, time_key="total_avg_us", dev_key="npu_avg_us"):
    tot = defaultdict(list)
    dev = defaultdict(list)
    for r in read_rows(path):
        N = fnum(r, "N")
        t = fnum(r, time_key)
        if N is None or t is None:
            continue
        tot[int(N)].append(t)
        d = fnum(r, dev_key)
        if d is not None:
            dev[int(N)].append(d)
    out = {}
    for n, v in tot.items():
        out[n] = {"total": summarise(v), "device": summarise(dev.get(n, []))}
    return out


def best_kernel_by_n(path):
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
                       "cpu_us": fnum(r, "cpu_us"),
                       "impl": (r.get("impl") or "").strip(),
                       "variant": (r.get("variant") or "").strip(),
                       "tile": (r.get("tile") or "").strip(),
                       "cores": (r.get("cores") or "").strip()}
    return best


def crossover(points):
    """Find where a ratio series crosses 1.0, interpolating in log-N.

    points: list of (N, ratio), ratio > 1 meaning the numerator is faster.
    Returns a list of (N_lo, N_hi, N_est) for each sign change.
    """
    pts = sorted(points)
    out = []
    for (n0, r0), (n1, r1) in zip(pts, pts[1:]):
        if r0 is None or r1 is None:
            continue
        if (r0 - 1.0) == 0:
            out.append((n0, n0, n0))
            continue
        if (r0 - 1.0) * (r1 - 1.0) < 0:
            # linear interpolation of log(ratio) against log(N)
            l0, l1 = math.log(r0), math.log(r1)
            x0, x1 = math.log(n0), math.log(n1)
            if l1 == l0:
                est = n0
            else:
                est = math.exp(x0 + (0 - l0) * (x1 - x0) / (l1 - l0))
            out.append((n0, n1, est))
    return out


def main():
    p = argparse.ArgumentParser(
        description="Cross-implementation analysis across every measured "
                    "length: what the kernel bought, and what the device "
                    "bought, separated.")
    p.add_argument("--matrix", default="./output/matrix.csv")
    p.add_argument("--numpy", default="./output/numpy_baseline.csv")
    p.add_argument("--mindspore", default="./output/mindspore_baseline.csv")
    p.add_argument("--split", default="./output/mindspore_split_all_runs.csv")
    p.add_argument("--out-dir", default="./output")
    p.add_argument("--label", default="",
                   help="project name to put in the generated text")
    p.add_argument("--figure", action="store_true")
    args = p.parse_args()

    kern = best_kernel_by_n(args.matrix)
    npb = baseline_by_n(args.numpy)
    msb = baseline_by_n(args.mindspore)

    all_n = sorted(set(list(kern) + list(npb) + list(msb)))
    if not all_n:
        print("[ERROR ] no data found")
        return 1

    os.makedirs(args.out_dir, exist_ok=True)
    rows = []

    for n in all_n:
        k = kern.get(n)
        ms = msb.get(n, {}).get("total")
        np_ = npb.get(n, {}).get("total")
        msd = msb.get(n, {}).get("device")

        row = {"N": n}
        row["kernel_us"] = "%.2f" % k["total_us"] if k else ""
        row["kernel_variant"] = ("%s/%s t%s c%s" % (k["impl"], k["variant"],
                                                    k["tile"], k["cores"])) if k else ""
        row["kernel_device_us"] = ("%.2f" % k["npu_us"]) if k and k["npu_us"] is not None else ""
        row["mindspore_us"] = "%.2f" % ms["mean"] if ms else ""
        row["mindspore_sd"] = "%.2f" % ms["sd"] if ms else ""
        row["mindspore_n"] = str(ms["n"]) if ms else "0"
        row["mindspore_device_us"] = "%.2f" % msd["mean"] if msd else ""
        row["numpy_us"] = "%.2f" % np_["mean"] if np_ else ""
        row["numpy_sd"] = "%.2f" % np_["sd"] if np_ else ""
        row["numpy_n"] = str(np_["n"]) if np_ else "0"

        def rat(a, b):
            if a and b and a > 0:
                return b / a
            return None

        r_k_ms = rat(k["total_us"] if k else None, ms["mean"] if ms else None)
        r_k_np = rat(k["total_us"] if k else None, np_["mean"] if np_ else None)
        r_ms_np = rat(ms["mean"] if ms else None, np_["mean"] if np_ else None)

        row["kernel_vs_mindspore"] = "%.4f" % r_k_ms if r_k_ms else ""
        row["kernel_vs_numpy"] = "%.4f" % r_k_np if r_k_np else ""
        row["mindspore_vs_numpy"] = "%.4f" % r_ms_np if r_ms_np else ""
        rows.append(row)

    csv_path = os.path.join(args.out_dir, "cross_implementation.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---- console ------------------------------------------------------
    print("=" * 86)
    print("Cross-implementation comparison%s" % (": " + args.label if args.label else ""))
    print("=" * 86)
    hdr = "%-9s %13s %13s %13s %11s %11s %11s" % (
        "N", "this work", "MindSpore", "NumPy",
        "vs MS", "vs NumPy", "MS/NumPy")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        def sp(key):
            v = r[key]
            if not v:
                return "---"
            f = float(v)
            return "%.2fx" % f if f >= 1 else "%.2fx-" % (1 / f)
        print("%-9s %13s %13s %13s %11s %11s %11s"
              % (r["N"], r["kernel_us"] or "---", r["mindspore_us"] or "---",
                 r["numpy_us"] or "---", sp("kernel_vs_mindspore"),
                 sp("kernel_vs_numpy"), sp("mindspore_vs_numpy")))
    print("")
    print("A trailing '-' marks a ratio below parity (the first named")
    print("implementation is the slower one).")
    print("")
    print("[WROTE ] %s" % csv_path)

    # ---- crossovers ---------------------------------------------------
    def series(key):
        return [(int(r["N"]), float(r[key])) for r in rows if r[key]]

    cross_md = ["# Crossover points", ""]
    for key, title in (("kernel_vs_numpy",
                        "This work overtakes the host reference"),
                       ("kernel_vs_mindspore",
                        "This work overtakes the vendor library"),
                       ("mindspore_vs_numpy",
                        "The vendor library overtakes the host reference")):
        s = series(key)
        cs = crossover(s)
        cross_md.append("## %s" % title)
        cross_md.append("")
        if not s:
            cross_md.append("No data.")
        elif not cs:
            allabove = all(r >= 1 for _, r in s)
            allbelow = all(r < 1 for _, r in s)
            if allabove:
                cross_md.append(
                    "No crossover in the measured range: the first "
                    "implementation is ahead at every length measured, from "
                    "N = %d to N = %d." % (s[0][0], s[-1][0]))
            elif allbelow:
                cross_md.append(
                    "No crossover in the measured range: the first "
                    "implementation is behind at every length measured, from "
                    "N = %d to N = %d." % (s[0][0], s[-1][0]))
            else:
                cross_md.append("Ratios are not monotone; see the CSV.")
        else:
            for lo, hi, est in cs:
                cross_md.append(
                    "Crosses parity between N = %d and N = %d, "
                    "interpolated at N ~ %.0f." % (lo, hi, est))
        cross_md.append("")
        cross_md.append("Measured ratios: " +
                        ", ".join("N=%d: %.2f" % (n, r) for n, r in s))
        cross_md.append("")
    cm = os.path.join(args.out_dir, "crossover.md")
    with open(cm, "w") as f:
        f.write("\n".join(cross_md) + "\n")
    print("[WROTE ] %s" % cm)

    # ---- the vendor-vs-host finding, stated with its evidence ----------
    ms_np = [(int(r["N"]), float(r["mindspore_vs_numpy"]))
             for r in rows if r["mindspore_vs_numpy"]]
    vnd = ["# The vendor accelerator library against the host reference", ""]
    if not ms_np:
        vnd.append("Not measured.")
    else:
        ratios = [r for _, r in ms_np]
        gm = math.exp(sum(math.log(r) for r in ratios) / len(ratios))
        wins = sum(1 for r in ratios if r > 1.0)
        vnd += [
            "Measured at %d lengths, from N = %d to N = %d."
            % (len(ms_np), ms_np[0][0], ms_np[-1][0]),
            "",
            "- The vendor library is faster than the host reference at "
            "%d of %d lengths." % (wins, len(ratios)),
            "- Geometric mean of the ratio: %.3f "
            "(1.000 would be exact parity)." % gm,
            "- Range: %.3f to %.3f." % (min(ratios), max(ratios)),
            "",
            "Per length: " + ", ".join("N=%d: %.3f" % (n, r) for n, r in ms_np),
            "",
            "## Why this matters",
            "",
            "Both of these are implementations their vendors tuned: one for "
            "the accelerator, one for the host CPU. Holding implementation "
            "quality roughly fixed and varying only the hardware, the ratio "
            "above is what the accelerator is worth on this workload. A "
            "figure near 1.0 says the device is not delivering an advantage "
            "through its own vendor's transform, which in turn means a "
            "speedup obtained by a custom kernel on the same device is "
            "attributable to the kernel and not to the hardware.",
            "",
            "This is the claim that separates the contribution from the "
            "platform, and it is the one worth stating in the abstract.",
        ]
    vp = os.path.join(args.out_dir, "vendor_vs_host.md")
    with open(vp, "w") as f:
        f.write("\n".join(vnd) + "\n")
    print("[WROTE ] %s" % vp)

    # ---- where the vendor spends its time ------------------------------
    if os.path.exists(args.split):
        by_n = defaultdict(lambda: defaultdict(list))
        for r in read_rows(args.split):
            N = fnum(r, "N")
            if N is None:
                continue
            for k2 in ("h2d_us", "compute_us", "d2h_us", "total_us"):
                v = fnum(r, k2)
                if v is not None:
                    by_n[int(N)][k2].append(v)
        srows = []
        for n in sorted(by_n):
            d = by_n[n]
            tot = summarise(d.get("total_us", []))
            cmp_ = summarise(d.get("compute_us", []))
            h2d = summarise(d.get("h2d_us", []))
            d2h = summarise(d.get("d2h_us", []))
            if not tot:
                continue
            srows.append({
                "N": n,
                "total_us": "%.2f" % tot["mean"],
                "compute_us": "%.2f" % cmp_["mean"] if cmp_ else "",
                "h2d_us": "%.2f" % h2d["mean"] if h2d else "",
                "d2h_us": "%.2f" % d2h["mean"] if d2h else "",
                "compute_pct": "%.1f" % (100 * cmp_["mean"] / tot["mean"]) if cmp_ else "",
                "transfer_pct": "%.1f" % (100 * ((h2d["mean"] if h2d else 0) +
                                                 (d2h["mean"] if d2h else 0)) / tot["mean"]),
                "runs": tot["n"],
            })
        if srows:
            sp_path = os.path.join(args.out_dir, "time_split.csv")
            with open(sp_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(srows[0].keys()))
                w.writeheader()
                w.writerows(srows)
            print("[WROTE ] %s" % sp_path)
            print("")
            print("Where the vendor transform spends its time")
            print("%-9s %11s %11s %11s" % ("N", "compute%", "transfer%", "runs"))
            for r in srows:
                print("%-9s %11s %11s %11s"
                      % (r["N"], r["compute_pct"], r["transfer_pct"], r["runs"]))

    if args.figure:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("[WARN  ] matplotlib missing; figure skipped")
            return 0
        fig, ax = plt.subplots(figsize=(7.4, 4.4))
        for key, lbl, mk in (
                ("kernel_vs_mindspore", "This work / MindSpore (same device)", "o"),
                ("kernel_vs_numpy", "This work / NumPy (host)", "s"),
                ("mindspore_vs_numpy", "MindSpore / NumPy (device vs host)", "^")):
            s = series(key)
            if s:
                ax.plot([n for n, _ in s], [r for _, r in s], marker=mk, label=lbl)
        ax.axhline(1.0, linestyle="--", color="0.4", linewidth=1)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("Transform length N")
        ax.set_ylabel("Ratio (>1 = numerator faster)")
        ax.set_title("Three implementations, pairwise")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fp = os.path.join(args.out_dir, "cross_implementation.png")
        fig.savefig(fp, dpi=200)
        print("[WROTE ] %s" % fp)

    return 0


if __name__ == "__main__":
    sys.exit(main())
