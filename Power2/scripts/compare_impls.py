#!/usr/bin/python3
# coding=utf-8
#
# Every implementation of this transform measured on this device, side by
# side, and the one question the new rows were collected to answer.
#
# THE QUESTION
#
# Section 6.5 established a length-independent fixed cost of roughly 845 us
# in the vendor transform and decomposed it into dispatch, planning and
# copies. That measurement went through MindSpore alone, and a reader may
# reasonably object that the finding belongs to that framework rather than
# to the device -- which would make it a much narrower claim, and a less
# interesting one.
#
# PyTorch on the same chip settles it. The two frameworks are maintained
# separately, reach the device through different runtimes, and share no FFT
# code. If both carry a fixed cost of the same order, the cost is the
# device's. If PyTorch does not, the cost is MindSpore's, and the thesis
# says so instead. Either way the claim stops being an inference from one
# sample.
#
# HOW THE FIXED COST IS SEPARATED
#
# For a transform the work grows as N log2 N, so a total time that is
#
#     t(N) = fixed + rate * N * log2(N)
#
# has an intercept that no amount of work explains. Fitting that line over
# the measured lengths gives the intercept directly, with the residual
# spread as a check that the model is not being forced. The fit is
# unweighted least squares on the raw times, which is the right choice here
# because the question is about absolute microseconds at small N, not about
# relative error at large N.
#
# A caution the output repeats: an intercept is an extrapolation to N = 0,
# which is not a transform anybody runs. It is read as "the part of the cost
# that does not come from the work", and nothing more.
#
# Usage:
#   python3 scripts/compare_impls.py --dir ./output
#   python3 scripts/compare_impls.py --dir ./output --md ./output/impls.md

import argparse
import csv
import glob
import math
import os
import sys

# label -> (file stem, what it is, does it run on the device)
SOURCES = [
    ("this work",   None,                    "custom AscendC kernels",      True),
    ("ops.fft",     "mindspore_baseline",    "MindSpore 2.4, ops.fft",      True),
    ("FFTWithSize", "fftwithsize_baseline",  "MindSpore 2.4, primitive",    True),
    ("torch.fft",   "torch_npu_baseline",    "PyTorch 2.1 via torch_npu",   True),
    ("ASD_FFT",     "asdsip_baseline",       "Ascend signal-processing lib", True),
    ("NumPy",       "numpy_baseline",        "host reference",              False),
]


def load(path):
    """length -> mean total time, averaging repeated rows at a length."""
    if not path or not os.path.exists(path):
        return {}
    acc = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                n, t = int(r["N"]), float(r["total_avg_us"])
            except (KeyError, ValueError, TypeError):
                continue
            if t > 0:
                acc.setdefault(n, []).append(t)
    return {n: sum(v) / len(v) for n, v in acc.items()}


def load_kernels(d):
    """Best verified kernel time per length, from the matrix."""
    best = {}
    for name in ("matrix_final.csv", "matrix.csv"):
        p = os.path.join(d, name)
        if not os.path.exists(p):
            continue
        with open(p) as f:
            for r in csv.DictReader(f):
                try:
                    n, t = int(r["N"]), float(r["total_us"])
                except (KeyError, ValueError, TypeError):
                    continue
                st = (r.get("status") or "").strip().upper()
                if t > 0 and st in ("", "OK", "PASS"):
                    if n not in best or t < best[n]:
                        best[n] = t
        if best:
            break
    return best


def fit_fixed(series):
    """Least squares of t = fixed + rate * N log2 N. Returns the intercept,
    the slope, the residual RMS and the number of points."""
    pts = [(n, t) for n, t in series.items() if n >= 2]
    if len(pts) < 3:
        return None
    xs = [n * math.log2(n) for n, _ in pts]
    ys = [t for _, t in pts]
    k = len(xs)
    mx, my = sum(xs) / k, sum(ys) / k
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return None
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    inter = my - slope * mx
    rms = math.sqrt(sum((y - (inter + slope * x)) ** 2
                        for x, y in zip(xs, ys)) / k)
    # A fit whose residual is the size of the data is not a fit. FFTWithSize
    # in the first sweep gave an intercept of 91 us with a residual RMS of
    # 14605, and the intercept was reported anyway -- which then dragged the
    # "best vendor fixed cost" down to a number that means nothing. The
    # intercept is only reported when the straight line actually describes
    # the times.
    span = max(ys) - min(ys)
    ok = span > 0 and rms < 0.25 * span
    return inter, slope, rms, k, ok


def geomean(vals):
    vals = [v for v in vals if v > 0]
    if not vals:
        return float("nan")
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="./output")
    p.add_argument("--md", default=None, help="also write a markdown table")
    a = p.parse_args()

    data, present = {}, []
    for label, stem, what, on_dev in SOURCES:
        s = (load_kernels(a.dir) if stem is None
             else load(os.path.join(a.dir, stem + ".csv")))
        if s:
            data[label] = s
            present.append((label, what, on_dev))

    if "NumPy" not in data:
        print("no numpy_baseline.csv in %s; nothing to compare against" % a.dir)
        return 2

    lengths = sorted(set().union(*[set(s) for s in data.values()]))
    labels = [l for l, _, _ in present]

    print("=" * 78)
    print("Implementations found in %s" % a.dir)
    print("=" * 78)
    for label, what, on_dev in present:
        print("  %-12s %-32s %s  (%d lengths)"
              % (label, what, "device" if on_dev else "host ",
                 len(data[label])))

    print()
    print("Total time, microseconds")
    print("-" * 78)
    print("%-9s" % "N" + "".join("%12s" % l for l in labels))
    for n in lengths:
        row = "%-9d" % n
        for l in labels:
            v = data[l].get(n)
            row += "%12s" % ("%.1f" % v if v else "-")
        print(row)

    print()
    print("Against the host reference, times faster")
    print("-" * 78)
    dev = [l for l in labels if l != "NumPy"]
    print("%-9s" % "N" + "".join("%12s" % l for l in dev))
    ratios = {l: [] for l in dev}
    for n in lengths:
        h = data["NumPy"].get(n)
        if not h:
            continue
        row = "%-9d" % n
        for l in dev:
            v = data[l].get(n)
            if v:
                ratios[l].append(h / v)
                row += "%11.2fx" % (h / v)
            else:
                row += "%12s" % "-"
        print(row)
    print("-" * 78)
    print("%-9s" % "geomean" + "".join("%11.2fx" % geomean(ratios[l])
                                       for l in dev))
    print("%-9s" % "wins" + "".join("%12s" % ("%d/%d" % (
        sum(1 for r in ratios[l] if r > 1), len(ratios[l]))) for l in dev))

    print()
    print("Fixed cost: the intercept of t = fixed + rate * N log2 N")
    print("-" * 78)
    print("%-13s %12s %14s %12s %7s" %
          ("", "fixed, us", "rate, us/op", "resid RMS", "points"))
    fits = {}
    for l in labels:
        f = fit_fixed(data[l])
        if not f:
            continue
        inter, slope, rms, k, ok = f
        print("%-13s %12.1f %14.3e %12.1f %7d  %s"
              % (l, inter, slope, rms, k,
                 "" if ok else "<- residual too large, fit rejected"))
        if ok:
            fits[l] = f

    print()
    print("What this says")
    print("-" * 78)
    devfits = {l: f for l, f in fits.items()
               if l not in ("NumPy", "this work")}
    if len(devfits) >= 2:
        vals = {l: f[0] for l, f in devfits.items()}
        lo, hi = min(vals.values()), max(vals.values())
        names = ", ".join("%s %.0f us" % (l, v) for l, v in sorted(
            vals.items(), key=lambda kv: kv[1]))
        print("  Device implementations and their fixed cost: %s." % names)
        if lo > 0 and hi / lo < 2.0:
            print("  These agree within a factor of %.2f across separately"
                  % (hi / lo))
            print("  maintained frameworks, so the fixed cost is a property of")
            print("  the platform rather than of any one framework. That is the")
            print("  stronger form of the Section 6.5 claim and it is now")
            print("  supported by more than one sample.")
        else:
            print("  These differ by a factor of %.2f, which is too much to" % (hi / lo))
            print("  call one platform cost. Report them separately and say")
            print("  that the Section 6.5 figure belongs to MindSpore.")
    else:
        print("  Only one device implementation has enough lengths to fit.")
        print("  Run the sweep for the others before drawing the conclusion.")

    if "this work" in fits and devfits:
        own = fits["this work"][0]
        best = min(f[0] for f in devfits.values())
        print()
        print("  These kernels carry %.0f us of fixed cost against the best"
              % own)
        print("  vendor figure of %.0f us." % best)

    if a.md:
        with open(a.md, "w") as f:
            f.write("# Implementations compared\n\n")
            f.write("| N | " + " | ".join(labels) + " |\n")
            f.write("|---" * (len(labels) + 1) + "|\n")
            for n in lengths:
                f.write("| %d | " % n + " | ".join(
                    ("%.1f" % data[l][n]) if n in data[l] else "--"
                    for l in labels) + " |\n")
            f.write("\n## Fixed cost\n\n")
            f.write("| implementation | fixed, us | rate, us/op | resid RMS |\n")
            f.write("|---|---|---|---|\n")
            for l, (i, s, r, k, _ok) in fits.items():
                f.write("| %s | %.1f | %.3e | %.1f |\n" % (l, i, s, r))
        print("\nwritten to %s" % a.md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
