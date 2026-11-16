#!/usr/bin/python3
# coding=utf-8
#
# Analysis of output/matrix.csv — the table the binary itself writes (a row
# per matrix cell, with all the axes and the result of verification against
# golden).
#
# It prints four blocks:
#
#   1. Per N — a table of the variants, each in its BEST configuration
#      (the minimum total_us among those that passed verification) and the
#      ratio to NumPy.
#
#   2. PAIRED comparisons along the axes — the main block for making a
#      decision. Only cells differing in EXACTLY one axis are compared: the
#      same N, tile, radix, algorithm, layout and so on. See below for why
#      that matters.
#
#   3. The overview table "the best time per value of an axis" — it is NOT
#      paired and is good only for a general look.
#
#   4. The cells without a PASS — as a list of their own, so they do not get
#      lost.
#
# WHY PAIRED COMPARISONS MATTER MORE THAN OVERVIEW ONES
#
# The overview table takes the best time for every value of an axis ANYWHERE in
# the matrix. On real data that produced a trap immediately: in the "algorithm"
# row dit stood at 195.95 (that is radix-8 on 40 cores) and stockham at 254.29
# (that is radix-4 on one core). The 1.30x difference looks like a superiority
# of DIT, while in fact it is a superiority of the radix and of multicore: an
# honest paired comparison on one and the same radix-4 gives DIT against
# Stockham only 1.24x, and on radix-2 — 1.17x.
#
# So the decisions are made by block 2, and block 3 stays as an overview.
#
# Dependencies — the standard library only.
#
# Usage:
#   python3 scripts/analyze_matrix.py
#   python3 scripts/analyze_matrix.py -N 4096
#   python3 scripts/analyze_matrix.py --axis granularity --all-pairs
#   python3 scripts/analyze_matrix.py --export ./output/pivot.csv

import argparse
import csv
import os
import sys

# The axes, and which other fields have to be EXCLUDED from the grouping key in
# a paired comparison. What has to be excluded is everything rigidly tied to the
# axis itself:
#   * radix — it is encoded in impl ("r4"), so impl is out of the key too;
#   * unit  — for the cube variants algo also equals "cube", otherwise no pair
#             is ever found;
#   * terms — it means something only inside cube, and variant differs there
#             as well.
# The fields the cells are grouped by for the paired comparisons: when
# comparing along axis X, EVERYTHING else out of this list is held fixed.
# "cores" is mandatory here: the moment every variant got a multicore twin,
# without it the axes "radix", "algorithm" and "granularity" would start
# mixing single-core and multicore cells into one group and comparing the
# incomparable — exactly the error that forced paired comparisons in instead
# of the overview table.
# "mm" is in for the same reason: cube1 and cube2 differ ONLY in the number of
# matmuls per term, and without that field they would land in one group along
# the "terms" axis (where only terms differ) and spoil it.
ALL_KEYS = ["impl", "radix", "algo", "granularity", "layout", "unit", "terms",
            "mm", "N", "tile", "cube_batch", "cores"]
AXES = [
    ("radix",        "radix",          {"impl", "radix", "variant"}),
    ("algo",         "algorithm",      {"algo", "variant"}),
    ("granularity",  "granularity",    {"granularity", "variant"}),
    ("layout",       "layout",         {"layout", "variant"}),
    # A Vector variant and a Cube variant differ in more than the unit field:
    # Vector has algo=dit, terms=0, mm=0 and cube_batch=0, Cube has algo=cube,
    # terms>0, mm>0 and a meaningful batch. All those fields are consequences
    # of the choice of compute unit, so they leave the grouping key too —
    # otherwise a pair is never found.
    #
    # "cores" leaves it for exactly the same reason, and this one is easy to
    # get wrong. The chip has 40 AIV cores and 20 AIC cores; a Vector variant
    # can therefore never declare 20 and a Cube variant can never declare 40.
    # Core count is not an independent choice here — it IS the compute unit,
    # restated. Keeping it in the key made this axis report "no paired
    # comparisons" on every table ever collected, which read as "we have no
    # data" when in fact the data was there and the key was wrong.
    ("unit",         "compute unit",   {"unit", "algo", "terms", "mm", "cube_batch",
                                        "cores", "variant"}),
    ("terms",        "terms",          {"terms", "variant"}),
    # A pair like (r8/cube1x3_fs_xs, r8/cube2x3_fs_xs): one and the same
    # combine arithmetic, but in cube1 it is folded into a single block-real
    # 16x16 matmul, while in cube2 it is split into two matmuls with
    # A2 = [Re(M); Im(M)] plus the Re/Im combine on Vector. That is the "one
    # matmul or two" axis itself.
    ("mm",           "matmuls",        {"mm", "variant"}),
    # A pair like (r4/dit_fs_xs, r4/dit_fsm_xs): one and the same arithmetic,
    # one core against forty. This is the only axis along which what multicore
    # gives on its own is visible, with no radix or algorithm mixed in.
    ("cores",        "cores",          {"cores", "variant"}),
    ("tile",         "tile threshold", {"tile"}),
    ("cube_batch",   "Cube batch",     {"cube_batch"}),
]
AXIS_BY_NAME = {a[0]: a for a in AXES}

# Which field to compare by: "_total" (the full time, the default) or "_npu"
# (the NPU part only). Why the second one is needed: the CPU stages are shared
# by all the variants of one radix, and in total they dilute the effect. For
# instance multicore on r4 at N=262144 gives only 1.85x by total, but 12.9x by
# NPU, because 92% of total is taken by two stages on the processor, which
# multicore does not touch at all. For a conclusion about scaling across cores
# the right metric is npu.
METRIC = {"key": "_total", "label": "TOTAL_us"}


def fnum(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def load_matrix(path):
    if not os.path.exists(path):
        print("[ERROR] no %s — run run.sh or run_matrix.sh first" % path)
        sys.exit(1)
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            r["_total"] = fnum(r.get("total_us"))
            r["_npu"] = fnum(r.get("npu_us"))
            r["_cpu"] = fnum(r.get("cpu_us"))
            r["_N"] = int(r["N"]) if r.get("N") else 0
            r["_full"] = "%s/%s" % (r.get("impl", "?"), r.get("variant", "?"))
            # Tables taken before yet another axis appeared must not bring the
            # analysis down: a missing column reads as "0". That way an old
            # matrix.csv.bak stays analyzable (just without that axis).
            for k in ALL_KEYS:
                if r.get(k) is None:
                    r[k] = "0"
            rows.append(r)
    return rows


def load_numpy_baseline(path):
    """{N: total_avg_us} out of the table scripts/bench_numpy.py writes."""
    base = {}
    if not os.path.exists(path):
        return base
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("variant", "").strip().lower() != "numpy":
                continue
            n, t = r.get("N"), fnum(r.get("total_avg_us"))
            if n and t:
                base[int(n)] = t
    return base


def table(rows, header):
    if not rows:
        return
    cols = list(zip(*([header] + rows)))
    w = [max(len(str(c)) for c in col) for col in cols]
    print("  ".join(str(h).ljust(w[i]) for i, h in enumerate(header)))
    print("  ".join("-" * x for x in w))
    for r in rows:
        print("  ".join(str(c).ljust(w[i]) for i, c in enumerate(r)))


def best_per_variant(rows):
    """The best cell that PASSED verification per variant, by the chosen metric."""
    mk = METRIC["key"]
    best = {}
    for r in rows:
        if r.get("status") != "PASS" or r.get(mk) is None:
            continue
        key = r["_full"]
        if key not in best or r[mk] < best[key][mk]:
            best[key] = r
    return best


def group_label(r, excluded):
    """A short group caption: what exactly is held fixed in this comparison."""
    bits = []
    if "impl" not in excluded:
        bits.append(r.get("impl", "?"))
    if "algo" not in excluded:
        bits.append(r.get("algo", "?"))
    if "granularity" not in excluded:
        bits.append(r.get("granularity", "?"))
    if "layout" not in excluded:
        bits.append(r.get("layout", "?"))
    if "unit" not in excluded and r.get("unit") == "cube" and "cube" not in bits:
        bits.append("cube")
    if "cores" not in excluded:
        bits.append("%scores" % r.get("cores", "?"))
    bits.append("N=%s" % r.get("N"))
    if "tile" not in excluded:
        bits.append("tile=%s" % r.get("tile"))
    if "cube_batch" not in excluded and r.get("unit") == "cube":
        bits.append("B=%s" % r.get("cube_batch"))
    return " ".join(bits)


def paired_axis(rows, axis, excluded, show_all, limit=6, allow_fail=False):
    """Comparisons of cells differing in EXACTLY the axis `axis`.

    allow_fail=True is needed for the terms axis: the number of terms of the
    precision-recovery scheme changes first of all the ACCURACY, and the whole
    point of the comparison is that the variants with terms=1 and terms=2 do
    NOT pass verification. Discarding them as usual, we would lose exactly what
    we set out to measure — the cost of accuracy in time.
    """
    mk = METRIC["key"]
    keys = [k for k in ALL_KEYS if k not in excluded]
    groups = {}
    for r in rows:
        if r.get(mk) is None:
            continue
        if not allow_fail and r.get("status") != "PASS":
            continue
        gk = tuple(r.get(k) for k in keys)
        av = r.get(axis)
        if av is None or av == "":
            continue
        slot = groups.setdefault(gk, {})
        # within one group a value of the axis may have several cells (for
        # instance different variants at a fixed algo) — we take the best one
        if av not in slot or r[mk] < slot[av][mk]:
            slot[av] = r
    # we keep only the groups where there is something to compare
    cmps = [(gk, slot) for gk, slot in groups.items() if len(slot) >= 2]
    if not cmps:
        return None
    # ratios may stay empty if not one group turned out to have a variant that
    # passed verification (only for the allow_fail axes)

    wins = {}
    ratios = []
    lines = []
    for gk, slot in cmps:
        ordered = sorted(slot.items(), key=lambda kv: kv[1][mk])
        win_val, win_row = ordered[0]
        worst_val, worst_row = ordered[-1]
        ratio = worst_row[mk] / win_row[mk] if win_row[mk] else 0.0
        if allow_fail:
            # Here "faster" does not mean "better": the fastest variant is
            # usually precisely the one that does not pass verification. We
            # compute the COST OF ACCURACY — how many times dearer the
            # cheapest of those that passed verification is against the
            # fastest overall.
            detail = " | ".join("%s: %.1f us, %s, %s"
                                % (v, r[mk], r.get("max_abs", "-"), r.get("status", "?"))
                                for v, r in ordered)
            passed = [(v, r) for v, r in ordered if r.get("status") == "PASS"]
            if not passed:
                lines.append([group_label(win_row, excluded), detail,
                              "not one of them passes verification"])
                continue
            pv, pr = passed[0]
            cost = pr[mk] / ordered[0][1][mk] if ordered[0][1][mk] else 0.0
            wins[pv] = wins.get(pv, 0) + 1
            ratios.append(cost)
            lines.append([group_label(win_row, excluded), detail,
                          "take %s: cost of accuracy +%.0f%%" % (pv, 100.0 * (cost - 1.0))])
        else:
            wins[win_val] = wins.get(win_val, 0) + 1
            ratios.append(ratio)
            detail = ", ".join("%s=%.2f" % (v, r[mk]) for v, r in ordered)
            lines.append([group_label(win_row, excluded), detail,
                          "%s faster by %.2fx" % (win_val, ratio)])
    lines.sort(key=lambda x: x[0])
    return {"lines": lines, "wins": wins, "ratios": ratios,
            "shown": lines if show_all else lines[:limit],
            "hidden": 0 if show_all else max(0, len(lines) - limit)}


def main():
    p = argparse.ArgumentParser(description="Analysis of the FFTBench matrix.")
    p.add_argument("--matrix", default="./output/matrix.csv")
    p.add_argument("--baseline", default="./output/numpy_baseline.csv")
    p.add_argument("-N", "--size", type=int, default=None, help="this N only")
    p.add_argument("--axis", default=None,
                   help="in detail along one axis: %s" % ", ".join(a[0] for a in AXES))
    p.add_argument("--all-pairs", action="store_true",
                   help="show all the paired comparisons, not just the first six")
    p.add_argument("--metric", choices=["total", "npu"], default="total",
                   help="which time to compare by: total (the default) or npu")
    p.add_argument("--export", default=None, help="dump the summary into a CSV")
    a = p.parse_args()

    if a.metric == "npu":
        METRIC["key"], METRIC["label"] = "_npu", "NPU_us"

    rows = load_matrix(a.matrix)
    base = load_numpy_baseline(a.baseline)

    n_pass = sum(1 for r in rows if r.get("status") == "PASS")
    n_fail = sum(1 for r in rows if r.get("status") == "FAIL")
    n_none = sum(1 for r in rows if r.get("status") == "NO_OUTPUT")
    print("===== FFTBench: matrix summary (metric: %s) =====" % METRIC["label"])
    print("cells: %d   PASS=%d  FAIL=%d  no result=%d" % (len(rows), n_pass, n_fail, n_none))

    sizes = sorted({r["_N"] for r in rows})
    if a.size is not None:
        sizes = [s for s in sizes if s == a.size]
    scope = [r for r in rows if r["_N"] in sizes]

    # ---- 1. the best per variant, per N ----------------------------------
    # Under --axis we print ONLY the axis that was asked for: otherwise the
    # block of interest drowns between the big per-N table and the overview
    # table, which is exactly what happened on the very first real run with
    # --axis tile.
    # (--export keeps working through that: the rows are always collected,
    # only the printing is suppressed)
    focus = a.axis is not None
    export_rows = []
    for N in sizes:
        sub = [r for r in scope if r["_N"] == N]
        bp = best_per_variant(sub)
        bl = base.get(N)
        if not focus:
            print("")
            print("--- N=%d %s ---" % (N, ("(NumPy: %.2f us)" % bl) if bl else "(NumPy: no data)"))
        if not bp:
            if not focus:
                print("    not one cell passed verification")
            continue
        body = []
        for full, r in sorted(bp.items(), key=lambda kv: kv[1][METRIC["key"]]):
            vs = ("%.2fx" % (r["_total"] / bl)) if bl else "-"
            body.append([
                full, r.get("tile", "-"),
                r.get("cube_batch") if r.get("unit") == "cube" else "-",
                r.get("cores", "-"),
                "%.2f" % r["_npu"] if r["_npu"] is not None else "-",
                "%.2f" % r["_cpu"] if r["_cpu"] is not None else "-",
                "%.2f" % r["_total"], vs,
                r.get("stages_npu", "-"), r.get("stages_cpu", "-"),
                r.get("max_abs", "-"),
            ])
            export_rows.append([N, full, r.get("radix"), r.get("algo"),
                                r.get("granularity"), r.get("layout"), r.get("unit"),
                                r.get("tile"), r.get("cube_batch"), r.get("cores"),
                                r.get("npu_us"), r.get("cpu_us"), r.get("total_us"),
                                ("%.4f" % (r["_total"] / bl)) if bl else "",
                                r.get("max_abs")])
        if not focus:
            table(body, ["impl/variant", "tile", "B", "cores", "NPU_us", "CPU_us",
                         # The ratio against NumPy is ALWAYS total/NumPy, even
                         # under --metric npu: the NumPy baseline is an
                         # end-to-end CPU time, so comparing only our NPU part
                         # against it would flatter us by hiding the CPU stages
                         # that the same run still has to do. Under --metric
                         # npu the row is selected by NPU time while this
                         # column stays a total-time ratio, so the header says
                         # so explicitly rather than leaving the reader to
                         # assume the two track each other.
                         "TOTAL_us", "tot/NumPy", "st.NPU", "st.CPU", "max_abs"])

    # ---- 2. paired comparisons along the axes ----------------------------
    print("")
    if not focus:
        print("=" * 76)
        print("PAIRED COMPARISONS ALONG THE AXES — the cells compared differ in")
        print("EXACTLY one axis (the rest is held fixed). Make decisions by these.")
        print("Metric: %s." % METRIC["label"])
        print("=" * 76)
    axes = [AXIS_BY_NAME[a.axis]] if a.axis else AXES
    if a.axis and a.axis not in AXIS_BY_NAME:
        print("[ERROR] unknown axis '%s'" % a.axis)
        sys.exit(1)
    for key, title, excl in axes:
        # terms changes the accuracy above all, and the point of the comparison
        # is precisely that some of the variants do not pass verification —
        # they must not be discarded.
        acc_axis = (key == "terms")
        res = paired_axis(scope, key, excl, a.all_pairs, allow_fail=acc_axis)
        print("")
        if res is None:
            print("--- axis: %s — there are no paired comparisons" % title)
            print("    (there are no two cells differing only in this axis: for")
            print("     instance xi is implemented for radix-2 only, while tile/batch")
            print("     have to be run in at least two values — run_matrix.sh -t / -B)")
            continue
        win_str = ", ".join("%s: %d" % (v, c)
                            for v, c in sorted(res["wins"].items(), key=lambda kv: -kv[1]))
        if res["ratios"]:
            lo, hi = min(res["ratios"]), max(res["ratios"])
            if acc_axis:
                print("--- axis: %s --- comparisons: %d   choice: %s   cost of accuracy: %.2f–%.2fx"
                      % (title, len(res["lines"]), win_str or "-", lo, hi))
            else:
                print("--- axis: %s --- comparisons: %d   wins: %s   speedup: %.2f–%.2fx"
                      % (title, len(res["lines"]), win_str, lo, hi))
        else:
            print("--- axis: %s --- comparisons: %d (not one group has a PASS)"
                  % (title, len(res["lines"])))
        table(res["shown"], ["held fixed", "times, us", "conclusion"])
        if res["hidden"]:
            print("    ... %d more comparisons (--all-pairs, to show them all)" % res["hidden"])

    # ---- 3. the overview (NOT paired) table -------------------------------
    if not focus:
        print("")
        print("=" * 76)
        print("OVERVIEW (NOT paired, metric %s): the best time per value of an axis" % METRIC["label"])
        print("in the matrix. No good for conclusions — see the paired block above.")
        print("=" * 76)
        axis_body = []
        for key, title, _ in AXES:
            vals = sorted({r.get(key) for r in scope if r.get(key)},
                          key=lambda v: (len(str(v)), str(v)))
            for v in vals:
                cand = [r for r in scope if r.get(key) == v
                        and r.get("status") == "PASS" and r["_total"] is not None]
                if not cand:
                    continue
                b = min(cand, key=lambda r: r["_total"])
                axis_body.append([title, str(v), "%.2f" % b["_total"],
                                  "%s N=%s tile=%s" % (b["_full"], b.get("N"), b.get("tile"))])
        table(axis_body, ["axis", "value", "best " + METRIC["label"], "achieved by"])

    # ---- 4. the cells without a PASS -------------------------------------
    bad = [r for r in scope if r.get("status") in ("FAIL", "NO_OUTPUT")]
    if bad and not focus:
        print("")
        print("--- cells without a PASS ---")
        table([[r["_full"], r.get("N"), r.get("tile"),
                r.get("cube_batch") if r.get("unit") == "cube" else "-",
                r.get("status"), r.get("mismatches", "-"), r.get("max_abs", "-")]
               for r in bad],
              ["impl/variant", "N", "tile", "B", "status", "mismatches", "max_abs"])

    if a.export and export_rows:
        with open(a.export, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["N", "variant", "radix", "algo", "granularity", "layout", "unit",
                        "tile", "cube_batch", "cores", "npu_us", "cpu_us", "total_us",
                        "vs_numpy", "max_abs"])
            w.writerows(export_rows)
        print("")
        print("[INFO] the summary was dumped into %s" % a.export)


if __name__ == "__main__":
    main()
