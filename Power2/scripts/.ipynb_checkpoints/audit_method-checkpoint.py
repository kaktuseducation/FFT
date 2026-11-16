#!/usr/bin/python3
# coding=utf-8
"""
audit_method.py — three checks the paired-comparison method does not yet make.

The rule of Chapter 3 is that two measurements may be compared only if they
differ in exactly one axis.  It is a good rule and it is enforced by
construction when the pairs are built.  It has one blind spot, and this tool
addresses it: the rule protects against the axes that were declared, and
against nothing else.  A variable nobody listed can vary freely across a
sample and carry the whole effect, and the resulting conclusion will fit the
data, survive an out-of-sample test, and be wrong.

That failure is not hypothetical.  It happened three times while
characterising the vendor operator of Section 6.6.  The second attempt was a
quadratic model with R² = 0.9998 that predicted a held-out length to within
0.1 µs out of 681, and it was wrong, because fifteen of the sixteen lengths
used to fit and to test it were divisible by 64.  The held-out points came
from the same family as the fitted ones, so holding them out tested nothing.

Three checks follow from that, and this tool performs them.

  pairs    For each axis, are the comparisons that decide it complete and
           balanced over the other axes?  A verdict drawn from pairs that
           happen to sit mostly at one tile size is a verdict about that
           tile size.

  strata   Does the verdict survive when the sample is cut by each other
           axis in turn?  An axis that wins overall but loses inside one
           stratum is an interaction, and reporting only the overall
           verdict hides it.  This is the check that would have caught the
           vendor-operator error, and it is cheap.

  scan     Does any derived property of the transform length explain the
           residual?  N is treated as a number by the method and has
           structure as an integer: its residue modulo a tile, its 2-adic
           valuation, its odd part, its largest prime factor.  This
           enumerates them and reports which, if any, splits the data.

Usage:
  python3 audit_method.py pairs  --csv output/matrix_final.csv \\
      --axes radix,algo,granularity,layout,unit --response npu_us
  python3 audit_method.py strata --csv output/matrix_final.csv \\
      --axes radix,algo,granularity,layout,unit --response npu_us
  python3 audit_method.py scan   --csv output/aclrfft_dense.csv \\
      --n-col N --response npu_avg_us
"""

import argparse
import csv
import math
import sys
from collections import defaultdict
from math import comb


def binom_p(wins, n):
    """Exact two-sided probability of a split at least this lopsided under a
    fair coin. A verdict that does not clear this is not a verdict; it is the
    number of pairs that happened to fall one way.

    Summed in log space: the direct form overflows a float long before the
    sample sizes a sweep of this kind can reach.
    """
    if n <= 0:
        return 1.0
    d = abs(wins - n / 2.0)
    ks = [k for k in range(n + 1) if abs(k - n / 2.0) >= d]
    if not ks:
        return 1.0
    ln2 = math.log(2.0)
    terms = [math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
             - n * ln2 for k in ks]
    top = max(terms)
    return min(1.0, math.exp(top) * sum(math.exp(t - top) for t in terms))


def pairs_needed(share, alpha=0.05):
    """Roughly how many pairs an effect of this size needs before it can be
    told from chance, from the normal approximation. Reported whenever a
    verdict falls short, because "undecided" is only actionable with the
    number attached."""
    from statistics import NormalDist
    d = abs(share - 0.5)
    if d < 1e-9:
        return None
    z = NormalDist().inv_cdf(1 - alpha / 2.0)
    n = (z / (2.0 * d)) ** 2
    return int(math.ceil(n / 2.0) * 2)


# --------------------------------------------------------------------------
# integer structure of a transform length
# --------------------------------------------------------------------------

def v2(n):
    k = 0
    while n and n % 2 == 0:
        n //= 2
        k += 1
    return k


def factorise(n):
    f = {}
    m, p = n, 2
    while p * p <= m:
        while m % p == 0:
            f[p] = f.get(p, 0) + 1
            m //= p
        p += 1 if p == 2 else 2
    if m > 1:
        f[m] = f.get(m, 0) + 1
    return f


def odd_part(n):
    while n and n % 2 == 0:
        n //= 2
    return n


def derived_features(n):
    """Properties of N that the axis list does not name but which an
    implementation may well branch on."""
    f = factorise(n) if n > 1 else {}
    big = max(f) if f else 1
    nearest = 1 << round(math.log2(n)) if n > 0 else 1
    out = {
        "v2(N)": v2(n),
        "odd part of N": odd_part(n),
        "largest prime factor": big,
        "is a power of two": int(n == (n & -n)),
        "distance to nearest 2^k": abs(n - nearest),
    }
    for m in (2, 4, 8, 16, 32, 64, 128):
        out["N mod %d" % m] = n % m
    return out


FEATURE_NAMES = list(derived_features(96).keys())


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def read_rows(path, status_col=None, status_ok="PASS"):
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if status_col and rows and status_col in rows[0]:
        rows = [r for r in rows
                if (r.get(status_col) or "").strip().upper() == status_ok]
    return rows


def fnum(r, k):
    v = (r.get(k) or "").strip()
    try:
        return float(v)
    except ValueError:
        return None


def gmean(xs):
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else None


def build_pairs(rows, axes, axis, response, key_cols):
    """Every pair of rows that differs in `axis` and agrees on all other
    axes and on every key column (length, tile, cores ...)."""
    other = [a for a in axes if a != axis]
    groups = defaultdict(list)
    for r in rows:
        t = fnum(r, response)
        if t is None or t <= 0:
            continue
        key = tuple((r.get(c) or "").strip() for c in other + key_cols)
        groups[key].append(r)
    pairs = []
    for key, members in groups.items():
        by_level = defaultdict(list)
        for r in members:
            by_level[(r.get(axis) or "").strip()].append(r)
        levels = sorted(by_level)
        for i in range(len(levels)):
            for j in range(i + 1, len(levels)):
                a = min(by_level[levels[i]], key=lambda r: fnum(r, response))
                b = min(by_level[levels[j]], key=lambda r: fnum(r, response))
                pairs.append((key, levels[i], levels[j],
                              fnum(a, response), fnum(b, response), a, b))
    return pairs, other


# --------------------------------------------------------------------------
# check 1: are the deciding comparisons complete and balanced?
# --------------------------------------------------------------------------

IGNORE_COLS = {"status", "mismatches", "points", "err_ratio", "max_abs",
               "max_rel", "total_min_us", "total_max_us", "stages_npu",
               "stages_cpu", "npu_us", "cpu_us", "total_us", "variant"}


def uncontrolled(rows, axes, keys, response):
    """Columns that vary inside a pairing group without being named.

    This is the check the tool most needs to make about its own use. A pair
    is only a pair if everything except the axis is held; a column nobody
    listed that varies between the two rows makes the comparison differ on
    two axes at once, and the verdict then belongs to neither.
    """
    if not rows:
        return {}
    cols = [c for c in rows[0]
            if c not in IGNORE_COLS and c not in axes and c not in keys
            and c != response]
    groups = defaultdict(list)
    for r in rows:
        groups[tuple((r.get(c) or "").strip() for c in axes + keys)].append(r)
    varying = defaultdict(int)
    for members in groups.values():
        if len(members) < 2:
            continue
        for c in cols:
            if len({(m.get(c) or "").strip() for m in members}) > 1:
                varying[c] += 1
    return dict(varying)


def cmd_pairs(args):
    rows = read_rows(args.csv, args.status_col)
    axes = args.axes.split(",")
    keys = [c for c in args.keys.split(",") if c] if args.keys else []
    print("rows considered: %d" % len(rows))
    print("axes: %s" % ", ".join(axes))
    print("held equal within a pair: %s" % (", ".join(keys) or "(none)"))

    var = uncontrolled(rows, axes, keys, args.response)
    if var:
        print("")
        print("  [WARNING] these columns vary inside a pairing group and are")
        print("  named neither as an axis nor as a key. Every pair below may")
        print("  differ in them as well as in the axis, which is the error")
        print("  this tool exists to detect. Add them to --keys and re-run:")
        for c in sorted(var, key=lambda k: -var[k]):
            print("      %-14s varies in %d group(s)" % (c, var[c]))
        print("      suggested: --keys %s" % ",".join(keys + sorted(var)))
    else:
        print("  no unnamed column varies inside a pairing group.")

    for axis in axes:
        pairs, other = build_pairs(rows, axes, axis, args.response, keys)
        print("\n" + "=" * 74)
        print("axis: %s   (%d complete pairs)" % (axis, len(pairs)))
        print("=" * 74)
        if not pairs:
            print("  no pair differs in this axis alone — the axis is not "
                  "decidable from this matrix")
            continue
        for co in other + keys:
            counts = defaultdict(int)
            for key, *_ in pairs:
                idx = (other + keys).index(co)
                counts[key[idx]] += 1
            if len(counts) <= 1:
                print("  %-14s held at a single value %r — every verdict for "
                      "%s is a verdict at that value only"
                      % (co, list(counts)[0], axis))
                continue
            lo, hi = min(counts.values()), max(counts.values())
            flag = "" if hi <= 3 * lo else "   <-- IMBALANCED"
            dist = ", ".join("%s:%d" % (k, counts[k])
                             for k in sorted(counts, key=str)[:8])
            print("  %-14s %s%s" % (co, dist, flag))
    return 0


# --------------------------------------------------------------------------
# check 2: does the verdict survive cutting the sample?
# --------------------------------------------------------------------------

def cmd_strata(args):
    rows = read_rows(args.csv, args.status_col)
    axes = args.axes.split(",")
    keys = [c for c in args.keys.split(",") if c] if args.keys else []
    alpha = args.alpha
    reversals, undecided = [], []

    for axis in axes:
        pairs, other = build_pairs(rows, axes, axis, args.response, keys)
        if not pairs:
            continue
        print("\n" + "=" * 74)
        print("axis: %s" % axis)
        print("=" * 74)

        def verdict(ps):
            wins = defaultdict(int)
            for _k, la, lb, ta, tb, _ra, _rb in ps:
                wins[la if ta < tb else lb] += 1
            return wins

        overall = verdict(pairs)
        tot = sum(overall.values())
        best = max(overall, key=overall.get)
        p = binom_p(overall[best], tot)
        decided = p < alpha
        print("  overall: %s"
              % ", ".join("%s=%d" % (k, overall[k])
                          for k in sorted(overall, key=str)))
        print("  leading value %r: %d of %d pairs (%.0f%%), p = %.3f  %s"
              % (best, overall[best], tot, 100.0 * overall[best] / tot, p,
                 "DECIDED" if decided else "NOT DECIDED"))
        if not decided:
            need = pairs_needed(overall[best] / tot, alpha)
            print("     an effect of this size needs about %s pairs before it"
                  % (need if need else ">20000"))
            print("     can be told from chance; this matrix supplies %d."
                  % tot)
            print("     report the axis as undecided, not as a narrow win.")
            undecided.append((axis, overall[best], tot, p))

        for co in other + keys:
            idx = (other + keys).index(co)
            byval = defaultdict(list)
            for pr in pairs:
                byval[pr[0][idx]].append(pr)
            if len(byval) <= 1:
                continue
            lines_out, flagged = [], False
            for val in sorted(byval, key=str):
                w = verdict(byval[val])
                t = sum(w.values())
                b = max(w, key=w.get)
                pb = binom_p(w[b], t)
                if w[b] * 2 == t:
                    tag = "tie"
                elif pb >= alpha:
                    tag = "n.s."
                elif b != best:
                    tag = "REVERSED"
                    flagged = True
                else:
                    tag = "same"
                lines_out.append("      %-10s n=%-4d %-12s %3.0f%%  p=%.3f  %s"
                                 % (val, t, b, 100.0 * w[b] / t, pb, tag))
            print("    cut by %s:%s" % (co,
                                        "   <-- REVERSES" if flagged else ""))
            for ln in lines_out:
                print(ln)
            if flagged:
                reversals.append((axis, co))

    print("\n" + "=" * 74)
    print("SUMMARY")
    print("=" * 74)
    if reversals:
        print("Genuine reversals - the leading value changes, and both the")
        print("overall verdict and the stratum verdict clear p < %.2f:" % alpha)
        for axis, co in reversals:
            print("   %-14s reverses with %s" % (axis, co))
        print("These are conditional verdicts and must be reported as such.")
    else:
        print("No verdict reverses significantly under any single cut.")
        print("A stratum whose split does not clear p < %.2f is marked n.s."
              % alpha)
        print("and one split exactly in half is marked tie. Neither is")
        print("evidence of an interaction, and neither is reported as one.")
    if undecided:
        print("")
        print("Axes this matrix does not decide:")
        for axis, w, n, pv in undecided:
            print("   %-14s %d of %d pairs (%.0f%%), p = %.3f"
                  % (axis, w, n, 100.0 * w / n, pv))
    return 0


# --------------------------------------------------------------------------
# check 3: hidden structure in the transform length
# --------------------------------------------------------------------------

def cmd_scan(args):
    rows = read_rows(args.csv, args.status_col)
    data = []
    for r in rows:
        n = fnum(r, args.n_col)
        t = fnum(r, args.response)
        if n and t and n > 0 and t > 0:
            data.append((int(n), t))
    if len(data) < 8:
        print("[ERROR] need at least 8 measurements, have %d" % len(data))
        return 1

    # One time per length: repeated lengths are averaged before the scan, so
    # a length measured more often cannot weight the result.
    byn = defaultdict(list)
    for n, t in data:
        byn[n].append(t)
    data = sorted((n, sum(v) / len(v)) for n, v in byn.items())
    ns = [n for n, _ in data]
    print("lengths: %d distinct, from %d to %d" % (len(ns), min(ns), max(ns)))

    # The response is detrended first: without this, any feature correlated
    # with magnitude would look explanatory merely because cost grows with N.
    print("\ndetrending against the declared variable (N) before the scan")
    best_fit, best_r2 = None, -1e9
    for name, f in (("N^2", lambda n: n * n),
                    ("N*log2N", lambda n: n * math.log2(n)),
                    ("N", lambda n: float(n))):
        xs = [f(n) for n, _ in data]
        ys = [t for _, t in data]
        k = len(xs)
        mx, my = sum(xs) / k, sum(ys) / k
        sxx = sum((x - mx) ** 2 for x in xs)
        if sxx == 0:
            continue
        b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
        a = my - b * mx
        ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
        ss_tot = sum((y - my) ** 2 for y in ys)
        r2 = 1 - ss_res / ss_tot if ss_tot else 0.0
        print("   t = a + b*%-9s  R^2 = %.5f" % (name, r2))
        if r2 > best_r2:
            best_fit, best_r2 = (name, f, a, b), r2
    name, f, a, b = best_fit
    print("   using %s (R^2 = %.5f) as the trend" % (name, best_r2))

    resid = [(n, t - (a + b * f(n))) for n, t in data]

    # Coverage comes first. A feature the sample barely varies cannot be
    # found by the scan below, and its absence from the findings would be
    # read as evidence that it does not matter. That is the exact error
    # this tool exists to prevent, so it is reported before anything else.
    print("\ncoverage of the sample over each derived feature")
    print("  %-24s %7s %9s  %s" % ("feature", "levels", "dominant", ""))
    print("  " + "-" * 60)
    blind = []
    for feat in FEATURE_NAMES:
        counts = defaultdict(int)
        for n in ns:
            counts[derived_features(n)[feat]] += 1
        k = len(counts)
        share = max(counts.values()) / len(ns)
        note = ""
        if k == 1:
            note = "   <-- NOT VARIED: this sample cannot test it"
            blind.append(feat)
        elif share > 0.7:
            note = "   <-- %.0f%% of lengths share one value" % (100 * share)
            blind.append(feat)
        print("  %-24s %7d %8.0f%%%s" % (feat, k, 100 * share, note))
    if blind:
        print("\n  %d feature(s) the sample cannot speak about. Any conclusion"
              % len(blind))
        print("  drawn here is a conclusion about the values present, not a")
        print("  general one. Widen the sample before generalising.")

    print("\nscanning derived features of N for structure in the residual")
    print("  %-24s %6s %9s %9s %s" % ("feature", "levels", "eta^2",
                                        "adjusted", "verdict"))
    print("  " + "-" * 66)
    results = []
    for feat in FEATURE_NAMES:
        groups = defaultdict(list)
        for n, e in resid:
            groups[derived_features(n)[feat]].append(e)
        groups = {k: v for k, v in groups.items() if len(v) >= 2}
        if len(groups) < 2:
            continue
        allv = [e for v in groups.values() for e in v]
        gm = sum(allv) / len(allv)
        ss_tot = sum((e - gm) ** 2 for e in allv)
        if ss_tot == 0:
            continue
        ss_bet = sum(len(v) * (sum(v) / len(v) - gm) ** 2
                     for v in groups.values())
        eta2 = ss_bet / ss_tot
        # Adjusted for the number of levels. "N mod 32" is a refinement of
        # "N mod 16" and will always fit at least as well, so an unadjusted
        # score prefers the finer feature and invents structure. The
        # adjustment charges each extra level against the fit, which lets
        # the coarsest feature that explains the data win.
        nobs, k = len(allv), len(groups)
        adj = (1 - (1 - eta2) * (nobs - 1) / (nobs - k)) if nobs > k else 0.0
        results.append((adj, eta2, feat, k))
    results.sort(reverse=True)
    print("  (ranked by the adjusted score, which charges for extra levels)")
    for adj, eta2, feat, k in results:
        mark = ("   <-- EXPLAINS THE RESIDUAL" if adj > 0.5 else
                "   <-- worth checking" if adj > 0.25 else "")
        print("  %-24s %6d %9.4f %9.4f%s" % (feat, k, eta2, adj, mark))

    top = results[0] if results else None
    print()
    if top and top[0] > 0.5:
        adj, eta2, feat, _k = top
        print("FINDING: %r accounts for %.1f%% of the variance the declared"
              % (feat, 100 * eta2))
        print("model leaves behind. It is not in the axis list, so no paired")
        print("comparison in this matrix controls for it, and any conclusion")
        print("drawn from a sample unbalanced on it is unsafe.")
        groups = defaultdict(list)
        for n, t in data:
            groups[derived_features(n)[feat]].append(t)
        print("\n  %-10s %6s %12s %12s" % (feat, "n", "mean", "range"))
        for k in sorted(groups, key=lambda x: (isinstance(x, str), x))[:20]:
            v = groups[k]
            print("  %-10s %6d %12.2f %12s"
                  % (k, len(v), sum(v) / len(v),
                     "%.0f..%.0f" % (min(v), max(v))))
    else:
        print("No derived feature of N explains more than half of the")
        print("residual. That is not proof of absence — only the features")
        print("listed above were tried — but the obvious ones are excluded.")
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    for name, fn in (("pairs", cmd_pairs), ("strata", cmd_strata)):
        s = sub.add_parser(name)
        s.add_argument("--csv", required=True)
        s.add_argument("--axes", required=True)
        s.add_argument("--response", default="npu_us")
        s.add_argument("--keys", default="N,tile,cores")
        s.add_argument("--status-col", default="status")
        s.add_argument("--alpha", type=float, default=0.05)
        s.set_defaults(func=fn)

    s = sub.add_parser("scan")
    s.add_argument("--csv", required=True)
    s.add_argument("--n-col", default="N")
    s.add_argument("--response", default="npu_avg_us")
    s.add_argument("--status-col", default=None)
    s.set_defaults(func=cmd_scan)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())