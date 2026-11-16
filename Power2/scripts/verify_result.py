#!/usr/bin/python3
# coding=utf-8
#
# A check of output_x.bin against golden.bin — in one output format with
# everything else ([RESULT] / [VERIFY]) and with a write into
# output/validation.csv, so that the final table shows not only the time but
# also "the test passed".
#
# Backward compatibility is kept: the two positional arguments
#   python3 scripts/verify_result.py ./output/output_x.bin ./output/golden.bin
# work as before, exit code 0 = PASS, 1 = FAIL. Nothing in the project calls
# this script any more — the verification moved into the binary itself (see
# host/verify.h); it stays here for re-checking a single run by hand, so the
# exit codes are kept the way whatever used to drive it expected them.
#
# The run label (impl/variant/N/tile) is taken by default from the last
# NON-numpy row of output/results.csv — the binary itself just appended it.
# It can be given explicitly: --variant dit_fs_xs -N 4096 --tile 32

import argparse
import csv
import os
import re
import sys

import numpy as np

RELATIVE_TOL = 1e-3
ABSOLUTE_TOL = 1e-3
ERROR_TOL = 1e-3

VALIDATION_HEADER = [
    "impl", "variant", "N", "tile_size", "status",
    "mismatches", "points", "err_ratio", "max_abs_err", "max_rel_err",
]


def detect_impl(variant):
    """r2 / r4 / r8, taken from the variant label itself.

    This used to guess from the project folder name (Radix8 and the like) with
    a fallback that inspected the set of kernel files. Both halves of that were
    artefacts of the era of three separate per-radix projects, and both are
    wrong in the merged one: the folder is called FFTBench, which has no digit
    in it, so the guess always fell through to the file-set heuristic, which
    found an "*_xi.cpp" among the radix-2 kernels and returned "r2"
    unconditionally — no matter which radix had actually run.

    The variant label is authoritative instead: host/bench_harness.h writes it
    into results.csv as "impl/variant" (e.g. "r4/dit_fsm_xs"), so the impl is
    already there and does not need to be inferred at all.
    """
    if variant and "/" in variant:
        head = variant.split("/", 1)[0].strip()
        if re.fullmatch(r"r\d+", head):
            return head
    return "unknown"


def last_run_label(results_csv):
    """(variant, N, tile) of the last non-numpy row of results.csv."""
    if not os.path.exists(results_csv):
        return None, None, None
    row = None
    with open(results_csv, newline="") as f:
        for r in csv.DictReader(f):
            if r.get("variant", "").strip().lower() == "numpy":
                continue
            row = r
    if row is None:
        return None, None, None
    try:
        return row["variant"], int(row["N"]), int(row["tile_size"])
    except (KeyError, ValueError):
        return row.get("variant"), None, None


def verify_result(output_file, golden_file, show_max=10):
    output_data = np.fromfile(output_file, dtype=np.float32).reshape(-1)
    golden_data = np.fromfile(golden_file, dtype=np.float32).reshape(-1)

    if output_data.size != golden_data.size:
        raise ValueError("[ERROR ] size mismatch: output=%d floats, golden=%d floats"
                         % (output_data.size, golden_data.size))

    # Interleaved format: [r0, i0, r1, i1, ...]
    output = output_data[0::2] + 1j * output_data[1::2]
    golden = golden_data[0::2] + 1j * golden_data[1::2]

    different_mask = ~np.isclose(output, golden, rtol=RELATIVE_TOL, atol=ABSOLUTE_TOL,
                                 equal_nan=True)
    different_indexes = np.where(different_mask)[0]

    abs_err = np.abs(output - golden)
    denom = np.maximum(np.abs(golden), 1e-30)
    rel_err = abs_err / denom
    max_abs = float(abs_err.max()) if abs_err.size else 0.0
    max_rel = float(rel_err.max()) if rel_err.size else 0.0

    for idx in range(min(len(different_indexes), show_max)):
        i = different_indexes[idx]
        print("[DIFF  ] index=%06d expected=%s actual=%s" % (i, golden[i], output[i]))
    if len(different_indexes) > show_max:
        print("[DIFF  ] ... %d more mismatches not shown" % (len(different_indexes) - show_max))

    err_ratio = float(different_indexes.size) / float(golden.size)
    stats = {
        "mismatches": int(different_indexes.size),
        "points": int(golden.size),
        "err_ratio": err_ratio,
        "max_abs_err": max_abs,
        "max_rel_err": max_rel,
    }
    return err_ratio <= ERROR_TOL, stats


def append_validation(path, row):
    need_header = not os.path.exists(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=VALIDATION_HEADER)
        if need_header:
            w.writeheader()
        w.writerow(row)


def parse_args():
    p = argparse.ArgumentParser(description="Verify FFT output against golden (unified report).")
    p.add_argument("output", nargs="?", default="./output/output_x.bin")
    p.add_argument("golden", nargs="?", default="./output/golden.bin")
    p.add_argument("--variant", default=None)
    p.add_argument("-N", "--size", type=int, default=None)
    p.add_argument("--tile", type=int, default=None)
    p.add_argument("--impl", default=None)
    p.add_argument("--results", default="./output/results.csv")
    p.add_argument("--validation", default="./output/validation.csv")
    p.add_argument("--no-csv", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    # Resolve the variant FIRST, because the impl is derived from it. The old
    # order (impl, then variant) could not work: results.csv stores the label
    # already qualified as "r4/dit_fsm_xs", so an independently guessed impl
    # was then printed in front of it and the line came out as
    # "impl=r2/r4/dit_fsm_xs".
    variant, N, tile = args.variant, args.size, args.tile
    if variant is None or N is None or tile is None:
        v2, n2, t2 = last_run_label(args.results)
        variant = variant if variant is not None else (v2 or "unknown")
        N = N if N is not None else (n2 if n2 is not None else 0)
        tile = tile if tile is not None else (t2 if t2 is not None else 0)

    impl = args.impl or detect_impl(variant)
    # If the variant arrived already qualified, drop the prefix so that the
    # "impl=%s/%s" line below does not repeat it.
    if variant and "/" in variant and variant.split("/", 1)[0] == impl:
        variant = variant.split("/", 1)[1]

    try:
        ok, st = verify_result(args.output, args.golden)
    except Exception as e:  # file not found / sizes did not match
        print("[VERIFY] impl=%s/%s N=%s tile=%s status=ERROR msg=%s" % (impl, variant, N, tile, e))
        sys.exit(1)

    status = "PASS" if ok else "FAIL"
    print("[VERIFY] impl=%s/%s N=%s tile=%s status=%s mismatches=%d/%d err_ratio=%.6f "
          "tol=%.1e max_abs=%.3e max_rel=%.3e"
          % (impl, variant, N, tile, status, st["mismatches"], st["points"],
             st["err_ratio"], ERROR_TOL, st["max_abs_err"], st["max_rel_err"]))

    if not args.no_csv:
        row = {"impl": impl, "variant": variant, "N": N, "tile_size": tile, "status": status}
        row.update({
            "mismatches": st["mismatches"],
            "points": st["points"],
            "err_ratio": "%.6f" % st["err_ratio"],
            "max_abs_err": "%.6e" % st["max_abs_err"],
            "max_rel_err": "%.6e" % st["max_rel_err"],
        })
        append_validation(args.validation, row)

    # we print the old line too — in case something still greps for it
    print("test pass" if ok else "[ERROR ] result mismatch")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()