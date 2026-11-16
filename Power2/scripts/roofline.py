#!/usr/bin/python3
# coding=utf-8
#
# Normalised performance metrics for every measured cell: achieved GFLOP/s,
# achieved GM bandwidth, and arithmetic intensity.
#
# WHY THIS EXISTS
#
# There is no GPU in this project's measurement setup, so a wall-clock
# comparison against a GPU implementation cannot be made here, and quoting one
# from a paper next to a microsecond figure measured on this machine would
# compare two different machines, two different input sizes and two different
# precisions at once.  What *can* be compared across machines is how much of a
# device's own capability an implementation extracts.  That is what this script
# computes: it turns each measured time into GFLOP/s and into achieved
# bandwidth, which are the units the GPU FFT literature reports and which are
# therefore directly quotable alongside it.
#
# The claim this supports is not "this kernel beats a GPU".  It is "this kernel
# extracts X% of what this device can deliver, which is comparable to / below /
# above the fraction that published GPU implementations extract from theirs" --
# a statement that survives the difference in hardware, and a much harder
# statement to make than a speedup ratio.
#
# WHAT IS MEASURED AND WHAT IS ASSUMED
#
# Measured: the times, from the matrix.
# Assumed: the operation count, via a model (see --flop-model).  The classic
#   5*N*log2(N) figure for a complex-to-complex transform is the default
#   because it is what the comparison literature uses; a radix-specific count
#   is selectable, and a count measured by this project's own operation-count
#   analysis can be supplied per length with --flop-csv.  Which model was used
#   is printed and written into every output row, because the GFLOP/s number
#   means nothing without it.
# Assumed: the minimum compulsory GM traffic, 16*N bytes -- one complex64 read
#   and one complex64 write per point.  A real kernel moves more than this
#   (every stage that round-trips through GM adds 16*N), so the achieved
#   bandwidth computed here is a LOWER bound on what the device actually moved,
#   and is labelled as such.  --stages lets a per-length stage count be given,
#   in which case the traffic becomes 16*N*stages and the figure stops being a
#   lower bound and becomes an estimate.
#
# Peak figures are NOT built in.  A fraction-of-peak column appears only if
# --peak-gflops and/or --peak-bw-gbs are passed explicitly, so that no number
# in the output ever rests on a specification figure this script guessed.  Take
# them from the device specification, or from a measured ceiling, and say in
# the text which it was.
#
# Usage:
#   python3 scripts/roofline.py --matrix output/matrix.csv
#   python3 scripts/roofline.py --matrix output/matrix.csv \
#           --peak-bw-gbs <from spec> --peak-gflops <from spec>
#   python3 scripts/roofline.py --matrix output/matrix.csv \
#           --baselines output/numpy_baseline.csv output/mindspore_baseline.csv

import argparse
import csv
import math
import os
import sys

BYTES_PER_COMPLEX64 = 8


def flops_for(N, model, radix=None):
    """Operation count for one complex-to-complex transform of length N."""
    if N <= 1:
        return 0.0
    if model == "5nlogn":
        # The standard figure: 5 real operations per point per radix-2 stage.
        return 5.0 * N * math.log2(N)
    if model == "butterfly":
        # Radix-r: log_r(N) stages, each N/r butterflies.  A radix-r butterfly
        # costs about 8*(r-1) real operations once the twiddle multiplies are
        # counted (6 per complex multiply, 2 per complex add), which for r=2
        # collapses to the 5NlogN figure within rounding.
        if not radix or radix < 2:
            return 5.0 * N * math.log2(N)
        stages = math.log(N, radix)
        return stages * (N / radix) * 8.0 * (radix - 1)
    raise ValueError("unknown flop model: %s" % model)


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


def load_flop_override(path):
    """N -> flops, from a CSV with columns N,flops."""
    out = {}
    for r in read_rows(path):
        N = fnum(r, "N")
        f = fnum(r, "flops")
        if N is not None and f is not None:
            out[int(N)] = f
    return out


def load_stage_counts(path):
    """N -> stages, from a CSV with columns N,stages."""
    out = {}
    for r in read_rows(path):
        N = fnum(r, "N")
        s = fnum(r, "stages")
        if N is not None and s is not None:
            out[int(N)] = s
    return out


def main():
    p = argparse.ArgumentParser(
        description="Compute achieved GFLOP/s, achieved GM bandwidth and "
                    "arithmetic intensity for every measured cell, so that "
                    "this work can be set against published implementations "
                    "on other hardware in units that cross machines.")
    p.add_argument("--matrix", default="./output/matrix.csv")
    p.add_argument("--baselines", nargs="*", default=[],
                   help="baseline CSVs (numpy_baseline.csv, "
                        "mindspore_baseline.csv) to score on the same scale")
    p.add_argument("--out", default="./output/roofline.csv")
    p.add_argument("--flop-model", choices=["5nlogn", "butterfly"],
                   default="5nlogn",
                   help="operation-count model (default: 5nlogn, the figure "
                        "the comparison literature uses)")
    p.add_argument("--flop-csv", default=None,
                   help="CSV of N,flops to override the model per length, "
                        "e.g. counts from this project's own analysis")
    p.add_argument("--stages", default=None,
                   help="CSV of N,stages; without it GM traffic is the "
                        "compulsory minimum and bandwidth is a lower bound")
    p.add_argument("--peak-gflops", type=float, default=None,
                   help="device peak, from the specification or a measured "
                        "ceiling; omit and no fraction-of-peak is reported")
    p.add_argument("--peak-bw-gbs", type=float, default=None,
                   help="device peak GM bandwidth, same caveat")
    p.add_argument("--peak-source", default=None,
                   help="free text recorded in the output saying where the "
                        "peak figures came from")
    p.add_argument("--only-pass", action="store_true", default=True)
    args = p.parse_args()

    flop_override = load_flop_override(args.flop_csv) if args.flop_csv else {}
    stage_counts = load_stage_counts(args.stages) if args.stages else {}

    if (args.peak_gflops or args.peak_bw_gbs) and not args.peak_source:
        print("[WARN  ] peak figures given without --peak-source; the text "
              "will need to say where they came from")

    rows_out = []

    def score(impl, variant, N, total_us, npu_us, radix, extra):
        if not total_us or total_us <= 0:
            return
        flops = flop_override.get(N)
        model_used = "measured(flop-csv)" if flops is not None else args.flop_model
        if flops is None:
            flops = flops_for(N, args.flop_model, radix)

        stages = stage_counts.get(N)
        traffic_bytes = 2.0 * N * BYTES_PER_COMPLEX64
        traffic_kind = "compulsory-minimum"
        if stages:
            traffic_bytes *= stages
            traffic_kind = "stages=%g" % stages

        secs_total = total_us * 1e-6
        gflops_total = (flops / secs_total) / 1e9
        bw_total = (traffic_bytes / secs_total) / 1e9

        rec = {
            "impl": impl,
            "variant": variant,
            "N": N,
            "flops": "%.0f" % flops,
            "flop_model": model_used,
            "gm_bytes": "%.0f" % traffic_bytes,
            "gm_traffic_kind": traffic_kind,
            "arithmetic_intensity_flop_per_byte": "%.3f" % (flops / traffic_bytes),
            "total_us": "%.2f" % total_us,
            "gflops_end_to_end": "%.3f" % gflops_total,
            "gbs_end_to_end": "%.3f" % bw_total,
        }

        # The device-only figure matters because the end-to-end one is capped
        # by PCIe at small N and so says more about the bus than the kernel.
        if npu_us and npu_us > 0:
            secs_dev = npu_us * 1e-6
            rec["npu_us"] = "%.2f" % npu_us
            rec["gflops_device_only"] = "%.3f" % ((flops / secs_dev) / 1e9)
            rec["gbs_device_only"] = "%.3f" % ((traffic_bytes / secs_dev) / 1e9)
        else:
            rec["npu_us"] = ""
            rec["gflops_device_only"] = ""
            rec["gbs_device_only"] = ""

        if args.peak_gflops:
            rec["pct_peak_gflops_end_to_end"] = "%.2f" % (
                100.0 * gflops_total / args.peak_gflops)
            if rec["gflops_device_only"]:
                rec["pct_peak_gflops_device_only"] = "%.2f" % (
                    100.0 * float(rec["gflops_device_only"]) / args.peak_gflops)
            else:
                rec["pct_peak_gflops_device_only"] = ""
        if args.peak_bw_gbs:
            rec["pct_peak_bw_end_to_end"] = "%.2f" % (
                100.0 * bw_total / args.peak_bw_gbs)
            if rec["gbs_device_only"]:
                rec["pct_peak_bw_device_only"] = "%.2f" % (
                    100.0 * float(rec["gbs_device_only"]) / args.peak_bw_gbs)
            else:
                rec["pct_peak_bw_device_only"] = ""
        if args.peak_source:
            rec["peak_source"] = args.peak_source

        rec.update(extra)
        rows_out.append(rec)

    # ---- the kernels ---------------------------------------------------
    for r in read_rows(args.matrix):
        if args.only_pass and (r.get("status") or "").strip().upper() != "PASS":
            continue
        N = fnum(r, "N")
        if N is None:
            continue
        score(impl=(r.get("impl") or "").strip(),
              variant=(r.get("variant") or "").strip(),
              N=int(N),
              total_us=fnum(r, "total_us"),
              npu_us=fnum(r, "npu_us"),
              radix=int(fnum(r, "radix") or 0) or None,
              extra={"tile": (r.get("tile") or "").strip(),
                     "cores": (r.get("cores") or "").strip(),
                     "source": os.path.basename(args.matrix)})

    # ---- the baselines, on the same scale ------------------------------
    for bpath in args.baselines:
        for r in read_rows(bpath):
            N = fnum(r, "N")
            if N is None:
                continue
            score(impl=(r.get("variant") or "baseline").strip(),
                  variant="baseline",
                  N=int(N),
                  total_us=fnum(r, "total_avg_us"),
                  npu_us=fnum(r, "npu_avg_us"),
                  radix=None,
                  extra={"tile": "0", "cores": "",
                         "source": os.path.basename(bpath)})

    if not rows_out:
        print("[ERROR ] nothing to score")
        return 1

    fields = []
    for r in rows_out:
        for k in r:
            if k not in fields:
                fields.append(k)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows_out:
            w.writerow(r)

    print("=" * 72)
    print("Normalised performance, best cell per (source, N)")
    print("=" * 72)
    print("flop model     : %s" % args.flop_model)
    print("GM traffic     : %s" % ("16*N*stages" if stage_counts
                                   else "16*N (compulsory minimum -> "
                                        "bandwidth is a LOWER bound)"))
    if args.peak_gflops or args.peak_bw_gbs:
        print("peaks          : gflops=%s  bw=%s GB/s  source=%s"
              % (args.peak_gflops, args.peak_bw_gbs,
                 args.peak_source or "UNSTATED"))
    else:
        print("peaks          : not given, so no fraction-of-peak column")
        print("                 (pass --peak-gflops / --peak-bw-gbs from the")
        print("                  device specification to get one)")
    print("")

    best = {}
    for r in rows_out:
        key = (r["source"], int(r["N"]))
        t = float(r["total_us"])
        if key not in best or t < float(best[key]["total_us"]):
            best[key] = r

    hdr = "%-26s %-9s %12s %12s %12s %10s" % (
        "source", "N", "total_us", "GFLOP/s", "GB/s(>=)", "AI")
    print(hdr)
    print("-" * len(hdr))
    for key in sorted(best, key=lambda k: (k[0], k[1])):
        r = best[key]
        print("%-26s %-9s %12s %12s %12s %10s"
              % (r["source"][:26], r["N"], r["total_us"],
                 r["gflops_end_to_end"], r["gbs_end_to_end"],
                 r["arithmetic_intensity_flop_per_byte"]))

    print("")
    print("[WROTE ] %s  (%d rows)" % (args.out, len(rows_out)))
    print("")
    print("How to use this in the text: quote the GFLOP/s and the fraction of")
    print("peak, state the flop model in the same sentence, and set them")
    print("against the same two quantities as published for a GPU FFT.  The")
    print("comparison is then between two efficiencies, which is meaningful,")
    print("rather than between two wall clocks on different machines, which")
    print("is not.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
