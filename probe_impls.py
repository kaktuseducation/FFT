#!/usr/bin/python3
# coding=utf-8
#
# What else on this machine can compute a Fourier transform on the device.
#
# WHY THIS EXISTS
#
# The supervisor's objection to the November draft was that a single
# comparison against MindSpore is too thin a basis for a claim about the
# platform.  He is right, and the answer is not argument but more
# implementations measured on the same silicon.  Before writing a harness for
# any of them it is worth knowing which are actually installed, because the
# answer decides how the next week is spent -- and because "not available on
# this image" is itself a finding worth recording rather than a gap to hide.
#
# This script measures nothing.  It reports what is present, at which version,
# and for the C library which symbols it exports, then prints a short verdict
# per candidate.  Run it first; it takes seconds and costs nothing.
#
# THE CANDIDATES
#
#   mindspore ops.fft          already measured; checked here for the version
#   mindspore FFTWithSize      the older primitive, a different code path
#                              inside the same framework
#   mindscience ASD_FFT        a 1-D complex-to-complex transform from the
#                              Ascend signal-processing library, reached from
#                              Python.  This is a SECOND vendor FFT, separate
#                              from the CANN operator already probed
#   AsdSip / SiPBoost (C)      the same library at the C level: asdFftCreate,
#                              a plan, a workspace, an execute.  Same shape as
#                              the aclnn two-stage call already wrapped
#   torch_npu torch.fft        PyTorch on Ascend: a second framework, so a
#                              second answer to whether the fixed cost belongs
#                              to the device or to MindSpore
#   CANN aclRfft1D             already measured, listed for completeness
#
# Usage:
#   python3 scripts/probe_impls.py
#   python3 scripts/probe_impls.py --json probe.json

import argparse
import glob
import json
import os
import subprocess
import sys

ASCEND_ROOTS = ("/usr/local/Ascend", "/usr/local/Ascend/ascend-toolkit",
                os.path.expanduser("~/Ascend"))


def say(ok, name, detail):
    mark = "  yes " if ok else "  no  "
    print("%s %-26s %s" % (mark, name, detail))
    return ok


def probe_mindspore(out):
    try:
        import mindspore as ms
    except Exception as exc:
        out["mindspore"] = None
        return say(False, "mindspore", "not importable: %s" % exc)

    ver = getattr(ms, "__version__", "?")
    out["mindspore"] = {"version": ver}
    say(True, "mindspore", "version %s" % ver)

    has_fft = hasattr(ms.ops, "fft")
    has_fws = hasattr(ms.ops, "FFTWithSize")
    out["mindspore"]["ops_fft"] = has_fft
    out["mindspore"]["fftwithsize"] = has_fws
    say(has_fft, "  ops.fft", "complex-to-complex, already measured"
        if has_fft else "absent")
    say(has_fws, "  ops.FFTWithSize", "separate primitive, worth a run of its "
        "own" if has_fws else "absent")
    return True


def probe_mindscience(out):
    try:
        from mindscience.sciops import fft as sfft
    except Exception as exc:
        out["mindscience"] = None
        return say(False, "mindscience.sciops.fft",
                   "not importable: %s" % str(exc)[:70])

    names = [n for n in ("ASD_FFT", "asd_fft", "ASD_IFFT", "asd_fftn",
                         "asd_rfft") if hasattr(sfft, n)]
    out["mindscience"] = {"symbols": names}
    return say(bool(names), "mindscience.sciops.fft",
               "exposes %s" % ", ".join(names) if names
               else "imported but exposes no ASD_FFT")


def probe_torch_npu(out):
    try:
        import torch
    except Exception as exc:
        out["torch"] = None
        return say(False, "torch", "not importable: %s" % str(exc)[:70])
    say(True, "torch", "version %s" % torch.__version__)
    try:
        import torch_npu                                      # noqa: F401
    except Exception as exc:
        out["torch"] = {"version": torch.__version__, "npu": False}
        return say(False, "  torch_npu", "not importable: %s" % str(exc)[:60])
    n = torch.npu.device_count() if hasattr(torch, "npu") else 0
    out["torch"] = {"version": torch.__version__, "npu": True, "devices": n}
    return say(n > 0, "  torch_npu",
               "%d device(s) visible" % n if n else "imported, no device")


def probe_asdsip(out):
    """Look for the signal-processing library on disk.

    Reported in three parts because they fail separately: the headers say the
    API is documented here, the shared object says it can be linked, and the
    exported symbols say what the plan call is actually named -- which the
    public documentation does not state.
    """
    headers, libs = [], []
    for root in ASCEND_ROOTS:
        if not os.path.isdir(root):
            continue
        headers += glob.glob(root + "/**/asd*fft*.h", recursive=True)
        headers += glob.glob(root + "/**/*sip*/**/*.h", recursive=True)
        libs += glob.glob(root + "/**/libasd*.so*", recursive=True)
        libs += glob.glob(root + "/**/libsip*.so*", recursive=True)
    headers = sorted(set(headers))[:12]
    libs = sorted(set(libs))[:12]

    ok = say(bool(headers), "AsdSip headers",
             "%d found, e.g. %s" % (len(headers), headers[0]) if headers
             else "none under " + ", ".join(ASCEND_ROOTS))
    say(bool(libs), "AsdSip shared object",
        "%d found, e.g. %s" % (len(libs), libs[0]) if libs else "none")

    syms = []
    for lib in libs[:3]:
        try:
            raw = subprocess.run(["nm", "-D", "--defined-only", lib],
                                 capture_output=True, text=True, timeout=60)
            syms += [l.split()[-1] for l in raw.stdout.splitlines()
                     if "asdFft" in l or "AsdFft" in l]
        except Exception:
            pass
    syms = sorted(set(syms))
    if syms:
        print("       plan and execute entry points exported:")
        for s in syms[:24]:
            print("         %s" % s)
        if len(syms) > 24:
            print("         ... and %d more" % (len(syms) - 24))
    out["asdsip"] = {"headers": headers, "libs": libs, "symbols": syms}
    return ok or bool(libs)


def probe_cann(out):
    ver = None
    for root in ASCEND_ROOTS:
        for p in glob.glob(root + "/**/version.info", recursive=True)[:4]:
            try:
                ver = open(p).read().strip().replace("\n", " ")[:60]
                break
            except Exception:
                pass
        if ver:
            break
    out["cann"] = ver
    return say(bool(ver), "CANN", ver or "version.info not found")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--json", type=str, default=None,
                   help="also write the findings to this file")
    a = p.parse_args()

    print("=" * 68)
    print("What can compute an FFT on this device")
    print("=" * 68)
    out = {}
    probe_cann(out)
    print()
    probe_mindspore(out)
    print()
    probe_mindscience(out)
    print()
    probe_torch_npu(out)
    print()
    probe_asdsip(out)

    print()
    print("=" * 68)
    ms_ok = bool(out.get("mindspore"))
    plan = []
    if ms_ok and out["mindspore"].get("fftwithsize"):
        plan.append("bench_mindspore.py --op fftwithsize   (second primitive)")
    if out.get("mindscience"):
        plan.append("bench_asdsip.py                       (second vendor FFT)")
    if out.get("torch") and out["torch"].get("npu"):
        plan.append("bench_torch_npu.py                    (second framework)")
    if out.get("asdsip", {}).get("symbols"):
        plan.append("AsdSip at the C level is linkable; the exported symbols "
                    "above name the plan call")
    if plan:
        print("Runnable here, in the order that buys the most per hour:")
        for i, line in enumerate(plan, 1):
            print("  %d. %s" % (i, line))
    else:
        print("Nothing beyond what is already measured is available on this\n"
              "image. That is a reportable result, not a dead end: say in the\n"
              "text which implementations were sought and why each could not\n"
              "be run.")
    print("=" * 68)

    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=2, default=str)
        print("written to %s" % a.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
