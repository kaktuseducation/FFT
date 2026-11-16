#!/usr/bin/python3
# coding=utf-8
#
# Did torch.fft actually run on the device?
#
# WHY THIS HAS TO BE ASKED
#
# The week-2 sweep put PyTorch's transform at 1852 us for N = 262144 while
# MindSpore on the same chip took 25351 us for the same transform, and at
# every length below 15625 PyTorch came out flat near 400 us with no growth
# in N log N at all. Fitting the times gives it a per-work rate an order of
# magnitude below NumPy's. A device that MindSpore cannot make faster than
# the host does not plausibly run fifteen times faster under a different
# framework; something else is going on.
#
# The likeliest something else is a silent fallback. When a backend does not
# implement an operator, PyTorch can route it to the CPU rather than fail,
# and the call returns a correct answer with no error. The benchmark moves
# its input to the device and never checks where the output came back from,
# so a fallback would look exactly like a very fast device.
#
# If that is what happened, then "PyTorch beats this work on Ascend" is not a
# statement about Ascend at all, and it must not reach the dissertation. If
# it is not what happened, the result stands and is worth a section of its
# own. Either way it is one script and five minutes.
#
# WHAT IT CHECKS
#
#   1. where the output tensor says it lives
#   2. whether torch names a fallback in its warnings
#   3. the same transform forced onto the CPU, timed the same way --
#      if the two agree closely, the device was never involved
#   4. the NPU figure against MindSpore's at the same length, which is the
#      comparison that raised the question
#
# Usage:
#   python3 check_torch_device.py -N 262144
#   python3 check_torch_device.py -N 4096 -b 20

import argparse
import sys
import time
import warnings


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-N", type=int, default=262144)
    p.add_argument("-b", "--runs", type=int, default=10)
    p.add_argument("-w", "--warmup", type=int, default=5)
    a = p.parse_args()

    import numpy as np
    caught = []
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        import torch
        import torch_npu                                      # noqa: F401
        caught += [str(x.message)[:200] for x in w]

    N = a.N
    rng = np.random.default_rng(0)
    x_np = (rng.standard_normal(N) + 1j * rng.standard_normal(N)
            ).astype(np.complex64)
    x_cpu = torch.from_numpy(x_np)

    print("=" * 70)
    print("torch %s   torch_npu %s   N = %d"
          % (torch.__version__,
             getattr(__import__("torch_npu"), "__version__", "?"), N))
    print("=" * 70)

    dev = "npu:0"
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        t = x_cpu.to(dev)
        y = torch.fft.fft(t)
        torch.npu.synchronize()
        msgs = [str(x.message)[:200] for x in w]

    print("\n1. where the tensors live")
    print("   input  after .to('npu:0') : %s" % t.device)
    print("   OUTPUT of torch.fft.fft   : %s" % y.device)
    on_device = "npu" in str(y.device)
    print("   -> %s" % ("output is on the NPU" if on_device
                        else "OUTPUT CAME BACK ON THE CPU"))

    print("\n2. warnings torch raised")
    fall = [m for m in caught + msgs
            if "fallback" in m.lower() or "cpu" in m.lower()
            or "not implemented" in m.lower() or "unsupport" in m.lower()]
    if fall:
        for m in fall[:6]:
            print("   ! %s" % m)
    else:
        print("   (none mentioning a fallback)")

    def timeit(make, sync):
        for _ in range(max(a.warmup, 1)):
            make()
            sync()
        ts = []
        for _ in range(a.runs):
            t0 = time.perf_counter()
            make()
            sync()
            ts.append((time.perf_counter() - t0) * 1e6)
        ts.sort()
        return ts[len(ts) // 2], ts[0], ts[-1]

    tn = x_cpu.to(dev)
    npu_med, npu_lo, npu_hi = timeit(lambda: torch.fft.fft(tn),
                                     torch.npu.synchronize)
    cpu_med, cpu_lo, cpu_hi = timeit(lambda: torch.fft.fft(x_cpu),
                                     lambda: None)

    print("\n3. the transform alone, median of %d runs" % a.runs)
    print("   on 'npu:0'      %9.1f us   (min %.1f, max %.1f)"
          % (npu_med, npu_lo, npu_hi))
    print("   forced on cpu   %9.1f us   (min %.1f, max %.1f)"
          % (cpu_med, cpu_lo, cpu_hi))
    ratio = cpu_med / npu_med if npu_med else float("inf")
    print("   cpu / npu = %.2f" % ratio)

    t0 = time.perf_counter()
    ref = np.fft.fft(x_np.astype(np.complex128))
    numpy_us = (time.perf_counter() - t0) * 1e6
    err = float(np.abs(torch.fft.fft(tn).cpu().numpy() - ref).max()
                / max(np.abs(ref).max(), 1e-30))
    print("\n4. correctness and the host reference")
    print("   numpy single run %9.1f us" % numpy_us)
    print("   max rel error vs numpy: %.3e" % err)

    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    if not on_device:
        print("  The output is not on the device. torch.fft fell back to the")
        print("  host, so these rows are a CPU measurement wearing an NPU")
        print("  label. They must not be reported as a device comparison.")
    elif ratio < 1.5:
        print("  The output claims to be on the device, but forcing the same")
        print("  transform onto the CPU costs %.2f of the NPU time. A device" % ratio)
        print("  that offers no speedup over the host for the same call is")
        print("  the signature of a dispatch that never left the host.")
        print("  Check the warnings above and treat these rows as suspect.")
    else:
        print("  The output is on the device and the CPU path is %.1fx slower," % ratio)
        print("  so the device really is doing the work. The sweep figures")
        print("  stand, and PyTorch beating both MindSpore and these kernels")
        print("  is a real finding that belongs in the text.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
