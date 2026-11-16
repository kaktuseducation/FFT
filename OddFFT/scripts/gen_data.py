#!/usr/bin/env python3
"""Generate input and golden data for OddFFT.

N must be a power of 3, 5 or 7. Unlike the power-of-two project there is no N
shared by all three radices, so there is no single "default size everyone
agrees on": the comparable triples are 2187/3125/2401 (within 1.43x) and
19683/15625/16807 (within 1.26x).

Writes:
  <input-dir>/input_x.bin   interleaved complex (re, im, re, im, ...) float32
  <output-dir>/golden.bin   numpy.fft.fft of that input, same layout

The golden transform is computed in float64 and narrowed only at the end, so
the reference is not itself limited by float32 rounding.
"""
import argparse
import os
import sys

import numpy as np

POWERS = {
    3: [3**k for k in range(1, 12)],
    5: [5**k for k in range(1, 9)],
    7: [7**k for k in range(1, 8)],
}


def factor_check(n):
    """Factor n over the primes {2,3,5,7}, or return None.

    Returns (label, factors). A pure power is labelled as such; anything else
    gets its full factorisation, because the combined transform (mix/frank_*)
    accepts any N whose prime factors lie in this set, not only powers.
    """
    if n < 2:
        return None
    v = n
    fac = {}
    for pr in (2, 3, 5, 7):
        while v % pr == 0:
            v //= pr
            fac[pr] = fac.get(pr, 0) + 1
    if v != 1:
        return None
    label = " * ".join("%d^%d" % (pr, k) for pr, k in sorted(fac.items()))
    return (label, fac)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("N", nargs="?", type=int, default=None,
                    help="transform length, a power of 3, 5 or 7")
    ap.add_argument("-N", dest="N_flag", type=int, default=None,
                    help="same as the positional N (kept for symmetry with the "
                         "power-of-two project's run.sh)")
    ap.add_argument("--input-dir", default="./input")
    ap.add_argument("--output-dir", default="./output")
    ap.add_argument("--seed", type=int, default=42)
    # The {2,3,5,7} check is a typo guard for the odd-radix and combined cells,
    # where an N outside the set cannot be planned at all. Bluestein and Rader
    # take any N, so they need it switched off -- but only on request, so a
    # mistyped N in an odd-radix cell still fails loudly instead of silently
    # generating data no variant can consume.
    ap.add_argument("--any", dest="any_n", action="store_true",
                    help="accept any N >= 2 (for the Bluestein and Rader "
                         "variants, which do not need a smooth N)")
    a = ap.parse_args()

    n = a.N_flag if a.N_flag is not None else a.N
    if n is None:
        ap.error("N is required (positional or -N)")

    if n < 2:
        print(f"[ERROR] N={n} must be at least 2.")
        return 1

    fac = factor_check(n)
    if fac is None and a.any_n:
        # Report the real factorisation so the log says what this N is, then
        # carry on: next_pow2(2N-1) is a power of two whatever N looks like.
        v, pf = n, []
        d = 2
        while d * d <= v:
            while v % d == 0:
                pf.append(d); v //= d
            d += 1
        if v > 1:
            pf.append(v)
        lbl = " * ".join(str(x) for x in pf)
        kind = "PRIME" if len(pf) == 1 else "composite"
        print(f"[INFO ] N={n} = {lbl} ({kind}); not {{2,3,5,7}}-smooth, so only "
              f"blu/* and rad/* can run it")
        fac = (lbl, {})
    if fac is None:
        print(f"[ERROR] N={n} has a prime factor outside {{2,3,5,7}}.")
        for r, ps in POWERS.items():
            print(f"        powers of {r}: " + " ".join(str(p) for p in ps))
        print("        comparable triples: 2187/3125/2401, 19683/15625/16807")
        print("        the combined transform also takes any mixed N, e.g.")
        print("        720 1000 3600 30240 100800")
        print("        a prime factor outside the set is what Bluestein is for")
        return 1
    label, _factors = fac

    rng = np.random.default_rng(a.seed)
    x = (rng.uniform(-1.0, 1.0, n) + 1j * rng.uniform(-1.0, 1.0, n)).astype(np.complex128)
    y = np.fft.fft(x)

    os.makedirs(a.input_dir, exist_ok=True)
    os.makedirs(a.output_dir, exist_ok=True)

    xi = np.empty(2 * n, dtype=np.float32)
    xi[0::2] = x.real.astype(np.float32)
    xi[1::2] = x.imag.astype(np.float32)
    in_path = os.path.join(a.input_dir, "input_x.bin")
    xi.tofile(in_path)

    yi = np.empty(2 * n, dtype=np.float32)
    yi[0::2] = y.real.astype(np.float32)
    yi[1::2] = y.imag.astype(np.float32)
    gold_path = os.path.join(a.output_dir, "golden.bin")
    yi.tofile(gold_path)

    print(f"[GEN] N={n} = {label}  seed={a.seed}")
    print(f"[GEN] {in_path}   {xi.nbytes} bytes")
    print(f"[GEN] {gold_path}   {yi.nbytes} bytes")
    print(f"[GEN] |max golden| = {np.abs(y).max():.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
