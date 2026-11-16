#!/usr/bin/env python3
"""Compare two output_x.bin files from two runs of the SAME variant.

Used by scripts/check_determinism.sh when a byte comparison fails: `cmp` says
only that the runs differ, and the interesting question is HOW they differ,
because the answer points at the cause.

  * a handful of points differing in the last mantissa bits (ulp 1-2) is the
    signature of a synchronization race — the vector unit reading UB before
    MTE2 finished the load, or a missing PipeBarrier. That is what the
    VECIN/VECOUT bug in kernels/r8_cube1_unpack.cpp looked like: one or two
    elements out of 4096, drifting from run to run, while every run still
    passed the golden check.
  * many points differing, or differing by a large amount, is the signature of
    two cores writing the same GM cell — a tile-enumeration bug where the
    host's rule and the kernel's blockIdx arithmetic disagree.
  * NaN appearing in one run and not the other means uninitialized GM is being
    read: see the zero-fill of scratchX in run_cube_fs_xs (host/impl_r8.cpp).

Prints one line, suitable for embedding in the shell script's output.
"""
import sys
import numpy as np


def ulp_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance in representable fp32 steps between a and b, elementwise.

    Reinterpreting a float32 as int32 makes adjacent representable values
    adjacent integers (for same-sign values), which is what makes "differs by
    1 ulp" a meaningful statement rather than a guess about magnitude.
    """
    ai = a.view(np.int32).astype(np.int64)
    bi = b.view(np.int32).astype(np.int64)
    # Map the negative half of the range so that the ordering is monotone
    # across zero; without this, -0.0 and +0.0 come out 2^31 apart.
    ai = np.where(ai < 0, np.int64(-2**31) - ai, ai)
    bi = np.where(bi < 0, np.int64(-2**31) - bi, bi)
    return np.abs(ai - bi)


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: compare_runs.py <run1.bin> <run2.bin>", file=sys.stderr)
        return 2

    a = np.fromfile(sys.argv[1], dtype=np.float32)
    b = np.fromfile(sys.argv[2], dtype=np.float32)
    if a.size != b.size:
        print("sizes differ: %d vs %d floats" % (a.size, b.size))
        return 1
    if a.size == 0:
        print("both files are empty")
        return 1

    nan_a, nan_b = int(np.isnan(a).sum()), int(np.isnan(b).sum())
    if nan_a or nan_b:
        print("NaN present: run1=%d run2=%d of %d floats — uninitialized GM is "
              "being read" % (nan_a, nan_b, a.size))
        return 1

    diff = a != b
    n = int(diff.sum())
    if n == 0:
        # The shell script only calls this after `cmp` already reported a
        # difference, so getting here means the bytes differ while the decoded
        # floats do not. The one way that happens in practice is a -0.0 / +0.0
        # pair, which is worth saying out loud rather than crashing on an empty
        # reduction: it is not a race, and it is not a wrong answer.
        print("the bytes differ but every float compares equal "
              "(a -0.0 / +0.0 pair, most likely) — not a numerical difference")
        return 0
    absd = float(np.abs(a[diff] - b[diff]).max())
    ulp = int(ulp_distance(a[diff], b[diff]).max())

    # The outputs are interleaved Re,Im — report in points, which is what the
    # rest of the project counts in.
    points = a.size // 2
    print("%d of %d floats differ (%.3f%% of %d points), max |delta|=%.3e, "
          "max %d ulp" % (n, a.size, 100.0 * n / a.size, points, absd, ulp))
    return 1


if __name__ == "__main__":
    sys.exit(main())
