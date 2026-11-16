# The vendor accelerator library against the host reference

Measured at 22 lengths, from N = 101 to N = 100800.

- The vendor library is faster than the host reference at 2 of 22 lengths.
- Geometric mean of the ratio: 0.119 (1.000 would be exact parity).
- Range: 0.011 to 1.473.

Per length: N=101: 0.023, N=125: 0.011, N=211: 0.035, N=243: 0.012, N=256: 0.012, N=343: 0.017, N=720: 0.024, N=1009: 0.167, N=1331: 0.051, N=2187: 0.066, N=2401: 0.074, N=3125: 0.102, N=3600: 0.100, N=4096: 0.108, N=10007: 1.473, N=15625: 0.548, N=16807: 0.576, N=19683: 0.728, N=30240: 0.834, N=32768: 0.782, N=65536: 0.877, N=100800: 1.073

## Why this matters

Both of these are implementations their vendors tuned: one for the accelerator, one for the host CPU. Holding implementation quality roughly fixed and varying only the hardware, the ratio above is what the accelerator is worth on this workload. A figure near 1.0 says the device is not delivering an advantage through its own vendor's transform, which in turn means a speedup obtained by a custom kernel on the same device is attributable to the kernel and not to the hardware.

This is the claim that separates the contribution from the platform, and it is the one worth stating in the abstract.
