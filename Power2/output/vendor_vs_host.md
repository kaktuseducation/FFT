# The vendor accelerator library against the host reference

Measured at 3 lengths, from N = 64 to N = 262144.

- The vendor library is faster than the host reference at 1 of 3 lengths.
- Geometric mean of the ratio: 0.097 (1.000 would be exact parity).
- Range: 0.008 to 1.062.

Per length: N=64: 0.008, N=4096: 0.103, N=262144: 1.062

## Why this matters

Both of these are implementations their vendors tuned: one for the accelerator, one for the host CPU. Holding implementation quality roughly fixed and varying only the hardware, the ratio above is what the accelerator is worth on this workload. A figure near 1.0 says the device is not delivering an advantage through its own vendor's transform, which in turn means a speedup obtained by a custom kernel on the same device is attributable to the kernel and not to the hardware.

This is the claim that separates the contribution from the platform, and it is the one worth stating in the abstract.
