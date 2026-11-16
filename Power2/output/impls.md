# Implementations compared

| N | this work | ops.fft | FFTWithSize | torch.fft | NumPy |
|---|---|---|---|---|---|
| 64 | 43.1 | 813.1 | 872.9 | 402.1 | 6.9 |
| 4096 | 142.5 | 1074.8 | 1174.8 | 1963.7 | 108.9 |
| 65536 | -- | 5462.5 | 9812.8 | 737.2 | -- |
| 262144 | 6884.8 | 25351.5 | 37736.8 | 1852.7 | 24005.8 |

## Fixed cost

| implementation | fixed, us | rate, us/op | resid RMS |
|---|---|---|---|
| this work | 56.9 | 1.447e-03 | 11.8 |
| ops.fft | 582.0 | 5.222e-03 | 346.1 |
| FFTWithSize | 1056.5 | 7.800e-03 | 336.9 |
| torch.fft | 1004.3 | 1.613e-04 | 605.4 |
| NumPy | -68.1 | 5.102e-03 | 59.9 |
