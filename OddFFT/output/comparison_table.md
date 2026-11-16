Table 6-X  The best verified variant at each length, against both baselines

| N | Best verified variant | Total (us) | MindSpore, same device (us) | vs MindSpore | NumPy, host (us) | vs NumPy |
|---|---|---|---|---|---|---|
| 101 | rad/rader_r4_psm_xs, tile 352, 40 cores | **471.65** | 819.22 (n=1) | **1.74x faster** | 19.18 +- 0.42 (n=9) | 24.57x slower |
| 125 | r5/direct_psm_xs, tile 64, 40 cores | **27.83** | 840.63 +- 55.44 (n=2) | **30.21x faster** | 8.85 +- 0.21 (n=11) | 3.14x slower |
| 211 | rad/rader_r4_psm_xs, tile 64, 40 cores | **587.80** | 856.38 (n=1) | **1.46x faster** | 30.25 +- 0.30 (n=9) | 19.42x slower |
| 243 | r3/winograd_ps_xs, tile 8, 1 cores | **83.41** | 918.58 (n=1) | **11.01x faster** | 11.19 +- 0.25 (n=10) | 7.45x slower |
| 256 | r4/direct_psm_xs, tile 8, 40 cores | **83.39** | 844.53 (n=1) | **10.13x faster** | 9.83 +- 0.26 (n=19) | 8.49x slower |
| 343 | r7/winograd_psm_xs, tile 352, 40 cores | **48.82** | 797.65 (n=1) | **16.34x faster** | 13.65 +- 0.38 (n=10) | 3.58x slower |
| 720 | mix/frank_hi_psm_xs, tile 8, 40 cores | **105.14** | 808.10 (n=1) | **7.69x faster** | 19.71 +- 0.30 (n=18) | 5.34x slower |
| 1009 | rad/rader_r4_psm_xs, tile 352, 40 cores | **767.32** | 994.16 (n=1) | **1.30x faster** | 165.72 +- 3.23 (n=9) | 4.63x slower |
| 1331 | blu/bluestein_r4_ps_xs, tile 352, 1 cores | **921.55** | 869.14 (n=1) | 1.06x slower | 44.77 +- 0.48 (n=9) | 20.58x slower |
| 2187 | r3/direct_psm_xs, tile 64, 40 cores | **194.30** | 995.46 (n=1) | **5.12x faster** | 65.34 +- 3.08 (n=10) | 2.97x slower |
| 2401 | r7/winograd_psm_xs, tile 64, 40 cores | **101.34** | 955.00 (n=1) | **9.42x faster** | 70.95 +- 1.02 (n=10) | 1.43x slower |
| 3125 | r5/sym_psm_xs, tile 352, 40 cores | **129.85** | 971.97 (n=1) | **7.49x faster** | 99.67 +- 2.11 (n=10) | 1.30x slower |
| 3600 | mix/frank_hi_psm_xs, tile 352, 40 cores | **176.61** | 1037.72 (n=1) | **5.88x faster** | 103.38 +- 3.56 (n=18) | 1.71x slower |
| 4096 | r4/direct_psm_xs, tile 352, 40 cores | **198.92** | 974.80 +- 1.50 (n=2) | **4.90x faster** | 105.62 +- 2.40 (n=20) | 1.88x slower |
| 10007 | blu/bluestein_r4_psm_xs, tile 352, 40 cores | **3959.71** | 2790.70 (n=1) | 1.42x slower | 4111.51 +- 29.08 (n=17) | **1.04x faster** |
| 15625 | r5/winograd_psm_xs, tile 64, 40 cores | **277.84** | 1567.87 (n=1) | **5.64x faster** | 859.34 +- 10.07 (n=10) | **3.09x faster** |
| 16807 | r7/winograd_psm_xs, tile 64, 40 cores | **218.83** | 1676.83 (n=1) | **7.66x faster** | 965.45 +- 11.82 (n=10) | **4.41x faster** |
| 19683 | r3/sym_psm_xs, tile 352, 40 cores | **786.17** | 2137.65 (n=1) | **2.72x faster** | 1555.42 +- 32.39 (n=10) | **1.98x faster** |
| 30240 | mix/frank_hi_psm_xs, tile 352, 40 cores | **489.63** | 2724.64 (n=1) | **5.56x faster** | 2271.73 +- 19.27 (n=10) | **4.64x faster** |
| 32768 | r2/direct_psm_xs, tile 64, 40 cores | **2701.32** | 2654.68 (n=1) | 1.02x slower | 2075.13 +- 84.52 (n=10) | 1.30x slower |
| 65536 | r4/direct_psm_xs, tile 352, 40 cores | **1298.88** | 5067.67 (n=1) | **3.90x faster** | 4446.92 +- 204.92 (n=10) | **3.42x faster** |
| 100800 | mix/frank_hi_psm_xs, tile 352, 40 cores | **999.72** | 7801.85 +- 85.65 (n=2) | **7.80x faster** | 8370.00 +- 178.08 (n=13) | **8.37x faster** |

Every baseline figure is the mean over the runs recorded in the
measurement files, with the standard deviation and the run count
given, and every speedup is computed against that mean.  The ranges
behind each ratio are in output/comparison_by_length.csv.
