Table 6-X  The best verified variant at each length, against both baselines

| N | Best verified variant | Total (us) | MindSpore, same device (us) | vs MindSpore | NumPy, host (us) | vs NumPy |
|---|---|---|---|---|---|---|
| 64 | r4/dit_fsm_xs, tile 8, 40 cores | **43.10** | 813.55 +- 48.10 (n=2) | **18.88x faster** | 6.85 +- 0.53 (n=3) | 6.29x slower |
| 4096 | r4/dit_fsm_xs, tile 8, 40 cores | **142.46** | 1054.38 +- 127.10 (n=2) | **7.40x faster** | 108.87 +- 6.45 (n=3) | 1.31x slower |
| 262144 | r4/dit_fsm_xs, tile 8, 40 cores | **6884.78** | 22606.11 +- 431.08 (n=3) | **3.28x faster** | 24005.78 +- 1137.34 (n=4) | **3.49x faster** |

Every baseline figure is the mean over the runs recorded in the
measurement files, with the standard deviation and the run count
given, and every speedup is computed against that mean.  The ranges
behind each ratio are in output/comparison_by_length.csv.
