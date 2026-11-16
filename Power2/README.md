# FFTBench — one project for power-of-two radix FFT kernels on Ascend NPU

Built from three earlier projects (Radix2, Radix4, Radix8). One binary, one
output format, one results table. The kernel code and the bodies of the host
run-functions were carried over **unchanged** — this is code already verified
on hardware.

Target: Huawei Ascend 910B4 (CANN 8.0.RC3, `dav_c220`), 40 AIV cores,
20 AIC cores, 191 KB UB per core.

**If you are reading this to review the work, start with
[`INSIGHTS.md`](INSIGHTS.md).** It holds the findings — what was learned about
the hardware, which design choices won and by how much, and what is still open.
This file is the operating manual.

## What is in it: 31 variants

| radix | algorithm | granularity | layout | compute unit | cores | names |
|---|---|---|---|---|---|---|
| 2 | DIT, Stockham | fused, per-stage | xs, xi | Vector | 1 and 40 | `r2/dit_fs_xs` … `r2/st_psm_xi` (16) |
| 4 | DIT, Stockham | fused, per-stage | xs | Vector | 1 and 40 | `r4/dit_fs_xs` … `r4/st_psm_xs` (8) |
| 8 | DIT | fused | xs | Vector | 40 | `r8/dit_fs_xs` |
| 8 | one 16×16 matmul | fused | xs | **Cube** | 20 | `r8/cube1_fs_xs`, `r8/cube1x2_fs_xs`, `r8/cube1x3_fs_xs` |
| 8 | **two matmuls** | fused | xs | **Cube** | 20 | `r8/cube2_fs_xs`, `r8/cube2x2_fs_xs`, `r8/cube2x3_fs_xs` |

Reading the names: `fs` = fused (the twiddles for every stage in one buffer
with a META descriptor), `ps` = per-stage; **an `m` after the granularity code
means multicore** (`fsm`, `psm`); `xs` = Re and Im in separate arrays, `xi` =
interleaved; `x2`/`x3` on the Cube variants is the number of **terms** in the
precision-recovery scheme (not of matmuls — for `cube2` each term costs two
matmuls).

The binary is always the authority on what exists:

```bash
./build/ascendc_kernels_bbit --list
```

### Single-core / 40-core pairs

Every radix-2 and radix-4 Vector variant exists twice, differing in **nothing
but** the number of cores it occupies: `dit_fs_xs` (one core) and `dit_fsm_xs`
(forty). The pair shares its mathematics, its kernel and its buffers — in the
kernel it is the same code with two modes, switched by a sentinel in the
`stage` parameter. Pairs rather than replacement, for two reasons: the
single-core numbers stay comparable with the matrix already collected, and the
table gains an honest paired comparison on the "cores" axis.

That is also what makes the other axes honest. While only radix-8 was
multicore, comparing radices mixed in the core count: "radix-4 is 7.7× faster
than radix-2" was partly a statement about forty cores against one. The
analyser now holds `cores` fixed inside a group and reports two separate
comparisons, at one core and at forty.

Multicore mode splits the stages across separate launches (stage *s+1* reads
what *s* wrote across the whole array, so it needs a barrier visible to every
core; ordering between launches on one stream provides that for free, whereas a
hand-written cross-core barrier inside a kernel is a proven source of hangs).
Tiles of a stage are handed out across cores through `GetBlockIdx()` with
stride `blockDim`.

### cube1 against cube2: "one matmul or two"

The combine step of a radix-8 butterfly is a multiplication by a fixed complex
DFT8 matrix. There are two ways to hand that to the Cube unit, and the choice
is its own axis of comparison:

* **cube1** folds everything into a single block-real 16×16 matmul, with the ±
  signs built into the matrix itself:
  `[Re(y); Im(y)] = [[Re(M), −Im(M)], [Im(M), Re(M)]] · [Re(t); Im(t)]`.
  Vector only has to add the precision terms.
* **cube2** keeps Re and Im apart and does two matmuls against one "vertical"
  matrix `A2 = [Re(M); Im(M)]`: `P = A2·Re(t)`, `Q = A2·Im(t)`, with the
  combine step `Re(y) = P − Q'`, `Im(y) = P' + Q` done by Vector in unpack.

The number of launches is the same for both — three per group of tiles: both of
`cube2`'s matmuls run inside one kernel, through one Cube object. Otherwise the
comparison would be measuring the cost of an extra launch rather than the two
schemes.

Mathematically `A2` is 16×8, but it is stored as 16×16 with columns 8..15
zeroed. That way a row of A clears the 32-byte `DataCopy` granularity, the
tiling stays the one already proven on hardware, and — most usefully — `repack`
is reused by both variants without a single edit: it already puts Re(t) in rows
0..7 and Im(t) in rows 8..15, so both of `cube2`'s B operands are slices of one
buffer, and the zero columns cancel the contribution of the "wrong" rows. The
full argument is in the header of `kernels/r8_cube2_matmul.cpp`.

## Running it

```bash
# the full matrix — this is the "one big test"
bash run_matrix.sh

# a grid of your own: sizes, thresholds, Cube batch widths
bash run_matrix.sh -n "64 4096 262144" -t "8 64 512" -B "64 128 256"

# quick pass (1 warmup, 3 measurements)
bash run_matrix.sh -q

# a single variant
bash run.sh -c r4/dit_fs_xs -r npu -n 4096 -t 8

# bit-exactness: each variant twice, compared byte for byte
bash scripts/check_determinism.sh
bash scripts/check_determinism.sh -n 262144 -t 8 -r 5

# analysis of a table already collected
python3 scripts/analyze_matrix.py
python3 scripts/analyze_matrix.py --axis cores --all-pairs
python3 scripts/analyze_matrix.py -N 4096 --export ./output/pivot.csv
```

The default grid is the full one on purpose: 3 sizes × 3 thresholds × 3 batch
widths. Two of those axes only produce paired comparisons when the table holds
at least two values for them, so a narrower default silently costs conclusions.

## About N: a hard constraint

N comes from the size of `input/input_x.bin` and must be a **power of the
radix**: radix-2 takes any `2^k`, radix-4 only `4^k`, radix-8 only `8^k`.
Radices can therefore only be compared with one another at the values common to
all three, that is at `N = 2^(6k)`: **64, 4096, 262144**. `run_matrix.sh` does
not queue an incompatible pair, and the binary rejects one with exit code 1.

## The axes of comparison

`output/matrix.csv` — one row per cell, every axis as its own column:

```
impl, variant, radix, algo, granularity, layout, unit, terms, mm,
N, tile, cube_batch, cores,
npu_us, cpu_us, total_us, total_min_us, total_max_us, stages_npu, stages_cpu,
status, mismatches, points, err_ratio, max_abs, max_rel
```

`tile` is the `min_npu_*` threshold: stages smaller than the threshold are
computed on the CPU, the rest on the NPU. So `cpu_us` is not overhead, it is a
real part of the algorithm, and `total_us` is what takes part in a comparison.

`cores` is the variant's **declared** blockDim policy, not measured occupancy.
Every radix-2 and radix-4 variant comes as a single-core / 40-core pair, so for
those the field is exactly what tells the twins apart; `r8/dit` hands tiles out
across the 40 AIV cores and `r8/cube` across the 20 AIC cores.

`mm` is how many matmuls one precision term costs: 1 for `cube1`, 2 for
`cube2`, 0 for the Vector variants. It is a column of its own rather than
something inferred from the variant name, because only then does the analyser
produce a **paired** comparison on the "one matmul or two" question with
everything else held fixed (`--axis mm`).

`cube_batch` is a compile-time constant (`CUBE_BATCH`) and affects only the
Cube variants; sweeping it requires a rebuild, which is why it is the outer
loop in `run_matrix.sh`.

**Only the paired-comparison block of the analyser is fit for drawing
conclusions.** The overview table below it reports the best time per axis value
anywhere in the table, which mixes cells that differ in several axes at once;
it is printed for orientation and labelled as unusable for conclusions. The
reasoning is in `INSIGHTS.md`, section 2.1.

## Layout

```
INSIGHTS.md         the findings — read this first if you are reviewing
host/
  main.cpp          CLI, dispatch, golden comparison, matrix.csv writer
  registry.h        the list of variants and their axes — the single source of truth
  cell.h            RunResult + the description of the cell being run (g_cell)
  bench_harness.h   the shared output format and timing
  verify.h          golden comparison inside the binary (replicates numpy.isclose)
  fft_host_r2.h     radix-specific utilities, each in its own namespace
  fft_host_r4.h
  fft_host_r8.h
  cube_tiling.h     TCubeTiling for the Cube variants
  kernel_decls.h    declarations of every kernel
  impl_r2.cpp       the radix-2 run-functions (carried over unchanged)
  impl_r4.cpp
  impl_r8.cpp       radix-8: the Vector variant and the whole Cube pipeline
kernels/            18 .cpp with a radix prefix + cube1_common.h
                    Cube path: r8_cube1_repack.cpp is SHARED by cube1 and cube2;
                    r8_cube_fused_xs.cpp + r8_cube1_unpack.cpp are cube1;
                    r8_cube2_matmul.cpp + r8_cube2_unpack.cpp are cube2.
                    One __global__ per file is not style — it works around the
                    precompile classifier, see the pitfalls below.
scripts/
  gen_data.py            input + golden for a given N
  bench_numpy.py         NumPy baseline on the same input
  verify_result.py       manual re-check of a single run
  analyze_matrix.py      analysis of matrix.csv: per-size tables and paired axes
  check_determinism.sh   each variant twice, compared byte for byte
  compare_runs.py        tells apart the signatures of a race / a collision / NaN
```

## The binary's exit codes

| code | meaning |
|---|---|
| 0 | the run happened, verification PASSed |
| 1 | error: unknown variant, no input, N not a power of the radix |
| 2 | the variant produced no result (a stub, or a kernel error) |
| 3 | the run happened, verification FAILed |

## Pitfalls already stepped on

These places are deliberately kept as they are and carry comments in the code;
changing them without reading the reasoning first is a bad idea. The full
accounts, with how each was diagnosed, are in `INSIGHTS.md` section 1.

* **The Cube kernels are built by a separate `ascendc_library()`.** Mixed into
  one target with the Vector kernels, the build fails in
  `MatmulChipCap::GetFeatures()`. See `common/cmake/npu_lib.cmake`.
* **`-DHAVE_WORKSPACE/-DHAVE_TILING` only on the cube library.** Without them
  the wrapper never calls `matmul::clearWorkspace()`, the AIC↔AIV exchange
  region stays garbage, and the very first matmul launch is a real hardware
  deadlock. Applied globally these defines break the Vector kernels instead:
  their last parameters are scalars, and the wrapper treats the last two
  parameters of every kernel as GM addresses.
* **AIV kernels launch only through a manual `_do()` wrapper**, not through
  `ACLRT_LAUNCH_KERNEL`: the second path gave a deterministic aicore exception
  (`0x7150026`, unaligned UUB). For the two matmul kernels it is the reverse —
  `ACLRT_LAUNCH_KERNEL` is mandatory.
* **One `__global__` kernel per file.** While repack and unpack shared a file,
  CANN's precompile scanner tagged both `_mix_aic`/`_mix_aiv` and they took an
  aicore exception for no apparent reason.
* **`DataCopy` granularity between UB and GM is 32 bytes**: 8 `float` or 16
  `half`. A shorter copy silently transfers nothing.
* **`TQue<VECIN>` versus `TQue<VECOUT>` are synchronization directions**, not
  "input/output". VECIN gives MTE2→V, VECOUT gives V→MTE3. Receiving data from
  GM into a VECOUT buffer and then reading it with the vector unit is a race
  that shows up as rare nondeterminism in the low bits — and that passes the
  golden check every time. `scripts/check_determinism.sh` is the test for it.
* **Garbage in GM is not only wrong numbers, it is NaN.** `cube2`'s matmul
  inevitably reads rows that repack did not write (their contribution is
  cancelled by the zero columns of `A2`), but `0 * NaN = NaN`, and a fresh
  `aclrtMalloc` holds garbage in which NaN/Inf for `half` is common. Hence
  `scratchX` is zero-filled once before the run; without it the bug would
  appear every other time, depending on the contents of DDR.
* **The Cube scratch footprint has a cliff between 640 KB and 1122 KB.** Above
  it, and with enough groups to pay for it, the same code runs 10–40× slower —
  `cube1` at B=256 goes from 11437 µs to 486086 µs at N=262144. It is the byte
  count that predicts this, not the batch width and not the choice of scheme, so
  the host prints `scratch=... KB` on the `[CUBE-PLAN]` line and warns before a
  run that is about to cross it. The healthy configuration is `cube1` at B=128.
  Full table in `INSIGHTS.md` section 2.11.
* **`output_x.bin` is removed before every run.** The file survives between
  runs, and without removing it a variant that wrote nothing would be compared
  against the previous run's output — up to a false PASS.
