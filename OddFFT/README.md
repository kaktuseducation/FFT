# OddFFT — FFT kernels for the odd radices 3, 5 and 7 on an Ascend NPU

Companion project to the power-of-two benchmark (FFTBench). Same method:
every design choice is implemented in all its variants and decided by a
**paired measurement** rather than by argument.

Target: Ascend 910B4 (`dav_c220`), CANN 8.0.RC3 — 40 AIV (Vector) cores,
20 AIC (Cube) cores, 191 KB of UB per core.

---

## Status

Phase 1. One kernel end to end, with the host side, the test harness and the
analysis complete.

| | |
|---|---|
| kernels | **14**, generated: radix {3,5,7} × butterfly {direct, sym, winograd, rader}, plus radix {2,4} × direct for mixed-radix stages |
| variants | **32** — each kernel as a 1-core / 40-core pair, plus 4 combined-transform variants (`./build/oddfft --list`) |
| mixed radix | **working and verified**, over radices {2,3,4,5,7} in any order — see `docs/MIXED_RADIX.md` |
| combined transform | `mix/frank_*` — picks factorisation, stage order and butterfly per stage. Takes any N factorisable over {2,3,5,7} |
| verified without a device | the host arithmetic vs a naive DFT (incl. mixed lists); every generated kernel's **emitted body** run scalar vs a reference DFT; every kernel's **exact addressing** simulated for result and write coverage, 162 cases; 18 combined-transform runs end to end with N cross-checked |
| verified on a device | **full matrix measured: 600 cells, 600 PASS, 0 FAIL.** See "First matrix" below |

Everything below that is not marked as measured is a **design decision with a
stated reason**, not a result.

---

## Quick start

```bash
# host-side tests — no device, no numpy
bash tests/run_tests.sh

# regenerate the kernels (only needed after editing a butterfly)
python3 scripts/gen_kernels.py

# one variant on hardware
bash run.sh -c r3/sym_psm_xs -n 2187 -t 64 -M 8

# every butterfly of one radix, same N and tile -- the paired butterfly axis
bash run.sh -c "r3/sym_psm_xs r3/direct_psm_xs r3/winograd_psm_xs r3/rader_psm_xs" -n 2187

# the full matrix
bash run_matrix.sh

# the variant list
./build/oddfft --list
```

To **stop a matrix run in progress**: `touch STOP` in the project directory.
It is checked at every cell boundary. Ctrl-C is not reliable — SIGINT does not
reach a process sitting inside an NPU driver call, and bash then carries on to
the next cell.

---

## The two facts that shape this project

### 1. An odd stride is not representable, so nothing is strided

`DataCopyParams{blockCount, blockLen, srcStride, dstStride}` counts its strides
in **32-byte units**. A radix-3 stage needs a stride of `t` floats with `t` a
power of three — never a multiple of 8. Counted across stages:

| radix | stages whose stride is not representable |
|---|---|
| 8 | 0 of 0 |
| 4 | 1 |
| 2 | 2 |
| **3** | **12 of 12** |
| **5** | **8 of 8** |
| **7** | **6 of 6** |

Integer division truncates silently, so the wrong answer here is a quiet one.

The way out, confirmed on this device by a standalone probe (`AlignProbe`):
give up on strided copies and issue each butterfly's `r` operand groups as `r`
**separate contiguous runs from unaligned bases**. Plain `DataCopy` tolerates
an unaligned GM base at every odd offset tested, and the penalty is **fixed
per transfer, not per byte** (×1.34–1.40 at 64 floats, ×1.11–1.19 at 512).
That is why the tile is shaped along `j`: a longer run amortises the same
fixed cost.

### 2. `t` is never a multiple of 8, so the write can never round up

The power-of-two kernels rounded every transfer length up to 8 for free,
because the true length was already a multiple of 8. Here it never is.
Rounding the write up would overrun the end of a `j`-run: for the `m=0` group
into the `m=1` group of the same butterfly, and for the last group into the
**next k-block — a different tile, and under multicore a different core**.
That is a cross-core race on live data, and it would produce a plausible wrong
answer rather than a crash.

So the write is `DataCopyPad`, byte-granular, exactly as many floats as are
live. And because a `DataCopyPad` below the 32-byte granule is not something
this device has been shown to support, a trailing remainder shorter than 8 is
**absorbed into the previous tile** instead of becoming a tile of its own.
Every tile length is therefore in `[8, tile+7]`, and the buffers are sized
`tile+8`. The arithmetic lives in `common/tiling.h` as a macro used by the
kernel *and* the host — if the two ever disagreed on the tile count, blocks
would be handed the wrong tiles.

---

## There is no N common to radix 3, 5 and 7

In FFTBench every radix shared `N = 2^(6k)` (64, 4096, 262144), so every
cross-radix comparison was paired at identical N. No N above 1 is a power of
3 *and* 5 *and* 7, so that method is simply unavailable here.

What replaces it, and its limits:

* comparisons **within** a radix stay paired at identical N — these are the
  strong ones, and all the per-axis blocks of the analyzer are of this kind;
* comparisons **across** radices use the nearest sizes and a normalised figure
  `ns_per_pt = total_us * 1000 / (N * log2 N)`, computed by the binary and
  written into `output/matrix.csv` so every reader sees the same number;
* the nearest triples are **2187 / 3125 / 2401** (within 1.43×) and
  **19683 / 15625 / 16807** (within 1.26×). The analyzer prints that spread
  next to every cross-radix verdict, because it is the limit on what the
  verdict is worth.

`radix` is deliberately **not** one of the paired axes. An axis that can never
find a pair would print "no data" forever and invite the reader to assume the
comparison was simply not run.

---

## The 12 kernels are generated, not written

`scripts/gen_kernels.py` emits all of them from one template. The reason is not
only speed.

The butterfly algorithm is the main axis this project measures. For that
comparison to be **paired**, every variant must differ in the butterfly and in
nothing else: same tiling, same `DataCopyPad` strategy, same queue layout, same
twiddle handling, same tile cap. Twelve hand-written files guarantee drift.
Generating them makes "identical except the butterfly" a property of the build
rather than a claim in a comment.

The tile cap is shared for the same reason. The kernels' UB footprints differ
by more than 2× (radix 7 needs far more slices than radix 3), so a per-kernel
cap would have let radix-3 run a 352-wide tile while radix-7 `winograd` was
silently clamped to less — and then `sym` vs `winograd` at radix 7 would have
been measured at *different effective tiles* while the table recorded the same
requested one. One cap, set by the hungriest kernel, keeps every variant on the
same tile ladder. The host clamps with the same macro the kernel clamps with
(`kernels/kernel_limits.h`) and records the **effective** tile in the matrix.

## The butterfly algorithms

Operation counts as the **generated kernels actually issue them**
(`python3 scripts/gen_kernels.py` prints this table and reconciles it against
the reference, exactly, every time it runs). Each algorithm is verified against
`numpy.fft` three times over: as scalar Python in `docs/odd_butterflies.py`, as
the expression graph the generator builds from it, and as the emitted C++ body
in `tests/gen_butterfly_test.cpp`.

| r | algorithm | mults | adds | total | per point |
|---|---|---|---|---|---|
| 2 | direct | 0 | 4 | 4 | 2.0 |
| 4 | direct | 0 | 24 | 24 | 6.0 |
| 3 | direct | 16 | 20 | 36 | 12.0 |
| 3 | sym | 4 | 12 | 16 | 5.3 |
| 3 | **winograd** | 4 | 12 | **16** | **5.3** (ties `sym`) |
| 3 | rader | 16 | 20 | 36 | 12.0 |
| 5 | direct | 64 | 72 | 136 | 27.2 |
| 5 | **sym** | 16 | 32 | **48** | **9.6** |
| 5 | winograd | 10 | 42 | 52 | 10.4 |
| 5 | rader | 64 | 72 | 136 | 27.2 |
| 7 | direct | 144 | 156 | 300 | 42.9 |
| 7 | **sym** | 36 | 60 | **96** | **13.7** |
| 7 | winograd | 18 | 100 | 118 | 16.9 |
| 7 | rader | 144 | 156 | 300 | 42.9 |

These counts are **lower than `docs/odd_butterflies.py` reports**, and the
generator prints the reconciliation rather than hiding the gap. Three causes,
all of them the reference charging for work no kernel would do: accumulator
initialisations of the form `add(0.0, x)` (no kernel materialises a zero tensor
to add it), arithmetic on two literals, **multiplies by 0 or ±1** (a zero, a
copy or a sign — not arithmetic; every radix-4 twiddle is one of these, which
is why radix 4 issues no multiplies at all, and 16 of winograd-7's 32
"multiplies" turn out to be sign flips), and — found by generating from it —
**two dead adds in the radix-5 winograd reference**, whose results nothing
consumes. That last one corrects the earlier claim that winograd-5 costs 54
operations; it costs 52. It still loses to `sym`.

Three things worth noting, the first two of which cut against the classical
literature:

* **Total matters, not the multiply count.** A multiply and an add are each
  one vector instruction here. Winograd minimises multiplies and loses to
  `sym` at r = 5 and r = 7 on the total.
* **Rader is dominated everywhere** in this form (a direct cyclic
  convolution). It is kept as a variant anyway, because being able to say it
  was measured and lost is worth more than asserting it.
* **`sym` and `winograd` tie at r=3** on total operations (16 each), so that
  pair is a clean test of whether operation count predicts time at all on this
  hardware — same count, different dependency structure.

These are operation counts, not timings. Which of them wins on the device is
the question the project exists to answer, and the counts are only the prior.

---

## What is *not* carried over from the power-of-two project

FFTBench found DIT beats Stockham, fused beats per-stage, and `xs` beats `xi`.
**None of those is assumed here.** DIT won there partly because it is in place
and partly because every stride it needed was a multiple of 8 — and the second
reason cannot hold for an odd radix. Both schemes exist as axes here and will
be measured from scratch.

The first kernel is per-stage because per-stage gives a free inter-stage
barrier from stream ordering, which is what multicore needs — not because
per-stage is expected to win.

---

## The combined transform ("Frankenstein")

`mix/frank_hi_psm_xs` and friends. For a given N it makes three choices and
then runs each stage on the kernel that won it:

1. **Factorisation.** `3600 = 2⁴·3²·5²` can be `[5,5,4,4,3,3]` or
   `[2,2,2,2,3,3,5,5]` or others. Fewer, larger radices mean fewer **stages**,
   and a stage is a whole pass over the array plus a launch. Larger radices
   also mean more arithmetic per point. Which wins is not obvious — the
   power-of-two project found radix 4 beating radix 2 on all 56 paired cells
   *despite radix 2 having the lower operation count*, which is why the planner
   does not simply minimise operations.
2. **Stage order.** The operand distance `t` is the product of the earlier
   radices, and a stage with `t < 8` cannot be fed by `DataCopy` at all, so it
   runs on the host. Large-radix-first clears the floor sooner. This is
   registered as **two variants** (`frank_hi_*` and `frank_lo_*`) differing in
   nothing else, so the ordering effect is a paired measurement rather than an
   assumption baked into the planner.
3. **Butterfly per stage**, from the operation counts.

The cost model is `α · (ops per point) + β · (stages)`. The second term is what
a pure operation count misses, and **its coefficients are not yet fitted to
this hardware**. Until the matrix has been measured, every plan this produces
is a *prediction*; the binary prints it labelled as one, and so should the
thesis.

Example plans it currently produces (N = 3600):

```
frank_hi   5/sym 5/sym 4/direct 4/direct 3/sym 3/sym          6 stages
frank_lo   2/direct ×4  3/sym 3/sym  5/sym 5/sym              8 stages
```

## Mixed radix

**Working and verified** over radices {2, 3, 4, 5, 7}, in any order and any
mixture, so one plan covers `N = 2^a · 3^b · 4^c · 5^d · 7^e`.

It did not start that way. An earlier version of this project *refused* mixed
lists and carried a comment explaining why the permutation for them was hard.
That was a misdiagnosis: the digit-reverse formula was correct and the
direction it was applied in was not — a scatter where a gather was needed. For
a uniform list digit reversal is an involution, so the two are the same map and
the bug is invisible; the only non-uniform list that passed, `[3,5,3]`, passed
for being a palindrome.

`docs/MIXED_RADIX.md` has the evidence table, the involution rates, and the
method that found it. It is worth reading as a methodology note rather than a
bug report: the wrong diagnosis was more expensive than the bug, because it
closed the question.

## Layout

```
kernels/     GENERATED device kernels, one __global__ kernel per file (a CANN
             precompile classifier tags a file with two as _mix_* and the
             launch then dies with an aicore exception). Do not edit by hand;
             regenerate with  python3 scripts/gen_kernels.py
  oddfft_tiling.h  the tile-count macros, shared by the kernels AND the host
  kernel_limits.h  the shared tile cap, used by the kernels AND the host
  kernel_table.h   launch declarations + the dispatch table the host registers
  (all three live here, not in common/, because the kernel build target gets
   no extra include directories -- a kernel can only include headers beside it)
host/
  plan.h             stage plan, twiddle tables, digit reversal
  butterflies.h      the host butterflies (sym + direct, for cross-checking)
  host_fft.h         the whole transform on the host — for the self-test
  host_twiddles.h    the PRECOMPUTED table for the timed host prefix
  platform_cores.h   AIV/AIC core counts from the toolkit, not hardcoded
  cell.h registry.h  the axes of the matrix and the variant list
  impl_odd.cpp       ONE run-function for every variant, registered from
                     kernels/kernel_table.h -- no file per radix, because the
                     kernels share a launch signature by construction
  main.cpp           CLI, verification, the matrix row
common/
  data_utils.h       file IO
tests/
  host_selftest.cpp       host arithmetic vs a naive O(N^2) DFT, incl. mixed
  gen_butterfly_test.cpp  GENERATED: every kernel's emitted body, run scalar.
                          This is what checks the register allocation
  sim_odd.cpp             the kernels' exact addressing, every radix: result
                          AND that each element is written exactly once
docs/
  MIXED_RADIX.md       the permutation, the misdiagnosis, and the method
  ODD_RADIX_DESIGN.md  the design record
  odd_butterflies.py   the op-count table (self-testing)
  plan_model.py        the stage-plan model that found the mixed-radix failure
scripts/
  gen_kernels.py  THE GENERATOR: builds each butterfly's expression graph from
                  the verified reference, checks it against numpy, schedules
                  and register-allocates it, and emits the kernel, the test and
                  the dispatch table. Run it after changing any butterfly.
  gen_data.py, bench_numpy.py, analyze_matrix.py, check_determinism.sh
```

## One pitfall found in *this* project

A matrix cell can **pass on the wrong N**. If data generation fails and the
previous size's `input_x.bin` and `golden.bin` are still on disk, the binary
runs on the stale input and verification passes — input and golden are stale
*together*, so they agree with each other. It is a false PASS on a size that
was never tested, and nothing in the results table shows it.

This happened during development of the combined transform: eighteen cells
reported PASS while actually running three different sizes. It was caught only
because the planner prints its stage list and the list did not match the N.
`run_matrix.sh` now deletes both files before regenerating and aborts the cell
if generation fails, and the development checks cross-check the N the binary
reports against the N that was requested.

## The floor was wrong

Plain `DataCopy` moves whole 32-byte granules, so a run shorter than 8 floats
transfers nothing. That constraint is real, and it was carried over from the
power-of-two project as a floor of 8 on every stage.

**But these kernels do not use plain `DataCopy`.** They use `DataCopyPad`,
which is byte-granular by construction — and nothing had tested it below the
granule. It has now been tested: lengths 1, 2, 3, 5 and 7 at every unaligned
base, 96 cases, all correct, with no write past the end.

So the floor for this data path is **1, not 8**. With the floor at 8, the
leading stages of every odd radix were stranded on the host — `t = 1, 3` for
radix 3, `t = 1, 5` for radix 5, `t = 1, 7` for radix 7 — and that is where
80–93% of the runtime was going.

Possible is not profitable: a stage with `t = 1` moves one float per run, and
a transfer costs about the same whatever its length. So the threshold is a
matrix axis (`MIN_T ∈ {1, 8, 64}`), not a constant, and the question of
whether a tiny stage belongs on the device is left to measurement.

## First matrix

600 cells, every one passing against the NumPy reference. The axis verdicts:

| Axis | Winner | Comparisons | Range |
|---|---|---|---|
| butterfly | sym 46, winograd 33, direct 15, rader 14 | 108 | 1.00-4.15x |
| cores | 40 wins 236, 1 wins 40 | 276 | 1.00-4.09x |
| tile width | 352 wins 113, 64 wins 46, 8 wins 25 | 184 | 1.00-3.92x |
| host-prefix threshold | 8 wins 177, 64 wins 99 | 276 | 1.00-4.59x |

**The dominant finding is not on any axis.** Nothing in the odd-radix family
beats NumPy; the best cell is 1.27x slower. It loses on the *host*, not on the
device:

| N | device | host | host share |
|---|---|---|---|
| 19683 (radix 3) | 393.8 us | 1577.1 us | 80% |
| 15625 (radix 5) | 160.6 us | 1463.0 us | 90% |
| 16807 (radix 7) | 135.8 us | 1804.0 us | **93%** |

The kernels do their part in 136-394 us; the two sub-floor stages that fall
back to the CPU take five to thirteen times longer. This is the same wall the
power-of-two project hit at 92%, now measured directly.

**`sym` and `winograd` tie at radix 3 in time as well as in operations** --
3431.17 us against 3431.62 us, a 0.01% difference, on 16 operations each. But
`direct` at 36 operations is only 1.02-1.39x slower, not the 2.25x the counts
imply. Operation count predicts the tie and over-predicts the gap.

## First numbers from hardware

`r3/sym_psm_xs`, N = 2187, tile 64, 40 cores:

| | µs |
|---|---|
| NPU (3 stages) | 94.4 |
| host prefix (4 stages) | 340.6 |
| **total** | **435.0** |
| NumPy on the same input | 64.3 |

Two things to read off this, neither flattering and both useful.

**78% of the time is the host prefix.** That is the same shape as the
power-of-two project's wall, and here it was partly self-inflicted: the tile
width was being passed as the threshold for "may this stage run on the device",
so asking for `tile=64` pushed every stage with `t < 64` onto the host — four
stages of seven. The hardware floor is 8, not 64. The two are now separate
parameters (`MIN_T`, 5th argument, default 8) and at `MIN_T=8` the split
becomes 5 device stages against 2 host stages. `MIN_T` is also a matrix axis,
so what the host prefix actually costs is a measured number rather than a
story.

**The device is 6.8× slower than NumPy here.** At N=2187 the three device
stages take 94 µs, about 31 µs per launch, which is launch cost rather than
arithmetic — the transform is too small to pay for the launches. Whether that
reverses at larger N is exactly what the matrix is for; the power-of-two
project saw the same shape and the crossover moved with N.

## What AlignProbe settled

Two modes differing in exactly one line:

| mode | what it adds | result |
|---|---|---|
| `padodd` | `DataCopyPad`, unaligned base, length not a multiple of 8 | **56/56 OK**, no writes past the end |
| `vecunalign` | the same, plus `Duplicate(t[len], …)` at an unaligned offset | **faults (507035) on the first `len % 8 ≠ 0` case** |

So byte-granular `DataCopyPad` is sound at every base and length an odd radix
produces, and it costs the same as plain `DataCopy` (0.1629 vs 0.1640 µs per
64-float copy aligned; both ≈1.35× that at an unaligned base). The whole
odd-radix data-movement design rests on that and it holds.

## A third pitfall: a vector op needs an aligned OFFSET

The first device run of the odd-radix kernels failed every launch with
`aclrtSynchronizeStream` returning **507035**, an aicore exception. The 66 ms
"runtime" was the fault path, and the wrong numbers were simply whatever was in
memory.

The kernels did exactly one thing the power-of-two project never did:

```cpp
AscendC::Duplicate(x0r[curValid], 0.0f, tail);   // curValid is not a multiple of 8
```

A vector op's **length** may be anything, but the tensor it starts from must
begin on a 32-byte boundary. In the power-of-two project every such offset was
a multiple of 8 by construction, so the constraint never showed. Here
`curValid` is a tile length in a radix that is never a multiple of 8.

The fix removes the need rather than the symptom: the vector ops now run at the
**exact** live length instead of a rounded-up one, so there is no tail to zero
and no unaligned offset anywhere. Lanes past `curValid` are never read into a
result, because `CopyOut` writes exactly `curValid` floats.

`AlignProbe` gained two modes, `padodd` and `vecunalign`, which differ in
**exactly** this unaligned `Duplicate` and otherwise perform the same
byte-granular copy. Whichever faults is the cause; if both do, byte-granular
`DataCopyPad` cannot serve this design at all and the data movement needs
rethinking. Running a controlled pair rather than guessing is the point.

## A second pitfall found in *this* project

The kernel build target (`ascendc_library()`) gets **no extra include
directories**. A kernel can include the toolkit's headers and files sitting
beside it, and nothing else.

A shared header was put in `common/` and included as `"tiling.h"`. The local
syntax check passed on all fourteen kernels — because that check was handing
the compiler `-I common`, which the real build does not. Every kernel then
failed on the device with `'tiling.h' file not found`.

The lesson is about the check, not the header: **a check more permissive than
the build it stands in for will certify exactly the bugs it exists to catch.**
`scripts/chk_kernels.sh` now passes only the stub directory, mirroring the real
include set, and additionally verifies structurally that every quoted include
in a kernel names a file in `kernels/`. That second check was confirmed against
the real failure by putting the header back in `common/` and watching it fire.

## Pitfalls inherited from the power-of-two project

Recorded here so they are not rediscovered:

* **`DataCopy` granularity between UB and GM is 32 bytes** — 8 `float`. A
  shorter copy transfers *nothing*, silently. This is why the leading stages
  with `t < 8` run on the host.
* **A TQue's position is a synchronisation direction.** `VECIN` installs
  MTE2→V, `VECOUT` installs V→MTE3. Reading GM into a `VECOUT` buffer, or
  writing GM out of a `VECIN` one, leaves the dependency uninstalled and
  produces partial, data-dependent corruption that *passes* a
  tolerance-based check.
* **`0 × NaN = NaN`**, and a fresh `aclrtMalloc` is full of NaN. Any GM region
  a kernel can read must be initialised even where the arithmetic says the
  value is irrelevant.
* **`blk_num` must equal `blockDim`** — the kernel uses it as the stride with
  which a block walks the tile list.
* A control case that fails means a **broken instrument, not a finding**. The
  alignment probe's first version reported the power-of-two case wrong, which
  it had no business doing; the bug was in the probe.
