# FFTBench — findings

What was learned building 31 FFT kernel variants for the Huawei Ascend 910B4
NPU (CANN 8.0.RC3, `dav_c220`, 40 AIV cores / 20 AIC cores, 191 KB UB per
core), and how each conclusion was arrived at.

The file is split into things that are **properties of the hardware and the
toolchain** (section 1), things that are **measured comparisons between design
choices** (section 2), the **one wall the current design cannot get past**
(section 3), and **what is still open** (section 4). Section 5 says how to
reproduce any of it.

A note on epistemic status, because it matters for how much weight to put on
each line. Section 1 items are all *mechanisms*: each was isolated to a single
cause, and in most cases the fix was predicted before it was applied and then
confirmed. Section 2 items are *measurements* from `output/matrix.csv`, and the
numbers quoted are paired comparisons (see 2.1) — they are reproducible but
they are specific to this chip, this CANN version and this input size range.
Where something is a hypothesis rather than a result, it says so.

---

## 1. Hardware and toolchain

These cost the most time to find, and none of them is in the documentation in a
form that would have prevented the bug.

### 1.1 `DataCopy` between UB and GM has a granularity of 32 bytes, and a short copy silently copies nothing

32 bytes is 8 `float` or 16 `half`. A `DataCopy` of fewer elements than that
does not copy a partial block and does not fail — it transfers zero bytes and
returns.

How it showed up: after the Cube operand was switched from fp32 to fp16, the
whole radix-8 Cube pipeline started returning exactly zeros for all 4096
points. The write into `scratchX` was `validB` elements long. At the stage with
`eighth = 8` that is `validB = 8`, which for `half` is 16 bytes — half a block.
Nothing was written, the matmul computed `Y = M16 · 0 = 0`, unpack wrote zeros
back, and the following stages faithfully spread those zeros over the whole
array. While the operand was still `float` the same 8 elements were exactly 32
bytes, so the bug did not exist before the fp16 change and looked like an
fp16 problem rather than an alignment problem.

Consequence: see `AlignUp16` in `kernels/r8_cube1_repack.cpp`. The same
granularity is the reason for the hard floor in all three stage planners
(`if (min_npu_* < 8) min_npu_* = 8;`) — a stage smaller than 8 elements per
row physically cannot be moved to the NPU in this layout. That floor is the
subject of section 3.

### 1.2 A `TQue` position is a synchronization *direction*, not an "in" or "out" label

`TPosition::VECIN` installs an MTE2 → V dependency ("the data is being brought
in by MTE2 and will be read by the vector unit"). `TPosition::VECOUT` installs
V → MTE3 ("the vector unit computed it, now flush it to GM"). Receiving data
from GM into a `VECOUT` buffer and then reading it with vector operations is a
race, because the dependency installed is the opposite of the one needed.

How it showed up — and this is the part worth remembering: **the symptom was
invisible until a later, unrelated change.** `kernels/r8_cube1_unpack.cpp`
received the matmul result into a `VECOUT` queue from the start. While
`terms = 1` there were no vector operations over that data at all — the buffer
was only moved GM → UB → GM — so the wrong direction never mattered and
`cube1_fs_xs` reproduced bit-for-bit. The moment `terms >= 2` added the `Add`
for `Y0 + Y1 + Y2`, the vector unit began reading UB before MTE2 had finished,
and the result was **one or two elements out of 4096 drifting between runs**:
`max_abs` wandered between 9.8e-02 and 1.5e-01 with `max_rel` unchanged. Every
single run still passed the golden comparison.

Two conclusions follow, and the second is the more useful one:

* The fix is a separate `VECIN` queue for the read and a `VECOUT` queue for the
  write. Confirmed on hardware: `cube1x3` went from an unstable 1.2e-01 to a
  stable 1.956e-04 with 0 mismatches, and `terms = 1` stayed bit-identical —
  exactly as predicted, which is what makes it a diagnosis rather than a guess.
* A tolerance-based correctness check cannot catch this class of bug. That is
  why `scripts/check_determinism.sh` exists: it runs each variant twice and
  compares the outputs byte for byte. It is the only test in this project that
  can fail for a reason the matrix cannot see.

### 1.3 fp32 × fp32 through the high-level Matmul API does not work on this architecture

Tested in isolation against Huawei's own reference sample
(`samples/.../11_matmul_kernellaunch/matmul_custom.cpp`) on this same 910B4:

| A/B type | C type | shape | result |
|---|---|---|---|
| `half` | `float` | 512×512×1024 | error ratio 0.0000, passes |
| `float` | `float` | 512×512×1024 | does not hang, computes **pure zeros** (error ratio 1.0) |
| `float` | `float` | 16×16×128 | **hangs** — a real hardware deadlock |

So fp32 operands are not merely slow here, they are wrong, and on small shapes
they hang. The internal L0A/L0B layout is built around fp16 granularity. The
supported pattern is the one the reference uses: `A_T = half`, `B_T = half`,
`C_T = float` — fp16 inputs with an fp32 accumulator.

This is what forced the precision-recovery scheme of section 2.8: fp16 inputs
are not a performance choice here, they are the only thing that computes at
all, so the accuracy has to be bought back arithmetically.

### 1.4 The matmul deadlock was a missing `matmul::clearWorkspace()`, not a data type or a shape

A genuine hardware deadlock on the first launch of the matmul kernel: device
logs cycling `SyncTask` / `LogicCqReportV2`, CPU at ~0%, no timeout, no error.

Cause: the cube library was built **without** `-DHAVE_WORKSPACE -DHAVE_TILING`,
so the auto-generated wrapper never called `matmul::clearWorkspace()`. That
call zeroes the message-exchange region in the system workspace used by the
KFC protocol between the AIC and AIV halves of a MIX kernel (`MatmulClient` on
AIV ↔ `MatmulServer` on AIC). Without it the region holds garbage from previous
tasks and the handshake between the two halves never completes.

The proof that settled it: the identical kernel, compiled inside the official
reference's harness — where those defines are present — does not hang at all.

What was tested and **ruled out** as the cause before that, listed because the
list is itself the finding (none of these is the problem, and each cost a day):
dtype (fp32 vs fp16), the 16×16×128 shape, parameter order and names,
`SetDim(1)` vs `SetDim(GetCoreNumAiv())`, `KERNEL_TASK_TYPE_DEFAULT`,
`SetSingleShape`, having two AscendC libraries in one process, `aclrtMalloc`
flags, an explicit `aclrtCreateContext`, the order and volume of preceding
allocations, and the method of copying the tiling (manual loop vs `CopyTiling`).

### 1.5 `HAVE_WORKSPACE` / `HAVE_TILING` must be scoped to the cube library only

Applied globally they break every Vector kernel. The generated wrapper
reinterprets the **last two parameters of every kernel in the library** as
`workspace, tilingGm` addresses. Vector kernels end in scalars, so the build
fails with "incompatible integer-to-pointer conversion". This — not the Vector
kernels themselves — is what broke the earlier attempt to enable the defines,
and it is why the two matmul kernels deliberately carry
`GM_ADDR workspace, GM_ADDR tilingGm` as their final parameters with all
scalars moved in front. See `common/cmake/npu_lib.cmake`.

### 1.6 Matmul kernels need their own `ascendc_library()` target

Mixing them with ordinary Vector kernels in one target makes the build fail
inside `MatmulChipCap::GetFeatures()`. Hence `KERNEL_FILES_CUBE` in
`CMakeLists.txt`, which holds exactly the two files that
`#include "lib/matmul_intf.h"`.

### 1.7 One `__global__` kernel per file

While `fft8_cube1_repack` and `fft8_cube1_unpack` shared a file, CANN's
precompile scanner tagged both `_mix_aic` / `_mix_aiv`, and they took an aicore
exception for no apparent reason. Confirmed by disassembling trial `aic`/`aiv`
object files. Splitting them one kernel per file fixed it. This is why the Cube
path is five files rather than two, and why `fft8_cube2_matmul` was put in its
own file instead of next to `fft8_cube1_matmul`.

### 1.8 AIV kernels launch through a manual `_do()` wrapper; the matmul kernels must use `ACLRT_LAUNCH_KERNEL`

Both directions bite. Launching an AIV kernel with `ACLRT_LAUNCH_KERNEL` gave a
deterministic aicore exception (`0x7150026`, unaligned UUB). Putting a manual
`_do()` wrapper in the same file as a Matmul kernel breaks the build on
`MatmulChipCap::GetFeatures()` (1.6). So the two launch paths are not
interchangeable and the choice is per kernel, not per project.

### 1.9 `0 × NaN = NaN`, and a fresh `aclrtMalloc` is full of NaN

Found while reviewing `cube2` before shipping it, not from a failure — which is
the only reason it is in this list rather than in the bug history.

The `cube2` matmuls read 16 K-rows, but the useful operand is 8 rows, so they
inevitably take in rows that `repack` never wrote: at `terms = 1` the `Q0`
matmul reaches into the unwritten `X_lo` block, and `Q1` reaches 8 rows past
the end of its slot. Mathematically the contribution is exactly zero, because
columns 8..15 of the weight matrix `A2` are zero. But a fresh `aclrtMalloc`
holds garbage from previous tasks, and for `half` a NaN or Inf is just an
exponent of all ones — common enough in random bytes that a buffer is
essentially guaranteed to contain one. `0 × NaN` is `NaN`, and it propagates
into the accumulator and out into the result.

Without the zero-fill this would have produced NaN in the output *every other
run*, depending on what the previous process left in DDR — the worst kind of
bug to debug. The fix is one `aclrtMemcpy` of zeros before the run, outside the
timed region. The invariant worth keeping: **if a kernel can read a GM region
it did not write, that region must be initialized, even when the arithmetic
says the value is irrelevant.**

### 1.10 The Cube fractal is 16 wide in K for fp16, so `K = 8` costs the same as `K = 16`

Relevant to `cube2`, where the weight matrix is mathematically 16×8. It is
stored as 16×16 with columns 8..15 zeroed instead, for three reasons that all
turned out to matter: a row becomes 32 bytes and so clears the `DataCopy`
granularity of 1.1; the tiling stays the already-proven `(16, 16, B)`; and if
the library rounds `baseK` up to 16 internally, the zero columns make the
rounding harmless rather than wrong. The doubled multiply count is not an
artifact of that choice — it is what `K = 8` physically costs on a 16-wide
fractal either way.

---

## 2. Measured comparisons

### 2.1 Methodology: only paired comparisons are admissible

This is the most important methodological point in the project, and it was
learned by getting it wrong.

The first version of `scripts/analyze_matrix.py` printed, for each axis, the
best time achieved at each value of that axis anywhere in the table. That table
said "DIT is 1.30× faster than Stockham". The comparison was between
`r8/dit_fs_xs` on 40 cores and `r4/st_fs_xs` on 1 core — it was mostly
measuring the core count, with the radix mixed in for good measure.

The analyser now groups cells that differ in **exactly one** axis, with every
other axis held fixed, and reports per-group ratios. The honest DIT-vs-Stockham
number is 1.01–1.29×. The overview table is still printed, but under a header
saying it is not fit for drawing conclusions.

Two corollaries that are easy to get wrong in the grouping key:

* `cores` must be **in** the key for most axes. Once every radix-2 and radix-4
  variant had a multicore twin, leaving it out would have put 1-core and
  40-core cells in the same group.
* `cores` must be **out** of the key for the compute-unit axis. The chip has 40
  AIV and 20 AIC cores, so a Vector variant can never declare 20 and a Cube
  variant can never declare 40. There, core count is not an independent choice —
  it *is* the compute unit, restated. Keeping it in reported "no paired
  comparisons" on every table ever collected, which read as missing data when
  the data was in fact present and the key was wrong.

The same reasoning added the `mm` column: `cube1` and `cube2` differ only in
how many matmuls one precision term costs, so without a column for it they
differ only by variant name and never pair.

### 2.2 Headline numbers

Full matrix, 186 cells, PASS=162 FAIL=24 (all 24 are the deliberately
low-precision Cube variants, see 2.8), best cell per size:

| N | best variant | total | vs NumPy |
|---|---|---|---|
| 64 | `r4/dit_fsm_xs` tile=8 | 43.10 µs | slower |
| 4096 | `r4/dit_fsm_xs` tile=8 | 142.46 µs | 1.39× slower |
| 262144 | `r4/dit_fsm_xs` tile=8 | 6884.78 µs | **3.37× faster** |

`r4/dit_fsm_xs` is the best variant at every size, which is the one result in
this table that did not have to come out that way — the axes of section 2 are
independent, and nothing guaranteed their individual winners would compose into
a single cell. They did.

The NPU part at N=262144 is 560 µs against NumPy's 23074 — a factor of 41. The
reason the end-to-end number is 3.37× and not 41× is section 3.

### 2.3 Layout: separate Re/Im arrays beat interleaved, by a lot

48 paired comparisons, `xs` wins **48 of 48**, by 1.03–4.61×.

This is the largest single-decision effect in the project and the least
ambiguous. Interleaved complex data makes every vector operation a strided
access; separate arrays make it contiguous. Only radix-2 has both layouts
implemented, which is why the axis has 48 comparisons and not more — and after
this result, implementing `xi` for radix-4 and radix-8 would have been work
spent confirming a settled question.

### 2.4 Algorithm: DIT beats Stockham, narrowly but consistently

72 paired comparisons, DIT wins **72 of 72**, by 1.01–5.08×.

The bulk of the range sits at the low end: 1.01–1.07× on the single-core
radix-2 cells. Stockham's ping-pong buffers cost an extra GM round-trip per
stage and buy back nothing here, because the digit-reverse permutation is done
once on the host and is not in the timed region. The large ratios at the top of
the range come from cells where Stockham's extra traffic interacts with a bad
`tile` choice, not from the algorithm itself.

### 2.5 Granularity: fused beats per-stage

72 paired comparisons, fused wins 61, per-stage wins 11, by 1.00–4.58×.

Fused keeps the twiddle factors for all stages in one buffer with a META
descriptor, so a single launch can walk the stages; per-stage re-reads a
per-stage twiddle buffer. The 11 cells where per-stage wins are all at large
`tile`, where few stages run on the NPU at all and the difference is noise.

### 2.6 Radix: 4 beats 2 and 8; radix-16 is not worth implementing

48 paired comparisons on the radix axis, radix-4 wins 46, radix-2 wins 2, by
1.06–5.94×. Against radix-8 the picture is close: at N=4096 `r4/dit_fsm_xs`
(146.20 µs) beats `r8/dit_fs_xs` (193.88), and at N=262144 6848.91 beats
7385.78.

Radix-4 is the sweet spot because the butterfly still fits comfortably in
registers while halving the number of stages relative to radix-2. Radix-8
halves them again but the butterfly needs more live values than the vector unit
holds well, and each stage does more work per element. Radix-16 extends that
trend in the direction that is already losing, and it would restrict N to
powers of 16 — which would leave only 65536 in the usable range and no size
shared with radix-2, radix-4 and radix-8 at once. The axis is closed.

### 2.7 Multicore: the barrier comes free from stream ordering

72 paired comparisons on the cores axis, 40 cores wins **72 of 72**, by
1.06–13.38×. The largest single gain measured: `r4/dit_fs_xs` at N=262144 went
6717.59 → 560.57 µs on the NPU side, a factor of 12.

The structural decision that made it work: **one launch per stage**, with the
inter-stage barrier coming from stream ordering rather than from any
synchronization inside a kernel. Stage *s+1* reads what stage *s* wrote across
the whole array, so it needs a barrier visible to every core; launches on one
stream give that for free, while a hand-written cross-core barrier inside a
kernel is a proven source of hangs. Tiles of a stage are handed out by
`GetBlockIdx()` with stride `blockDim`.

Single-core variants were kept alongside the multicore ones rather than
replaced, for two reasons: the old numbers stay comparable, and the table gains
an honest paired axis for the effect of multicore itself.

**Where multicore loses, and it does.** At N=64 `r2/dit_fs_xs` on one core
(47.95 µs) beats its own multicore twin `r2/dit_fsm_xs` (64.99 µs). This is the
only region in the whole matrix where the single-core twin wins, and it is the
expected one: multicore mode splits the stages into separate launches, and at
N=64 there are six stages and almost no work in each, so the launch overhead is
the entire measurement. The cross-over is somewhere between N=64 and N=4096,
where the same pair reads 731.02 against 266.69 in multicore's favour.

Worth stating because it is the sort of thing that gets rounded off to
"multicore is faster": it is faster everywhere that matters, and it is a
pessimisation at the bottom of the size range.

One invariant the kernels trust the host on, documented here because nothing
enforces it: `blk_num` must equal the launch's `blockDim`. The kernel strides
`tid += blkNum` while the launch fans out over `blockDim` cores, so if the two
disagree, tiles are either skipped (`blk_num > blockDim`) or computed twice by
different cores (`blk_num < blockDim`) — and the second case is a write-write
race. At every call site the same variable is passed to both, so it holds, but
it holds by construction and not by check.

### 2.8 Precision recovery: fp16 matmul reaches fp32 accuracy at 3 terms

Because fp32 operands do not work at all (1.3), accuracy has to be recovered
arithmetically. The scheme is the standard Ozaki / split-matmul one: a value not
representable in `half` is split into a sum of two `half`, and the product is
expanded term by term, with accumulation in fp32 throughout.

| terms | computed | error vs fp32 | verification |
|---|---|---|---|
| 1 | `A_hi·X_hi` | ~5e-4 relative, `max_abs` ≈ 0.5 at N=4096 | FAIL |
| 2 | `+ A_hi·X_lo` | removes the dominant term (rounding of the data) | FAIL |
| 3 | `+ A_lo·X_hi` | drops only `A_lo·X_lo` (~1e-7) | **PASS** |

12 paired comparisons on the terms axis: 3 terms is the only choice that passes,
at a cost of 1.02–3.89× in time. At `cube1` with B=128 the cost is +2% to +26%,
which is the configuration one would actually ship.

The FAIL cells in the matrix are all `terms = 1` and `terms = 2`. They are kept
deliberately: the point of the axis is the price of accuracy, and dropping the
cells that fail would discard the measurement.

**The number of terms needed depends on N, which the table makes visible only
because N=64 is in the grid.** At N=64 two terms are already enough —
`cube1x2_fs_xs` and `cube2x2_fs_xs` both PASS, with `max_abs` 2.702e-02 for the
one-term variants against 0.5 at N=4096 and 7.3 at N=262144. The reason is
simply that error accumulates over stages: N=64 has two radix-8 stages,
N=262144 has six. So "three terms" is not a property of the scheme, it is the
requirement at the sizes that matter here; a shorter transform can buy back the
~10% that the third term costs.

### 2.9 Cube vs Vector, and one matmul vs two

The Cube path reaches the same accuracy as the Vector path but does not beat it:
at N=262144 the best Cube cell is `r8/cube1x3_fs_xs` at 11436.64 µs against
`r4/dit_fsm_xs` at 6884.78 and `r8/dit_fs_xs` at 7452.72.

The measured per-group cost model from the batch sweep is
`t(group) ≈ 40 µs + 2.7 µs × (tiles in the group)` — 44.8/49.1/61.2/83.8 µs per
group at B=256/128/64/32. Two things follow: the tiles inside a group do **not**
run in parallel across cores (otherwise the linear term would be flat), and at
large B everything is eaten by the constant 40 µs per group rather than by the
arithmetic. This rejected the core-starvation hypothesis the sweep was designed
to test — time *fell* as the batch grew, the opposite of the prediction.

On one matmul versus two (`cube1` vs `cube2`, the `mm` axis): **`cube1` wins by
1.1–1.7×**, and the margin grows slowly with N.

| N | `cube1x3` | `cube2x3` | ratio |
|---|---|---|---|
| 64 | 77.28 µs | 93.59 µs | 1.21× |
| 4096 | 262.84 µs | 389.05 µs | 1.48× |
| 262144 | 16285.85 µs | 27901.85 µs | 1.71× |

(all at B=64, tile=8 — see below for why the batch width has to be pinned there)

That margin is what the mechanism predicts: `cube2` does 2× the padded
multiplies, because a `K = 8` operand occupies a 16-wide fractal either way
(1.10), and it adds 16 Add/Sub per tile in unpack for the combine step that
`cube1` gets for free from the signs built into its matrix. So the answer to the
thesis question is **one matmul**, for a reason rather than by accident.

**A correction, and a methodological trap worth naming.** An earlier reading of
this axis put `cube2` at 7–24× slower, from comparing `cube1` and `cube2` at the
same `CUBE_BATCH`. That comparison is invalid, and it is exactly the error
section 2.1 is about — committed on an axis built to avoid it. At a given batch
width `cube2`'s scratch footprint is **twice** `cube1`'s, because it keeps six
Y blocks instead of three. So "same B" is not "everything else held fixed": it
silently varies the footprint, and above a threshold the footprint costs more
than everything else on the axis combined (2.11). Pinning B=64 keeps both
schemes inside the healthy range and leaves the schemes as the only difference.

The general lesson: a paired comparison has to hold fixed the quantity that
*matters*, not the knob that is convenient to name. `cube_batch` is a knob;
scratch bytes are the quantity.

### 2.10 The optimal `tile` threshold inverts with core count

54 paired comparisons: `tile=8` wins 30, `tile=512` wins 15, `tile=64` wins 9,
by 1.07–5.72×. The aggregate is misleading; the structure is the finding:

| | 1 core | 40 cores |
|---|---|---|
| `r2/dit_fused_xi`, N=262144 | 512 best, by 1.50× | 8 best, by 1.99× |
| `r2/dit_fused_xi`, N=4096 | 512 best, by 2.14× | 8 best, by 1.56× |

On one core the NPU is slow enough that leaving small stages on the CPU pays;
on forty it is fast enough that everything which *can* go to the NPU should.
The threshold is therefore not a tunable to be fixed once — it is a function of
the configuration, and reporting a single "best tile" for the project would be
wrong.

### 2.11 The scratch footprint cliff — what the "batch anomaly" actually was

Two things in this project looked like separate anomalies: a large Cube batch
collapsing at large N, and `cube2` collapsing at large N. **They are one
phenomenon, and the parameter is total scratch bytes.**

Measured at N=262144, tile=8, sorted by footprint:

| footprint | configuration | total | |
|---|---|---|---|
| 320 KB | `cube1` B=64 | 16286 µs | healthy |
| 561 KB | `cube2` B=64 | 27902 µs | healthy |
| 640 KB | `cube1` B=128 | 11437 µs | healthy — **fastest Cube cell** |
| 1122 KB | `cube2` B=128 | 272238 µs | collapse |
| 1280 KB | `cube1` B=256 | 486086 µs | collapse |
| 2244 KB | `cube2` B=256 | timed out at 600 s | collapse |

The three smallest footprints are healthy and the three largest collapse, with
the boundary between 640 KB and 1122 KB. The scheme does not predict it: at
561 KB `cube2` is fine, at 1280 KB `cube1` is not. The number of Y blocks does
not predict it either — `cube2` at B=64 has the same six blocks as `cube2` at
B=128 and behaves completely differently. Only the byte count predicts it, and
it does so across both schemes and all three batch widths without an exception.

Footprint here is `20 slots × (Y blocks × 16 × B × 4 bytes) + X`, which is
`3840·B` bytes of Y for `cube1` and `7680·B` for `cube2`; the host prints the
figure as `scratch=... KB` on the `[CUBE-PLAN]` line and warns when a run is
about to cross into the collapse region.

**Why it only shows at large N.** The penalty is paid once per *group*, not once
per run, so it needs many groups to become visible. At N=4096 there are 3 groups
and a larger batch is still mildly *better* (B=256 at 216.99 µs beats B=128 at
235.73); at N=64 it is flat. At N=262144 there are 64 groups and the same
configuration is 40× worse. The sign change on *size* that made this look
anomalous was never about the batch — it was the group count multiplying a
fixed per-group penalty.

This also settles what the right configuration is, which was previously guesswork:
**`cube1` at B=128** — the largest batch that stays inside the healthy range.

---

## 3. The wall: the floor of 8, and a 92% CPU residue

At the best cell in the whole matrix (`r4/dit_fsm_xs`, N=262144, tile=8), the
split is **560.57 µs on the NPU and 6288.34 µs on the CPU**. The CPU is 91.8%
of the total. Every optimisation in section 2 was applied to the 8% part.

This is not a tuning oversight, and the reason is worth being precise about.
Lowering the `tile` threshold does not help: at `tile=1` nothing changes,
because all three stage planners clamp with `if (min_npu_* < 8) min_npu_* = 8;`
and that clamp is not a heuristic — it is section 1.1. A stage whose rows are
shorter than 8 `float` cannot be moved to the NPU in this layout at all,
because the `DataCopy` that would feed it transfers nothing. The first stages of
any radix-8 FFT (`eighth = 1`) and the first stages of radix-2 and radix-4 are
all below the floor, and they are precisely the stages with the most butterflies.

The way past it is **k-block batching**: instead of giving the NPU one stage of
short rows, gather many independent k-blocks into one tile so that the copied
run is a multiple of 32 bytes even when a single block's row is not. The Cube
path already does exactly this (`gCount` k-blocks of `perBlock` butterflies
each, see `kernels/r8_cube1_repack.cpp`); the Vector stage planners do not.

Projection, by moving the currently-CPU stages onto the NPU at the measured
per-element NPU rate: ~670 µs total at N=262144, i.e. roughly **35× faster than
NumPy** instead of the current 3.37×. That projection is an extrapolation, not
a measurement, and it is the single largest remaining win in the project.

---

## 4. Open questions

### 4.1 What exactly the footprint cliff is, physically — RESOLVED in part

The *behaviour* is established and measured: section 2.11. A single parameter —
total scratch bytes — predicts the collapse across both schemes and all three
batch widths, the boundary sits between 640 KB and 1122 KB, and the penalty
scales with the group count. That is enough to choose a configuration and enough
to state as a result.

What is **not** established is which hardware structure the boundary belongs to.
The shape of the effect — a fixed penalty per group, appearing once the working
set exceeds a threshold — is what one expects when the scratch stops being
resident between consecutive groups and every group re-fetches it from DDR. A
~1 MB resident window is consistent with the bracket, but naming the specific
level would be a guess: the measurement brackets the threshold, it does not
identify the structure, and a DDR round trip alone is a 5–10× cost where the
observed per-group penalty is larger.

Two candidates remain for the amplification, and one measurement separates them:

* **Operand traffic.** If the extra time is in the `matmul` phase, it is the A/B
  CopyIn re-reading GM.
* **The vector side.** Every kernel in this project runs vector operations over
  UB lanes in `[alignedValid, computeAligned)` that were never loaded, holding
  whatever the previous tile left there. The results are never stored back, so
  correctness is unaffected — but arithmetic on stale floats can hit denormals,
  and denormal handling is a classic source of exactly this kind of nonlinear
  slowdown. That would explain an amplification that cache misses alone do not.

`RADIX8_PROFILE=1` splits the per-group time into repack / matmul / unpack and
distinguishes them directly. Run it on `r8/cube2x3_fs_xs` at N=262144 with
B=128 (collapsed) and B=64 (healthy) and compare the layout, not the totals —
the mode synchronizes after every launch, so its absolute numbers are inflated
by construction.

### 4.2 Whether k-block batching is worth it for the Vector path

Section 3 says it is the largest remaining win. It is also the largest remaining
piece of work: it changes all three stage planners and the tile enumeration in
every Vector kernel. The Cube path's implementation is the template.

### 4.3 A true `K = 8` tiling for `cube2`

Only interesting if `cube2` turns out competitive after 4.1 is understood.
Currently it is not, by a margin that a 2× reduction in padded MACs would not
close.

---

## 5. Reproducing any of this

```bash
bash run_matrix.sh                       # the full grid: 3 sizes x 3 tiles x 3 batches
python3 scripts/analyze_matrix.py        # per-size tables + every paired axis
python3 scripts/analyze_matrix.py --axis cores --all-pairs
bash scripts/check_determinism.sh        # bit-exactness, the test the matrix cannot do
```

What each check actually verifies, since "it passes" means different things:

* **Golden comparison** (in the binary, `host/verify.h`) — replicates
  `numpy.isclose` with `atol = rtol = 1e-3` on complex magnitudes; PASS if at
  most 0.1% of points miss. Catches wrong arithmetic. Blind to races smaller
  than the tolerance (1.2).
* **Determinism check** (`scripts/check_determinism.sh`) — two runs, byte for
  byte. Catches races, and `scripts/compare_runs.py` distinguishes the
  signatures: a few points at 1–2 ulp is a synchronization race, many points or
  a large delta is two cores writing the same GM cell, a NaN in one run only is
  uninitialized GM.
* **Paired axes** (`scripts/analyze_matrix.py`) — the only output of this
  project from which design conclusions should be drawn (2.1).
* **`output/matrix.csv`** — one row per cell with every axis as its own column,
  so any comparison in section 2 can be re-derived from the raw table.

`N` must be a power of the variant's radix. Only 64, 4096 and 262144 are powers
of 2, 4 **and** 8 at once, so those are the only sizes on which radices can be
compared with one another; the binary rejects an incompatible pair with exit
code 1 and `run_matrix.sh` does not queue one.
