# Odd-radix FFT on Ascend 910B4 — design

Target: radix 3, 5 and 7 FFT kernels with several butterfly algorithms each, and
an honest comparison between them. Same chip and toolkit as the power-of-two
project (Ascend 910B4, CANN 8.0.RC3, 40 AIV / 20 AIC cores, 191 KB UB per core).

This document exists because odd radices are **not** the power-of-two project
with a different constant. Two properties of the problem differ in ways that
change the architecture, and both were established before any kernel was
written. They are in section 1. What follows from them is in sections 2–5.

---

## 1. The two facts that shape everything

### 1.1 For radix 3, 5 and 7 the butterfly's data movement is not expressible in `DataCopyParams` — at any stage

All 18 kernels of the power-of-two project move butterfly operands with

```cpp
DataCopy(dst, src, DataCopyParams{blockCount, blockLen, srcStride, dstStride});
```

whose `blockLen` and strides are in units of **32 bytes**, i.e. 8 `float`. For a
radix-r stage with parameter `t` (= r^s), the r operands of one butterfly sit
`t` elements apart and consecutive k-blocks sit `r·t` apart. Both of those have
to be divisible by 8 for the stride to be representable at all.

Counted over every stage of every usable transform size:

| radix | stages with representable strides | without |
|---|---|---|
| 2 | 17 | 2 |
| 4 | 8 | 1 |
| 8 | 6 | **0** |
| **3** | **0** | **12** |
| **5** | **0** | **8** |
| **7** | **0** | **6** |

Zero. Not one stage, for any of the three odd radices. And the power-of-two
project never noticed because its two unrepresentable cases (radix-2 at t=2 and
t=4) are exactly the stages that already fall below the 32-byte `DataCopy` floor
and run on the CPU anyway.

This is not a misalignment to be worked around with a tail fixup. The stride is
a 16-bit field counting 32-byte units; a stride of 9 elements has no
representation in it. And the failure mode of porting the existing kernels
naively is the bad kind: `(r*t - perBlock)/8` is integer division, so it
**truncates silently** and the gather reads the wrong elements with no fault.
`AlignProbe` mode `trunc` demonstrates exactly this, with the power-of-two case
as a control (its stride comes out as exactly 1.000 where radix-3/5/7 give
2.375, 9.125, 14.625, 41.875).

**Consequence.** The odd-radix kernels need a different primitive for GM↔UB
movement, and which primitives actually work on this device is a question about
the hardware and the toolkit, not about the algorithm. `AlignProbe` asks it
directly: five modes, one process each so a faulting mode does not take the
others with it.

| probe mode | what it decides |
|---|---|
| `aligned` | the control — if this fails the probe itself is wrong |
| `unaligned` | does plain `DataCopy` tolerate an unaligned GM base? |
| `pad` | does `DataCopyPad` exist here, and is it byte-granular as documented? |
| `scalar` | `GetValue`/`SetValue` — certainly correct, how slow? |
| `trunc` | proves the naive port is silently wrong |

**ANSWERED — measured on the device, 2026-10-14.** `AlignProbe` ran all six
modes; the correctness verdicts are unambiguous and the throughput mode gives
the cost.

| probe mode | result |
|---|---|
| `aligned` | OK — the control passes, so the probe is sound |
| `unaligned` | **OK at every odd offset tested** (1, 3, 5, 7, 9, 25, 27, 49, 125, 343) |
| `pad` | OK — `DataCopyPad` exists here and is byte-granular as documented |
| `scalar` | OK |
| `trunc` | 4 odd cases WRONG, power-of-two control OK — the truncation is real and silent |

So plain `DataCopy` tolerates an unaligned GM base on this device, and the
simplest possible design is available: **the butterfly operands are read as r
separate contiguous runs straight from their natural positions, with no
realignment, no in-UB regather and no scalar fallback.** `DataCopyPad` is not
needed; it is kept as the portable alternative.

One caveat to carry into the write-up: this is a property of **this** chip and
**this** CANN version, not a guarantee of the API. The documentation asks for
32-byte alignment and the device does not enforce it. The project therefore
rests on measured behaviour, and `DataCopyPad` — which also passed — is the
declared route if the code ever moves.

**What unalignment costs.** Throughput mode, 2000 copies per launch, best of 3:

| primitive | len | aligned | unaligned | penalty |
|---|---|---|---|---|
| `DataCopy` | 64 | 0.157 µs | 0.211–0.221 | **×1.34–1.40** |
| `DataCopy` | 512 | 0.206 µs | 0.228–0.244 | **×1.11–1.19** |
| `DataCopyPad` | 64 | 0.154 µs | 0.210–0.216 | ×1.37–1.41 |
| `DataCopyPad` | 512 | 0.205 µs | 0.230–0.245 | ×1.13–1.20 |
| scalar | 64 | 0.637 µs | 0.637 µs | ×1.000 |

Three things follow, and all three are design decisions rather than
observations:

1. The penalty is **fixed per transfer, not per byte** — it falls from 1.37× at
   64 floats to 1.15× at 512. So the tile should be shaped to make the
   contiguous run along `j` as long as possible; that is the one knob that
   dilutes the cost of unalignment, and it is free to take.
2. `DataCopyPad` is indistinguishable from plain `DataCopy` (0.154 vs 0.157
   aligned). Use the simple one.
3. Scalar addressing is **exactly** alignment-independent (×1.000 to four
   decimals, as it must be — there are no rows, only elements) and only 4.05×
   slower per element than aligned DMA. That is much better than expected and
   it has a consequence beyond this project; see §1.3.

A note on the correctness modes' timings, which are in the logs and should not
be read: they copy 64 floats per launch at a launch cost of 20–55 µs, so they
measure the launch and nothing else. That is why the throughput mode exists.
And the first version of `trunc` had a VECIN/VECOUT race of its own — it read
into a `TQue<VECIN>` tensor and wrote to GM straight out of it, leaving the
MTE2→MTE3 dependency uninstalled. It was caught by the power-of-two control
failing, which it had no business doing: a control that fails is a broken probe,
not a finding.

### 1.3 A side result: scalar addressing may be the way past the power-of-two project's wall

Not part of this project's plan, but it came out of the probe and it bears on
the largest open item of the previous one, so it is recorded here.

The power-of-two project ends with the NPU doing 561 µs and the CPU 6288 µs at
N=262144 — the CPU is **92% of the total**, and it is there because stages whose
rows are shorter than 8 `float` cannot be fed by `DataCopy` at all (the 32-byte
granularity of §1.1). The planned fix was k-block batching: gather many
independent k-blocks so the copied run clears 32 bytes even when one block's row
does not. That is a rewrite of all three stage planners.

But a sub-floor stage does not need `DataCopy`. Under scalar addressing there
are no rows, only elements, and the 8-element floor does not exist — and the
probe says scalar costs 4.05× per element, not the order of magnitude one would
assume.

The arithmetic, stated as a projection and not a result: 0.637 µs per 64
elements is ~10 ns/element, so ~2.6 ms for a stage at N=262144 **on one core** —
the probe ran at blockDim=1. Spread over 40 AIV cores, if it scales, ~65 µs
against the CPU's ~3144 µs per sub-floor stage. Roughly 48×.

What to distrust in that: the probe timed a GM↔UB copy, not a butterfly, and GM
bandwidth may saturate well before 40 cores. But it is checkable with one
variant in the power-of-two project — the sub-floor stages on the NPU with
scalar addressing instead of on the CPU — and if it holds it is far cheaper than
k-block batching and attacks the same 92%.

### 1.2 Radix 3, 5 and 7 share no common N, so "same N, different radix" is impossible

3^a = 5^b = 7^c only at N = 1. The power-of-two project could compare radices at
N = 64, 4096, 262144 because those are powers of 2, 4 **and** 8 at once; here
there is no such point, and pretending otherwise would reproduce exactly the
error that made the first version of that project's analyser useless.

So cross-radix comparison uses two devices, both of which must be in the
analyser from the start:

1. **Nearest-size triples.** The usable sizes cluster well:

   | scale | radix 3 | radix 5 | radix 7 | spread |
   |---|---|---|---|---|
   | small | 243 (3⁵) | 125 (5³) | 343 (7³) | 2.7× |
   | medium | 2187 (3⁷) | 3125 (5⁵) | 2401 (7⁴) | 1.43× |
   | large | 19683 (3⁹) | 15625 (5⁶) | 16807 (7⁵) | 1.26× |
   | largest | 177147 (3¹¹) | 78125 (5⁷) | 117649 (7⁶) | 2.27× |

   The medium and large triples are within 1.5×, which is close enough to compare
   with a normalised metric and far too far apart to compare raw times.

2. **A normalised metric.** Time divided by `N·log₂N`, reported as ns per
   butterfly-equivalent. This is the only figure in which a 15625-point radix-5
   transform and a 16807-point radix-7 one are commensurable, and the analyser
   makes it the primary cross-radix column while keeping raw `total_us` as the
   within-radix one.

Every comparison in the project is therefore labelled by what it holds fixed:
**within-radix** comparisons (the butterfly-algorithm axis, the layout axis, the
core-count axis) are paired and exact, exactly as in the power-of-two project.
**Cross-radix** comparisons are normalised and approximate, and will say so.

---

## 2. The butterfly algorithms — measured before implementing

All four are implemented and verified against `numpy.fft` in
`odd_butterflies.py`, which also counts the real operations as the kernel will
issue them (separate Re/Im, one count per instruction). Running that file is the
self-test.

| r | algorithm | mults | adds | **total** | per point |
|---|---|---|---|---|---|
| 3 | direct | 16 | 20 | 36 | 12.0 |
| 3 | sym | 4 | 14 | 18 | 6.0 |
| 3 | **winograd** | 4 | 12 | **16** | **5.3** |
| 3 | rader | 16 | 24 | 40 | 13.3 |
| 5 | direct | 64 | 72 | 136 | 27.2 |
| 5 | **sym** | 16 | 36 | **52** | **10.4** |
| 5 | winograd | 10 | 44 | 54 | 10.8 |
| 5 | rader | 64 | 80 | 144 | 28.8 |
| 7 | direct | 144 | 156 | 300 | 42.9 |
| 7 | **sym** | 36 | 66 | **102** | **14.6** |
| 7 | winograd | 32 | 116 | 148 | 21.1 |
| 7 | rader | 144 | 168 | 312 | 44.6 |

Two of these were derived rather than transcribed, because a wrong constant here
would be silent:

* The **length-3 cyclic convolution** reconstruction, solved for over random
  data and coming out exactly rational — `3y₀ = m₀+3m₁−m₃`,
  `3y₁ = m₀−3m₁−3m₂+2m₃`, `3y₂ = m₀+3m₂−m₃` — with the division by 3 folding
  into the precomputed constants, so it costs 4 multiplies and no scaling.
* The **radix-7 Winograd** form, built as Rader (length-7 DFT → length-6 cyclic
  convolution) composed with Agarwal-Cooley (6 = 2·3 coprime, so the length-6
  convolution nests a length-2 and a length-3 one: 2 × 4 = 8 complex multiplies).

**What the arithmetic already settles.** On a vector unit a multiply and an add
are both one instruction, so the figure to compare is their sum — not the
multiply count the classical literature minimises, which was written for
hardware where a multiply cost many times an add. By that measure:

* `rader` with a directly computed convolution is **dominated everywhere**: it
  has exactly `direct`'s multiply count plus extra adds. It earns one measured
  line in the results, not an implementation.
* `winograd` wins at r=3 (16 vs 18).
* At r=5 it is a wash (54 vs 52, 1.04× worse).
* At r=7 it is **45% worse** than `sym` (148 vs 102) — it trades 4 multiplies
  for 50 adds, a good trade on a DSP and a bad one here.

So the prediction going in is `winograd` at r=3 and `sym` at r=5 and r=7. What
arithmetic cannot predict is the instruction mix, register pressure and UB
traffic each form produces on the actual AIV — which is why all of them get
built and measured, and why a result that contradicts this table would be the
interesting one.

A fifth variant, **`cube`**, has no entry above because its cost is not an
operation count: it is the fixed 2r×2r block-real DFT matrix handed to the AIC
matmul unit, exactly as `cube1` did for radix-8. Its arithmetic is `direct`'s;
what changes is which unit pays for it. Note that the padding waste is now a
variable rather than a constant: the fp16 fractal is 16 wide, so radix-7's 14×14
wastes 23% of it while radix-3's 6×6 wastes 86%. The prediction is that the Cube
path improves monotonically with r, and radix-3 is where it should be hopeless.

---

## 3. Architecture

### 3.1 The stage plan is a list of radices from the start

A stage plan is `std::vector<StageInfo>` where each stage carries its own radix.
A pure power is the special case whose list is `[r, r, …, r]`. This costs nothing
now and it is what makes mixed radix a host-side change later rather than a
kernel rewrite: `N = 3^a·5^b·7^c` is just a plan with a mixed list, and the
question "which factorisation of the same N wins" becomes answerable with the
kernels already built.

That ordering matters for the comparison, not only for the effort. The
butterfly-algorithm axis can only be measured cleanly on pure powers, where a
transform uses one radix and one algorithm throughout. Build mixed radix first
and every cell confounds the two.

### 3.2 What carries over from the power-of-two project, and what does not

Carried over, with the mechanism that justifies it:

* **xs beats xi** (48 of 48 paired wins there). The mechanism is strided vs
  contiguous vector access, which is radix-independent. Only `xs` is built;
  the assumption is recorded here rather than silently inherited.
* **fused beats per-stage** (74 of 84). Twiddles for all stages in one buffer
  with a META descriptor.
* **multicore wins except at the bottom of the size range**, where the launch
  count is the whole measurement. Both twins are built — the sentinel-in-`stage`
  trick makes the single-core one free — so the cores axis stays measurable.
* **The `tile` threshold is a runtime knob and its optimum inverts with core
  count.** Not a constant to be fixed once.
* **The Cube scratch footprint cliff**: healthy at or below 480 KB of matmul
  output, collapsed at or above 960 KB, the penalty paid once per group. The
  `cube` variants here inherit the `scratch=… KB` reporting and the warning.

**Not** carried over — and this is the interesting one:

* **DIT beats Stockham** held by 1.01–1.07× on powers of two, where both have
  the same access pattern. For odd radices they do not: in-place DIT needs the
  r operands of a butterfly at mutual offsets of `t` (unrepresentable, §1.1),
  while Stockham writes to a separate output buffer and can **choose** its
  output layout, which gives it a route to keeping accesses contiguous that DIT
  does not have. So the expectation is that this result **reverses** for odd
  radices, and both are built and measured. If it does reverse, that is a
  finding about the interaction of the DMA engine's granularity with the
  algorithm, not about the algorithm.

### 3.3 The floor problem is worse here, and differently per radix

The 32-byte `DataCopy` floor puts any stage with fewer than 8 elements per row
on the CPU. For radix r the sub-floor stages are `t = 1` and `t = r` (and `t = 9`
for r=3), so 2–3 stages per transform — comparable to the power-of-two project
in count. But the **number of stages** differs sharply: log₃N against log₇N is a
factor of 1.77, so the same floor removes a very different fraction of the work
depending on the radix. Measuring that fraction per radix is one of the planned
results, and it interacts with the k-block batching that the power-of-two
project left as its largest open item.

---

## 4. The comparison matrix

Axes, with the new ones first:

| axis | values | kind |
|---|---|---|
| **radix** | 3, 5, 7 | cross-radix, normalised |
| **butterfly** | direct, sym, winograd, cube | within-radix, paired |
| algorithm | DIT, Stockham | within-radix, paired (expected to reverse) |
| cores | 1, 40 (20 for cube) | paired |
| tile threshold | 8, 64, 512 | paired |
| N | 4 sizes per radix (§1.2) | — |
| terms (cube only) | 1, 2, 3 | paired |
| cube batch | 64, 128 | paired, below the cliff |

Reusing the power-of-two harness: `matrix.csv` with one row per cell and every
axis as its own column, paired comparisons as the only admissible conclusions,
the overview table printed but labelled unusable, `check_determinism.sh` as the
test the matrix cannot do, and the `STOP`-file escape hatch.

New in the analyser: the normalised `ns/(N·log₂N)` column, and a cross-radix
block that is explicitly separated from the paired blocks and labelled as
approximate.

---

## 5. Order of work

1. ~~**Run `AlignProbe`.**~~ Done — §1.1 is answered: plain `DataCopy` from an
   unaligned base, operands as r separate contiguous runs, tile shaped for the
   longest possible run along `j` to dilute the fixed ×1.15–1.40 penalty.
2. **Host side, which is probe-independent** — generic radix-list stage planner,
   twiddle tables, digit-reverse permutation for an arbitrary radix list, the
   CPU reference for all four butterfly algorithms (also the sub-floor fallback
   and the correctness oracle), `gen_data` for arbitrary N, and the harness,
   registry, verifier and analyser adapted from the power-of-two project.
3. **One kernel, one radix, one algorithm** — radix-3 `sym`, DIT, fused,
   multicore — end to end, PASS against golden, bit-exact across runs. The point
   is to settle the data-movement design on hardware with the smallest possible
   thing.
4. **Fan out** over the butterfly algorithms, then the radices, then Stockham.
5. **Cube variants**, reusing the radix-8 pipeline structure with 2r×2r weights.
6. **Mixed radix**, which by §3.1 is a host-side plan change.
