# Mixed radix: the permutation, and why the first verdict on it was wrong

## The claim that was wrong

An early model of the stage plan reported mixed radix as broken. `[3,5]`,
`[5,3]`, `[3,5,7]`, `[7,5,3]` and `[3,3,5]` all came out with errors of order
the signal itself, while `[3,3,3]`, `[5,5]`, `[7,7]` and `[3,3,3,3]` were
exact. The conclusion drawn was that the digit-reverse **formula** does not
generalise to non-uniform lists, and `build_plan()` was made to refuse them.

That conclusion was wrong, and the guard was protecting the wrong thing.

## What was actually wrong

The formula was right. The **direction it was applied in** was not.

The permutation the plan needs is a **gather**:

```
z[i] = x[rev(i)]
```

and what was being done was a **scatter**:

```
z[rev(i)] = x[i]
```

For a **uniform** radix list, digit reversal is an **involution** — reversing
the digit string twice returns it — so `rev⁻¹ = rev` and the gather and the
scatter are *the same map*. The bug cannot be observed. For a non-uniform list
the digit ranges differ, reversal is no longer an involution, and the two maps
diverge.

## The evidence

Involution rate is the fraction of indices with `rev(rev(i)) == i`:

| radices | N | involution rate | scatter | gather |
|---|---|---|---|---|
| [3,3,3] | 27 | 1.00 | ok | ok |
| [5,5] | 25 | 1.00 | ok | ok |
| [7,7] | 49 | 1.00 | ok | ok |
| [3,3,3,3] | 81 | 1.00 | ok | ok |
| **[3,5,3]** | 45 | **1.00** | **ok** | ok |
| [3,5] | 15 | 0.20 | **WRONG** | ok |
| [5,3] | 15 | 0.20 | **WRONG** | ok |
| [3,7] | 21 | 0.24 | **WRONG** | ok |
| [5,7] | 35 | 0.09 | **WRONG** | ok |
| [3,5,7] | 105 | 0.09 | **WRONG** | ok |
| [3,3,5] | 45 | 0.07 | **WRONG** | ok |
| [2,3] | 6 | 0.33 | **WRONG** | ok |
| [4,3] | 12 | 0.17 | **WRONG** | ok |
| [2,2,3] | 12 | 0.17 | **WRONG** | ok |
| [2,3,5] | 30 | 0.07 | **WRONG** | ok |
| [4,4,3] | 48 | 0.08 | **WRONG** | ok |
| [2,3,3,5] | 90 | 0.02 | **WRONG** | ok |
| [3,3,3,5,7] | 945 | 0.01 | **WRONG** | ok |
| [2,2,2,3,3] | 72 | 0.06 | **WRONG** | ok |

Scatter: wrong on 20 of 25 lists tested. Gather: wrong on none.

The involution column is the whole story. **Every** list on which the scatter
happened to work has involution rate exactly 1.00 — and that includes the one
non-uniform list that passed, `[3,5,3]`, which passes because it is a
**palindrome**.

## The method lesson

This is the part worth keeping.

A test suite of uniform lists plus one accidental palindrome cannot see this
bug at all, and that is exactly what the suite was. Worse, the failure was
*misdiagnosed in a direction that closed the question*: a guard was added, a
comment was written explaining why mixed radix was hard, and the feature was
deferred to a later phase. The wrong diagnosis was more expensive than the bug.

What finally found it was not more algebra. It was a **derivation**: apply the
stage plan to each basis vector, match the resulting columns against the DFT
matrix, and read the permutation off directly — then compare that against both
readings of the formula. That procedure cannot be fooled by a palindrome,
because it never assumes a form for the answer.

## What this unlocks

The plan is now verified for every order and mixture of radices in
**{2, 3, 4, 5, 7}**, so a single plan covers

```
N = 2^a · 3^b · 4^c · 5^d · 7^e
```

That is what makes the final combined implementation possible at all: a stage
plan can pick, per stage, whichever radix and butterfly measured best, instead
of being locked to one radix for the whole transform. Together with Bluestein
for the leftovers, it covers arbitrary N.

Note the distinction the code keeps: having a radix in a *plan* is not the same
as having a *kernel* for it. `StageInfo::radix` is what a launch dispatches on,
and a stage whose radix has no registered kernel has to run on the host.
