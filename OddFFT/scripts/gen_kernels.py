#!/usr/bin/env python3
"""Generate one AscendC kernel per (radix, butterfly algorithm).

WHY A GENERATOR AND NOT TWELVE HAND-WRITTEN FILES
Two reasons, and the second matters more than the first.

1. Speed. Twelve kernels differing only in a straight-line arithmetic body is
   exactly the shape a generator is for.

2. METHOD. The butterfly algorithm is the main axis this project measures. For
   that comparison to be paired, every variant must differ in the butterfly
   and in NOTHING else -- same tiling, same DataCopy strategy, same queue
   layout, same twiddle handling. Hand-writing twelve files guarantees they
   drift. Generating them from one skeleton makes "identical except the
   butterfly" a property of the build rather than a claim in a comment.

HOW THE ARITHMETIC GETS HERE
It is not transcribed. docs/odb_butterflies.py (the reference that was checked
against numpy.fft and whose operation counts are the table in README.md) does
all its arithmetic through three helpers: mul, add, sub. This script imports
that module and replaces those three helpers with IR-node constructors, so
running the SAME verified functions builds an expression DAG instead of
computing floats. Then:

  * the DAG is evaluated numerically and compared against numpy.fft again, so
    a generated kernel whose math is wrong cannot be emitted;
  * the DAG is scheduled and register-allocated onto UB slices;
  * AscendC vector ops are emitted.

So the emitted kernel issues the operations of an algorithm that was verified
twice: once as scalar Python against numpy, and once as the DAG that becomes
the kernel.

ONE DELIBERATE DIFFERENCE FROM THE REFERENCE COUNTS
The reference initialises accumulators with the Python literal 0.0 and then
does add(0.0, ...). Those adds are counted in the table but are an artifact of
writing a loop, not arithmetic: no kernel would materialise a zero tensor to
add it. The generator folds exactly those, and reports both numbers, so the
difference is visible instead of silent.
"""
import argparse
import importlib.util
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)


# =========================================================================
# the IR
# =========================================================================
class Node:
    __slots__ = ("op", "a", "b", "k", "name", "idx", "uses", "slot", "from_add")

    def __init__(self, op, a=None, b=None, k=None, name=None):
        # A `neg` node emits one Muls by -1 either way, but it has to be
        # CHARGED to the column the reference charged it to: a negation born
        # from mul(-1, x) belongs to the multiply column, and one born from
        # sub(0.0, x) belongs to the add column. Mixing them up double-counted
        # across columns and was why the reconciliation failed.
        self.from_add = False
        self.op = op          # 'in' | 'add' | 'sub' | 'muls' | 'mul'
        self.a = a
        self.b = b
        self.k = k            # scalar constant for 'muls'
        self.name = name      # input name
        self.idx = None
        self.uses = 0
        self.slot = None


NODES = []
FOLDED_ADDS = [0]      # add/sub against a literal 0.0
FOLDED_CONST = [0]     # add/sub of two literals: constant arithmetic
FOLDED_MULS = [0]      # multiply by 0 or 1 (a -1 becomes a `neg` node)

# Constants that are exactly 0, +1 or -1 after the trigonometry. cos and sin of
# exact multiples of pi/2 land on these, and sin(pi) lands a rounding step away
# from zero, hence a tolerance rather than ==.
#
# WHY FOLD THESE AT ALL, when the rest of the generator deliberately emits the
# reference's operation count verbatim:
#   * Radix 2 and 4 need it to exist. Every twiddle in a radix-4 butterfly is
#     0, +-1 or +-i, so without folding the generated radix-4 would issue 64
#     multiplies by a literal 1.0 instead of the textbook zero.
#   * It is the FAIR choice for the butterfly comparison. Muls(d, a, 1.0f) is
#     not arithmetic, it is a copy nobody would write. Leaving it in would make
#     `direct` look worse than any real implementation of it, and the point of
#     the axis is to compare algorithms, not to compare one algorithm against a
#     strawman of another.
# Folds are counted and reported, like everything else the generator removes.
FOLD_TOL = 1e-12


def _mk(op, a=None, b=None, k=None, name=None):
    n = Node(op, a, b, k, name)
    n.idx = len(NODES)
    NODES.append(n)
    return n


def inp(name):
    return _mk("in", name=name)


def _is_num(x):
    return isinstance(x, (int, float))


def _muls(node, k):
    """Build a scale, folding the three constants that are not arithmetic."""
    if abs(k) < FOLD_TOL:
        FOLDED_MULS[0] += 1
        return 0.0
    if abs(k - 1.0) < FOLD_TOL:
        FOLDED_MULS[0] += 1
        return node
    if abs(k + 1.0) < FOLD_TOL:
        # NOT counted as folded: a negation is a real node. It usually
        # disappears anyway, absorbed into the sign of a neighbouring add by
        # the peephole below, and then shows up in the `dead` column -- which
        # is the honest place for it. Counting it here as well was a double
        # count and made the reconciliation fail.
        return _mk("neg", a=node)   # born from a multiply
    return _mk("muls", a=node, k=k)


def ir_mul(x, y):
    """Replaces odd_butterflies.mul. One side is always a constant here."""
    if _is_num(x) and _is_num(y):
        raise AssertionError("const*const in the reference -- unexpected")
    if _is_num(x):
        return _muls(y, float(x))
    if _is_num(y):
        return _muls(x, float(y))
    return _mk("mul", a=x, b=y)


def ir_add(x, y):
    if _is_num(x) and _is_num(y):
        FOLDED_CONST[0] += 1
        return x + y
    # folding an add against a literal zero: an accumulator-init artifact
    if _is_num(x):
        if x == 0.0:
            FOLDED_ADDS[0] += 1
            return y
        raise AssertionError("add against a non-zero constant %r" % x)
    if _is_num(y):
        if y == 0.0:
            FOLDED_ADDS[0] += 1
            return x
        raise AssertionError("add against a non-zero constant %r" % y)
    # A negation folded into the sign of the add costs nothing and keeps the
    # emitted sequence the one a person would write.
    if y.op == "neg":
        return _mk("sub", a=x, b=y.a)
    if x.op == "neg":
        return _mk("sub", a=y, b=x.a)
    return _mk("add", a=x, b=y)


def ir_sub(x, y):
    if _is_num(x) and _is_num(y):
        FOLDED_CONST[0] += 1
        return x - y
    if _is_num(y):
        if y == 0.0:
            FOLDED_ADDS[0] += 1
            return x
        raise AssertionError("sub of a non-zero constant %r" % y)
    if _is_num(x):
        if x == 0.0:
            if y.op == "neg":
                # -(-y) = y. The reference charged an add for this one; the
                # kernel issues nothing, so it is a folded add. (The inner neg
                # is left to the dead-code counter, which charges it to
                # whichever column it was born in.)
                FOLDED_ADDS[0] += 1
                return y.a
            n = _mk("neg", a=y)
            n.from_add = True         # born from a subtraction
            return n
        raise AssertionError("sub from a non-zero constant %r" % x)
    if y.op == "neg":
        return _mk("add", a=x, b=y.a)
    return _mk("sub", a=x, b=y)


def eval_dag(node, env):
    """Numeric value of a DAG node, for verification."""
    memo = {}

    def go(n):
        if _is_num(n):
            return float(n)
        if n.idx in memo:
            return memo[n.idx]
        if n.op == "in":
            v = env[n.name]
        elif n.op == "add":
            v = go(n.a) + go(n.b)
        elif n.op == "sub":
            v = go(n.a) - go(n.b)
        elif n.op == "neg":
            v = -go(n.a)
        elif n.op == "muls":
            v = go(n.a) * n.k
        elif n.op == "mul":
            v = go(n.a) * go(n.b)
        else:
            raise AssertionError(n.op)
        memo[n.idx] = v
        return v

    return go(node)


# =========================================================================
# scheduling + register allocation
# =========================================================================
def schedule(roots):
    """Topological order of the nodes actually needed by `roots`."""
    order, seen = [], set()

    def visit(n):
        if _is_num(n) or n.idx in seen:
            return
        seen.add(n.idx)
        for ch in (n.a, n.b):
            if ch is not None:
                visit(ch)
        order.append(n)

    for r in roots:
        visit(r)
    return order


def allocate(order, roots, out_slot_of):
    """Assign a UB slice to every non-input node.

    Roots go straight into their output slice where that is safe -- a root is
    safe if nothing else consumes it. Everything else gets a calc slot from a
    free list, released at its last use, so the calc buffer is sized by the
    PEAK live count rather than by the node count.
    """
    use_count = {}
    for n in order:
        for ch in (n.a, n.b):
            if ch is not None and not _is_num(ch):
                use_count[ch.idx] = use_count.get(ch.idx, 0) + 1

    # `next_slot` is what matters: the number of DISTINCT calc slots the
    # schedule touches, hence how many have to be declared. An earlier version
    # of this returned a "peak live" count instead and got it wrong (it did not
    # count a slot taken back off the free list as live again), so the emitted
    # body referenced c3 while only c0..c2 were declared. The compiler caught
    # it; the lesson is that the slot count is a property of the allocation, not
    # something to track separately alongside it.
    free = []
    next_slot = 0
    remaining = dict(use_count)

    for n in order:
        if n.op == "in":
            continue
        root_safe = (n in roots) and use_count.get(n.idx, 0) == 0
        if root_safe:
            n.slot = ("out", out_slot_of[id(n)])
        elif free:
            n.slot = ("calc", free.pop())
        else:
            n.slot = ("calc", next_slot)
            next_slot += 1
        # release the children whose last use this was
        for ch in (n.a, n.b):
            if ch is None or _is_num(ch) or ch.op == "in":
                continue
            remaining[ch.idx] -= 1
            if remaining[ch.idx] == 0 and ch.slot and ch.slot[0] == "calc":
                free.append(ch.slot[1])
    return next_slot


# =========================================================================
# emission
# =========================================================================
def slot_name(kind, i, radix):
    """Plain identifiers, not array indexing. LocalTensor is a handle and an
    array of them would rely on copy-assignment semantics that are not worth
    betting a kernel on."""
    if kind == "calc":
        return "c%d" % i
    p, half = divmod(i, 2)
    return "y%d%s" % (p, "r" if half == 0 else "i")


def ref(n, in_names, radix):
    if n.op == "in":
        return in_names[n.name]
    return slot_name(n.slot[0], n.slot[1], radix)


def emit_body(order, in_names, radix, indent="        "):
    lines, nmul, nadd = [], 0, 0
    for n in order:
        if n.op == "in":
            continue
        d = ref(n, in_names, radix)
        if n.op == "add":
            lines.append("%sAscendC::Add(%s, %s, %s, len);"
                         % (indent, d, ref(n.a, in_names, radix), ref(n.b, in_names, radix)))
            nadd += 1
        elif n.op == "sub":
            lines.append("%sAscendC::Sub(%s, %s, %s, len);"
                         % (indent, d, ref(n.a, in_names, radix), ref(n.b, in_names, radix)))
            nadd += 1
        elif n.op == "neg":
            # A surviving negation: one scale by -1. A real instruction either
            # way; charged to the column it came from.
            lines.append("%sAscendC::Muls(%s, %s, (float)(-1.0), len);"
                         % (indent, d, ref(n.a, in_names, radix)))
            if n.from_add:
                nadd += 1
            else:
                nmul += 1
        elif n.op == "muls":
            lines.append("%sAscendC::Muls(%s, %s, (float)(%.17g), len);"
                         % (indent, d, ref(n.a, in_names, radix), n.k))
            nmul += 1
        elif n.op == "mul":
            lines.append("%sAscendC::Mul(%s, %s, %s, len);"
                         % (indent, d, ref(n.a, in_names, radix), ref(n.b, in_names, radix)))
            nmul += 1
    return lines, nmul, nadd


def build(radix, algo, fn):
    """Build, verify and schedule one butterfly. Returns everything the
    template needs."""
    global NODES
    NODES = []
    FOLDED_ADDS[0] = 0
    FOLDED_CONST[0] = 0
    FOLDED_MULS[0] = 0

    xr = [inp("u%dr" % m) for m in range(radix)]
    xi = [inp("u%di" % m) for m in range(radix)]
    yr, yi = fn(xr, xi)

    # --- verify the DAG against numpy, independently of the Python run ---
    rng = np.random.default_rng(1234 + radix)
    for _ in range(4):
        z = rng.standard_normal(radix) + 1j * rng.standard_normal(radix)
        env = {}
        for m in range(radix):
            env["u%dr" % m] = float(z[m].real)
            env["u%di" % m] = float(z[m].imag)
        got = np.array([eval_dag(yr[p], env) + 1j * eval_dag(yi[p], env)
                        for p in range(radix)])
        err = float(np.abs(got - np.fft.fft(z)).max())
        if err > 1e-11:
            raise AssertionError("r=%d %s: DAG disagrees with numpy, err=%.3e"
                                 % (radix, algo, err))

    roots, out_slot_of = [], {}
    for p in range(radix):
        for half, lst in ((0, yr), (1, yi)):
            n = lst[p]
            if _is_num(n):
                raise AssertionError("output %d is a constant" % p)
            roots.append(n)
            out_slot_of[id(n)] = 2 * p + half

    order = schedule(roots)
    peak = allocate(order, roots, out_slot_of)

    # Nodes the reference computed but nothing consumes: DEAD CODE in the
    # reference. The reference's counter charges for them because it executed
    # them; the generated kernel does not issue them because the scheduler only
    # walks what the outputs actually depend on. Counted and reported rather
    # than quietly absorbed -- an unexplained gap between the table in the
    # README and the kernel is exactly the kind of thing that has to be either
    # explained or wrong.
    live = {n.idx for n in order}
    dead_mul = sum(1 for n in NODES
                   if n.idx not in live
                   and (n.op in ("mul", "muls")
                        or (n.op == "neg" and not n.from_add)))
    dead_add = sum(1 for n in NODES
                   if n.idx not in live
                   and (n.op in ("add", "sub")
                        or (n.op == "neg" and n.from_add)))

    in_names = {}
    for m in range(radix):
        in_names["u%dr" % m] = "u%dr" % m
        in_names["u%di" % m] = "u%di" % m

    body, nmul, nadd = emit_body(order, in_names, radix)

    # A root that something else consumes could not take an output slot, so its
    # value sits in a calc slot and has to be copied out.
    copies = []
    for p in range(radix):
        for half, lst in ((0, yr), (1, yi)):
            n = lst[p]
            if n.slot[0] != "out":
                copies.append("        AscendC::Adds(%s, %s, (float)0.0, len);"
                              % (slot_name("out", 2 * p + half, radix),
                                 ref(n, in_names, radix)))
    return {"body": body, "copies": copies, "peak": peak,
            "nmul": nmul, "nadd": nadd, "folded": FOLDED_ADDS[0],
            "foldedc": FOLDED_CONST[0], "foldedm": FOLDED_MULS[0],
            "dead_mul": dead_mul, "dead_add": dead_add}


# =========================================================================
# the kernel template
# =========================================================================
TEMPLATE = r'''// kernels/{fname}
// GENERATED by scripts/gen_kernels.py -- DO NOT EDIT BY HAND.
// Regenerate with:  python3 scripts/gen_kernels.py
//
// R{radix}-{ALGO}-DIT-PS-XS  (radix {radix}, `{algo}` butterfly, DIT,
//                   per-stage launch, multicore, X = Split re/im arrays)
//
// Everything outside Compute() is the SAME TEXT in all the generated kernels,
// by construction: the tiling, the DataCopyPad strategy, the queue layout and
// the twiddle handling come from one template. That is what makes the
// butterfly axis a paired comparison rather than a claim -- two of these
// kernels differ in the arithmetic and in nothing else.
//
// ARITHMETIC PROVENANCE
// The body of Compute() below was not written by hand and not transcribed. It
// was produced by running docs/odd_butterflies.py's `{algo}` function for
// r={radix} -- the same function that is checked against numpy.fft and whose
// operation count is the table in README.md -- with its mul/add/sub helpers
// replaced by expression-graph constructors, then scheduling and
// register-allocating that graph. The graph was evaluated against numpy.fft
// again before this file was emitted.
//
// OPERATIONS ISSUED BY THIS KERNEL, PER TILE OF `len` POINTS
//   butterfly:        {nmul} multiplies + {nadd} adds = {ntot}
//   twiddle premult:  {twmul} multiplies + {twadd} adds   (4 mul + 2 add per m>=1)
//   TOTAL:            {totalops} vector instructions
// The reference table counts {refmul} multiplies and {refadd} adds for the
// butterfly alone. The adds reconcile exactly as
//     {nadd} issued + {folded} zero-inits + {foldedc} constant folds
//         + {deada} dead = {refadd}
// where a zero-init is an add(0.0, x) the reference performs because it writes
// an accumulator loop -- no kernel would materialise a zero tensor to add it --
// and a constant fold is an add of two literals, which is compile-time
// arithmetic the reference happens to do at runtime. The multiply counts are
// reconcile as {nmul} issued + {foldedm} folded (x0 or x+-1, a copy or a sign,
// not arithmetic; a x-1 becomes a negation, which the peephole usually folds
// into a neighbouring add and which then appears as dead) + {deadm} dead
// = {refmul}. A "dead" operation is one the reference
// computes and never uses: the scheduler walks only what the outputs depend on,
// so the kernel never issues it. Nothing else differs.
//
// ===========================================================================
// WHY THE DATA MOVEMENT LOOKS LIKE THIS (identical in every generated kernel)
// ===========================================================================
// The operand distance t is a power of an odd number, so it is NEVER a
// multiple of 8. Two consequences:
//
//   1. THE WRITE CANNOT ROUND UP. Rounding a write length to the 32-byte
//      granule would overrun the end of the j-run: for the m=0 group into the
//      m=1 group of the same butterfly, and for the last group into the NEXT
//      k-block -- a different tile, and under multicore a different core.
//      That is a cross-core race on live data, and it produces a plausible
//      wrong answer rather than a crash. So the write is DataCopyPad,
//      byte-granular, exactly `valid` floats.
//
//   2. THE OPERAND BASES ARE UNALIGNED. i_m = i_0 + m*t with t odd, so at most
//      one of the {radix} can sit on a 32-byte boundary. This is the same fact
//      as "DataCopyParams strides are counted in 32-byte units, so an odd
//      stride is not representable". The way out, confirmed on this device by
//      AlignProbe, is to abandon strided copies and issue the {radix} operand
//      groups as {radix} separate contiguous runs from unaligned bases. The
//      measured penalty for an unaligned base is fixed per transfer, not per
//      byte, which is why the tile is shaped along j: a longer run amortises
//      the same fixed cost.
//
// The short-tail absorb (a trailing remainder below 8 joins the previous tile
// instead of becoming its own) keeps every transfer at or above the granule
// without ever rounding up. See kernels/oddfft_tiling.h.

#include "kernel_operator.h"
// Both of these sit in this same directory on purpose: the kernel build target
// gets no extra include directories, so a kernel can only include headers
// beside it. See kernels/oddfft_tiling.h.
#include "oddfft_tiling.h"
#include "kernel_limits.h"
#include <cstdint>

class {cls} {{
public:
    __aicore__ inline {cls}() {{}}

    __aicore__ inline void Init(GM_ADDR XR, GM_ADDR XI, GM_ADDR W,
                                uint32_t n, uint32_t t, uint32_t block,
                                uint32_t wOffset, uint32_t rowLen,
                                uint32_t tile, uint32_t blk_num)
    {{
        this->n       = n;
        this->t       = t;
        this->block   = block;
        this->wOffset = wOffset;
        this->rowLen  = rowLen;
        this->tile    = (tile > MAX_TILE) ? MAX_TILE : (tile < 8u ? 8u : tile);
        this->blkNum  = blk_num > 0 ? blk_num : 1;

        xrGm.SetGlobalBuffer((__gm__ float *)XR, this->n);
        xiGm.SetGlobalBuffer((__gm__ float *)XI, this->n);
        wGm.SetGlobalBuffer((__gm__ float *)W, this->wOffset + NW * this->rowLen);

        const uint32_t bufS = TILE_F * sizeof(float);
        // One input queue and one output queue, not one per tensor: the number
        // of hardware event IDs is limited (8 per pipeline pair) and more TQue
        // than that makes the events collide and deadlock.
        pipe.InitBuffer(inQ,  2, NIN_T  * bufS);
        pipe.InitBuffer(outQ, 2, NOUT_T * bufS);
        pipe.InitBuffer(calcBuf, NCALC_T * bufS);
    }}

    __aicore__ inline void Process()
    {{
        const uint32_t nShift = ODDFFT_SHIFT_COUNT(this->t, this->tile);
        const uint32_t nBlk   = this->n / this->block;
        const uint32_t nTiles = nBlk * nShift;
        const uint32_t bid    = (uint32_t)AscendC::GetBlockIdx();

        // The (k, shift) double loop is flattened into one linear tile index so
        // that BOTH loops are handed out across cores. Handing out only the
        // outer loop would leave most cores idle on the late stages, where the
        // number of k-blocks falls towards 1.
        //
        // Correctness: tile (k, shift) reads and writes exactly its own
        // {radix} runs of length `valid`; runs of different tiles are disjoint
        // because the shifts partition [0, t) and the k-blocks lie end to end.
        // There are no intra-stage dependencies, and the inter-stage barrier
        // comes free from the ordering of the per-stage launches on the stream.
        for (uint32_t tid = bid; tid < nTiles; tid += this->blkNum) {{
            const uint32_t k = (tid / nShift) * this->block;
            const uint32_t s = tid % nShift;
            const uint32_t shift = s * this->tile;
            const uint32_t valid = ODDFFT_TILE_VALID(this->t, this->tile, s, nShift);
            ProcessTile(k + shift, shift, valid);
        }}
    }}

private:
    __aicore__ inline void ProcessTile(uint32_t i0, uint32_t shift, uint32_t valid)
    {{
        this->curValid = valid;
        CopyIn(i0, shift);
        Compute();
        CopyOut(i0);
    }}

    __aicore__ inline void Load(const AscendC::LocalTensor<float> &dst,
                                const AscendC::GlobalTensor<float> &src, uint32_t idx)
    {{
        AscendC::DataCopyExtParams ext{{1, (uint32_t)(this->curValid * sizeof(float)), 0, 0, 0}};
        AscendC::DataCopyPadExtParams<float> pad{{false, 0, 0, 0.0f}};
        AscendC::DataCopyPad(dst, src[idx], ext, pad);
    }}

    __aicore__ inline void Store(const AscendC::GlobalTensor<float> &dst,
                                 const AscendC::LocalTensor<float> &src, uint32_t idx)
    {{
        AscendC::DataCopyExtParams ext{{1, (uint32_t)(this->curValid * sizeof(float)), 0, 0, 0}};
        AscendC::DataCopyPad(dst[idx], src, ext);
    }}

    __aicore__ inline void CopyIn(uint32_t i0, uint32_t shift)
    {{
        auto inT = inQ.AllocTensor<float>();
        // The {radix} operand groups as {radix} separate contiguous runs: no
        // strided copy is attempted, because an odd stride is not
        // representable in 32-byte units.
{copyin}
        inQ.EnQue(inT);
    }}

    __aicore__ inline void Compute()
    {{
        auto inT = inQ.DeQue<float>();
{decl_x}
{decl_w}
        auto calcT = calcBuf.Get<float>();
{decl_u}
{decl_c}

        // The vector length is EXACTLY the live element count. Not rounded up.
        //
        // An earlier version rounded to a multiple of 8 and zeroed the tail
        // with Duplicate(x[curValid], 0, tail). That faulted on hardware --
        // every launch returned 507035, an aicore exception -- because
        // curValid is not a multiple of 8 and a vector op needs its tensor to
        // START on a 32-byte boundary. The LENGTH may be anything; the OFFSET
        // may not. The power-of-two project never hit this because there every
        // such offset was a multiple of 8 by construction.
        //
        // With the length exact there is nothing to zero: lanes past curValid
        // are never read into a result, because CopyOut writes exactly
        // curValid floats. The hardware may process whole 256-byte blocks
        // internally and touch those lanes, but they stay inside this tensor
        // and never leave it.
        const uint32_t len = this->curValid;

        // --- twiddle premultiply: u_0 = x_0 ; u_m = x_m * W_block^(m*j) -----
        // m = 0 needs no multiply (W^0 = 1), which is the one saving every
        // butterfly algorithm shares and therefore does not distinguish them.
{twiddle}
        AscendC::PipeBarrier<PIPE_V>();

        auto outT = outQ.AllocTensor<float>();
{decl_y}

        // --- the butterfly: {algo}, r={radix} ------------------------------
        // GENERATED. See the provenance note in the file header.
{body}
{copies}
        AscendC::PipeBarrier<PIPE_V>();

        outQ.EnQue(outT);
        inQ.FreeTensor(inT);
    }}

    __aicore__ inline void CopyOut(uint32_t i0)
    {{
        auto outT = outQ.DeQue<float>();
        // Exactly `valid` floats per run. No rounding up anywhere.
{copyout}
        outQ.FreeTensor(outT);
    }}

private:
    AscendC::TPipe pipe;
    AscendC::GlobalTensor<float> xrGm, xiGm, wGm;

    static constexpr uint32_t R       = {radix};
    static constexpr uint32_t NW      = {nw};        // twiddle rows: 2*(r-1)
    static constexpr uint32_t NIN_T   = {nin};       // x (2r) + w (2r-2)
    static constexpr uint32_t NOUT_T  = {nout};      // y (2r)
    static constexpr uint32_t NCALC_T = {ncalc};     // u (2r-2) + temps ({peak}) + 2

    // MAX_TILE is the longest j-run a tile may cover; TILE_F adds the 8 floats
    // the short-tail absorb can push past it.
    // UB cost = ((NIN_T + NOUT_T) * 2 + NCALC_T) * TILE_F * 4 B
    //         = {slices} slices * {tile_f} float * 4 B = {ub_kb} KB of 191 KB.
    // MAX_TILE is chosen per kernel so this fits with headroom -- r=7 needs
    // far more slices than r=3 and cannot carry the same tile.
    // From kernels/kernel_limits.h, which the host reads too -- one number, so
    // the kernel and the host cannot disagree about what tile was used.
    static constexpr uint32_t MAX_TILE = ODDFFT_MAX_TILE;
    static constexpr uint32_t TILE_F   = MAX_TILE + 8;

    AscendC::TQue<AscendC::TPosition::VECIN,  2> inQ;
    AscendC::TQue<AscendC::TPosition::VECOUT, 2> outQ;
    AscendC::TBuf<AscendC::TPosition::VECCALC>   calcBuf;

    uint32_t n, t, block, wOffset, rowLen, tile;
    uint32_t blkNum;
    uint32_t curValid;
}};

extern "C" __global__ __aicore__ void {kname}(
    GM_ADDR XR, GM_ADDR XI, GM_ADDR W,
    uint32_t n, uint32_t t, uint32_t block, uint32_t wOffset, uint32_t rowLen,
    uint32_t tile, uint32_t blk_num)
{{
    {cls} op;
    op.Init(XR, XI, W, n, t, block, wOffset, rowLen, tile, blk_num);
    op.Process();
}}

#ifndef ASCENDC_CPU_DEBUG
void {kname}_do(uint32_t blockDim, void *stream,
                void *xr, void *xi, void *w,
                uint32_t n, uint32_t t, uint32_t block, uint32_t wOffset,
                uint32_t rowLen, uint32_t tile, uint32_t blk_num)
{{
    {kname}<<<blockDim, nullptr, stream>>>(xr, xi, w, n, t, block, wOffset,
                                           rowLen, tile, blk_num);
}}
#endif
'''


# =========================================================================
# the generated self-test
# =========================================================================
# Compiling a kernel proves the body PARSES. It does not prove the register
# allocation is right, and that is the dangerous class of bug here: a calc slot
# released one step too early is reused while still live, and the kernel then
# computes a plausible wrong answer. (The first version of the allocator had a
# bug of exactly this family -- it miscounted how many slots it needed. The
# compiler caught that one only because the symbol was missing; a slot reused
# too early has no missing symbol.)
#
# So the generator also emits this: the SAME body text, verbatim, run scalar
# against a reference DFT. The shim turns each tensor name into a float and
# each vector op into a scalar one, so what is tested is the exact instruction
# sequence with the exact allocation.
TEST_HEADER = r'''// tests/gen_butterfly_test.cpp
// GENERATED by scripts/gen_kernels.py -- DO NOT EDIT BY HAND.
//
// Runs the EMITTED body of every generated kernel, verbatim, as scalar code
// against a reference DFT. This is what checks the register allocation: a calc
// slot freed one step too early would be reused while still live and would
// produce a plausible wrong answer that compiles perfectly.
#include <cmath>
#include <cstdint>
#include <cstdio>

// The shim: a "tensor" is one float, a vector op is a scalar op.
namespace AscendC {
inline void Add(float &d, float a, float b, uint32_t) {{ d = a + b; }}
inline void Sub(float &d, float a, float b, uint32_t) {{ d = a - b; }}
inline void Mul(float &d, float a, float b, uint32_t) {{ d = a * b; }}
inline void Muls(float &d, float a, float k, uint32_t) {{ d = a * k; }}
inline void Adds(float &d, float a, float k, uint32_t) {{ d = a + k; }}
}}

static int failures = 0;

static void check(const char *name, uint32_t r,
                  const float *yr, const float *yi,
                  const float *ur, const float *ui)
{{
    double maxe = 0.0, scale = 0.0;
    for (uint32_t p = 0; p < r; ++p) {{
        double sr = 0.0, si = 0.0;
        for (uint32_t m = 0; m < r; ++m) {{
            const double a = -6.28318530717958647692 * (double)p * (double)m / (double)r;
            const double c = std::cos(a), s = std::sin(a);
            sr += (double)ur[m] * c - (double)ui[m] * s;
            si += (double)ur[m] * s + (double)ui[m] * c;
        }}
        maxe  = std::fmax(maxe, std::fabs((double)yr[p] - sr));
        maxe  = std::fmax(maxe, std::fabs((double)yi[p] - si));
        scale = std::fmax(scale, std::fabs(sr));
        scale = std::fmax(scale, std::fabs(si));
    }}
    const double rel = scale > 0.0 ? maxe / scale : maxe;
    const bool ok = rel < 1e-5;
    if (!ok) ++failures;
    printf("%-22s r=%u  rel=%.3e  %s\n", name, r, rel, ok ? "OK" : "<<<< FAIL");
}}
'''

TEST_FN = r'''
static void test_{radix}_{algo}()
{{
    const uint32_t len = 1;
    (void)len;
    // the butterfly inputs (already twiddled, as the kernel sees them)
{decl_in}
{init_in}
    // the calc slots and the outputs, exactly as the kernel declares them
{decl_c}
{decl_y}
{body}
{copies}
    const float ur[] = {{ {list_r} }};
    const float ui[] = {{ {list_i} }};
    const float yr[] = {{ {list_yr} }};
    const float yi[] = {{ {list_yi} }};
    check("{radix}/{algo}", {radix}, yr, yi, ur, ui);
}}
'''

TEST_MAIN = r'''
int main()
{{
    printf("--- generated butterfly bodies, run scalar vs a reference DFT ---\n");
{calls}
    printf("\n%s  (failures=%d)\n",
           failures ? "SOME GENERATED BODIES ARE WRONG" : "ALL GENERATED BODIES PASS",
           failures);
    return failures ? 1 : 0;
}}
'''


DECLS_TEMPLATE = """// kernels/kernel_table.h
// GENERATED by scripts/gen_kernels.py -- DO NOT EDIT BY HAND.
//
// Launch-wrapper declarations for every generated kernel, plus the dispatch
// table the host registers from.
//
// WHY A TABLE AND NOT A FILE PER RADIX ON THE HOST SIDE
// All {count} kernels have the same launch signature by construction, so the
// host needs no per-radix code at all: one loop over this table registers
// every variant. A hand-written impl_r3.cpp / impl_r5.cpp / impl_r7.cpp would
// be three copies of one function, and three copies drift.
#pragma once
#include <cstdint>

#ifndef ASCENDC_CPU_DEBUG

// One launch signature for all of them:
//   n       transform length
//   t       operand distance of this stage
//   block   t * radix
//   wOffset where this stage's twiddles start in W (in floats)
//   rowLen  twiddle row length, ceil(t/8)*8
//   tile    the j-run length one tile covers (already clamped by the host)
//   blk_num must equal blockDim; the kernel uses it as the tile stride
using OddLaunchFn = void (*)(uint32_t blockDim, void *stream,
                             void *xr, void *xi, void *w,
                             uint32_t n, uint32_t t, uint32_t block,
                             uint32_t wOffset, uint32_t rowLen,
                             uint32_t tile, uint32_t blk_num);

{protos}

#endif  // ASCENDC_CPU_DEBUG

struct OddKernelEntry {{
    uint32_t    radix;
    const char *butterfly;
    const char *scheme;
#ifndef ASCENDC_CPU_DEBUG
    OddLaunchFn launch;
#endif
}};

static const OddKernelEntry ODD_KERNELS[] = {{
{entries}
}};
static constexpr uint32_t ODD_KERNEL_COUNT =
    sizeof(ODD_KERNELS) / sizeof(ODD_KERNELS[0]);
"""


# =========================================================================
# the driver
# =========================================================================
LIMITS_TEMPLATE = '''// kernels/kernel_limits.h
// GENERATED by scripts/gen_kernels.py -- DO NOT EDIT BY HAND.
//
// The tile cap, shared by every generated kernel AND by the host.
//
// The host must clamp the requested tile to exactly the same value the kernel
// clamps it to, and must RECORD the clamped value in the comparison matrix.
// Otherwise a run asked for tile=512 would be recorded as tile=512 while the
// kernel quietly used {max_tile}, and the tile axis would be comparing cells
// that are not what the table says they are.
//
// The value is set by the hungriest kernel ({worst} UB slices) against a
// {budget_kb} KB budget of the 191 KB per core, and is shared by all of them so
// that two variants differing only in the butterfly are measured at the same
// effective tile. See the pass-1 comment in scripts/gen_kernels.py.
#pragma once

#define ODDFFT_MAX_TILE {max_tile}u

// The clamp, written once so the kernel and the host cannot disagree.
#define ODDFFT_CLAMP_TILE(t) \\
    ( ((t) > ODDFFT_MAX_TILE) ? ODDFFT_MAX_TILE : (((t) < 8u) ? 8u : (t)) )
'''

UB_BYTES = 191 * 1024
UB_BUDGET = 170 * 1024          # headroom: the TPipe itself needs some UB


def pick_max_tile(slices):
    """The largest tile (a multiple of 8) whose UB footprint fits the budget."""
    t = (UB_BUDGET // (slices * 4)) - 8
    t = (t // 8) * 8
    return max(8, min(512, t))


def load_reference():
    path = os.path.join(PROJ, "docs", "odd_butterflies.py")
    spec = importlib.util.spec_from_file_location("oddbf", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(PROJ, "kernels"))
    # 2 and 4 are here for the mixed-radix plan: a stage of a combined
    # transform may be a power-of-two stage, and it must go through a kernel
    # with the same launch signature and the same tiling as the odd ones.
    # They have `direct` only -- sym, winograd and rader are odd-radix forms.
    ap.add_argument("--radices", default="2,4,3,5,7")
    ap.add_argument("--algos", default="direct,sym,winograd,rader")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    mod = load_reference()
    # Replace the three arithmetic helpers with IR constructors. The butterfly
    # functions are untouched -- this is the same verified code, building a
    # graph instead of computing floats.
    mod.mul, mod.add, mod.sub = ir_mul, ir_add, ir_sub

    algos = {
        2: {"direct": mod.direct_generic(2)},
        4: {"direct": mod.direct_generic(4)},
        3: {"direct": mod.bf3_direct, "sym": mod.bf_sym(3),
            "winograd": mod.bf3_wino, "rader": mod.bf_rader(3)},
        5: {"direct": mod.direct_generic(5), "sym": mod.bf_sym(5),
            "winograd": mod.bf5_wino, "rader": mod.bf_rader(5)},
        7: {"direct": mod.direct_generic(7), "sym": mod.bf_sym(7),
            "winograd": mod.bf7_wino, "rader": mod.bf_rader(7)},
    }
    # The reference counts, recomputed here rather than pasted, so the note in
    # each generated header cannot go stale against the table.
    refcount = {}
    for r in (3, 5, 7):
        for name, fn in algos[r].items():
            pass
    # recount with the ORIGINAL helpers on a fresh import
    plain = load_reference()
    plain_algos = {
        2: {"direct": plain.direct_generic(2)},
        4: {"direct": plain.direct_generic(4)},
        3: {"direct": plain.bf3_direct, "sym": plain.bf_sym(3),
            "winograd": plain.bf3_wino, "rader": plain.bf_rader(3)},
        5: {"direct": plain.direct_generic(5), "sym": plain.bf_sym(5),
            "winograd": plain.bf5_wino, "rader": plain.bf_rader(5)},
        7: {"direct": plain.direct_generic(7), "sym": plain.bf_sym(7),
            "winograd": plain.bf7_wino, "rader": plain.bf_rader(7)},
    }
    rng = np.random.default_rng(5)
    for r in (2, 4, 3, 5, 7):
        for name, fn in plain_algos[r].items():
            plain.Count.reset()
            z = rng.standard_normal(r) + 1j * rng.standard_normal(r)
            yr, yi = fn(list(z.real), list(z.imag))
            got = np.array(yr) + 1j * np.array(yi)
            err = float(np.abs(got - np.fft.fft(z)).max())
            assert err < 1e-11, "reference %d/%s is wrong: %.3e" % (r, name, err)
            refcount[(r, name)] = (plain.Count.m, plain.Count.a)

    radices = [int(x) for x in a.radices.split(",")]
    want = [x.strip() for x in a.algos.split(",")]

    # PASS 1: build every butterfly and size its UB footprint, so a single
    # MAX_TILE can be chosen for ALL of them.
    #
    # WHY ONE SHARED CAP AND NOT A PER-KERNEL ONE
    # The footprints differ a lot -- r=7 needs more than twice the slices of
    # r=3 -- so a per-kernel cap would let r=3 run a 512-wide tile while r=7
    # sym was silently clamped to 424. The damage is not the lost width, it is
    # that `sym` and `direct` at r=7 would then be measured at DIFFERENT
    # effective tiles while the table recorded the same requested one, and the
    # butterfly axis would stop being paired exactly where it matters most.
    # One cap, set by the hungriest kernel, keeps every variant on the same
    # tile ladder. The tile sweep then says whether the lost width mattered.
    built = {}
    for radix in radices:
        for algo in want:
            if algo not in algos[radix]:
                continue   # sym/winograd/rader are odd-radix only
            info = build(radix, algo, algos[radix][algo])
            nw = 2 * (radix - 1)
            nin = 2 * radix + nw
            nout = 2 * radix
            ncalc = nw + info["peak"] + 2
            info["slices"] = (nin + nout) * 2 + ncalc
            built[(radix, algo)] = info
    GLOBAL_MAX_TILE = min(pick_max_tile(i["slices"]) for i in built.values())

    rows = []
    tests, test_calls = [], []
    for radix in radices:
        for algo in want:
            if (radix, algo) not in built:
                continue
            info = built[(radix, algo)]
            nw = 2 * (radix - 1)
            nin = 2 * radix + nw
            nout = 2 * radix
            ncalc = nw + info["peak"] + 2
            slices = info["slices"]
            max_tile = GLOBAL_MAX_TILE
            tile_f = max_tile + 8
            ub_kb = slices * tile_f * 4 // 1024

            kname = "fft%d_%s_dit" % (radix, algo)
            cls = "KernelR%d%sDit" % (radix, algo.capitalize())
            fname = "r%d_%s_dit_perstage_xs.cpp" % (radix, algo)

            # --- the declaration blocks -----------------------------------
            ind = "        "
            copyin = []
            for m in range(radix):
                base = "i0" if m == 0 else ("i0 + %du * this->t" % m)
                copyin.append("%sLoad(inT[%d * TILE_F], xrGm, %s);"
                              % (ind, 2 * m, base))
                copyin.append("%sLoad(inT[%d * TILE_F], xiGm, %s);"
                              % (ind, 2 * m + 1, base))
            copyin.append("")
            copyin.append("%sconst uint32_t wb = this->wOffset + shift;" % ind)
            for q in range(nw):
                copyin.append("%sLoad(inT[%d * TILE_F], wGm, wb + %du * this->rowLen);"
                              % (ind, 2 * radix + q, q))

            decl_x = []
            for m in range(radix):
                decl_x.append("%sauto x%dr = inT[%d * TILE_F];  auto x%di = inT[%d * TILE_F];"
                              % (ind, m, 2 * m, m, 2 * m + 1))
            decl_w = []
            for m in range(1, radix):
                q = 2 * (m - 1)
                decl_w.append("%sauto w%dr = inT[%d * TILE_F];  auto w%di = inT[%d * TILE_F];"
                              % (ind, m, 2 * radix + q, m, 2 * radix + q + 1))

            # u_0 aliases x_0 (no twiddle); u_m for m>=1 lives in calc slots
            decl_u = ["%sauto u0r = x0r;  auto u0i = x0i;   // W^0 = 1" % ind]
            for m in range(1, radix):
                q = 2 * (m - 1)
                decl_u.append("%sauto u%dr = calcT[%d * TILE_F];  auto u%di = calcT[%d * TILE_F];"
                              % (ind, m, q, m, q + 1))
            decl_c = []
            for i in range(info["peak"]):
                decl_c.append("%sauto c%d = calcT[%d * TILE_F];" % (ind, i, nw + i))
            decl_c.append("%sauto ta = calcT[%d * TILE_F];  auto tb = calcT[%d * TILE_F];"
                          % (ind, nw + info["peak"], nw + info["peak"] + 1))

            twiddle, twmul, twadd = [], 0, 0
            for m in range(1, radix):
                twiddle.append("%sAscendC::Mul(ta, x%dr, w%dr, len); "
                               "AscendC::Mul(tb, x%di, w%di, len); "
                               "AscendC::Sub(u%dr, ta, tb, len);" % (ind, m, m, m, m, m))
                twiddle.append("%sAscendC::Mul(ta, x%dr, w%di, len); "
                               "AscendC::Mul(tb, x%di, w%dr, len); "
                               "AscendC::Add(u%di, ta, tb, len);" % (ind, m, m, m, m, m))
                twmul += 4
                twadd += 2

            decl_y = []
            for p in range(radix):
                decl_y.append("%sauto y%dr = outT[%d * TILE_F];  auto y%di = outT[%d * TILE_F];"
                              % (ind, p, 2 * p, p, 2 * p + 1))

            # CopyOut re-derives the slices from outT: the y* names are
            # locals of Compute() and do not exist here.
            copyout = []
            for p in range(radix):
                base = "i0" if p == 0 else ("i0 + %du * this->t" % p)
                copyout.append("%sStore(xrGm, outT[%d * TILE_F], %s);  "
                               "Store(xiGm, outT[%d * TILE_F], %s);"
                               % (ind, 2 * p, base, 2 * p + 1, base))

            rm, ra = refcount[(radix, algo)]
            text = TEMPLATE.format(
                fname=fname, radix=radix, algo=algo, ALGO=algo.upper(),
                cls=cls, kname=kname,
                nw=nw, nin=nin, nout=nout, ncalc=ncalc, peak=info["peak"],
                slices=slices, tile_f=tile_f, ub_kb=ub_kb, max_tile=max_tile,
                nmul=info["nmul"], nadd=info["nadd"],
                ntot=info["nmul"] + info["nadd"],
                twmul=twmul, twadd=twadd,
                totalops=info["nmul"] + info["nadd"] + twmul + twadd,
                refmul=rm, refadd=ra, folded=info["folded"],
                foldedc=info["foldedc"], foldedm=info["foldedm"],
                deadm=info["dead_mul"],
                deada=info["dead_add"],
                copyin="\n".join(copyin),
                decl_x="\n".join(decl_x), decl_w="\n".join(decl_w),
                decl_u="\n".join(decl_u), decl_c="\n".join(decl_c),
                twiddle="\n".join(twiddle),
                decl_y="\n".join(decl_y),
                body="\n".join(info["body"]),
                copies="\n".join(info["copies"]),
                copyout="\n".join(copyout),
            )

            if not a.dry_run:
                os.makedirs(a.out, exist_ok=True)
                with open(os.path.join(a.out, fname), "w", encoding="utf-8") as f:
                    f.write(text)

            # --- the scalar test of this very body ------------------------
            decl_in, init_in = [], []
            for m in range(radix):
                decl_in.append("    float u%dr, u%di;" % (m, m))
                # a fixed, reproducible input; nothing about it is special
                init_in.append("    u%dr = (float)(%.9g);  u%di = (float)(%.9g);"
                               % (m, math.cos(0.7 * m + 0.3 * radix),
                                  m, math.sin(1.1 * m + 0.9 * radix)))
            tdecl_c = ["    float c%d = 0.0f;" % i for i in range(info["peak"])]
            tdecl_c.append("    float ta = 0.0f, tb = 0.0f;  (void)ta; (void)tb;")
            tdecl_y = []
            for q in range(radix):
                tdecl_y.append("    float y%dr = 0.0f, y%di = 0.0f;" % (q, q))
            tbody = [ln.replace("        ", "    ", 1) for ln in info["body"]]
            tcopies = [ln.replace("        ", "    ", 1) for ln in info["copies"]]
            tests.append(TEST_FN.format(
                radix=radix, algo=algo,
                decl_in="\n".join(decl_in), init_in="\n".join(init_in),
                decl_c="\n".join(tdecl_c), decl_y="\n".join(tdecl_y),
                body="\n".join(tbody), copies="\n".join(tcopies),
                list_r=", ".join("u%dr" % m for m in range(radix)),
                list_i=", ".join("u%di" % m for m in range(radix)),
                list_yr=", ".join("y%dr" % q for q in range(radix)),
                list_yi=", ".join("y%di" % q for q in range(radix)),
            ))
            test_calls.append("    test_%d_%s();" % (radix, algo))
            rows.append((radix, algo, info["nmul"], info["nadd"],
                         info["nmul"] + info["nadd"], rm, ra, info["folded"],
                         info["peak"], slices, max_tile, ub_kb,
                         len(info["copies"]), fname, info["foldedc"],
                         info["dead_mul"], info["dead_add"], info["foldedm"]))

    if not a.dry_run:
        tdir = os.path.join(PROJ, "tests")
        os.makedirs(tdir, exist_ok=True)
        with open(os.path.join(tdir, "gen_butterfly_test.cpp"), "w",
                  encoding="utf-8") as f:
            f.write(TEST_HEADER.replace("{{", "{").replace("}}", "}"))
            for t in tests:
                f.write(t)
            f.write(TEST_MAIN.format(calls="\n".join(test_calls))
                    .replace("{{", "{").replace("}}", "}"))

        protos, entries = [], []
        for (rx, al, *_rest) in rows:
            kn = "fft%d_%s_dit" % (rx, al)
            protos.append(
                "void %s_do(uint32_t blockDim, void *stream,\n"
                "                void *xr, void *xi, void *w,\n"
                "                uint32_t n, uint32_t t, uint32_t block,\n"
                "                uint32_t wOffset, uint32_t rowLen,\n"
                "                uint32_t tile, uint32_t blk_num);" % kn)
            entries.append(
                '#ifndef ASCENDC_CPU_DEBUG\n'
                '    { %d, "%s", "dit", %s_do },\n'
                '#else\n'
                '    { %d, "%s", "dit" },\n'
                '#endif' % (rx, al, kn, rx, al))
        with open(os.path.join(a.out, "kernel_table.h"), "w",
                  encoding="utf-8") as f:
            f.write(DECLS_TEMPLATE.format(count=len(rows),
                                          protos="\n\n".join(protos),
                                          entries="\n".join(entries)))

        lim = os.path.join(a.out, "kernel_limits.h")
        with open(lim, "w", encoding="utf-8") as f:
            f.write(LIMITS_TEMPLATE.format(
                max_tile=GLOBAL_MAX_TILE,
                worst=max(i["slices"] for i in built.values()),
                budget_kb=UB_BUDGET // 1024))

    print("%-3s %-9s %6s %6s %6s  %11s %6s %6s %6s %6s %5s %7s %5s %6s"
          % ("r", "algo", "mults", "adds", "total", "ref(m,a)", "fold0",
             "foldK", "foldM", "dead", "peak", "slices", "tile", "UB_KB"))
    print("-" * 104)
    for (radix, algo, nm, na, tot, rm, ra, fold, peak, slices,
         max_tile, ub_kb, ncopy, fname, foldc, dm, da, fm) in rows:
        print("%-3d %-9s %6d %6d %6d  %11s %6d %6d %6d %6s %5d %7d %5d %6d"
              % (radix, algo, nm, na, tot, "%d,%d" % (rm, ra), fold, foldc, fm,
                 "%d,%d" % (dm, da), peak, slices, max_tile, ub_kb))
    print()
    print("mults/adds: what the GENERATED kernel issues for the butterfly.")
    print("ref(m,a):   what docs/odd_butterflies.py counts, table in README.")
    print("fold0:      add(0.0, x) accumulator inits the reference counts and")
    print("            the kernel does not issue -- no zero tensor is made.")
    print("foldK:      adds of two LITERALS, which are compile-time constant")
    print("            arithmetic the reference happens to perform at runtime.")
    print("dead:       (mults,adds) the reference COMPUTES AND NEVER USES.")
    print("            The kernel does not issue them: the scheduler walks only")
    print("            what the outputs depend on. This is dead code in the")
    print("            reference, found by generating from it.")
    print("foldM:      multiplies by 0 or 1 -- a zero or a copy, not")
    print("            arithmetic. Every radix-4 twiddle is one of these, which")
    print("            is why radix 4 issues no multiplies at all.")
    print("            mults + foldM + dead_m == ref m, and")
    print("            adds + fold0 + foldK + dead_a == ref a, exactly.")
    print("tile/UB_KB: the per-kernel tile cap and its UB footprint.")
    ok = all(na + fold + foldc + da == ra and nm + fm + dm == rm for
             (_, _, nm, na, _, rm, ra, fold, _, _, _, _, _, _, foldc, dm, da, fm)
             in rows)
    print()
    print("counts reconcile with the reference table: %s" % ("YES" if ok else "NO"))
    if not ok:
        return 1
    print("%d kernels %s" % (len(rows), "checked (dry run)" if a.dry_run else "written"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
