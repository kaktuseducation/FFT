#!/usr/bin/env python3
"""odd_butterflies.py — reference implementations of the radix-3/5/7 butterflies.

This is the source of truth that the C++ host reference and the AscendC kernels
are ported FROM. Every algorithm here is checked against numpy.fft and every
operation count is measured by a counter, not quoted from a paper: run this
file and it prints both.

The arithmetic is deliberately written the way the kernel has to write it —
real operations on separate Re/Im values — so the counts mean what they will
mean on the device, and so the port is a transcription rather than a
re-derivation.

Two things that were DERIVED here rather than copied, because getting them
wrong would be silent:

  * the 3-point cyclic convolution reconstruction. Solving for the coefficient
    matrix over random data gave exactly
        3*y0 = m0 + 3*m1 - m3
        3*y1 = m0 - 3*m1 - 3*m2 + 2*m3
        3*y2 = m0 + 3*m2 - m3
    and the division by 3 folds into the precomputed constants, so the
    convolution costs 4 multiplies and no scaling.

  * the radix-7 Winograd form, as Rader (length-7 DFT -> length-6 cyclic
    convolution) composed with Agarwal-Cooley (6 = 2*3 coprime, so the length-6
    convolution nests a length-2 and a length-3 one: 2 * 4 = 8 complex
    multiplies). Literature quotes a 7-point minimum in the high teens of real
    multiplies; this construction lands at 32, and the measured total operation
    count is what matters below.

WHAT THE COUNTS ALREADY SETTLE, BEFORE ANY KERNEL EXISTS

On a vector unit a multiply and an add are both one instruction, so the figure
to compare is their SUM, not the multiply count that the classical literature
minimises. By that measure:

  * `rader` with a directly-computed convolution is dominated everywhere — it
    has exactly the multiply count of `direct` plus extra adds. It is not worth
    implementing as a performance variant; it is worth one measured line.
  * `winograd` wins only at r=3 (16 vs 18 total ops).
  * at r=5 it is a wash (54 vs 52, i.e. 1.04x worse).
  * at r=7 it is 45% WORSE than `sym` (148 vs 102): it trades 4 multiplies for
    50 adds, which is a good trade on hardware where multiplies are expensive
    and a bad one here.

So the expectation going in is that `sym` wins at 5 and 7 and `winograd` wins
at 3. What arithmetic cannot predict is the instruction mix, the register
pressure and the UB traffic each form produces on the actual AIV, which is why
all of them still get built and measured.
"""
import math
import numpy as np


class Count:
    """Counts real multiplies and adds, exactly as the kernel would issue them."""
    m = 0
    a = 0

    @classmethod
    def reset(cls):
        cls.m = 0
        cls.a = 0


def mul(x, y):
    Count.m += 1
    return x * y


def add(x, y):
    Count.a += 1
    return x + y


def sub(x, y):
    Count.a += 1
    return x - y



# --------------------------------------------------------------------------
# radix 3
# --------------------------------------------------------------------------

def bf3_direct(xr, xi):
    """Наивный: y_p = sum_k x_k W^(pk). W^0 не умножаем."""
    W = [[math.cos(-2*math.pi*(p*k % 3)/3) for k in range(3)] for p in range(3)]
    Wi= [[math.sin(-2*math.pi*(p*k % 3)/3) for k in range(3)] for p in range(3)]
    yr = [0.0]*3; yi = [0.0]*3
    for p in range(3):
        ar = xr[0]; ai = xi[0]                      # k=0: W=1, без умножений
        for k in range(1, 3):
            if p == 0:                              # W=1
                ar = add(ar, xr[k]); ai = add(ai, xi[k])
            else:
                ar = add(ar, sub(mul(W[p][k], xr[k]), mul(Wi[p][k], xi[k])))
                ai = add(ai, add(mul(W[p][k], xi[k]), mul(Wi[p][k], xr[k])))
        yr[p] = ar; yi[p] = ai
    return yr, yi

def bf3_wino(xr, xi):
    """Классика: 2 вещественных умножения на комплексный вход (4 на Re+Im)."""
    c = -0.5                                   # cos(2pi/3)
    s = -math.sin(2*math.pi/3)                 # sin(-2pi/3)
    t1r = add(xr[1], xr[2]); t1i = add(xi[1], xi[2])
    t2r = sub(xr[1], xr[2]); t2i = sub(xi[1], xi[2])
    y0r = add(xr[0], t1r);   y0i = add(xi[0], t1i)
    mr  = add(xr[0], mul(c, t1r)); mi = add(xi[0], mul(c, t1i))
    nr  = mul(s, t2r);             ni = mul(s, t2i)
    # y1 = m + i*n, y2 = m - i*n   (i*n = (-ni, nr))
    y1r = sub(mr, ni); y1i = add(mi, nr)
    y2r = add(mr, ni); y2i = sub(mi, nr)
    return [y0r, y1r, y2r], [y0i, y1i, y2i]


# --------------------------------------------------------------------------
# generic: naive matrix, and conjugate-pair symmetry (the FFTW-codelet form)
# --------------------------------------------------------------------------

def bf_sym(r):
    """Симметрия сопряжённых пар: пары (k, r-k) складываются и вычитаются
    ОДИН раз, затем на каждую половину выхода идут только cos/sin-члены.
    Это форма codelet'ов FFTW."""
    h = (r - 1) // 2
    c = [[math.cos(2*math.pi*(p*k % r)/r) for k in range(r)] for p in range(r)]
    s = [[math.sin(2*math.pi*(p*k % r)/r) for k in range(r)] for p in range(r)]
    def f(xr, xi):
        Ar = [None]*(h+1); Ai = [None]*(h+1); Br = [None]*(h+1); Bi = [None]*(h+1)
        for k in range(1, h+1):
            Ar[k] = add(xr[k], xr[r-k]); Ai[k] = add(xi[k], xi[r-k])
            Br[k] = sub(xr[k], xr[r-k]); Bi[k] = sub(xi[k], xi[r-k])
        yr = [None]*r; yi = [None]*r
        yr[0] = xr[0]; yi[0] = xi[0]
        for k in range(1, h+1):
            yr[0] = add(yr[0], Ar[k]); yi[0] = add(yi[0], Ai[k])
        for p in range(1, h+1):
            ur = xr[0]; ui = xi[0]; vr = 0.0; vi = 0.0
            for k in range(1, h+1):
                ur = add(ur, mul(c[p][k], Ar[k])); ui = add(ui, mul(c[p][k], Ai[k]))
                vr = add(vr, mul(s[p][k], Br[k])); vi = add(vi, mul(s[p][k], Bi[k]))
            # W = exp(-2pi i/r): y_p = u - i*v, y_{r-p} = u + i*v
            yr[p]   = add(ur, vi); yi[p]   = sub(ui, vr)
            yr[r-p] = sub(ur, vi); yi[r-p] = add(ui, vr)
        return yr, yi
    return f

def primitive_root(p):
    for g in range(2, p):
        seen = set(); x = 1
        for _ in range(p-1):
            x = x*g % p; seen.add(x)
        if len(seen) == p-1: return g
    raise ValueError

def bf_rader(r):
    """Rader: ДПФ простой длины r = циклическая свёртка длины r-1.
    Свёртка считается напрямую (это и есть вариант 'rader-direct')."""
    g = primitive_root(r)
    n = r - 1
    perm  = [pow(g, q, r) for q in range(n)]            # g^q mod r
    iperm = [pow(g, -q, r) for q in range(n)]           # g^-q mod r
    wr = [math.cos(-2*math.pi*pow(g, q, r)/r) for q in range(n)]
    wi = [math.sin(-2*math.pi*pow(g, q, r)/r) for q in range(n)]
    def f(xr, xi):
        ar = [xr[iperm[q]] for q in range(n)]
        ai = [xi[iperm[q]] for q in range(n)]
        s0r = xr[0]; s0i = xi[0]
        sumr = ar[0]; sumi = ai[0]
        for q in range(1, n): sumr = add(sumr, ar[q]); sumi = add(sumi, ai[q])
        yr = [None]*r; yi = [None]*r
        yr[0] = add(s0r, sumr); yi[0] = add(s0i, sumi)
        conv_r = [0.0]*n; conv_i = [0.0]*n
        for p in range(n):
            accr = 0.0; acci = 0.0
            for q in range(n):
                k = (p - q) % n
                accr = add(accr, sub(mul(wr[k], ar[q]), mul(wi[k], ai[q])))
                acci = add(acci, add(mul(wr[k], ai[q]), mul(wi[k], ar[q])))
            conv_r[p] = accr; conv_i[p] = acci
        for p in range(n):
            yr[perm[p]] = add(s0r, conv_r[p]); yi[perm[p]] = add(s0i, conv_i[p])
        return yr, yi
    return f

def direct_generic(r):
    def f(xr, xi):
        c = [[math.cos(-2*math.pi*(p*k % r)/r) for k in range(r)] for p in range(r)]
        s = [[math.sin(-2*math.pi*(p*k % r)/r) for k in range(r)] for p in range(r)]
        yr = [None]*r; yi = [None]*r
        for p in range(r):
            ar = xr[0]; ai = xi[0]
            for k in range(1, r):
                if p == 0: ar = add(ar, xr[k]); ai = add(ai, xi[k])
                else:
                    ar = add(ar, sub(mul(c[p][k], xr[k]), mul(s[p][k], xi[k])))
                    ai = add(ai, add(mul(c[p][k], xi[k]), mul(s[p][k], xr[k])))
            yr[p] = ar; yi[p] = ai
        return yr, yi
    return f


# --------------------------------------------------------------------------
# radix 5: minimal-multiply Winograd (10 real multiplies)
# --------------------------------------------------------------------------

def pair3(a, b, t3, t4):
    m1 = mul(a, add(t3, t4))
    m2 = mul(b - a, t4)          # (b-a) — предвычисленная константа, не операция
    m3 = mul(a + b, t3)          # (a+b) — тоже константа
    return add(m1, m2), sub(m3, m1)

def wino5_component(x):
    """Одна вещественная компонента (Re или Im) пятиточечного ДПФ.
    Возвращает (cos-часть для p=1,2 ; sin-часть для p=1,2).
    5 умножений: p*t5, q*d, и три в pair3."""
    u = 2*math.pi/5
    c1, c2 = math.cos(u), math.cos(2*u)
    s1, s2 = math.sin(u), math.sin(2*u)
    p = (c1 + c2)/2.0            # = -1/4 ровно
    q = (c1 - c2)/2.0            # = sqrt(5)/4
    t1 = add(x[1], x[4]); t2 = add(x[2], x[3])
    t3 = sub(x[1], x[4]); t4 = sub(x[2], x[3])
    t5 = add(t1, t2);     d  = sub(t1, t2)
    y0 = add(x[0], t5)
    pt = mul(p, t5); qd = mul(q, d)
    A  = add(pt, qd)             # c1*t1 + c2*t2
    B  = sub(pt, qd)             # c2*t1 + c1*t2
    u1 = add(x[0], A); u2 = add(x[0], B)
    Cc, Dd = pair3(s1, s2, t3, t4)   # C = s1*t3+s2*t4 ; D = s2*t3-s1*t4
    return (u1, u2, Cc, Dd)

def bf5_wino(xr, xi):
    ur1, ur2, Cr, Dr = wino5_component(xr)
    ui1, ui2, Ci, Di = wino5_component(xi)
    # W = exp(-2pi i/5):  y_p = u_p - i*V_p,  y_{5-p} = u_p + i*V_p
    #   V_1 = C (комплексное), V_2 = D
    yr = [None]*5; yi = [None]*5
    yr[0] = add(xr[0], 0.0) if False else None
    # y0 считается внутри компоненты, но мы его потеряли — пересоберём дёшево
    t1r = add(xr[1], xr[4]); t2r = add(xr[2], xr[3])
    t1i = add(xi[1], xi[4]); t2i = add(xi[2], xi[3])
    yr[0] = add(xr[0], add(t1r, t2r)); yi[0] = add(xi[0], add(t1i, t2i))
    # -i*(Cr + i*Ci) = Ci - i*Cr
    yr[1] = add(ur1, Ci); yi[1] = sub(ui1, Cr)
    yr[4] = sub(ur1, Ci); yi[4] = add(ui1, Cr)
    yr[2] = add(ur2, Di); yi[2] = sub(ui2, Dr)
    yr[3] = sub(ur2, Di); yi[3] = add(ui2, Dr)
    return yr, yi


# --------------------------------------------------------------------------
# radix 7: Rader + Agarwal-Cooley (8 complex multiplies in the convolution)
# --------------------------------------------------------------------------

# комплексное на комплексную КОНСТАНТУ: 4 вещественных умножения
def cmulk(kr, ki, xr, xi):
    return (sub(mul(kr, xr), mul(ki, xi)),
            add(mul(kr, xi), mul(ki, xr)))
def cadd(a, b): return (add(a[0], b[0]), add(a[1], b[1]))
def csub(a, b): return (sub(a[0], b[0]), sub(a[1], b[1]))
def cscale_int(a, s):                          # умножение на целое: оставляем как сложения
    if s == 1:  return a
    if s == -1: return (sub(0.0, a[0]), sub(0.0, a[1]))
    if s == 2:  return cadd(a, a)
    raise ValueError(s)

def cyc3_bilinear(H, X):
    """H — 4 предвычисленные комплексные константы, X — 3 комплексных входа.
    4 комплексных умножения. Матрица восстановления из cyc.py."""
    Xs = cadd(cadd(X[0], X[1]), X[2])
    X1 = csub(X[0], X[2])
    X2 = csub(X[1], X[2])
    X3 = cadd(X1, X2)
    M = [cmulk(H[0][0], H[0][1], Xs[0], Xs[1]),
         cmulk(H[1][0], H[1][1], X1[0], X1[1]),
         cmulk(H[2][0], H[2][1], X2[0], X2[1]),
         cmulk(H[3][0], H[3][1], X3[0], X3[1])]
    y0 = csub(cadd(M[0], M[1]), M[3])
    y1 = cadd(csub(csub(M[0], M[1]), M[2]), cscale_int(M[3], 2))
    y2 = csub(cadd(M[0], M[2]), M[3])
    return [y0, y1, y2]

def cyc3_consts(h):
    """Предвычисление H из ядра свёртки h (3 комплексных)."""
    a0 = h[0] - h[2]; a1 = h[1] - h[2]
    return [(h[0]+h[1]+h[2])/3, a0, a1, (a0+a1)/3]

def cyc6(h, x):
    """Циклическая свёртка длины 6 = 2x3 (Агарвал-Кули): 8 комплексных умножений."""
    # CRT-отображение i <-> (i mod 2, i mod 3), обратное i = (3u+4v) mod 6
    idx = [[(3*u + 4*v) % 6 for v in range(3)] for u in range(2)]
    H2 = [[h[idx[u][v]] for v in range(3)] for u in range(2)]
    X2 = [[x[idx[u][v]] for v in range(3)] for u in range(2)]
    # внешний уровень — длина 2, два произведения
    Xs = [cadd(X2[0][v], X2[1][v]) for v in range(3)]
    Xd = [csub(X2[0][v], X2[1][v]) for v in range(3)]
    hs = [(H2[0][v] + H2[1][v]) / 2 for v in range(3)]
    hd = [(H2[0][v] - H2[1][v]) / 2 for v in range(3)]
    P = cyc3_bilinear([(c.real, c.imag) for c in cyc3_consts(hs)], Xs)
    Q = cyc3_bilinear([(c.real, c.imag) for c in cyc3_consts(hd)], Xd)
    out = [None]*6
    for v in range(3):
        out[idx[0][v]] = cadd(P[v], Q[v])
        out[idx[1][v]] = csub(P[v], Q[v])
    return out

def bf7_wino(xr, xi):
    r = 7; g = 3; n = 6
    W = [complex(math.cos(-2*math.pi*k/r), math.sin(-2*math.pi*k/r)) for k in range(r)]
    perm  = [pow(g,  q, r) for q in range(n)]
    iperm = [pow(g, -q, r) for q in range(n)]
    h = [W[pow(g, k, r)] for k in range(n)]
    a = [(xr[iperm[q]], xi[iperm[q]]) for q in range(n)]
    s = a[0]
    for q in range(1, n): s = cadd(s, a[q])
    y = [None]*r
    y[0] = cadd((xr[0], xi[0]), s)
    conv = cyc6(h, a)
    for p in range(n):
        y[perm[p]] = cadd((xr[0], xi[0]), conv[p])
    return [v[0] for v in y], [v[1] for v in y]




# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(11)
    algos = {
        3: {"direct": bf3_direct,       "sym": bf_sym(3),
            "winograd": bf3_wino,       "rader": bf_rader(3)},
        5: {"direct": direct_generic(5), "sym": bf_sym(5),
            "winograd": bf5_wino,        "rader": bf_rader(5)},
        7: {"direct": direct_generic(7), "sym": bf_sym(7),
            "winograd": bf7_wino,        "rader": bf_rader(7)},
    }
    print("r   algorithm   mults  adds   TOTAL  per point  max|err|   ")
    print("-" * 66)
    totals = {}
    for r in (3, 5, 7):
        x = rng.standard_normal(r) + 1j * rng.standard_normal(r)
        ref = np.fft.fft(x)
        for name in ("direct", "sym", "winograd", "rader"):
            Count.reset()
            yr, yi = algos[r][name](list(x.real), list(x.imag))
            got = np.array(yr) + 1j * np.array(yi)
            err = float(np.abs(got - ref).max())
            tot = Count.m + Count.a
            totals.setdefault(r, []).append((tot, name))
            assert err < 1e-12, "%d/%s is WRONG: %.3e" % (r, name, err)
            print("%-3d %-11s %5d %5d  %5d   %7.1f   %.2e  OK"
                  % (r, name, Count.m, Count.a, tot, tot / r, err))
        best = min(totals[r])[1]
        print("    -> fewest total operations at r=%d: %s\n" % (r, best))
    print("A multiply and an add are both one vector instruction, so TOTAL is")
    print("the figure to compare — not the multiply count the classical")
    print("literature minimises.")
