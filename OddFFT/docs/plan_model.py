"""Модель алгоритма до написания C++: цифровая инверсия + план стадий для
ПРОИЗВОЛЬНОГО списка радиксов, проверка против numpy.fft."""
import numpy as np

def digit_rev_index(i, radices):
    """Индекс i раскладывается как i = sum_s d_s * t_s, где t_s = prod(r_0..r_{s-1}).
    Инверсия ставит разряд d_s в вес prod(r_{s+1}..r_{L-1})."""
    L = len(radices)
    d = []
    v = i
    for r in radices:
        d.append(v % r); v //= r
    j = 0
    for s in range(L):
        w = 1
        for u in range(s+1, L):
            w *= radices[u]
        j += d[s] * w
    return j

def dft_r(v, r):
    W = np.exp(-2j*np.pi*np.outer(np.arange(r), np.arange(r))/r)
    return W @ v

def fft_plan(x, radices, perm_input=True, stage_order=None):
    """DIT: перестановка входа, затем стадии от t=1 вверх.
    Стадия s: r=radices[s], t=prod(radices[:s]), block=t*r,
    бабочка (k,j): x[k*block + j + m*t], твиддл W_block^(m*j)."""
    N = int(np.prod(radices))
    order = stage_order if stage_order is not None else list(range(len(radices)))
    y = np.zeros(N, dtype=complex)
    if perm_input:
        for i in range(N):
            y[digit_rev_index(i, radices)] = x[i]
    else:
        y[:] = x
    t = 1
    for s in order:
        r = radices[s]
        block = t * r
        W = np.exp(-2j*np.pi/block)
        for k in range(N // block):
            for j in range(t):
                idx = [k*block + j + m*t for m in range(r)]
                vec = np.array([y[i] for i in idx])
                vec = vec * np.array([W**(m*j) for m in range(r)])
                res = dft_r(vec, r)
                for m, i in enumerate(idx):
                    y[i] = res[m]
        t = block
    return y

# сначала проверим, что инверсия — биекция
for radices in ([3,3,3],[5,5],[3,5,7],[7,5,3],[3,3,5]):
    N = int(np.prod(radices))
    img = sorted(digit_rev_index(i, radices) for i in range(N))
    assert img == list(range(N)), ("не биекция", radices)
print("цифровая инверсия — биекция на всех проверенных списках: OK")
print()

rng = np.random.default_rng(3)
print("%-20s %-7s %-12s %s" % ("радиксы", "N", "перм.входа", "без перестановки"))
print("-"*58)
for radices in ([3,3,3],[5,5],[7,7],[3,3,3,3],[3,5],[5,3],[3,5,7],[7,5,3],[3,3,5]):
    N = int(np.prod(radices))
    x = rng.standard_normal(N) + 1j*rng.standard_normal(N)
    ref = np.fft.fft(x)
    e1 = np.abs(fft_plan(x, radices, True) - ref).max()
    e2 = np.abs(fft_plan(x, radices, False) - ref).max()
    print("%-20s %-7d %-12s %s" % (radices, N,
          "OK" if e1 < 1e-10 else "%.1e" % e1,
          "OK" if e2 < 1e-10 else "%.1e" % e2))
