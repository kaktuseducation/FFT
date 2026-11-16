#!/usr/bin/python3
# coding=utf-8
#
# Разбор output/matrix.csv — таблицы, которую пишет сам бинарник (по строке на
# клетку матрицы, со всеми осями и результатом сверки с golden).
#
# Печатает четыре блока:
#
#   1. По каждому N — таблицу вариантов в ЛУЧШЕЙ для каждого конфигурации
#      (минимум total_us среди прошедших сверку) и отношение к NumPy.
#
#   2. ПАРНЫЕ сравнения по осям — главный блок для выбора решения. Сравниваются
#      только клетки, различающиеся РОВНО по одной оси: одинаковые N, tile,
#      радикс, алгоритм, раскладка и так далее. См. ниже, почему это важно.
#
#   3. Обзорная таблица "лучшее время на каждое значение оси" — она НЕ парная
#      и годится только для общего взгляда.
#
#   4. Клетки без PASS — отдельным списком, чтобы не терялись.
#
# ПОЧЕМУ ПАРНЫЕ СРАВНЕНИЯ ВАЖНЕЕ ОБЗОРНЫХ
#
# Обзорная таблица берёт лучшее время по каждому значению оси ГДЕ УГОДНО в
# матрице. На реальных данных это сразу дало ловушку: в строке "алгоритм" у
# dit стояло 195.95 (это radix-8 на 40 ядрах), а у stockham — 254.29 (это
# radix-4 на одном ядре). Разница 1.30x выглядит как превосходство DIT, хотя
# на самом деле это превосходство радикса и многоядерности: честное парное
# сравнение на одном и том же radix-4 даёт для DIT против Stockham всего
# 1.24x, а на radix-2 — 1.17x.
#
# Поэтому решения принимаются по блоку 2, а блок 3 остаётся как обзор.
#
# Зависимости — только стандартная библиотека.
#
# Использование:
#   python3 scripts/analyze_matrix.py
#   python3 scripts/analyze_matrix.py -N 4096
#   python3 scripts/analyze_matrix.py --axis granularity --all-pairs
#   python3 scripts/analyze_matrix.py --export ./output/pivot.csv

import argparse
import csv
import os
import sys

# Оси и то, какие ещё поля нужно ИСКЛЮЧИТЬ из ключа группировки при парном
# сравнении. Исключать приходится всё, что жёстко связано с самой осью:
#   * radix — закодирован в impl ("r4"), поэтому impl тоже вне ключа;
#   * unit  — у cube-вариантов algo тоже равен "cube", иначе пар не найдётся;
#   * terms — имеет смысл только внутри cube, там же различается и variant.
ALL_KEYS = ["impl", "radix", "algo", "granularity", "layout", "unit", "terms",
            "N", "tile", "cube_batch"]
AXES = [
    ("radix",       "радикс",        {"impl", "radix", "variant"}),
    ("algo",        "алгоритм",      {"algo", "variant"}),
    ("granularity", "гранулярность", {"granularity", "variant"}),
    ("layout",      "раскладка",     {"layout", "variant"}),
    # Vector-вариант и Cube-вариант различаются не только полем unit: у
    # Vector'а algo=dit, terms=0 и cube_batch=0, у Cube — algo=cube, terms>0 и
    # осмысленный батч. Все эти поля — следствия выбора вычислителя, поэтому
    # из ключа группировки они тоже уходят, иначе пара не находится никогда.
    ("unit",        "вычислитель",   {"unit", "algo", "terms", "cube_batch", "variant"}),
    ("terms",       "членов схемы",  {"terms", "variant"}),
    # Пара вроде (r4/dit_fs_xs, r4/dit_fsm_xs): одна и та же математика, одно
    # ядро против сорока. Это единственная ось, по которой видно, что даёт
    # многоядерность сама по себе, без примеси радикса и алгоритма.
    ("cores",       "число ядер",    {"cores", "variant"}),
    ("tile",        "порог tile",    {"tile"}),
    ("cube_batch",  "Cube-батч",     {"cube_batch"}),
]
AXIS_BY_NAME = {a[0]: a for a in AXES}

# По какому полю сравнивать: "_total" (полное время, по умолчанию) или "_npu"
# (только NPU-часть). Зачем нужен второй вариант: CPU-стадии общие у всех
# вариантов одного радикса, и в total они разбавляют эффект. Например
# многоядерность r4 при N=262144 даёт по total всего 1.85x, а по NPU — 12.9x,
# потому что 92% total занимают две стадии на процессоре, которых
# многоядерность вообще не касается. Для вывода про масштабирование по ядрам
# правильная метрика — npu.
METRIC = {"key": "_total", "label": "TOTAL_us"}


def fnum(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def load_matrix(path):
    if not os.path.exists(path):
        print("[ERROR] нет %s — сначала прогони run.sh или run_matrix.sh" % path)
        sys.exit(1)
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            r["_total"] = fnum(r.get("total_us"))
            r["_npu"] = fnum(r.get("npu_us"))
            r["_cpu"] = fnum(r.get("cpu_us"))
            r["_N"] = int(r["N"]) if r.get("N") else 0
            r["_full"] = "%s/%s" % (r.get("impl", "?"), r.get("variant", "?"))
            rows.append(r)
    return rows


def load_numpy_baseline(path):
    """{N: total_avg_us} из таблицы, которую пишет scripts/bench_numpy.py."""
    base = {}
    if not os.path.exists(path):
        return base
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("variant", "").strip().lower() != "numpy":
                continue
            n, t = r.get("N"), fnum(r.get("total_avg_us"))
            if n and t:
                base[int(n)] = t
    return base


def table(rows, header):
    if not rows:
        return
    cols = list(zip(*([header] + rows)))
    w = [max(len(str(c)) for c in col) for col in cols]
    print("  ".join(str(h).ljust(w[i]) for i, h in enumerate(header)))
    print("  ".join("-" * x for x in w))
    for r in rows:
        print("  ".join(str(c).ljust(w[i]) for i, c in enumerate(r)))


def best_per_variant(rows):
    """Лучшая ПРОШЕДШАЯ сверку клетка на вариант, по выбранной метрике."""
    mk = METRIC["key"]
    best = {}
    for r in rows:
        if r.get("status") != "PASS" or r.get(mk) is None:
            continue
        key = r["_full"]
        if key not in best or r[mk] < best[key][mk]:
            best[key] = r
    return best


def group_label(r, excluded):
    """Короткая подпись группы: что именно зафиксировано в этом сравнении."""
    bits = []
    if "impl" not in excluded:
        bits.append(r.get("impl", "?"))
    if "algo" not in excluded:
        bits.append(r.get("algo", "?"))
    if "granularity" not in excluded:
        bits.append(r.get("granularity", "?"))
    if "layout" not in excluded:
        bits.append(r.get("layout", "?"))
    if "unit" not in excluded and r.get("unit") == "cube" and "cube" not in bits:
        bits.append("cube")
    bits.append("N=%s" % r.get("N"))
    if "tile" not in excluded:
        bits.append("tile=%s" % r.get("tile"))
    if "cube_batch" not in excluded and r.get("unit") == "cube":
        bits.append("B=%s" % r.get("cube_batch"))
    return " ".join(bits)


def paired_axis(rows, axis, excluded, show_all, limit=6, allow_fail=False):
    """Сравнения клеток, различающихся РОВНО по оси axis.

    allow_fail=True нужен для оси terms: число членов схемы восстановления
    точности меняет в первую очередь ТОЧНОСТЬ, и вся суть сравнения в том, что
    варианты с terms=1 и terms=2 сверку НЕ проходят. Отбросив их как обычно,
    мы потеряли бы ровно то, что хотели измерить, — цену точности во времени.
    """
    mk = METRIC["key"]
    keys = [k for k in ALL_KEYS if k not in excluded]
    groups = {}
    for r in rows:
        if r.get(mk) is None:
            continue
        if not allow_fail and r.get("status") != "PASS":
            continue
        gk = tuple(r.get(k) for k in keys)
        av = r.get(axis)
        if av is None or av == "":
            continue
        slot = groups.setdefault(gk, {})
        # внутри одной группы у значения оси может быть несколько клеток
        # (например разные variant при фиксированном algo) — берём лучшую
        if av not in slot or r[mk] < slot[av][mk]:
            slot[av] = r
    # оставляем только группы, где есть что сравнивать
    cmps = [(gk, slot) for gk, slot in groups.items() if len(slot) >= 2]
    if not cmps:
        return None
    # ratios может остаться пустым, если ни в одной группе не нашлось
    # прошедшего сверку варианта (только для allow_fail-осей)

    wins = {}
    ratios = []
    lines = []
    for gk, slot in cmps:
        ordered = sorted(slot.items(), key=lambda kv: kv[1][mk])
        win_val, win_row = ordered[0]
        worst_val, worst_row = ordered[-1]
        ratio = worst_row[mk] / win_row[mk] if win_row[mk] else 0.0
        if allow_fail:
            # Здесь "быстрее" не значит "лучше": самый быстрый вариант обычно
            # как раз тот, что не проходит сверку. Считаем ЦЕНУ ТОЧНОСТИ —
            # во сколько раз дороже самый дешёвый из прошедших сверку по
            # сравнению с самым быстрым вообще.
            detail = " | ".join("%s: %.1f us, %s, %s"
                                % (v, r[mk], r.get("max_abs", "-"), r.get("status", "?"))
                                for v, r in ordered)
            passed = [(v, r) for v, r in ordered if r.get("status") == "PASS"]
            if not passed:
                lines.append([group_label(win_row, excluded), detail,
                              "сверку не проходит ни один"])
                continue
            pv, pr = passed[0]
            cost = pr[mk] / ordered[0][1][mk] if ordered[0][1][mk] else 0.0
            wins[pv] = wins.get(pv, 0) + 1
            ratios.append(cost)
            lines.append([group_label(win_row, excluded), detail,
                          "брать %s: цена точности +%.0f%%" % (pv, 100.0 * (cost - 1.0))])
        else:
            wins[win_val] = wins.get(win_val, 0) + 1
            ratios.append(ratio)
            detail = ", ".join("%s=%.2f" % (v, r[mk]) for v, r in ordered)
            lines.append([group_label(win_row, excluded), detail,
                          "%s быстрее в %.2fx" % (win_val, ratio)])
    lines.sort(key=lambda x: x[0])
    return {"lines": lines, "wins": wins, "ratios": ratios,
            "shown": lines if show_all else lines[:limit],
            "hidden": 0 if show_all else max(0, len(lines) - limit)}


def main():
    p = argparse.ArgumentParser(description="Разбор матрицы FFTBench.")
    p.add_argument("--matrix", default="./output/matrix.csv")
    p.add_argument("--baseline", default="./output/numpy_baseline.csv")
    p.add_argument("-N", "--size", type=int, default=None, help="только это N")
    p.add_argument("--axis", default=None,
                   help="подробно по одной оси: %s" % ", ".join(a[0] for a in AXES))
    p.add_argument("--all-pairs", action="store_true",
                   help="показать все парные сравнения, а не первые шесть")
    p.add_argument("--metric", choices=["total", "npu"], default="total",
                   help="по какому времени сравнивать: total (по умолчанию) или npu")
    p.add_argument("--export", default=None, help="выгрузить сводку в CSV")
    a = p.parse_args()

    if a.metric == "npu":
        METRIC["key"], METRIC["label"] = "_npu", "NPU_us"

    rows = load_matrix(a.matrix)
    base = load_numpy_baseline(a.baseline)

    n_pass = sum(1 for r in rows if r.get("status") == "PASS")
    n_fail = sum(1 for r in rows if r.get("status") == "FAIL")
    n_none = sum(1 for r in rows if r.get("status") == "NO_OUTPUT")
    print("===== FFTBench: сводка матрицы (метрика: %s) =====" % METRIC["label"])
    print("клеток: %d   PASS=%d  FAIL=%d  без результата=%d" % (len(rows), n_pass, n_fail, n_none))

    sizes = sorted({r["_N"] for r in rows})
    if a.size is not None:
        sizes = [s for s in sizes if s == a.size]
    scope = [r for r in rows if r["_N"] in sizes]

    # ---- 1. лучшее на вариант, по каждому N ------------------------------
    # При --axis печатаем ТОЛЬКО запрошенную ось: иначе интересующий блок
    # тонет между большой таблицей по N и обзорной таблицей, как и вышло на
    # первом же реальном запуске с --axis tile.
    # (--export при этом продолжает работать: строки собираются всегда,
    # подавляется только печать)
    focus = a.axis is not None
    export_rows = []
    for N in sizes:
        sub = [r for r in scope if r["_N"] == N]
        bp = best_per_variant(sub)
        bl = base.get(N)
        if not focus:
            print("")
            print("--- N=%d %s ---" % (N, ("(NumPy: %.2f us)" % bl) if bl else "(NumPy: нет данных)"))
        if not bp:
            if not focus:
                print("    ни одна клетка не прошла сверку")
            continue
        body = []
        for full, r in sorted(bp.items(), key=lambda kv: kv[1][METRIC["key"]]):
            vs = ("%.2fx" % (r["_total"] / bl)) if bl else "-"
            body.append([
                full, r.get("tile", "-"),
                r.get("cube_batch") if r.get("unit") == "cube" else "-",
                r.get("cores", "-"),
                "%.2f" % r["_npu"] if r["_npu"] is not None else "-",
                "%.2f" % r["_cpu"] if r["_cpu"] is not None else "-",
                "%.2f" % r["_total"], vs,
                r.get("stages_npu", "-"), r.get("stages_cpu", "-"),
                r.get("max_abs", "-"),
            ])
            export_rows.append([N, full, r.get("radix"), r.get("algo"),
                                r.get("granularity"), r.get("layout"), r.get("unit"),
                                r.get("tile"), r.get("cube_batch"), r.get("cores"),
                                r.get("npu_us"), r.get("cpu_us"), r.get("total_us"),
                                ("%.4f" % (r["_total"] / bl)) if bl else "",
                                r.get("max_abs")])
        if not focus:
            table(body, ["impl/variant", "tile", "B", "ядер", "NPU_us", "CPU_us",
                         "TOTAL_us", "vs_NumPy", "ст.NPU", "ст.CPU", "max_abs"])

    # ---- 2. парные сравнения по осям -------------------------------------
    print("")
    if not focus:
        print("=" * 76)
        print("ПАРНЫЕ СРАВНЕНИЯ ПО ОСЯМ — сравниваются клетки, различающиеся")
        print("РОВНО по одной оси (прочее зафиксировано). Решения принимать по ним.")
        print("Метрика: %s." % METRIC["label"])
        print("=" * 76)
    axes = [AXIS_BY_NAME[a.axis]] if a.axis else AXES
    if a.axis and a.axis not in AXIS_BY_NAME:
        print("[ERROR] неизвестная ось '%s'" % a.axis)
        sys.exit(1)
    for key, title, excl in axes:
        # terms меняет прежде всего точность, и смысл сравнения как раз в том,
        # что часть вариантов сверку не проходит — их нельзя отбрасывать.
        acc_axis = (key == "terms")
        res = paired_axis(scope, key, excl, a.all_pairs, allow_fail=acc_axis)
        print("")
        if res is None:
            print("--- ось: %s — парных сравнений нет" % title)
            print("    (нет двух клеток, различающихся только этой осью: например")
            print("     xi реализован лишь для radix-2, а tile/батч надо прогнать")
            print("     хотя бы в двух значениях — run_matrix.sh -t / -B)")
            continue
        win_str = ", ".join("%s: %d" % (v, c)
                            for v, c in sorted(res["wins"].items(), key=lambda kv: -kv[1]))
        if res["ratios"]:
            lo, hi = min(res["ratios"]), max(res["ratios"])
            if acc_axis:
                print("--- ось: %s --- сравнений: %d   выбор: %s   цена точности: %.2f–%.2fx"
                      % (title, len(res["lines"]), win_str or "-", lo, hi))
            else:
                print("--- ось: %s --- сравнений: %d   побед: %s   выигрыш: %.2f–%.2fx"
                      % (title, len(res["lines"]), win_str, lo, hi))
        else:
            print("--- ось: %s --- сравнений: %d (ни в одной группе нет PASS)"
                  % (title, len(res["lines"])))
        table(res["shown"], ["зафиксировано", "времена, us", "вывод"])
        if res["hidden"]:
            print("    ... ещё %d сравнений (--all-pairs, чтобы показать все)" % res["hidden"])

    # ---- 3. обзорная (НЕ парная) таблица ---------------------------------
    if not focus:
        print("")
        print("=" * 76)
        print("ОБЗОРНО (НЕ парно, метрика %s): лучшее время на каждое значение оси" % METRIC["label"])
        print("в матрице. Для выводов не годится — см. блок парных сравнений выше.")
        print("=" * 76)
        axis_body = []
        for key, title, _ in AXES:
            vals = sorted({r.get(key) for r in scope if r.get(key)},
                          key=lambda v: (len(str(v)), str(v)))
            for v in vals:
                cand = [r for r in scope if r.get(key) == v
                        and r.get("status") == "PASS" and r["_total"] is not None]
                if not cand:
                    continue
                b = min(cand, key=lambda r: r["_total"])
                axis_body.append([title, str(v), "%.2f" % b["_total"],
                                  "%s N=%s tile=%s" % (b["_full"], b.get("N"), b.get("tile"))])
        table(axis_body, ["ось", "значение", "лучшее " + METRIC["label"], "кем достигнуто"])

    # ---- 4. клетки без PASS ----------------------------------------------
    bad = [r for r in scope if r.get("status") in ("FAIL", "NO_OUTPUT")]
    if bad and not focus:
        print("")
        print("--- клетки без PASS ---")
        table([[r["_full"], r.get("N"), r.get("tile"),
                r.get("cube_batch") if r.get("unit") == "cube" else "-",
                r.get("status"), r.get("mismatches", "-"), r.get("max_abs", "-")]
               for r in bad],
              ["impl/variant", "N", "tile", "B", "status", "расхождений", "max_abs"])

    if a.export and export_rows:
        with open(a.export, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["N", "variant", "radix", "algo", "granularity", "layout", "unit",
                        "tile", "cube_batch", "cores", "npu_us", "cpu_us", "total_us",
                        "vs_numpy", "max_abs"])
            w.writerows(export_rows)
        print("")
        print("[INFO] сводка выгружена в %s" % a.export)


if __name__ == "__main__":
    main()