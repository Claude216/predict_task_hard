"""Differential test: stock ezr.acquire() vs the harness's EZR adapter.

Run from the project directory:
    EZR_PATH=/path/to/ezr.py python check_ezr.py ../data/moot/optimize/config/SS-A.csv

WHY TWO TESTS

The two paths do not use the same yardstick, by design:

  stock EZR  reads each row's TRUE y values from the table and scores with its
             own disty() -- a LOGISTIC normalisation of every goal about that
             goal's running mean.
  harness    scores with the frozen RF oracle and the paper's Eq.1, a MIN-MAX
             normalisation, collapsed into one synthetic "D2h-" column.

So a bare comparison only rediscovers that these differ. Test 1 removes the
difference by giving BOTH paths the same scalar -- the true Eq.1 d2h computed
from the table's own y values, no surrogate -- so any divergence left is the
adapter's fault. Test 2 then puts the surrogate back and measures how much it
moves EZR.

TEST 1 MUST PASS EXACTLY. It is the regression test for the warm-start
pre-labelling fix: acquire() clones the starting rows BEFORE label() runs, so
under lazy labelling the D2h column is cloned with n=0, sd=0, norm() saturates
every value to one number, warm_start's sort degenerates and best/rest becomes
arbitrary. If that fix regresses, Test 1 diverges within the first few
acquisitions while Test 2 still looks superficially reasonable.
"""

from __future__ import annotations

import math
import random
import sys

import numpy as np

from data import Dataset
from oracle import BudgetGate, Oracle
from optimizers.ezr_opt import EZR, ezr, ezr_options, _native

START = 4          # Table IV: init labels 4


# --------------------------------------------------------------------------- #
def true_d2h_table(ds: Dataset) -> np.ndarray:
    """Eq.1 computed from the table's REAL y values. Same formula as
    Oracle.d2h, with the surrogate taken out of the loop."""
    bounds = {y: (float(ds.df[y].min()), float(ds.df[y].max())) for y in ds.y_cols}
    cols = []
    for y in ds.y_cols:
        lo, hi = bounds[y]
        raw = ds.df[y].to_numpy(dtype=float)
        f = np.zeros_like(raw) if hi == lo else (raw - lo) / (hi - lo)
        cols.append(f if ds.goals[y] == "min" else 1.0 - f)
    return np.sqrt(np.mean(np.square(np.column_stack(cols)), axis=1))


def key_of(ds: Dataset, row: dict) -> tuple:
    return tuple(_native(row[c]) for c in ds.x_cols)


class TrueOracle:
    """Duck-types Oracle for BudgetGate: exact table lookup, no RF.

    Rows that repeat in x-space but differ in y keep their FIRST d2h, which is
    only reached if the table has duplicate configurations; the count is
    reported so a silent collision cannot be mistaken for adapter drift."""

    def __init__(self, ds: Dataset, truth: np.ndarray):
        self.ds = ds
        self.map: dict[tuple, float] = {}
        self.collisions = 0
        for row, v in zip(ds.pool, truth):
            k = key_of(ds, row)
            if k in self.map:
                self.collisions += 1
            else:
                self.map[k] = float(v)

    def d2h(self, row: dict) -> float:
        return self.map[key_of(self.ds, row)]


# --------------------------------------------------------------------------- #
def stock_run(ds: Dataset, oracle, budget: int, few: int, seed: int):
    """oracle supplies the scalar for BOTH paths, so duplicate x-rows cannot
    diverge. The real RF oracle collapses duplicates the same way."""
    header = [*ds.x_cols, "D2h-"]
    rows = [[_native(r[c]) for c in ds.x_cols] + [oracle.d2h(r)]
            for r in ds.pool]

    order: list[tuple] = []

    def label(_data, row):
        order.append(tuple(row[:-1]))
        return row

    with ezr_options(**{"p": 2, "few": min(few, len(rows)),
                        "learn__start": START,
                        "learn__budget": max(0, budget - START)}):
        random.seed(seed)
        data = ezr.Data([header] + rows)
        lab = ezr.acquire(data, score=ezr.acquireWithCentroid, label=label)

    return order, min(r[-1] for r in lab.rows)


def harness_run(ds: Dataset, oracle, budget: int, few: int, seed: int):
    gate = BudgetGate(oracle, budget)
    best_cfg = EZR(start=START, few=few).run(ds, gate, seed)
    order = [key_of(ds, r) for r, _ in gate.trace]
    return order, gate.best, gate.used, best_cfg


# --------------------------------------------------------------------------- #
def test1(ds: Dataset, truth: np.ndarray, budgets, seeds) -> bool:
    probe = TrueOracle(ds, truth)
    print("=" * 72)
    print("TEST 1  same yardstick (true Eq.1, no surrogate) -- must be identical")
    print("=" * 72)
    ok_all = True
    for B in budgets:
        few = max(128, 2 * B)
        for s in seeds:
            o_a, best_a = stock_run(ds, probe, B, few, s)
            o_b, best_b, used, _ = harness_run(ds, probe, B, few, s)

            same_len = len(o_a) == len(o_b) == B == used
            first_bad = next((i for i, (a, b) in enumerate(zip(o_a, o_b)) if a != b),
                             None)
            same_order = first_bad is None and same_len
            same_best = math.isclose(best_a, best_b, rel_tol=0, abs_tol=1e-12)
            ok = same_order and same_best
            ok_all &= ok

            note = "OK" if ok else (
                f"FAIL diverges at acquisition {first_bad}"
                if first_bad is not None else
                f"FAIL len {len(o_a)}/{len(o_b)}/used={used}"
                if not same_len else "FAIL best differs")
            print(f"  B={B:<4} seed={s:<3} n={len(o_a):<4} "
                  f"stock={best_a:.6f} harness={best_b:.6f}  {note}")
    print(f"\nTEST 1: {'PASS' if ok_all else 'FAIL'}")
    return ok_all


def test2(ds: Dataset, truth: np.ndarray, budgets, seeds) -> None:
    print()
    print("=" * 72)
    print("TEST 2  surrogate put back -- expected to differ, reported not asserted")
    print("=" * 72)
    probe = TrueOracle(ds, truth)
    oracle = Oracle(ds)
    lookup = {key_of(ds, r): float(v) for r, v in zip(ds.pool, truth)}
    print(f"  {'B':>4} {'seed':>5} {'stock(true)':>12} {'harness(true)':>14} "
          f"{'harness(oracle)':>16} {'overlap':>8}")
    for B in budgets:
        few = max(128, 2 * B)
        for s in seeds:
            o_a, best_a = stock_run(ds, probe, B, few, s)
            o_b, best_orc, _, cfg = harness_run(ds, oracle, B, few, s)
            # both bests re-measured on the TRUE scale, so the columns compare
            best_b_true = lookup[key_of(ds, cfg)]
            overlap = len(set(o_a) & set(o_b)) / max(1, len(o_a))
            print(f"  {B:>4} {s:>5} {best_a:>12.6f} {best_b_true:>14.6f} "
                  f"{best_orc:>16.6f} {overlap:>7.0%}")
    print("\n  stock(true)     : best true d2h among rows stock EZR labelled")
    print("  harness(true)   : true d2h of the config the RF-driven run returned")
    print("  harness(oracle) : what the run itself saw; gap to harness(true) is")
    print("                    surrogate error, and it is NOT a bug")
    print("  overlap         : fraction of labelled rows the two paths share")


# --------------------------------------------------------------------------- #
def main():
    if len(sys.argv) < 2:
        sys.exit("usage: python check_ezr.py <moot.csv> [budget ...]")
    path = sys.argv[1]
    budgets = [int(x) for x in sys.argv[2:]] or [30, 50]
    seeds = list(range(3))

    ds = Dataset.load(path)
    truth = true_d2h_table(ds)
    probe = TrueOracle(ds, truth)
    print(f"task {path}: rows={len(ds.df)} x={len(ds.x_cols)} goals={ds.goals}")
    print(f"true Eq.1 d2h over table: best={truth.min():.6f} "
          f"median={np.median(truth):.6f}")
    if probe.collisions:
        print(f"WARNING {probe.collisions} rows repeat in x-space; Test 1 keeps "
              f"the first d2h for each and a mismatch may not be adapter drift")
    print(f"ezr loaded from {ezr.__file__}\n")

    if test1(ds, truth, budgets, seeds):
        test2(ds, truth, budgets, seeds)
    else:
        print("\nskipping TEST 2: fix the adapter first, its numbers would be "
              "uninterpretable")


if __name__ == "__main__":
    main()