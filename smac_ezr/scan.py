"""Cheap profile of every MOOT table. No oracle, no optimizer, no RF.

Run this BEFORE committing to a long batch. It answers three things:

  * which files parse at all, and which blow up (catch this now, not 8 hours in)
  * whether the exclusion rules select the paper's ~106 SE tasks or all 127
  * which tables sit BETWEEN the two extremes we have measured so far

On coverage = log2(distinct x-rows) - sum log2|Xi|:
    SS-A       ~0    the table is close to a full factorial, so the RF oracle
                     interpolates and stock-vs-harness EZR overlapped 87-100%
    nasa93dem  ~-18  93 rows over 22 variables, so the RF extrapolates nearly
                     everywhere and that overlap fell to 10-77%
A mid-coverage table is the missing third data point.

On x_dup_rate: the fraction of rows whose decision values repeat. Where this is
high, y is not a function of x, so no oracle can be accurate -- the error is
irreducible rather than a tuning problem.

Usage:
    python scan.py ../data/moot/optimize --out scan.csv
"""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from data import Dataset

FIELDS = ["task_id", "rows", "n_decisions", "n_goals", "objective",
          "space_size", "coverage", "x_dup_rate", "input_shape"]


def task_id(path: Path, root: Path) -> str:
    return str(path.relative_to(root).with_suffix("")).replace("/", "__")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--out", default=None, help="write the profile as CSV")
    ap.add_argument("--exclude-dir", nargs="*",
                    default=["sales", "finance", "misc", "hpo"])
    args = ap.parse_args()

    root = Path(args.root).resolve()
    excl = {d.lower() for d in args.exclude_dir}

    rows, skipped, failed = [], [], []
    for p in sorted(root.rglob("*.csv")):
        tid = task_id(p, root)
        if {part.lower() for part in p.relative_to(root).parts[:-1]} & excl:
            skipped.append(tid)
            continue
        try:
            d = Dataset.load(str(p)).describe()
            d["task_id"] = tid
            rows.append({k: d[k] for k in FIELDS})
        except Exception as exc:
            failed.append((tid, f"{type(exc).__name__}: {exc}"))

    df = pd.DataFrame(rows).sort_values("coverage", ascending=False)

    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(df.to_string(index=False))

    print(f"\nselected={len(df)}  excluded_by_dir={len(skipped)}  "
          f"failed={len(failed)}")
    if failed:
        print("failed to parse:")
        for tid, err in failed:
            print(f"  {tid}: {err}")

    print("\ncoverage terciles -- pick one task from each for the density probe:")
    d = df.sort_values("coverage", ascending=False).reset_index(drop=True)
    n = len(d)
    if n >= 3:
        for name, frac in [("dense", 1/6), ("mid", 1/2), ("sparse", 5/6)]:
            r = d.iloc[int(n * frac)]
            print(f"  {name:<7} {r.task_id:<30} coverage={r.coverage:>8} "
                  f"rows={r.rows:>7} x={r.n_decisions:>4} dup={r.x_dup_rate}")

    print("\nhighest x_dup_rate -- oracle cannot be trusted on these:")
    for _, r in df.nlargest(5, "x_dup_rate").iterrows():
        print(f"  {r.task_id:<28} dup={r.x_dup_rate:<8} rows={r.rows}")

    if args.out:
        df.to_csv(args.out, index=False)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()