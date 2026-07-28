"""Compute the Dimensionality Reduction Ratio for every MOOT table, once.

    DRR_SRC=../external/dimensionality_reduction_ratio/src \
    python compute_drr.py ../data/moot/optimize --out drr.csv

DRR is a per-task constant, it is not cheap (pairwise distances over up to 2000
rows), and the package that computes it is a pinned upstream plus local fixes.
So it is computed offline into a CSV that analysis scripts join against, rather
than imported at plot time. The CSV is a reviewable artifact; recomputing on
every plot and hoping for the same numbers is not.

WHAT IS FED IN

ds.df[ds.x_cols], NOT ds.df. DataProcessor strips columns ending in + - !, but
NOT the "X" suffix that MOOT uses for "ignore this column" -- nasa93dem has
four of them (idX, centerX, YearX, MonthsX). Passing the whole frame would
count those as features and inflate R, the denominator of 1 - I/R.

Our Dataset.load has already decided column types from the header's initial
case and dropped rows with missing values, so what is measured here is the same
table the optimizers actually search.

KNOWN PROPERTY OF THESE NUMBERS

DataProcessor applies no scaling, and the distance metric is l1. Where column
ranges differ by orders of magnitude -- SS-A's Spout_wait spans 1-10000 while
Spliters spans 1-6 -- the wide column dominates the distance, so the estimated
intrinsic dimension partly reflects range rather than geometry. Kept as-is
deliberately: DRR is an estimator, not a claim about true geometry, and
changing it here would break comparability with the upstream work. Recorded so
that the x-axis of any DRR plot is read for what it is.

SAMPLING

Two stages, both seeded: DataProcessor takes 5000 rows if the table is larger,
then IntrinsicDimensionEstimator takes 2000 of those with a hard-coded
default_rng(42). Reproducible, which is not the same as stable: x264 is
estimated from 1.2% of its rows. --seeds re-runs the first stage under other
seeds to measure the spread. Tables of 2000-5000 rows cannot be probed this way
-- only the estimator's fixed-seed stage samples them -- so they report nan.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from data import Dataset


def load_drr(src: str | None):
    """Put the DRR package on sys.path here and nowhere else: it ships
    top-level-ish module names (data_processor, main) and is a pinned upstream
    carrying local patches, so its scope is kept to this one script."""
    src = src or os.environ.get("DRR_SRC")
    if not src:
        raise SystemExit("set DRR_SRC (or pass --drr-src) to the directory "
                         "containing the 'drr' package")
    src = str(Path(src).expanduser().resolve())
    if not (Path(src) / "drr").is_dir():
        raise SystemExit(f"no 'drr' package under {src}")
    sys.path.insert(0, src)
    import drr
    return drr


def one_task(drr, ds: Dataset, seed: int) -> dict:
    proc = drr.DataProcessor(random_seed=seed)
    est = drr.IntrinsicDimensionEstimator()
    frame = ds.df[ds.x_cols]                 # X-suffix columns already excluded
    arr, meta = proc.process_dataset(frame)
    R, I, ratio = est.estimate(arr)
    return dict(R=int(R), I=float(I), drr=float(ratio),
                sampling_applied=bool(meta.get("sampling_applied", False)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--out", default="drr.csv")
    ap.add_argument("--drr-src", default=None)
    ap.add_argument("--seeds", type=int, nargs="*", default=[42],
                    help="first is the reported value; extras measure spread")
    ap.add_argument("--quiet", action="store_true", default=True)
    args = ap.parse_args()

    if args.quiet:
        logging.getLogger("drr").setLevel(logging.ERROR)
        logging.basicConfig(level=logging.ERROR)

    drr = load_drr(args.drr_src)
    version = getattr(drr, "__version__", "unknown")
    print(f"drr package {version} from {Path(drr.__file__).parent}")
    print(f"seeds={args.seeds}  (first is reported, rest measure spread)\n")

    root = Path(args.root).resolve()
    files = sorted(root.rglob("*.csv"))
    print(f"found {len(files)} tables\n")

    rows, failed = [], []
    for n, p in enumerate(files, 1):
        tid = str(p.relative_to(root).with_suffix("")).replace("/", "__")
        try:
            ds = Dataset.load(str(p))
            base = one_task(drr, ds, args.seeds[0])

            spread = np.nan
            if base["sampling_applied"] and len(args.seeds) > 1:
                vals = [base["drr"]] + [one_task(drr, ds, s)["drr"]
                                        for s in args.seeds[1:]]
                spread = float(max(vals) - min(vals))

            rows.append(dict(
                task_id=tid, dir=p.parent.name, rows=len(ds.df),
                n_decisions=len(ds.x_cols), n_goals=len(ds.goals),
                drr_spread=spread, drr_version=version, drr_seed=args.seeds[0],
                **base))
            print(f"[{n}/{len(files)}] {tid}: R={base['R']} "
                  f"I={base['I']:.2f} DRR={base['drr']:.3f}"
                  + ("" if np.isnan(spread) else f"  spread={spread:.3f}"),
                  flush=True)
        except Exception as exc:
            failed.append((tid, f"{type(exc).__name__}: {exc}"))
            print(f"[{n}/{len(files)}] {tid}: FAILED {type(exc).__name__}: {exc}",
                  flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"\nwrote {len(df)} rows -> {args.out}   failed={len(failed)}")
    for tid, err in failed:
        print(f"  {tid}: {err}")

    if len(df):
        print(f"\nDRR distribution: min={df.drr.min():.3f} "
              f"q25={df.drr.quantile(.25):.3f} median={df.drr.median():.3f} "
              f"q75={df.drr.quantile(.75):.3f} max={df.drr.max():.3f}")
        sub = df.dropna(subset=["drr_spread"])
        if len(sub):
            print(f"seed spread on the {len(sub)} sampled tables: "
                  f"median={sub.drr_spread.median():.3f} "
                  f"max={sub.drr_spread.max():.3f}")
            worst = sub.nlargest(3, "drr_spread")
            for _, r in worst.iterrows():
                print(f"  {r.task_id}: DRR={r.drr:.3f} +/- {r.drr_spread:.3f} "
                      f"over {r.rows} rows")
            print("If the spread is comparable to the gaps you plan to read off"
                  " the x-axis, the DRR plot needs error bars.")


if __name__ == "__main__":
    main()