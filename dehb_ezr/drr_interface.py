#!/usr/bin/env python3
"""The interface to the DRR estimator: R, I and DRR = 1 - I/R per MOOT task.

    conda run -n drr python dehb_ezr/drr_interface.py \
        --root data/moot/optimize --out results/dehb_ezr/drr.csv

SOLE SOURCE
`drr`, the editable install of
  /Users/claudeli/NCSU/Research/26fa_aise/dimensionality_reduction_ratio
resolved from the active environment. drr_mine/ and drr_cal/ are gone and no
other estimator appears anywhere in this study.

WHAT IS FED IN
ds.df[ds.x_cols], not the raw csv. DataProcessor strips columns ending in
+ - !, but NOT the "X" suffix MOOT uses for "ignore this column" -- nasa93dem
has four (idX, centerX, YearX, MonthsX). Passing the whole frame would count
those as features and inflate R, the denominator of 1 - I/R. Dataset.load has
also already decided column types from the header's initial case and dropped
rows with missing values, so what is measured is the table the optimizers
actually search.

PROVENANCE, WHICH IS NOT OPTIONAL HERE
The estimator checkout carries UNCOMMITTED edits to
src/drr/intrinsic_dimension_estimator.py. These numbers depend on that working
tree, and a commit sha alone does not identify it. Every run therefore records
the version string, the sha, a dirty flag, and dumps `git diff` beside the csv.
Delete that patch file and the numbers become unreproducible.

KNOWN PROPERTY
DataProcessor applies no scaling and the default metric is l1. Where column
ranges differ by orders of magnitude -- SS-A's Spout_wait spans 1..10000 while
Spliters spans 1..6 -- the wide column dominates the distance, so I partly
reflects range rather than geometry. Left as-is: changing it would break
comparability with the upstream work. Recorded so the figures' axes are read
for what they are.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import subprocess
import sys
import traceback
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dehb_ezr.shared import Dataset, task_id      # noqa: E402

FIELDS = ["task_id", "task", "path", "group", "n_rows", "n_used",
          "R", "I", "drr", "seed", "drr_version", "drr_commit", "drr_dirty",
          "status"]


def provenance(module) -> dict:
    """Where drr came from and whether that checkout is dirty."""
    pkg = Path(module.__file__).resolve().parent
    repo = pkg.parent.parent                       # src/drr -> repo root

    def git(*args) -> str:
        try:
            return subprocess.run(["git", "-C", str(repo), *args],
                                  capture_output=True, text=True,
                                  check=True).stdout
        except (subprocess.CalledProcessError, FileNotFoundError):
            return ""

    return {
        "repo": repo,
        "drr_version": getattr(module, "__version__", "?"),
        "drr_commit": git("rev-parse", "--short", "HEAD").strip() or "?",
        "drr_dirty": bool(git("status", "--porcelain").strip()),
        "diff": git("diff"),
    }


def drr_for(drr, ds: Dataset, seed: int = 42, max_rows: int = 5000) -> dict:
    """(R, I, DRR) for one already-loaded table. Raises; the caller logs."""
    proc = drr.DataProcessor(max_rows_for_processing=max_rows, random_seed=seed)
    est = drr.IntrinsicDimensionEstimator()
    arr, _meta = proc.process_dataset(ds.df[ds.x_cols])
    if not proc.validate_processed_data(arr):
        raise ValueError("DataProcessor.validate_processed_data failed")
    R, I, ratio = est.estimate(arr)
    return dict(R=int(R), I=float(I), drr=float(ratio), n_used=int(arr.shape[0]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data/moot/optimize")
    ap.add_argument("--out", default="results/dehb_ezr/drr.csv")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-rows", type=int, default=5000)
    args = ap.parse_args()

    logging.basicConfig(level=logging.ERROR)
    warnings.filterwarnings("ignore")

    try:
        import drr
    except ModuleNotFoundError:
        sys.exit("no `drr` on the path -- run under the 'drr' conda env")

    prov = provenance(drr)
    print(f"drr {prov['drr_version']} @ {prov['drr_commit']}"
          f"{' (DIRTY)' if prov['drr_dirty'] else ''}\n  from {prov['repo']}\n")

    root = Path(args.root)
    paths = sorted(root.rglob("*.csv"))
    if not paths:
        sys.exit(f"no csv under {root}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if prov["diff"]:
        patch = out.with_name("drr_estimator.patch")
        patch.write_text(prov["diff"])
        print(f"wrote {patch} -- these numbers depend on it\n")

    rows, failed = [], []
    for p in paths:
        row = {f: "" for f in FIELDS}
        row.update(task_id=task_id(p, root), task=p.stem,
                   path=os.path.relpath(p), group=p.parent.name,
                   seed=args.seed, drr_version=prov["drr_version"],
                   drr_commit=prov["drr_commit"], drr_dirty=prov["drr_dirty"])
        try:
            ds = Dataset.load(str(p))
            row["n_rows"] = len(ds.df)
            row.update(drr_for(drr, ds, seed=args.seed, max_rows=args.max_rows))
            row["status"] = "ok"
            print(f"  {row['task_id']:<40} R={row['R']:<5} I={row['I']:<8.3f} "
                  f"DRR={row['drr']:.4f}")
        except Exception as e:      # skip-and-log: one bad table must not stop the sweep
            row["status"] = f"error: {type(e).__name__}: {e}"
            failed.append((row["task_id"], e))
            print(f"  {row['task_id']:<40} FAILED: {type(e).__name__}: {e}")
        rows.append(row)

    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    ok = sum(r["status"] == "ok" for r in rows)
    print(f"\nwrote {out}: {ok}/{len(rows)} tasks estimated")

    for tid, e in failed:
        print(f"\n--- {tid} ---", file=sys.stderr)
        traceback.print_exception(type(e), e, e.__traceback__)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
