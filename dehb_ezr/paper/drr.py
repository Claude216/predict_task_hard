#!/usr/bin/env python3
"""DRR for the 49 Table 1 datasets, using the paper's R rule.

    conda run -n drr python -m dehb_ezr.paper.drr

R = every column MINUS every dependent (`+ - !`). MOOT's `X`-suffix "ignore"
columns are therefore KEPT, which is what reproduces Table 2 for xomo (27) and
adult (14). This differs from results/dehb_ezr/drr.csv, which used ezr's x_cols
and so dropped them -- hence a separate sweep rather than a re-use.

The frame handed to the estimator is exactly the feature matrix the RandomForest
is trained on (loader.encode), so the x-axis and the optimization describe the
same data, including the 5,000-row cap.

PROVENANCE. The estimator checkout carries uncommitted edits; a commit sha alone
does not identify these numbers. The version, sha, dirty flag and `git diff` are
all recorded. Delete the patch file and the numbers stop being reproducible.
"""

from __future__ import annotations

import argparse
import csv
import logging
import subprocess
import sys
import traceback
import warnings
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    __package__ = "dehb_ezr.paper"

from .datasets import DATASETS, task_id
from .loader import load

FIELDS = ["task_id", "label", "path", "is_se", "kind", "R", "paper_R", "I", "drr",
          "n_rows", "n_used", "seed", "drr_version", "drr_commit", "drr_dirty", "status"]
OUT = Path(__file__).resolve().parent.parent / "paper_figs" / "drr.csv"


def provenance(module) -> dict:
    pkg = Path(module.__file__).resolve().parent
    repo = pkg.parent.parent

    def git(*a) -> str:
        try:
            return subprocess.run(["git", "-C", str(repo), *a], capture_output=True,
                                  text=True, check=True).stdout
        except (subprocess.CalledProcessError, FileNotFoundError):
            return ""

    return {"repo": repo, "drr_version": getattr(module, "__version__", "?"),
            "drr_commit": git("rev-parse", "--short", "HEAD").strip() or "?",
            "drr_dirty": bool(git("status", "--porcelain").strip()),
            "diff": git("diff")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-rows", type=int, default=5000)
    args = ap.parse_args()

    logging.basicConfig(level=logging.ERROR)
    warnings.filterwarnings("ignore")
    try:
        import drr
    except ModuleNotFoundError:
        sys.exit("no `drr` on the path -- run under the 'drr' conda env")

    import pandas as pd

    prov = provenance(drr)
    print(f"drr {prov['drr_version']} @ {prov['drr_commit']}"
          f"{' (DIRTY)' if prov['drr_dirty'] else ''}\n  from {prov['repo']}\n")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if prov["diff"]:
        patch = out.with_name("drr_estimator.patch")
        patch.write_text(prov["diff"])
        print(f"wrote {patch} -- these numbers depend on it\n")

    rows, failed = [], []
    for e in DATASETS:
        row = {f: "" for f in FIELDS}
        row.update(task_id=task_id(e.label), label=e.label, path=e.path,
                   is_se=e.is_se, paper_R=e.paper_R, seed=args.seed,
                   drr_version=prov["drr_version"], drr_commit=prov["drr_commit"],
                   drr_dirty=prov["drr_dirty"])
        try:
            t = load(e, seed=args.seed, row_cap=args.max_rows)
            row.update(kind=t.kind, R=t.R, n_rows=len(t.y))
            frame = pd.DataFrame(t.X, columns=t.features)
            proc = drr.DataProcessor(max_rows_for_processing=args.max_rows,
                                     random_seed=args.seed)
            est = drr.IntrinsicDimensionEstimator()
            arr, _ = proc.process_dataset(frame)
            R_est, I, ratio = est.estimate(arr)
            row.update(I=float(I), drr=float(ratio), n_used=int(arr.shape[0]),
                       status="ok")
            flag = "" if R_est == t.R else f"  (estimator saw R={R_est})"
            note = "" if e.paper_R in (None, t.R) else f"  [paper R={e.paper_R}]"
            print(f"  {e.label:<22} R={t.R:<5} I={I:<6.2f} DRR={ratio:.4f}{flag}{note}")
        except Exception as ex:      # skip-and-log
            row["status"] = f"error: {type(ex).__name__}: {ex}"
            failed.append((e.label, ex))
            print(f"  {e.label:<22} FAILED: {type(ex).__name__}: {ex}")
        rows.append(row)

    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    ok = sum(r["status"] == "ok" for r in rows)
    print(f"\nwrote {out}: {ok}/{len(rows)} estimated")
    for label, ex in failed:
        print(f"\n--- {label} ---", file=sys.stderr)
        traceback.print_exception(type(ex), ex, ex.__traceback__)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
