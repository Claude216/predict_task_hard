"""Sweep data/moot/optimize/**/*.csv through drr_mine/intrinsic_dim.py.

Writes drr_mine/results/moot_drr.csv (one row per task) and appends failures
to drr_mine/results/errors.log. Skip-and-log, never crash.

Usage (drr conda env has numpy/pandas):
    conda run -n drr python drr_mine/run_moot_drr.py [--seed 42] [--p 2]
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from intrinsic_dim import estimate_intrinsic_dim  # noqa: E402

FIELDS = ["task", "path", "n_rows", "n_dedup", "dup_row_frac", "n_used", "R",
          "I_fit", "n_live", "flat_frac", "fit_argmax_frac", "slope_iqr",
          "I_maxgrad", "drr_fit", "drr_maxgrad", "dup_pair_frac", "seed",
          "runtime_s", "status"]

# the local slope should settle inside a real scaling region; a spread this
# wide across the live pieces means it never did (replaces the old fit r2)
HIGH_SPREAD_IQR = 1.0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=os.path.join(REPO_ROOT, "data", "moot", "optimize"))
    ap.add_argument("--out", default=os.path.join(HERE, "results", "moot_drr.csv"))
    ap.add_argument("--errors", default=os.path.join(HERE, "results", "errors.log"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-rows", type=int, default=2000)
    ap.add_argument("--p", type=float, default=2, help="Minkowski p (ezr the.p default 2)")
    ap.add_argument("--norm", default="ezr", choices=["ezr", "minmax"])
    ap.add_argument("--num-radii", type=int, default=100)
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.data, "**", "*.csv"), recursive=True))
    if not paths:
        sys.exit(f"no csvs under {args.data}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    rows, errors = [], []

    for path in paths:
        task = os.path.relpath(path, args.data)
        t0 = time.time()
        try:
            res = estimate_intrinsic_dim(path, seed=args.seed, max_rows=args.max_rows,
                                         p=args.p, num_radii=args.num_radii,
                                         norm=args.norm)
        except Exception as e:  # skip-and-log, never crash the sweep
            res = {f: "" for f in FIELDS}
            res["status"] = f"error: {type(e).__name__}: {e}"
            errors.append(f"{task}: {type(e).__name__}: {e}\n{traceback.format_exc()}")
        row = {"task": task, "path": os.path.relpath(path, REPO_ROOT),
               "runtime_s": round(time.time() - t0, 2), **res}
        rows.append(row)
        if row["status"] != "ok":
            errors.append(f"{task}: {row['status']}")

        iqr = row.get("slope_iqr")
        flag = ""
        if isinstance(iqr, float) and iqr == iqr and iqr > HIGH_SPREAD_IQR:
            flag = f"  [low trust: slope_iqr={iqr:.2f}]"
        ifit = row.get("I_fit")
        dfit = row.get("drr_fit")
        print(f"{task:55s} R={row.get('R', ''):<4} "
              f"I_fit={ifit if not isinstance(ifit, float) else round(ifit, 2):<7} "
              f"DRR={dfit if not isinstance(dfit, float) else round(dfit, 3):<7} "
              f"status={row['status']}{flag}")

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v)
                        for k, v in row.items()})

    if errors:
        with open(args.errors, "a") as f:
            for line in errors:
                f.write(line + "\n")

    ok = sum(1 for r in rows if r["status"] == "ok")
    print(f"\n{ok}/{len(rows)} tasks ok -> {os.path.relpath(args.out, os.getcwd())}")
    print(f"settings: seed={args.seed} max_rows={args.max_rows} p={args.p} "
          f"norm={args.norm} num_radii={args.num_radii}")
    if errors:
        print(f"{len(errors)} problem(s) logged to {os.path.relpath(args.errors, os.getcwd())}")


if __name__ == "__main__":
    main()
