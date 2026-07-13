"""Stage 1 sanity checks (CLAUDE.md):
  - eff_rank_x <= n_x
  - A5-A7 finite on tasks with Sym columns
  - one row per MOOT task; report NaN counts per feature
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
df = pd.read_csv(ROOT / "results" / "features.csv")
gt = pd.read_csv(ROOT / "results" / "ground_truth.csv")

print(f"tasks in features.csv: {len(df)}; in ground_truth.csv: {len(gt)}")
missing = set(gt.task) ^ set(df.task)
print(f"task-name mismatches: {sorted(missing) or 'none'}")

print("\nNaN count per column:")
print(df.isna().sum().to_string())

bad = df[df.eff_rank_x > df.n_x]
print(f"\neff_rank_x > n_x violations: {len(bad)}")
if len(bad):
  print(bad[["task", "n_x", "eff_rank_x", "sym_ratio"]].to_string(index=False))

sym = df[df.sym_ratio > 0]
print(f"\ntasks with Sym X columns: {len(sym)}")
for c in ["eff_rank_x", "cca_1", "linear_r2"]:
  n = sym[c].isna().sum()
  print(f"  {c}: {n} NaN on Sym tasks")

print("\nfeature summary:")
print(df.describe().T[["min", "50%", "max"]].to_string())
sys.exit(0)
