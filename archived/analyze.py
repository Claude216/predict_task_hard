"""Stage 4: correlate meta-features against ground truth.

Outputs
  results/analysis_spearman.csv : feature x {score, stability} Spearman
      rho, raw p, Holm-adjusted p (step-down, corrected within each target's
      family of tests), and confirmatory/exploratory flag.
  results/analysis_tree.txt     : depth-<=3 decision tree predicting score
      from all features, plus leave-one-task-out (LOO) validation summary.

Statistical choices (fixed by CLAUDE.md):
  - Spearman only, never Pearson.
  - Holm multiple-comparison control. Pre-registered confirmatory features:
    B1 fdc, B5 d2h_tail_gap, C1 nb_auc (nb_auc_med here); everything else
    is exploratory and flagged as such.
  - Tree depth <= 3, LOO validation, report LOO Spearman of predictions.

Choices the spec left open (stated, not silent):
  - Holm families: one family per target = all 17 feature columns tested
    against that target (17 tests per family).
  - Tree NaN handling: median imputation, with the median computed inside
    each LOO training fold (no leakage). Tree seed fixed at 1.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.tree import DecisionTreeRegressor, export_text

ROOT = Path(__file__).resolve().parent.parent
CONFIRMATORY = {"fdc", "d2h_tail_gap", "nb_auc_med"}
TREE_SEED = 1


def holm(pvals):
  """Holm step-down adjusted p-values (monotone, clipped at 1)."""
  p = np.asarray(pvals, float)
  m = len(p)
  order = np.argsort(p)
  adj = np.empty(m)
  running = 0.0
  for rank, i in enumerate(order):
    running = max(running, (m - rank) * p[i])
    adj[i] = min(1.0, running)
  return adj


def spearman_table(df, feats):
  rows = []
  for target in ["score", "stability"]:
    fam = []
    for c in feats:
      sub = df[[c, target]].dropna()
      rho, p = spearmanr(sub[c], sub[target])
      fam.append({"target": target, "feature": c, "n": len(sub),
                  "spearman_rho": rho, "p_raw": p,
                  "kind": ("confirmatory" if c in CONFIRMATORY
                           else "exploratory")})
    adj = holm([r["p_raw"] for r in fam])
    for r, a in zip(fam, adj):
      r["p_holm"] = a
    rows += fam
  out = pd.DataFrame(rows)[["target", "feature", "kind", "n",
                            "spearman_rho", "p_raw", "p_holm"]]
  return out.sort_values(["target", "p_holm"]).reset_index(drop=True)


def loo_tree(df, feats):
  """Depth-<=3 tree, leave-one-task-out; fold-internal median imputation."""
  x_all = df[feats].to_numpy(float)
  y = df["score"].to_numpy(float)
  preds = np.empty(len(df))
  for i in range(len(df)):
    mask = np.ones(len(df), bool)
    mask[i] = False
    xtr = x_all[mask].copy()
    med = np.nanmedian(xtr, axis=0)
    xtr = np.where(np.isnan(xtr), med, xtr)
    xte = np.where(np.isnan(x_all[i]), med, x_all[i])
    tree = DecisionTreeRegressor(max_depth=3, random_state=TREE_SEED)
    tree.fit(xtr, y[mask])
    preds[i] = tree.predict(xte.reshape(1, -1))[0]
  rho, p = spearmanr(preds, y)
  # one tree on all data, for inspection only (not a validation result)
  med = np.nanmedian(x_all, axis=0)
  xfull = np.where(np.isnan(x_all), med, x_all)
  full = DecisionTreeRegressor(max_depth=3, random_state=TREE_SEED)
  full.fit(xfull, y)
  return preds, rho, p, export_text(full, feature_names=feats)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--ground-truth", default="results/ground_truth.csv",
                  help="ground truth csv (task, score, stability)")
  ap.add_argument("--suffix", default="",
                  help="suffix for output files, e.g. _v2")
  args = ap.parse_args()

  f = pd.read_csv(ROOT / "results" / "features.csv")
  g = pd.read_csv(ROOT / args.ground_truth)
  df = f.merge(g, on="task", validate="one_to_one")
  feats = [c for c in f.columns if c not in ("task", "labels_used_total")]

  tab = spearman_table(df, feats)
  tab.to_csv(ROOT / "results" / f"analysis_spearman{args.suffix}.csv",
             index=False)
  pd.set_option("display.width", 120)
  print("== Spearman vs ground truth (Holm-corrected within target) ==")
  print(tab.round(4).to_string(index=False))

  preds, rho, p, tree_txt = loo_tree(df, feats)
  report = (f"ground truth: {args.ground_truth}\n"
            f"LOO (n={len(df)}) depth<=3 tree predicting score\n"
            f"LOO Spearman(pred, true) rho={rho:.4f} p={p:.3e}\n\n"
            f"Tree fit on ALL tasks (inspection only):\n{tree_txt}")
  (ROOT / "results" / f"analysis_tree{args.suffix}.txt").write_text(report)
  print("\n== " + report)


if __name__ == "__main__":
  main()
