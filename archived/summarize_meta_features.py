"""
Aggregate the 127 per-dataset files from results/meta_features/ (made by
analysis/per_dataset_meta.py) into:

1. analysis/meta_features_summary.csv -- one row per dataset:
     dataset_name, mean_feature_var, mean_target_var, mean_pairwise_cor,
     stability_score, performance_score
   where mean_* average that file's x_variation / y_variation / correlation
   rows (undefined correlations -- pairs with a constant column, stored as
   empty values -- are excluded), stability_score is the EZR default-settings
   stability agreement, and performance_score is the EZR default-settings
   RMSE. (The files' extra jaccard_mean stability measure is carried along as
   a supplementary column, stability_jaccard.)

2. analysis/meta_features_score_correlations.csv -- Pearson and Spearman
   correlation (with p-values) of each of the 3 means against
   stability_score and performance_score (plus supplementary rows vs
   stability_jaccard), over the 127 datasets.

Run: /Users/claudeli/Tool/miniconda3/envs/caus26/bin/python analysis/summarize_meta_features.py
"""
import glob
import os

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

REPO = "/Users/claudeli/NCSU/Research/26sp_aise/stability_evaluation"
IN_DIR = os.path.join(REPO, "results/meta_features")

files = sorted(glob.glob(os.path.join(IN_DIR, "*.csv")))
assert len(files) == 127, f"expected 127 files, got {len(files)}"

rows = []
for path in files:
    d = pd.read_csv(path)
    d["value"] = pd.to_numeric(d["value"], errors="coerce")
    by = d.groupby("kind")["value"].mean()  # NaN (undefined pairs) excluded
    score = d.set_index(["kind", "name"])["value"]
    rows.append({
        "dataset_name": os.path.basename(path)[:-4],
        "mean_feature_var": by["x_variation"],
        "mean_target_var": by["y_variation"],
        "mean_pairwise_cor": by.get("correlation", np.nan),
        "stability_score": score[("stability", "agreement")],
        "performance_score": score[("performance", "rmse")],
        "stability_jaccard": score[("stability", "jaccard_mean")],
    })

summary = pd.DataFrame(rows)
summary_path = os.path.join(REPO, "analysis/meta_features_summary.csv")
summary.to_csv(summary_path, index=False)
print(f"wrote {len(summary)} rows to {summary_path}")
print("NaN counts:", summary.isna().sum().sum())

means = ["mean_feature_var", "mean_target_var", "mean_pairwise_cor"]
targets = ["stability_score", "performance_score", "stability_jaccard"]
out = []
for target in targets:
    for m in means:
        ok = summary[[m, target]].dropna()
        pr, pp = pearsonr(ok[m], ok[target])
        sr, sp = spearmanr(ok[m], ok[target])
        out.append({"target": target, "meta_feature_mean": m, "n": len(ok),
                    "pearson_r": round(pr, 4), "pearson_p": round(pp, 6),
                    "spearman_r": round(sr, 4), "spearman_p": round(sp, 6)})

corr = pd.DataFrame(out)
corr_path = os.path.join(REPO, "analysis/meta_features_score_correlations.csv")
corr.to_csv(corr_path, index=False)
print(f"wrote {len(corr)} rows to {corr_path}\n")
print(corr.to_string(index=False))
