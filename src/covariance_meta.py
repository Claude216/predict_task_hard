"""
Pairwise feature COVARIANCE meta-features, extending the per-dataset files in
results/meta_features/ (from analysis/per_dataset_meta.py).

For each of the 127 datasets:
  - every numeric-numeric feature pair (ezr naming: uppercase-initial = Num;
    covariance is undefined for symbolic columns, those pairs are skipped)
    gets cov(a, b) computed on the min-max [0,1]-scaled columns
    (pairwise-complete rows, sample covariance ddof=1), so values are
    comparable across datasets; range is [-0.25, 0.25].
  - rows `covariance,<f1>|<f2>,<value>` are inserted into the dataset's
    results/meta_features/{dataset}.csv right after the correlation block
    (score rows stay last). Re-running replaces any existing covariance rows.

Aggregates -> analysis/meta_features_covariance.csv, one row per dataset:
  dataset_name, n_cov_pairs, mean_pairwise_cov (signed), mean_abs_pairwise_cov,
  stability_score, performance_score, stability_jaccard (scores merged from
  analysis/meta_features_summary.csv). Datasets with no numeric pair have NaN.

Correlations -> analysis/meta_features_cov_correlations.csv:
  Pearson and Spearman (with p-values) of mean_pairwise_cov and
  mean_abs_pairwise_cov against stability_score, performance_score and
  stability_jaccard, over the datasets where the mean is defined.

Run: /Users/claudeli/Tool/miniconda3/envs/caus26/bin/python analysis/covariance_meta.py
"""
import csv as _csv
import glob
import os

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

REPO = "/Users/claudeli/NCSU/Research/26sp_aise/stability_evaluation"
META_DIR = os.path.join(REPO, "results/meta_features")


def col_role(name):
    if name[-1] == "X":
        return "ignore"
    if name[-1] in "-+":
        return "y"
    return "x"


def pair_covariances(path):
    "list of (f1, f2, cov) over numeric-numeric x pairs, [0,1]-scaled columns."
    df = pd.read_csv(path, dtype=str, skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    num_x = [c for c in df.columns if col_role(c) == "x" and c[0].isupper()]
    if len(num_x) < 2:
        return []
    sub = df[num_x].apply(
        lambda s: pd.to_numeric(s.replace("?", np.nan), errors="coerce"))
    rng = sub.max() - sub.min()
    keep = [c for c in num_x if rng[c] > 0]  # constant columns: cov trivially 0/undefined
    scaled = (sub[keep] - sub[keep].min()) / rng[keep]
    cov = scaled.cov(min_periods=3)  # pairwise-complete, ddof=1
    out = []
    for i, a in enumerate(keep):
        for b in keep[i + 1:]:
            v = cov.loc[a, b]
            if not np.isnan(v):
                out.append((a, b, float(v)))
    return out


def rewrite_meta_file(dataset, covs):
    "Insert covariance rows after the correlation block; scores stay last."
    path = os.path.join(META_DIR, f"{dataset}.csv")
    with open(path, encoding="utf-8") as fh:
        rows = [r for r in _csv.reader(fh)][1:]
    body = [r for r in rows if r[0] not in ("covariance", "performance", "stability")]
    scores = [r for r in rows if r[0] in ("performance", "stability")]
    cov_rows = [["covariance", f"{a}|{b}", repr(v)] for a, b, v in covs]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = _csv.writer(fh)
        w.writerow(["kind", "name", "value"])
        w.writerows(body + cov_rows + scores)


def main():
    files = sorted(glob.glob(os.path.join(REPO, "data/optimize/*/*.csv")))
    assert len(files) == 127, f"expected 127 files, got {len(files)}"

    agg = []
    for path in files:
        dataset = os.path.basename(path)[:-4]
        covs = pair_covariances(path)
        rewrite_meta_file(dataset, covs)
        vals = np.array([v for _, _, v in covs])
        agg.append({
            "dataset_name": dataset,
            "n_cov_pairs": len(vals),
            "mean_pairwise_cov": vals.mean() if len(vals) else np.nan,
            "mean_abs_pairwise_cov": np.abs(vals).mean() if len(vals) else np.nan,
        })
        print(f"{dataset}: pairs={len(vals)} "
              f"mean_cov={vals.mean() if len(vals) else float('nan'):.4f}")

    cov_df = pd.DataFrame(agg)
    summary = pd.read_csv(os.path.join(REPO, "analysis/meta_features_summary.csv"))
    cov_df = cov_df.merge(
        summary[["dataset_name", "stability_score", "performance_score",
                 "stability_jaccard"]], on="dataset_name", validate="1:1")
    cov_path = os.path.join(REPO, "analysis/meta_features_covariance.csv")
    cov_df.to_csv(cov_path, index=False)
    print(f"\nwrote {len(cov_df)} rows to {cov_path}")
    print(f"datasets with no numeric pair (NaN mean): "
          f"{cov_df.mean_pairwise_cov.isna().sum()}")

    out = []
    for target in ["stability_score", "performance_score", "stability_jaccard"]:
        for m in ["mean_pairwise_cov", "mean_abs_pairwise_cov"]:
            ok = cov_df[[m, target]].dropna()
            pr, pp = pearsonr(ok[m], ok[target])
            sr, sp = spearmanr(ok[m], ok[target])
            out.append({"target": target, "covariance_mean": m, "n": len(ok),
                        "pearson_r": round(pr, 4), "pearson_p": round(pp, 6),
                        "spearman_r": round(sr, 4), "spearman_p": round(sp, 6)})
    corr = pd.DataFrame(out)
    corr_path = os.path.join(REPO, "analysis/meta_features_cov_correlations.csv")
    corr.to_csv(corr_path, index=False)
    print(f"wrote {len(corr)} rows to {corr_path}\n")
    print(corr.to_string(index=False))


if __name__ == "__main__":
    main()
