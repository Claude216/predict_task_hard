# Statistical Feature Impact on EZR Performance and Stability

**Question:** do simple statistical properties of a dataset (spread of its
columns, association between its features) predict how well — and how
stably — EZR optimizes it under **default settings**?

**Corpus:** all 127 MOOT datasets under `data/optimize/*/*.csv`. Column roles
follow ezr.py's naming convention (`Cols`, ezr.py:54-63): uppercase-initial =
numeric, trailing `-`/`+` = optimization target (y), trailing `X` = ignored,
everything else = feature (x).

---

## 1. What we checked, and how each number was obtained

### 1.1 Meta-features (from the raw CSVs)

| Meta-feature | Definition | Range |
|---|---|---|
| **Variation of each feature column** (targets excluded) | Numeric: column min-max scaled to [0,1], variation = 2 × population std (a [0,1] variable maxes at std 0.5, so this is exactly [0,1]; 0 = constant, 1 = mass split between the extremes). Symbolic: Gini impurity `1 − Σp²` normalized by its maximum `1 − 1/k` (matches ezr's default `-I Impurity=gini`). **Note: this is a normalized standard-deviation-family measure, not variance.** | [0, 1] |
| **Correlation of every feature pair** | numeric–numeric: \|Pearson r\| (pairwise-complete rows); symbolic–symbolic: Cramér's V; numeric–symbolic: correlation ratio η. Pairs involving a constant column are undefined and excluded. | [0, 1] |
| **Variation of each target column** | Same standardized variation as above, applied to the y columns. | [0, 1] |
| **Covariance of every feature pair** (follow-up round) | Sample covariance (ddof=1) of the two min-max [0,1]-scaled columns; numeric–numeric pairs only (covariance is undefined for categoricals). 31/127 datasets (mostly `FM/FFM-*` binary_config) have no numeric pair and drop out. | [−0.25, 0.25] |

Per-dataset means aggregate these: `mean_feature_var`, `mean_target_var`,
`mean_pairwise_cor`, `mean_pairwise_cov` (signed), `mean_abs_pairwise_cov`.

### 1.2 EZR scores (fresh runs, default settings — not copied from prior results)

Neither `results/9` (refined config: `Budget=50, Check=10`) nor the root
`jaccard.csv` (init/refined configs) used ezr defaults, so both scores were
**re-run from scratch** with `the` left at ezr.py's defaults
(`acq=near, Any=4, Budget=30, Check=5, leaf=3, Impurity=gini`), replicating
9.py's `ezr` treatment procedure otherwise line-for-line:

- **`performance_score` (RMSE, lower = better):** 20 train/holdout half-splits
  (seeds 0,10,…,190). Per split, error = win(best reachable in
  holdout+labels) − win(best of the tree-guided `Check=5` picks or the
  labels); score = √(mean error²). (9.py Part 1.)
- **`stability_score` (agreement %, higher = more stable):** fixed seed-42
  half-split; 20 trees built from 20 independent `likely()` labelings
  (seeds 0–19); for each of ≤100 test rows, the sd of its 20 predicted win
  scores; agreement = % of rows with sd < 0.35 × sd(win over the whole
  dataset). (9.py Part 2.)
- **`stability_jaccard` (higher = more structurally stable, supplementary):**
  mean pairwise weighted Jaccard of the split-feature sets of those same 20
  trees (jaccard.py's measure).

### 1.3 Pipeline (scripts → outputs)

| Step | Script | Output |
|---|---|---|
| Per-column variation, per-pair correlation | `analysis/compute_meta_features.py` | `analysis/meta_features_detail.csv` (1.29M rows), `analysis/meta_features.csv` (superseded merged file; its scores are the *refined*-config ones from `results/9`/`jaccard.csv`, not defaults) |
| Default-settings EZR scores + one file per dataset | `analysis/per_dataset_meta.py` | `results/meta_features/{dataset}.csv` × 127, long format `kind,name,value` |
| Per-dataset means + score correlations | `analysis/summarize_meta_features.py` | `analysis/meta_features_summary.csv`, `analysis/meta_features_score_correlations.csv` |
| Pairwise covariances + score correlations | `analysis/covariance_meta.py` | covariance rows added to the 127 files; `analysis/meta_features_covariance.csv`, `analysis/meta_features_cov_correlations.csv` |

All runs: `/Users/claudeli/Tool/miniconda3/envs/caus26/bin/python` (conda env `caus26`).

---

## 2. Result data

### 2.1 Distribution of the per-dataset values (127 datasets)

From `analysis/meta_features_summary.csv` / `analysis/meta_features_covariance.csv`:

| quantity | min | median | max |
|---|---:|---:|---:|
| mean_feature_var | 0.013 | 0.681 | 1.000 |
| mean_target_var | 0.061 | 0.383 | 0.996 |
| mean_pairwise_cor | 0.000 | 0.014 | 0.624 |
| stability_score (agreement %) | 0 | 5 | 100 |
| performance_score (RMSE) | 0.00 | 20.00 | 87.54 |
| stability_jaccard | 0.106 | 0.432 | 0.859 |
| mean_pairwise_cov (n=96) | −0.0078 | −0.0000 | 0.0274 |
| mean_abs_pairwise_cov (n=96) | 0.0000 | 0.0007 | 0.0515 |

### 2.2 Meta-feature means vs. EZR scores

From `analysis/meta_features_score_correlations.csv` (n = 127):

| target | meta-feature mean | Pearson r | p | Spearman r | p |
|---|---|---:|---:|---:|---:|
| stability_score | mean_feature_var | 0.2366 | 0.0074 | 0.2439 | 0.0057 |
| stability_score | mean_target_var | −0.0338 | 0.7059 | −0.0858 | 0.3374 |
| stability_score | mean_pairwise_cor | 0.1584 | 0.0752 | −0.0167 | 0.8525 |
| performance_score | mean_feature_var | −0.1560 | 0.0799 | −0.3181 | 0.0003 |
| performance_score | mean_target_var | **−0.5958** | <1e-6 | **−0.6185** | <1e-6 |
| performance_score | mean_pairwise_cor | 0.1828 | 0.0397 | 0.0970 | 0.2777 |
| stability_jaccard | mean_feature_var | **0.4775** | <1e-6 | **0.6092** | <1e-6 |
| stability_jaccard | mean_target_var | 0.1261 | 0.1578 | 0.1660 | 0.0622 |
| stability_jaccard | mean_pairwise_cor | −0.1298 | 0.1459 | −0.3023 | 0.0006 |

### 2.3 Covariance means vs. EZR scores

From `analysis/meta_features_cov_correlations.csv` (n = 96 datasets with ≥1
numeric feature pair):

| target | covariance mean | Pearson r | p | Spearman r | p |
|---|---|---:|---:|---:|---:|
| stability_score | mean_pairwise_cov | −0.1689 | 0.1000 | −0.0822 | 0.4259 |
| stability_score | mean_abs_pairwise_cov | −0.0366 | 0.7230 | −0.1328 | 0.1972 |
| performance_score | mean_pairwise_cov | −0.0609 | 0.5553 | −0.1026 | 0.3200 |
| performance_score | mean_abs_pairwise_cov | −0.0881 | 0.3935 | 0.1135 | 0.2710 |
| stability_jaccard | mean_pairwise_cov | **−0.3368** | 0.0008 | −0.3002 | 0.0030 |
| stability_jaccard | mean_abs_pairwise_cov | −0.1562 | 0.1285 | **−0.4823** | <1e-5 |

---

## 3. Findings

1. **Target variation is the strongest performance predictor.** Datasets whose
   objective columns spread evenly over their range are optimized with lower
   RMSE (r ≈ −0.60, ρ ≈ −0.62, p < 1e-6). Since `performance_score` is an
   error, the negative sign means high target variation → *better*
   performance.
2. **Feature variation predicts structural stability.** More-varied feature
   columns → more consistent tree feature sets across seeds
   (ρ = 0.61 vs. Jaccard, p < 1e-6); the link to the agreement-style
   stability score is much weaker (r ≈ 0.24).
3. **Feature redundancy hurts structural stability.** Higher absolute pairwise
   covariance → lower Jaccard (ρ = −0.48, p ≈ 1e-6); the pairwise-correlation
   mean points the same way (ρ = −0.30). Interpretation: strongly co-varying
   features are interchangeable in tree splits, so different seeds pick
   different ones.
4. **Neither correlation nor covariance means predict the main scores.**
   Against the agreement stability score and the performance RMSE, no
   association survives both Pearson and Spearman (best single test
   p ≈ 0.04).

**Caveats.** `performance_score` is lower-is-better while both stability
measures are higher-is-better — mind the signs. The two stability measures
disagree with each other about which meta-features matter (agreement is
prediction-variance-based; Jaccard is structure-based). The covariance
analysis covers only the 96 datasets with ≥2 non-constant numeric features.
These are correlations over 127 heterogeneous datasets, not causal or
cross-validated predictive claims; see `analysis/summary_report.md` for the
earlier prediction-oriented validations under the refined config.
