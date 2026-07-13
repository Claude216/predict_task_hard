# predict_task_hard — Results Summary

Predicting per-task EZR performance from dataset meta-features (127 MOOT tasks).
Ground truth: `score` (distance-like, **lower = better** EZR performance) and
`stability` from `results/ground_truth.csv`.

**Verdict rule** used throughout: a metric **matters** if its Spearman correlation
with `score` survives Holm correction (p_holm < 0.05, from Stage 4). Metrics
significant only for `stability` are marked "No (hardness)" with a note, since the
objective is task-hardness prediction. Full table: `results/analysis_spearman.csv`.

## How the performance scores were produced (EZR configuration)

**Ground truth** (pre-existing; produced by `src/per_dataset_meta.py` with an
*older* ezr variant, run at that version's default settings:
`acq=near, Any=4, Budget=30, Check=5, leaf=3, Impurity=gini`):

- **score** (= `performance_score`, RMSE, lower = better): 20 train/holdout
  half-splits (seeds 0, 10, …, 190); per split, error = win(best row reachable in
  holdout + labels) − win(best of the tree-guided `Check=5` picks or the labels);
  score = √(mean error²). Win scores are on ezr's 100 = best scale.
- **stability** (= `stability_score`, agreement %, higher = more stable): fixed
  seed-42 half-split; 20 trees from 20 independent labelings (seeds 0–19); for each
  of ≤100 test rows take the sd of its 20 predicted win scores; agreement = % of
  rows with sd < 0.35 × sd(win over the whole dataset).

Note the ground-truth EZR **budget was 30 labels** (that version's default), not the
50 of the frozen `ezr.py` in this repo — so "hardness" here means hardness under a
30-label default-config EZR run.

**Feature side** (this repo's frozen `ezr.py`, defaults from its docstring:
`p=2, learn.budget=50, learn.start=4, learn.check=5, few=128, bayes.m=2, bayes.k=1`):

- Groups A/B consume no labels (MOOT csvs carry objective values in every row).
- C1 `nb_auc`: 32-row warm start, best/rest split at int(√32) = 5, NB via
  `ezr.likes`; held-out = min(500, n−32) unseen rows, "best" = top int(√|H|) by d2h.
- C2 `landmark_10`: `ezr.acquire()` with `the.learn.budget` temporarily set to
  **10** (+4 warm start = 14 labels; restored afterwards), default acquisition
  (`acquireWithCentroid`); reported value = `ezr.wins` score (100 = best) of the
  best row found.
- Both probes: 20 repeats with explicit seeds 1..20, reported as median + IQR;
  global driver seed 1 (`src/run_all.py`).

---

## Stage 1 — Group A: static, label-free features

Computed 9 structural X/Y features (sizes, type mix, missingness, effective rank,
canonical correlation, ridge R², objective conflict) on all 127 tasks — no labels used.

| feature | ρ vs score | p_holm | ρ vs stability | matters? |
|---|---:|---:|---:|---|
| n_y | +0.40 | <0.0001 | −0.45 | **Yes** |
| n_rows | +0.27 | 0.028 | −0.28 | **Yes** |
| linear_r2 | −0.23 | 0.090 | −0.16 | No (borderline) |
| y_corr_mean | +0.17 | 0.79 | +0.06 | No |
| eff_rank_x | +0.13 | 1.0 | −0.14 | No |
| n_x | +0.09 | 1.0 | −0.20 | No |
| missing_density | −0.05 | 1.0 | −0.08 | No |
| cca_1 | +0.04 | 1.0 | −0.18 | No |
| sym_ratio | +0.02 | 1.0 | −0.03 | No |

**Conclusion:** only task dimensionality matters — more objectives (n_y) and more
rows mean harder, less stable tasks — while every data-type/structure feature
(sym_ratio, missingness, cca_1, eff_rank_x, n_x, y_corr_mean, linear_r2) does not.

---

## Stage 2 — Group B: landscape geometry

Computed 5 features of the distance-to-heaven (d2h) landscape (fitness–distance
correlation, smoothness, d2h variance/skew/tail-gap); gate check first confirmed B3
reproduces the known variance signal (ρ = −0.614 vs reference −0.619 ✅).

| feature | ρ vs score | p_holm | ρ vs stability | matters? |
|---|---:|---:|---:|---|
| d2h_var (B3) | −0.61 | <0.0001 | +0.07 | **Yes** |
| smoothness (B2) | −0.27 | 0.031 | +0.13 | **Yes** |
| fdc (B1, confirmatory) | −0.24 | 0.070 | −0.16 | No (narrowly missed) |
| d2h_skew (B4) | −0.01 | 1.0 | **+0.45** (p_holm<0.0001) | No (hardness); Yes for stability |
| d2h_tail_gap (B5, confirmatory) | +0.12 | 1.0 | **−0.42** (p_holm<0.0001) | No (hardness); Yes for stability |

**Conclusion:** the *spread* and *smoothness* of the objective landscape matter for
hardness (high d2h variance / smooth landscapes → easier tasks), fdc just misses the
Holm bar, and the d2h shape features (skew, tail_gap) do not predict hardness but are
the strongest stability predictors found.

---

## Stage 3 — Group C: mechanism probes (46 labels/task)

Ran two label-consuming probes per task, 20 seeded repeats each: NB best-vs-rest AUC
from a 32-row warm start (C1) and the win score of a budget-10 mini-EZR run (C2).

| feature | ρ vs score | p_holm | ρ vs stability | matters? |
|---|---:|---:|---:|---|
| landmark_10_med (C2) | **−0.72** | <0.0001 | +0.12 | **Yes** (strongest overall) |
| landmark_10_iqr (C2) | +0.58 | <0.0001 | −0.16 | **Yes** |
| nb_auc_med (C1, confirmatory) | −0.42 | <0.0001 | +0.07 | **Yes** (pre-registration confirmed) |
| nb_auc_iqr (C1) | +0.11 | 1.0 | +0.06 | No |

**Conclusion:** all three probe medians/spreads except nb_auc_iqr matter — a cheap
10-budget run (its result *and* its seed-to-seed instability) is the best hardness
signal available, and NB separability (the one confirmed pre-registered hypothesis)
directly validates EZR's acquisition mechanism as the source of task difficulty.

---

## Stage 4 — Confirmatory analysis & predictive gate

Built the Holm-corrected Spearman table (feeding the verdicts above) and a depth-≤3
decision tree predicting score, validated leave-one-task-out (LOO).

| analysis | result | verdict |
|---|---|---|
| C1 nb_auc (pre-registered) | ρ = −0.42, p_holm < 0.0001 | **Confirmed** |
| B1 fdc (pre-registered) | ρ = −0.24, p_holm = 0.070 | Not confirmed (near miss) |
| B5 d2h_tail_gap (pre-registered) | ρ = +0.12 vs score, p_holm = 1.0 | Not confirmed (strong on stability instead) |
| depth-≤3 tree, LOO (n=127) | **Spearman(pred, true) = 0.68**, p < 1e-18 | Hardness is predictable |
| tree root split | landmark_10_med ≤ 78 | probe-based gate (costs 46 labels) |

**Conclusion:** task hardness is genuinely predictable (LOO ρ = 0.68), with the
label-costing landmark probe mattering most and one of three pre-registered
hypotheses (nb_auc) confirmed.

---

## Ground truth v2 — rerun under current ezr.py defaults (2026-07-13)

Re-generated both scores with the SAME metric definitions but this repo's frozen
`ezr.py` at its defaults (`p=2, learn.budget=50, learn.start=4, learn.check=5,
few=128, bayes.m=2, bayes.k=1`), porting the old API (`likely()` → `acquire()`,
`Tree`/leaf `.mu` → `treeGrow`/leaf `.ynum.mu`; win formula unchanged).
Performance: 20 half-splits, seeds 0,10,…,190. Stability: seed-42 split, 20 trees
seeded 0..19. Outputs: `ground_truth_v2.csv`, `analysis_spearman_v2.csv`,
`analysis_tree_v2.txt` (v1 files and the stage tables above are unchanged).

v1 ↔ v2 agreement: score ρ = 0.85 (budget 30→50 makes tasks easier: median score
20.0 → 6.5); stability ρ = 0.33 (the extra labels change tree consistency a lot —
v2 stability is effectively a new target).

| feature (Holm-surviving, v2) | ρ vs score | ρ vs stability | change from v1 |
|---|---:|---:|---|
| landmark_10_med | **−0.87** | +0.39 | stronger (was −0.72) |
| d2h_var | **−0.76** | +0.40 | stronger (was −0.61) |
| landmark_10_iqr | +0.59 | −0.45 | ≈ same |
| n_y | +0.52 | −0.26 | stronger (was +0.40) |
| nb_auc_med (confirmatory) | −0.39 | +0.37 | **confirmed again** |
| fdc (confirmatory) | −0.27 | n.s. | **now CONFIRMED** (p_holm = 0.034; was a near miss) |
| n_rows | +0.27 | n.s. | ≈ same |
| smoothness | n.s. (−0.22) | **+0.61** | lost score signal; now the top stability predictor |
| d2h_tail_gap (confirmatory) | n.s. | n.s. | still not confirmed; v1 stability link gone |
| depth-≤3 tree, LOO | **Spearman 0.73** | — | up from 0.68 |

**Conclusion (v2):** hardness signals sharpen across the board (2 of 3
pre-registered features now confirmed: nb_auc and fdc; d2h_tail_gap fails on both
targets), and v2 stability becomes an "easy tasks are repeatable tasks" axis
(smoothness +0.61, landmark/d2h_var positive) rather than v1's d2h-shape story.

---

## Overall conclusion

To determine how hard a task is: if you can spend ~14 labels, run a budget-10 EZR
probe — its win score (ρ = −0.72 v1, **−0.87 v2**) and seed instability (ρ ≈ +0.58)
are the dominant signals; at zero label cost, objective-space spread (d2h_var,
−0.61 v1, **−0.76 v2**), objective count (n_y, +0.40/+0.52), and landscape structure
(fdc, confirmed under v2) carry most of the signal. A depth-3 tree over these
features ranks unseen tasks by hardness at LOO Spearman 0.68 (v1) / **0.73 (v2)**,
so a routing gate (e.g., EZR→SNAP cascade) is viable — stability is a separate,
ground-truth-sensitive axis (d2h shape under v1; smoothness +0.61 under v2).
