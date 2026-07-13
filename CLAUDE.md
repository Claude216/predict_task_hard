# CLAUDE.md — predict_task_hard

## Goal

Predict per-task EZR performance from dataset meta-features, so that a "task
complexity predictor" gate can decide when to route a task to complementary
processing (e.g., an EZR→SNAP cascade). Ground-truth per-task EZR performance
and stability scores ALREADY EXIST (see `results/ground_truth.csv`). Do NOT
re-run full EZR experiments to regenerate them.

Deliverable: `results/features.csv` (one row per MOOT task, one column per
meta-feature) plus a Stage-4 analysis correlating features against ground truth.

## Repository layout

```
predict_task_hard/
├── CLAUDE.md                # this file
├── ezr.py                   # frozen — DO NOT MODIFY
├── data/moot/optimize/           # MOOT benchmark tasks (csv), via sparse checkout
├── src/
│   ├── static.py            # Group A (label-free, X/Y structure)
│   ├── landscape.py         # Group B (landscape geometry)
│   ├── probes.py            # Group C (mechanism-specific probes)
│   └── run_all.py           # loop: tasks × features → results/features.csv
└── results/
    ├── ground_truth.csv     # task, score, stability (pre-existing, read-only)
    └── features.csv         # output of Stages 1–3
```

## EZR interface contract (ezr.py)

Import, never modify. Key entry points:

- `Data(csv(path))` — load a MOOT csv into a Data (rows + summarized cols).
- `data.cols.xs` / `data.cols.ys` — independent / objective columns.
- `disty(data, row)` — distance-to-heaven over Y columns (lower = better).
- `distx(data, r1, r2)` — Minkowski distance over X columns.
- `wins(data)` — returns scoring fn mapping row → win score (100 = best).
- `acquire(data, score=..., label=...)` — active learner; consumes
  `the.learn.budget` labels (default 50), warm-start `the.learn.start` (4),
  subsample cap `the.few` (128).
- `likes(data, row, n_rows, n_klasses)` — NB log-likelihood.
- `the` — global config namespace parsed from the module docstring
  (e.g., `the.learn.budget`, `the.p`, `the.stats.eps`).

Column-name conventions (CRITICAL — do not guess):
- Header ending `+` = objective to maximize; `-` = minimize; `!` = klass.
- Header ending `X` = ignored column.
- Header starting with an UPPERCASE letter = numeric (Num); otherwise Sym.
- Missing values are the string `"?"` — handle explicitly everywhere.
- `Num.heaven` is 1 unless the name ends `-`.

MOOT-specific fact: objective values are present in every csv row, so
`disty` is FREE to compute on this benchmark. Group B features are therefore
label-free *here*, even though they would cost labels in deployment. Record
this distinction in feature metadata (`labels_used` column: 0 for A and B on
MOOT; actual count for Group C).

## Meta-features to implement

### Group A — static, label-free (metafeatures/static.py)

| id | feature | definition |
|----|---------|-----------|
| A1 | n_rows | row count |
| A2 | n_x, n_y | count of X and Y columns |
| A3 | sym_ratio | fraction of X columns that are Sym |
| A4 | missing_density | fraction of "?" cells in X |
| A5 | eff_rank_x | effective rank of X correlation matrix: exp(entropy of normalized eigenvalues) |
| A6 | cca_1 | first canonical correlation between X and Y |
| A7 | linear_r2 | R² of ridge regression X → d2h (5-fold CV) |
| A8 | y_corr_mean | mean pairwise correlation among Y columns (objective conflict) |

Encoding rule for A5–A7: one-hot encode Sym columns; standardize Num
columns; impute "?" with column mean/mode. State this in code comments —
do not silently choose a different scheme.

### Group B — landscape geometry (metafeatures/landscape.py)

| id | feature | definition |
|----|---------|-----------|
| B1 | fdc | fitness–distance correlation: Spearman ρ between distx(row, best_row) and disty(row), over min(500, n_rows) sampled rows; best_row = argmin disty |
| B2 | smoothness | Spearman ρ between distx(r1,r2) and \|disty(r1)−disty(r2)\| over 1000 sampled pairs |
| B3 | d2h_var | variance of disty over all rows (the confirmed baseline signal — keep as reference) |
| B4 | d2h_skew | skewness of the disty distribution |
| B5 | d2h_tail_gap | (median(d2h) − p10(d2h)) / IQR(d2h) — heavy-left-tail indicator |

### Group C — mechanism probes (metafeatures/probes.py)

| id | feature | definition | labels used |
|----|---------|-----------|-------------|
| C1 | nb_auc | warm-start sample of 32 rows, split best/rest by disty at sqrt(n) as acquire() does, train NB (likes), AUC of best-vs-rest ranking on held-out sample | 32 |
| C2 | landmark_10 | run acquire() with the.learn.budget=10, report win score of best found row | 10 (+4 warm start) |

C1 and C2: 20 repeats per task with seeds 1..20; report median and IQR as
two separate columns each. Restore `the.learn.budget` after C2.

## Hard constraints

1. `ezr.py` is frozen. Wrap it; never edit it.
2. All randomness seeded and recorded. Global driver seed in run_all.py;
   per-repeat seeds explicit.
3. Every feature function signature: `f(data: Data, rng: random.Random) -> dict`
   returning `{feature_name: value}`; run_all.py merges dicts per task.
4. Output `results/features.csv` columns: task, then all features, then
   labels_used_total. One row per MOOT task under `moot/optimize/`.
5. Skip-and-log, never crash: if a feature fails on a task (e.g., all-Sym X
   breaks CCA), write NaN and append a line to `results/errors.log`.
6. Stage 4 correlations use Spearman, not Pearson.
7. Multiple-comparison control: report Holm-corrected p-values.
   Pre-registered (confirmatory) features: B1 fdc, B5 d2h_tail_gap, C1 nb_auc.
   All others are exploratory — label them as such in the output table.

## Staged execution plan

- **Stage 1**: implement + run Group A on all tasks. Sanity checks: eff_rank_x
  ≤ n_x; A5–A7 finite on tasks with Sym columns; runtime < ~1 min/task.
- **Stage 2**: Group B. Verify B3 reproduces the previously confirmed
  objective-variance signal (Spearman vs. ground truth, sign and rough
  magnitude) before proceeding — if it does not, stop and report; there is
  likely a d2h or ground-truth alignment bug.
- **Stage 3**: Group C with repeats.
- **Stage 4**: `experiments/analyze.py` — Spearman table (feature × ground
  truth score, with Holm correction and confirmatory/exploratory flag), plus
  a depth-≤3 decision tree predicting score from features with
  leave-one-task-out validation. Report LOO Spearman of predictions.

Pause after each stage for human review. Do not proceed to the next stage
uninvited.

## Do NOT

- Do not modify ezr.py or files under moot/.
- Do not regenerate ground truth by running full-budget EZR sweeps.
- Do not use Pearson correlation in Stage 4.
- Do not silently choose encodings, imputation, or sample sizes different
  from those specified above; if a spec is ambiguous, ask.
- Do not add heavy dependencies. Allowed: numpy, scipy, scikit-learn, pandas.