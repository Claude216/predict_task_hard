# Feature Index

The 17 meta-feature columns of `results/features.csv` (127 MOOT tasks), grouped as
in the pipeline: **A** static/label-free (`src/static.py`), **B** landscape geometry
(`src/landscape.py`), **C** mechanism probes (`src/probes.py`). "d2h" =
`ezr.disty`, distance-to-heaven over the Y (objective) columns, lower = better row.
Groups A and B cost 0 labels on MOOT (objective values are present in every csv
row); Group C costs 46 labels per task (`labels_used_total`).

---

## Group A — static, label-free

### n_rows (A1)
**One-liner:** number of data rows in the task.

Sheer task size. More rows mean a larger search space for the active learner to
cover with the same fixed label budget, so size alone can make a task harder; it is
also the cheapest feature to obtain. In our results it carries a real but modest
hardness signal (ρ = +0.27).

### n_x, n_y (A2)
**One-liner:** counts of independent (X) and objective (Y) columns.

Task dimensionality on both sides. n_x measures how many knobs describe a row;
n_y measures how many objectives must be traded off at once. Since d2h aggregates
all objectives into one distance, each extra objective dilutes the geometry that
guides acquisition — which is why n_y turned out to be the strongest static
hardness predictor (ρ = +0.40) and also the strongest static *instability*
predictor (ρ = −0.45), while n_x barely matters.

### sym_ratio (A3)
**One-liner:** fraction of X columns that are symbolic (Sym) rather than numeric.

Data-type mix. EZR treats Syms and Nums differently in distance (`aha`: equality
test vs normalized difference), so a Sym-heavy task lives in a coarser, less smooth
metric space. Plausible as a hardness driver, but empirically it matters not at all
here (ρ ≈ 0).

### missing_density (A4)
**One-liner:** fraction of "?" cells among all X cells.

Data quality. Missing values force ezr's pessimistic distance fallback and shrink
the evidence NB can use per row, so high missingness could plausibly hurt. MOOT
tasks are mostly complete (max ≈ 1.9%), and the feature shows no signal — a floor
effect as much as a verdict.

### eff_rank_x (A5)
**One-liner:** effective rank of the encoded-X correlation matrix,
exp(entropy of its normalized eigenvalues).

Intrinsic dimensionality: how many *independent* directions X really has after
one-hot/standardize/impute encoding. A task with 50 correlated columns may be
"really" 3-dimensional and easy to search. Caveat: one-hot encoding lets the value
exceed raw n_x, and high-cardinality ID-like Sym columns inflate it to the
thousands on a few tasks (Loan, all_players). No significant hardness signal.

### cca_1 (A6)
**One-liner:** first canonical correlation between the encoded X and Y blocks.

The strongest single linear bridge between decisions and objectives: how well some
linear combination of X predicts some linear combination of Y. If nothing in X
linearly relates to any objective, no learner has an easy starting point. In
practice almost every MOOT task has a high cca_1 (median 0.91), so it separates
nothing (ρ ≈ 0).

### linear_r2 (A7)
**One-liner:** 5-fold CV R² of a ridge regression from encoded X to d2h.

A direct "is the landscape globally linear?" test: if a linear model explains d2h,
the task should be nearly trivial for any sensible optimizer. Points the expected
direction (linear tasks are easier, ρ = −0.23) but just misses the Holm bar —
mostly superseded by the sharper Group B geometry features.

### y_corr_mean (A8)
**One-liner:** mean pairwise Pearson correlation among the Y columns
(objective conflict; NaN when n_y < 2).

Multi-objective conflict. Negatively correlated objectives (improving one worsens
another) force a Pareto trade-off; positively correlated objectives collapse into
what is effectively one goal. Sensible in theory, but undefined on the 29
single-objective tasks and no significant signal on the rest.

---

## Group B — landscape geometry
*(label-free on MOOT only — in deployment these would cost labels)*

### fdc (B1) — pre-registered confirmatory
**One-liner:** fitness–distance correlation: Spearman ρ between distx(row,
best_row) and d2h(row) over min(500, n) sampled rows.

The classic optimization-hardness measure: does getting *closer* (in X) to the best
known row make rows *better*? High fdc means a single global basin that
distance-guided acquisition can ride downhill; low or negative fdc means deceptive
or scattered optima. Points the right way (ρ = −0.24) but narrowly missed
confirmation after Holm correction (p_holm = 0.070).

### smoothness (B2)
**One-liner:** Spearman ρ between distx(r1, r2) and |d2h(r1) − d2h(r2)| over
1000 sampled row pairs.

A local-continuity probe: do nearby rows have similar quality? Smooth landscapes
let a learner generalize from few labels ("this region is good"); rugged ones make
every label nearly useless one step away. A real, Holm-surviving hardness signal
(ρ = −0.27): smoother → easier.

### d2h_var (B3)
**One-liner:** variance of d2h over all rows — the previously confirmed baseline
signal, kept as reference.

How spread out row quality is. When variance is high, good rows are *much* better
than average and even a handful of labels finds the gradient; when all rows score
alike, acquisition has nothing to grip. Reproduced the prior finding almost exactly
(ρ = −0.61 vs the reference −0.62) — the strongest label-free hardness feature.

### d2h_skew (B4)
**One-liner:** skewness of the d2h distribution over all rows.

Shape of the quality distribution: positive skew = most rows good with a long bad
tail; negative skew = rare elite rows in a sea of mediocrity. No hardness signal at
all (ρ ≈ 0), but the strongest *stability* predictor found (ρ = +0.45): skewed-bad
tasks give EZR consistent, repeatable outcomes.

### d2h_tail_gap (B5) — pre-registered confirmatory
**One-liner:** (median(d2h) − p10(d2h)) / IQR(d2h) — how far the good tail
sits below the bulk.

A heavy-left-tail indicator: large values mean the best ~10% of rows are far
detached from typical rows, i.e. there is a distinct elite worth hunting for. The
pre-registered hardness hypothesis was **not** confirmed (ρ = +0.12 vs score);
instead it strongly predicts *instability* (ρ = −0.42): a detached elite makes
results depend on whether a given seed stumbles into it.

---

## Group C — mechanism probes (46 labels/task; median + IQR over 20 seeded repeats)

### nb_auc_med, nb_auc_iqr (C1) — pre-registered confirmatory
**One-liner:** AUC of a Naive Bayes best-vs-rest ranker trained on a 32-row warm
start and evaluated on a held-out sample (median and IQR over seeds 1..20).

A direct probe of EZR's inner loop: acquire() steers with exactly this kind of NB
best/rest model, so C1 asks "on this task, does that steering mechanism work from
32 labels?" (Split at int(√32)=5 best like `warm_start`; held-out = min(500, n−32)
rows, top int(√|H|) by d2h labeled best.) The hypothesis was **confirmed**:
learnable tasks are easier for EZR (median ρ = −0.42). The IQR column (seed
sensitivity of the AUC) carries no signal.

### landmark_10_med, landmark_10_iqr (C2)
**One-liner:** win score (ezr.wins, 100 = best) of the best row found by
acquire() run with budget 10 (+4 warm start), median and IQR over seeds 1..20.

Landmarking: the cheapest honest preview of the real thing — run a 14-label
mini-EZR and see how far it gets. Unsurprisingly but usefully, it is the strongest
hardness predictor overall (median ρ = −0.72), and its seed-to-seed spread is an
independent warning sign (IQR ρ = +0.58: tasks where the cheap run is erratic end
badly). The catch for a routing gate: 46 labels of probing against a 30-label
ground-truth budget — Group C only pays off if probe labels are reusable or much
cheaper than run labels.
