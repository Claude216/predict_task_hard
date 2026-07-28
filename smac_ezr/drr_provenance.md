# DRR provenance

How the Dimensionality Reduction Ratio numbers in `drr.csv` were produced, and
every way they depart from a naive reading of "intrinsic dimension".

Fill in the bracketed fields before relying on this file.

---

## Source

| | |
|---|---|
| Package | `dimensionality_reduction_ratio` (Lustosa) |
| Upstream commit | `94c41c84357e095b54e9013af9ca3cf2db71ac4e` |
| Local state | pinned upstream + local patches, see below |
| Location in this project | `external/dimensionality_reduction_ratio/src` |
| Version string recorded in `drr.csv` | column `drr_version` |

The package is **not** vendored into `smac_ezr/`. `compute_drr.py` puts
`$DRR_SRC` on `sys.path` for the duration of that one script and nothing else
imports it. Analysis scripts join against `drr.csv` instead.

### Local patches

Reproduce with `git -C external/dimensionality_reduction_ratio diff > drr.patch`.

| Fix | Why |
|---|---|
| `log(0)` sentinel | The original code somehow included the head and the tail of the curve which led issues as the algorithm is trying to find the max slope of the curve (early range or near head would have a super high gradient which cannot reflect the valid max slope so we need to omit this part for a valid max slope). |
| Missing smoothing | TODO |
| Constant columns not filtered | This doesn't matter. |

Only `estimate()` and a few functions it calls were changed. The call
signatures and the `(R, I, drr)` return are unchanged, so `compute_drr.py` uses
the upstream API as documented.

---

## Definition

    DRR = 1 - I / R

- `R` — number of decision columns
- `I` — intrinsic dimension, Levina-Bickel correlation function method
- Range [0, 1). Higher means the table collapses into fewer effective
  dimensions.

---

## What is measured

Input is `ds.df[ds.x_cols]` from this project's `Dataset.load`, **not** the raw
CSV and **not** `ds.df`.

Two reasons, both load-bearing:

1. **`Dataset.load` has already changed the table.** Column types are decided
   from the header's initial case, per the MOOT convention that `ezr.Col()`
   also follows, rather than from pandas dtype inference — a single `?` missing
   marker is enough to make pandas read a numeric column as strings, which is
   how `dataset120` came to have five coverage columns typed as text. Rows with
   missing values are then dropped. What the optimizers search is the resulting
   table, so that is what DRR must describe.

2. **`DataProcessor` does not strip `X`-suffix columns.** It removes `+`, `-`
   and `!` only. MOOT uses `X` for "ignore this column" and `nasa93dem` has
   four (`idX`, `centerX`, `YearX`, `MonthsX`). Handing it the whole frame
   would count those as features and inflate `R`, which is the denominator.

Symbolic columns are label-encoded by `DataProcessor`, not one-hot encoded, so
`R` equals the decision-column count.

---

## Known properties of these numbers

### No scaling, with an l1 metric

`DataProcessor._process_columns` performs numeric conversion and label encoding
and applies **no normalisation**. The distance metric is `l1`.

Where column ranges differ by orders of magnitude the wide column dominates the
distance. `SS-A` is the clear case: `Spout_wait` spans 1–10000 while `Spliters`
spans 1–6, so the estimated intrinsic dimension partly reflects **range**
rather than geometry.

**Kept as-is deliberately.** DRR is being used as an estimator and an axis to
sort tasks along, not as a claim about true geometry. Normalising would give
numbers incomparable with the upstream work for no gain against the question
being asked.

### Label encoding imposes an order that may not exist

`nasa93dem`'s 22 COCOMO ratings become integers. Under l1 this asserts that
`vl` is as far from `l` as `l` is from `n`, and that the ratings are ordered at
all. For genuinely ordinal ratings that is roughly right; for nominal columns
it is not.

### Sampling: reproducible, which is not the same as stable

Two stages, both seeded:

| Stage | Cap | Seed |
|---|---|---|
| `DataProcessor._sample_data` | 5000 rows | `random_seed`, default 42 |
| `IntrinsicDimensionEstimator.estimate` | 2000 rows | hard-coded `default_rng(42)` |

So `x264` (167k rows) is estimated from roughly 1.2% of itself. Rerunning gives
the identical number; that is reproducibility, not evidence that the number is
insensitive to which rows were drawn.

**Measured, not assumed.** `compute_drr.py --seeds 42 1 2 3 4` varies the first
stage. Observed spread on the sampled tables: **below 0.01**, against an
observed DRR range of roughly **0.2 to 1.0**. Sampling noise is one to two
orders of magnitude below the gaps read off the axis, so DRR plots are drawn
without error bars.

Tables of 2000–5000 rows cannot be probed this way — only the estimator's
hard-coded stage samples them — and report `nan` in `drr_spread`.

---

## Reproducing

```bash
DRR_SRC=../external/dimensionality_reduction_ratio/src \
python compute_drr.py ../data/moot/optimize --out drr.csv --seeds 42 1 2 3 4
```

Recompute whenever `Dataset.load` changes how columns are typed or how rows are
dropped: that changes the input table, and therefore the DRR.

---

## Not established here

- Whether DRR is telling us anything that `coverage`
  (`log2(distinct rows) - sum log2|Xi|`) or `x_dup_rate` do not. Not yet
  checked. If DRR turns out to be strongly correlated with either, a DRR plot
  and a coverage plot are the same plot.
- Whether DRR relates to SMAC's `trace_on_table_rate`. SMAC's advantage grows
  where the surrogate extrapolates, so if DRR tracks sparsity then "SMAC wins
  at high DRR" and "SMAC exploits the surrogate at high DRR" are not yet
  distinguishable. The rate is carried in the analysis CSV for this reason.