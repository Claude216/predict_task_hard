# DEHB (HEAVY) vs EZR (LITE) on MOOT, read against DRR

Run 2026-08-10. 126 of 127 MOOT `optimize` tasks, 10 seeds each (0..9), four arms.

## Setup

| arm | budget | what it is |
|---|---|---|
| `ezr` | 30 labels | LITE: pool-based active learning, every evaluation at full fidelity |
| `random_pool` | 30 | floor for EZR — uniform table rows |
| `dehb` | 3000 evals | HEAVY: DE + Hyperband over a tree-count fidelity |
| `random_cs` | 3000 | floor for DEHB — uniform draws from the *same* ConfigSpace |

Shared evaluation surface: one frozen `RandomForestRegressor` per goal, fit on
the whole table, collapsed to Eq.1 d2h. DEHB's fidelity is how many of the 100
trees vote; fidelity 100 is bit-identical to what EZR is scored on. Every
oracle call costs 1 regardless of fidelity, so 3000 is 3000. DEHB's reported
score counts only full-fidelity evaluations (~161 per run), and its 3000th
evaluation is reserved to confirm its incumbent at 100 trees.

Verdicts are `stats.top()` (Cliff's delta + KS) over each arm's 10 best-d2h
values, with `eps = 0.01 x d2h_spread`. A 2-element result is a tie.

Config space: `Ordinal` over each column's sorted observed values, `Categorical`
for symbolic. DEHB may propose configurations that are not rows in the table —
and its answers routinely are — but never a value the column has not taken.

## Headline

| verdict | tasks | share |
|---|---|---|
| DEHB wins | 102 | 81% |
| tie | 17 | 13% |
| EZR wins | 7 | 6% |

At 100x the evaluation budget, HEAVY beats LITE on four tasks in five. That
number on its own is close to a foregone conclusion and is **not** the result.
The two informative facts are below.

### 1. On 46% of tasks, DEHB's win is budget, not search

**58 of DEHB's 102 wins did not beat `random_cs@3000`** — 3000 blind uniform
draws from the identical config space. On those tasks DEHB's advantage over EZR
is attributable to having 100x the evaluations, not to differential evolution
or successive halving finding anything. Those points are drawn hollow in both
figures.

(For symmetry: 4 of EZR's 7 wins did not beat `random_pool@30`.)

### 2. DRR does NOT predict when LITE keeps up with HEAVY — null result

This was the question the study was built to answer, and the answer is no.
Comparing the 24 tasks where LITE was competitive (EZR win or tie) against the
102 where HEAVY was clearly better:

| feature | LITE competitive (n=24) | HEAVY better (n=102) | Mann-Whitney p |
|---|---|---|---|
| DRR | 0.677 | 0.667 | 0.252 |
| I (intrinsic dim) | 3.00 | 3.00 | 0.959 |
| R (decision cols) | 9.00 | 8.00 | 0.368 |
| rows | 3389 | 10000 | 0.093 |

No separation on DRR, on absolute intrinsic dimension, or on raw dimension.
EZR's 7 wins do sit at a higher median DRR (0.889 vs 0.667), but with n=7 and
p=0.152 that is descriptive, not evidence.

**There is no DRR threshold in this data above which 30 labels match 3000
evaluations.** The figures show why: colour is scattered throughout, not banded.

### 3. DRR does track whether search beats blind sampling — but it is confounded

Among all 126 tasks, whether DEHB beat its own random floor correlates with DRR
(Spearman rho = 0.286, p = 0.0012; among DEHB's wins alone, rho = 0.307,
p = 0.0017). Floor-beating tasks have median DRR 0.809 against 0.613 for the
rest (p = 0.002). Read naively: structured search pays off over blind sampling
where the table collapses into fewer effective dimensions.

**Do not report that without the confound.** DRR band membership is largely
task family:

| DRR band | dominant family | floor-beat rate |
|---|---|---|
| (0.0, 0.5] | 35/49 `hpo` | 0.33 |
| (0.5, 0.65] | mixed | 0.60 |
| (0.65, 0.8] | 21/28 `config` | 0.14 |
| (0.8, 0.9] | mixed | 0.53 |
| (0.9, 1.0] | 12/20 `binary_config` | 0.85 |

Floor-beat rate by family runs from `systems` 0.08 and `config` 0.14 up to
`binary_config` 1.00 and `misc`/`rl` 1.00. The FFM SAT instances in
`binary_config` all have very high DRR *and* DEHB always beats random there;
`config`/`systems` sit mid-DRR and DEHB rarely does. The rate is also
non-monotone across bands (0.33, 0.60, **0.14**, 0.53, 0.85), which a genuine
DRR mechanism would not predict. Family, not DRR, is the more parsimonious
explanation on this evidence.

## Figures

- `dim_intrinsic_vs_raw.png` — x = R, y = I, both log
- `drr_vs_raw.png` — x = R, y = DRR = 1 − I/R

Yellow = EZR wins, green = tie, red = DEHB wins; hollow red = DEHB win that did
not beat its own floor. 126 tasks occupy only 64 distinct (R, I) cells (largest
holds 35 tasks), so co-located points are fanned horizontally by up to ±8% of R
for legibility — y is never displaced, and `verdicts.csv` carries exact values.
Figure 2 is a deterministic transform of figure 1 (same x, y = 1 − y/x).

## A DEHB defect found, patched, and re-run

An integrity pass over the 1260 DEHB runs (`n_evals == budget`?) found 5 runs on
2 tasks that had stopped after 15–1006 evaluations instead of 3000.

**Cause, in DEHB itself** (`dehb/optimizers/de.py:188-190`):

```python
ranges = np.arange(start=0, stop=1, step=1/len(hyper.sequence))
param_value = hyper.sequence[np.where((vector[i] < ranges) == False)[0][-1]]
```

`np.arange(0, 1, 1/n)` does not always yield `n` elements. For `n = 49`,
`49 * (1/49) == 0.9999999999999999 < 1`, so arange emits a 50th edge and a DE
coordinate at ~1.0 selects `sequence[49]` of a 49-element tuple →
`IndexError: tuple index out of range`. 328 of the first 5000 lengths are
affected; 11 of the 126 tasks carry at least one such column.

**Why it was silent.** DEHB decorates `run()` with loguru's `@logger.catch`, so
the IndexError was logged and swallowed — `dehb.run()` returned normally and the
arm was recorded as a completed 3000-evaluation run. Nothing in DEHB's own
output flagged it.

**Fix.** `DEHBOptimizer._patch_vector_to_configspace` replaces the mapping with
one that derives the index from the *same* edges
(`np.searchsorted(ranges, v, "right") - 1`) and then clamps to `n-1` — i.e. the
round-and-clamp index interpolation this study specified; DEHB omitted the
clamp. The only behavioural change is at the coordinate that previously raised.
`dehb_ezr/tests/test_dehb_patch.py` asserts equivalence with DEHB's mapping over
a dense sample and at the failing coordinate.

**Validation.** All 11 at-risk tasks were re-run. The 9 that never hit the bug
came back **bit-identical** (same verdicts, same median d2h), confirming the
patch is inert elsewhere; the 2 broken tasks improved (`Car_price_cleaned` DEHB
median 0.2317 → 0.2186, `xomo_osp` 0.0224 → 0.0215) and **kept their verdicts**,
so no headline number in this document changed. Post-fix integrity: 0 problems
across 126 tasks, every DEHB run confirming exactly 161 configs at full fidelity.

A guard now raises if DEHB returns having spent less than its budget, so this
class of silent truncation cannot be recorded as a valid run again.

## Caveats and deviations

- **DRR provenance.** Estimator is `dimensionality_reduction_ratio` at commit
  `defce44` **with uncommitted working-tree edits**; the diff is archived as
  `drr_estimator.patch`. These numbers are not reproducible without it.
- **Synthetic fidelity.** MOOT has no real fidelity axis. Tree count makes low
  fidelity noisier but not meaningfully cheaper in wall time.
- **DEHB imposes an order on symbolic levels.** `vector_to_configspace`
  (de.py:191-193) index-interpolates `Categorical` exactly as it does `Ordinal`,
  so DE arithmetic treats adjacent levels in the sorted list as similar. Not
  fixable without forking DEHB.
- **Budget asymmetry is deliberate**, so `lite_vs_heavy` is not a like-for-like
  method comparison. The floor arms exist to make it interpretable anyway.
- **Not comparable to `smac_ezr/results_b*`.** DEHB pins `numpy<2`, so this ran
  in a separate env (`dmoot`, numpy 1.26.4 / sklearn 1.9.0) and the RF oracle's
  predictions differ from that earlier cluster run. `env.lock.txt` has the
  exact versions.
- **One task dropped**: `health_data/Data_COVID19_Indonesia` — no row is free of
  missing values, so it fails both `Dataset.load` and the DRR estimator.
- Oracle RF uses sklearn defaults; absolute d2h is not comparable with the
  paper's, only rankings.

## Reproduce

```bash
conda run -n drr   python dehb_ezr/drr_interface.py --root data/moot/optimize --out results/dehb_ezr/drr.csv
conda run -n dmoot python -m pytest dehb_ezr/tests -q
conda run -n dmoot python -m dehb_ezr.batch --root data/moot/optimize --out dehb_ezr/results --seeds 10 -j 6
conda run -n dmoot python -m dehb_ezr.analyze --results dehb_ezr/results --drr results/dehb_ezr/drr.csv --out results/dehb_ezr
```

Files: `verdicts.csv` (one row per task), `bands.csv`, `drr.csv`,
`drr_estimator.patch`, `env.lock.txt`, `dehb_ezr/results/*.json` and
`*.runs.jsonl` (per-run traces).
