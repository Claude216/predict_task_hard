"""Evaluate one RF configuration: fit, predict, score (§3.5).

REGRESSION            MRE = mean |actual - pred| / actual        minimize
                      PRED40 = fraction of MREs <= 0.40          maximize
                      SA = (1 - MAE/MAE_dumb) * 100              maximize
                      MAE_dumb = MAE of predicting the training mean, the
                      paper's "simplest reasonable estimator".

CLASSIFICATION        accuracy, precision, recall, F1            maximize
                      The paper's formulas are binary (A,B,C,D). Several of
                      these datasets are multiclass (iris 3, heart num! 5,
                      power consumption many), so precision/recall/F1 are
                      MACRO-averaged. Declared deviation.

TWO SCORES, DELIBERATELY
The paper's D2H (Eq. 8) is a Zitzler RANK over "all evaluated examples from all
techniques", so it cannot be the online search signal: a configuration's score
would depend on configurations not yet evaluated. So:

  * search scalar  -- `d2h` here: distance to heaven over the metric vector
    using NATURAL bounds (all metrics live in [0,1] with a known heaven), so it
    is well defined from the first evaluation and identical for both arms.
  * reported score -- the Zitzler rank, computed post hoc in zitzler.py and used
    for the Scott-Knott verdict, exactly as the paper does.

Both are written to the run records so the verdicts' sensitivity to that choice
is checkable rather than assumed.

MRE IS UNDEFINED WHEN actual == 0. Those test rows are excluded from MRE and
PRED40 (and the count is reported). Silently adding an epsilon would turn a
divide-by-zero into an arbitrarily large error term that then dominates d2h.
"""

from __future__ import annotations

import warnings

import numpy as np

from . import space as S

warnings.filterwarnings("ignore")

PRED40_THRESHOLD = 0.40      # MRE is a ratio here, so 40% is 0.40

REG_METRICS = ["MRE", "PRED40", "SA"]
CLF_METRICS = ["accuracy", "precision", "recall", "f1"]
# +1 maximize, -1 minimize -- the w_j of the paper's Zitzler indicator
REG_W = {"MRE": -1, "PRED40": +1, "SA": +1}
CLF_W = {m: +1 for m in CLF_METRICS}


def metric_names(kind: str) -> list[str]:
    return REG_METRICS if kind == "regression" else CLF_METRICS


def weights(kind: str) -> dict[str, int]:
    return REG_W if kind == "regression" else CLF_W


# ------------------------------------------------------------------ scoring
def regression_metrics(actual: np.ndarray, pred: np.ndarray,
                       train_mean: float) -> tuple[dict, int]:
    nonzero = actual != 0
    n_skipped = int((~nonzero).sum())
    if nonzero.any():
        mre_i = np.abs(actual[nonzero] - pred[nonzero]) / np.abs(actual[nonzero])
        mre = float(np.mean(mre_i))
        pred40 = float(np.mean(mre_i <= PRED40_THRESHOLD))
    else:
        mre, pred40 = float("inf"), 0.0
    mae = float(np.mean(np.abs(actual - pred)))
    mae_dumb = float(np.mean(np.abs(actual - train_mean)))
    sa = (1 - mae / mae_dumb) * 100 if mae_dumb > 0 else 0.0
    return {"MRE": mre, "PRED40": pred40, "SA": float(sa)}, n_skipped


def classification_metrics(actual: np.ndarray, pred: np.ndarray) -> dict:
    from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                                 recall_score)
    kw = dict(average="macro", zero_division=0)
    return {"accuracy": float(accuracy_score(actual, pred)),
            "precision": float(precision_score(actual, pred, **kw)),
            "recall": float(recall_score(actual, pred, **kw)),
            "f1": float(f1_score(actual, pred, **kw))}


def d2h(metrics: dict, kind: str) -> float:
    """Distance to heaven over natural metric bounds. 0 is best.

    Each term is mapped to a [0,1] "distance from perfect":
      accuracy/precision/recall/F1, PRED40  -> 1 - m           (heaven 1)
      SA (percent, can go negative when the model loses to the
         training mean)                     -> 1 - clip(SA/100, 0, 1)
      MRE (unbounded above)                 -> clip(MRE, 0, 1)
    Clipping SA and MRE keeps one unboundedly-bad metric from swamping the rest;
    a model with MRE > 1 or SA < 0 is already worthless, so the lost resolution
    is in a region no optimizer should be choosing between.
    """
    if kind == "regression":
        v = [min(max(metrics["MRE"], 0.0), 1.0),
             1 - min(max(metrics["SA"] / 100.0, 0.0), 1.0),
             1 - metrics["PRED40"]]
    else:
        v = [1 - metrics[m] for m in CLF_METRICS]
    return float(np.sqrt(np.mean(np.square(v))))


WORST = 1.0      # d2h of a configuration that could not be fitted


# --------------------------------------------------------------- objective
class Objective:
    """Fit-and-score for one (dataset, repeat). Deterministic and cached.

    Fidelity is the number of TRAINING ROWS the forest sees -- the classic
    multi-fidelity axis for HPO, and the only sensible one here since
    n_estimators is itself part of the search space. The rows are a prefix of a
    seeded permutation, so a promoted configuration sees a superset of what it
    saw at the lower rung.
    """

    def __init__(self, xtr, ytr, xte, yte, kind: str, seed: int):
        self.xtr, self.ytr, self.xte, self.yte = xtr, ytr, xte, yte
        self.kind, self.seed = kind, seed
        self.n_train = len(ytr)
        self.train_mean = float(np.mean(ytr)) if kind == "regression" else 0.0
        self._order = np.random.default_rng(seed).permutation(self.n_train)
        self._cache: dict[tuple, dict] = {}
        self.n_fits = 0
        self.n_failed = 0
        self.n_skipped_zero_actual = 0

    # ------------------------------------------------------------------ #
    def evaluate(self, cfg: dict, n_rows: int | None = None) -> dict:
        cfg = S.native(cfg)
        n = self.n_train if n_rows is None else int(np.clip(n_rows, 1, self.n_train))
        ck = (S.key(cfg), n)
        hit = self._cache.get(ck)
        if hit is not None:
            return hit
        out = self._fit_score(cfg, n)
        self._cache[ck] = out
        return out

    def _fit_score(self, cfg: dict, n: int) -> dict:
        from sklearn.ensemble import (RandomForestClassifier,
                                      RandomForestRegressor)
        idx = self._order[:n]
        xtr, ytr = self.xtr[idx], self.ytr[idx]
        Model = RandomForestRegressor if self.kind == "regression" else RandomForestClassifier
        try:
            m = Model(**S.as_rf_kwargs(cfg, self.kind, self.seed)).fit(xtr, ytr)
            pred = m.predict(self.xte)
            self.n_fits += 1
        except Exception as e:
            # e.g. criterion='poisson' rejects negative targets. A config the
            # learner cannot accept is a legitimately worst-possible config, not
            # a crash -- but it is counted, so a dataset where everything fails
            # is visible instead of silently scoring 1.0 everywhere.
            self.n_failed += 1
            names = metric_names(self.kind)
            return {"metrics": {k: (float("inf") if k == "MRE" else 0.0) for k in names},
                    "d2h": WORST, "n_rows": n, "failed": f"{type(e).__name__}: {e}"}

        if self.kind == "regression":
            met, skipped = regression_metrics(self.yte, pred, self.train_mean)
            self.n_skipped_zero_actual = skipped
        else:
            met = classification_metrics(self.yte, pred)
        return {"metrics": met, "d2h": d2h(met, self.kind), "n_rows": n, "failed": None}
