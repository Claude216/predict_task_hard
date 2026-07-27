"""The shared evaluation layer (paper IV-A, IV-B).

Oracle    = one frozen RF per goal, fit on the whole table, shared by EVERY
            optimizer so any surrogate bias is common to all of them (VI-E).
            Nearest-neighbour lookup is deliberately not used: it rewards
            solutions close to existing rows and so is unfair to
            membership-query methods like SMAC (IV-A).
BudgetGate= the only way to obtain a score.  Counts evaluations and records the
            full trajectory, which RQ3's GD/IGD/HV need.

Known deviation: the paper says "we tuned the surrogate" without giving the
search space, so RF hyperparameters here are scikit-learn defaults.  Absolute
d2h values are therefore not comparable with the paper's; rankings are.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from data import Dataset


class Oracle:
    """Frozen surrogate + d2h scoring.  Construct ONCE per dataset."""

    def __init__(self, ds: Dataset, seed: int = 0, **rf_kwargs):
        self.ds = ds
        X = ds.encode(ds.pool)
        self.models = {
            y: RandomForestRegressor(random_state=seed, n_jobs=-1, **rf_kwargs).fit(
                X, ds.df[y].values
            )
            for y in ds.y_cols
        }
        # Normalisation bounds come from the TRUE table values and never change,
        # so d2h stays comparable across methods, budgets and seeds.
        self.bounds = {
            y: (float(ds.df[y].min()), float(ds.df[y].max())) for y in ds.y_cols
        }

        self._spread: float | None = None

    # ------------------------------------------------------------------ #
    def predict(self, row: dict) -> dict[str, float]:
        """Raw per-goal predictions, before orientation or normalisation."""
        x = self.ds.encode(row)
        return {y: float(m.predict(x)[0]) for y, m in self.models.items()}

    def _oriented(self, y: str, raw: float) -> float:
        lo, hi = self.bounds[y]
        if hi == lo:
            return 0.0
        f = (raw - lo) / (hi - lo)
        return f if self.ds.goals[y] == "min" else 1.0 - f     # 0 = best

    def d2h(self, row: dict) -> float:
        """Eq.1: normalised Euclidean distance to the ideal point.  Lower better.

        Multi-objective tasks are collapsed here, BEFORE search, so every
        optimizer sees a single scalar (VI-C).
        """
        p = self.predict(row)
        vals = [self._oriented(y, p[y]) for y in self.ds.y_cols]
        return float(np.sqrt(np.mean(np.square(vals))))

    def d2h_many(self, rows: list[dict]) -> np.ndarray:
        """Vectorised d2h; used for whole-table reference statistics."""
        x = self.ds.encode(rows)
        cols = [
            np.array([self._oriented(y, v) for v in self.models[y].predict(x)])
            for y in self.ds.y_cols
        ]
        return np.sqrt(np.mean(np.square(np.column_stack(cols)), axis=1))

    # oracle.py,每个 task 算一次并缓存
    @property
    def d2h_spread(self) -> float:
        """Robust attainable spread on this task.  p90-p10 rather than max-min:
        the table's worst rows are outliers no optimizer goes near, and they
        would inflate eps."""
        if self._spread is None:
            v = self.d2h_many(self.ds.pool)
            self._spread = float(np.percentile(v, 90) - np.percentile(v, 10))
        return self._spread

# ---------------------------------------------------------------------- #
class BudgetExhausted(Exception):
    """Raised when an optimizer asks for one evaluation too many."""


@dataclass
class BudgetGate:
    """Per-run scoring gate.  Optimizers may not score anything except through
    this object, so the labelling budget cannot be silently overspent."""

    oracle: Oracle
    budget: int
    used: int = 0
    trace: list[tuple[dict, float]] = field(default_factory=list)

    def __call__(self, row: dict) -> float:
        if self.used >= self.budget:
            raise BudgetExhausted(f"budget {self.budget} exhausted")
        self.used += 1
        v = self.oracle.d2h(row)
        self.trace.append((dict(row), v))
        return v

    @property
    def remaining(self) -> int:
        return self.budget - self.used

    @property
    def best(self) -> float:
        return min(v for _, v in self.trace)

    @property
    def best_config(self) -> dict:
        return min(self.trace, key=lambda t: t[1])[0]