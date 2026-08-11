"""Tree-count fidelity on top of smac_ezr's frozen-RF oracle.

WHY A FIDELITY AT ALL
DEHB is Hyperband + differential evolution: successive halving needs a cheap
approximation of the objective. A MOOT table is a static lookup, so there is no
natural fidelity axis. The one used here is the number of trees in the frozen
RF that vote on the prediction: fidelity k averages the first k of 100 trees.
Low k is a genuinely noisier estimate of the same quantity; k = 100 is
bit-identical to the value EZR and the floors are scored on, so the arms remain
comparable at the point where it matters.

It is a SYNTHETIC fidelity: k = 1 is noisier than k = 100 but not meaningfully
cheaper in wall time. That is recorded as a known deviation rather than hidden.

WHY tree_.predict AND NOT rf.predict
Measured on one MOOT task (Scrum1k, 1 row, 100 trees), all returning the
identical value 259.220000000000:

    rf.predict, n_jobs=-1 (smac_ezr's setting)   13.48 ms
    rf.predict, n_jobs=None                       1.85 ms
    loop over rf.estimators_                       2.93 ms
    loop over est.tree_.predict                    0.16 ms

sklearn's predict pays joblib dispatch overhead that dwarfs the work on a
one-row input. At 3000 evals x 10 seeds x 127 tasks that difference is the
whole feasibility of the sweep. The last path is exact, not an approximation:
sklearn's own _validate_X_predict casts to float32 before reaching the tree, so
this does the same cast and averages the same numbers. test_fidelity.py asserts
the equality rather than trusting this paragraph.

The per-tree vector is also exactly what fidelity needs, so the speedup and the
mechanism are one piece of code.

WHAT "BEST" MEANS
gate.best is the best score over FULL-fidelity evaluations only. A 1-tree
evaluation that happens to return a low number is not a result. DEHB therefore
reports the best config it actually confirmed at 100 trees -- see
DEHBOptimizer.py, which reserves one evaluation to confirm its incumbent.
"""

from __future__ import annotations

import numpy as np

from .shared import BudgetExhausted, Dataset, Oracle

N_TREES = 100          # eta=3 over 1..100 gives fidelities {1, 3, 11, 33, 100}


class FidelityOracle(Oracle):
    """Oracle + per-tree predictions + d2h at a partial tree count."""

    def __init__(self, ds: Dataset, seed: int = 0, n_trees: int = N_TREES, **rf_kwargs):
        super().__init__(ds, seed=seed, n_estimators=n_trees, **rf_kwargs)
        self.n_trees = n_trees
        for y, m in self.models.items():
            assert len(m.estimators_) == n_trees, (
                f"goal {y}: asked for {n_trees} trees, got {len(m.estimators_)}")

    # ------------------------------------------------------------------ #
    def tree_preds(self, x: np.ndarray) -> dict[str, np.ndarray]:
        """Per-goal array of this row's prediction from each individual tree.

        `x` is an already-encoded (1, d) row -- callers encode once and reuse
        the same array for the cache key, since Dataset.encode is a Python loop
        over columns and Scrum1k has 124 of them.
        """
        x32 = np.ascontiguousarray(x, dtype=np.float32)
        return {
            y: np.fromiter((e.tree_.predict(x32)[0, 0] for e in m.estimators_),
                           dtype=float, count=self.n_trees)
            for y, m in self.models.items()
        }

    def d2h_from_preds(self, preds: dict[str, np.ndarray], n_trees: int | None) -> float:
        """Eq.1 d2h using the first `n_trees` trees. None means full fidelity."""
        k = self.n_trees if n_trees is None else int(np.clip(n_trees, 1, self.n_trees))
        vals = [self._oriented(y, float(preds[y][:k].mean())) for y in self.ds.y_cols]
        return float(np.sqrt(np.mean(np.square(vals))))

    def d2h(self, row: dict) -> float:
        """Override: same value as Oracle.d2h, ~84x faster. See module docstring."""
        return self.d2h_from_preds(self.tree_preds(self.ds.encode(row)), None)


# ---------------------------------------------------------------------- #
class FidelityGate:
    """Per-run scoring gate. The ONLY way any arm may obtain a score.

    Deliberately not a subclass of smac_ezr's BudgetGate: the trace here carries
    a third field (the fidelity) and `best` filters on it, which is a different
    contract, and pretending otherwise would let a low-fidelity number leak into
    a comparison.

    Every call costs exactly 1 regardless of fidelity. DEHB's 3000 is therefore
    3000 oracle calls, directly countable against EZR's 30. Charging
    fidelity-proportionally was considered and rejected: it would let DEHB see
    far more than 3000 configurations and break comparability.
    """

    def __init__(self, oracle: FidelityOracle, budget: int):
        self.oracle = oracle
        self.budget = budget
        self.used = 0
        self.trace: list[tuple[dict, float, int]] = []   # (row, d2h, fidelity)
        # The encoded key of trace[i], kept in step with it. Dataset.encode is a
        # Python loop over columns and costs 1.8 ms/row at 128 decisions (14 ms
        # at FFM-1000's 1044). Every caller needs this key -- the tree cache
        # here, trace_on_table_rate in run_task, _confirm in DEHBOptimizer --
        # so computing it once and sharing it removes two thirds of all encode
        # calls in the sweep. Encoding is deterministic, so this is purely a
        # cost saving: no number changes.
        self.trace_keys: list[tuple] = []
        self._cache: dict[tuple, dict[str, np.ndarray]] = {}

    def __call__(self, row: dict, fidelity: int | None = None) -> float:
        if self.used >= self.budget:
            raise BudgetExhausted(f"budget {self.budget} exhausted")
        self.used += 1
        x = self.oracle.ds.encode(row)
        key = tuple(x[0])
        preds = self._cache.get(key)
        if preds is None:
            # A config promoted up the bracket is re-scored at a higher fidelity
            # from the SAME tree predictions -- no recomputation, and the cache
            # never bypasses the counter above.
            preds = self._cache[key] = self.oracle.tree_preds(x)
        k = self.oracle.n_trees if fidelity is None else int(
            np.clip(fidelity, 1, self.oracle.n_trees))
        v = self.oracle.d2h_from_preds(preds, k)
        self.trace.append((dict(row), v, k))
        self.trace_keys.append(key)
        return v

    # ------------------------------------------------------------------ #
    @property
    def remaining(self) -> int:
        return self.budget - self.used

    @property
    def full_trace(self) -> list[tuple[dict, float]]:
        """Only the evaluations that used every tree. See module docstring."""
        return [(r, v) for r, v, k in self.trace if k >= self.oracle.n_trees]

    @property
    def best(self) -> float:
        ft = self.full_trace
        if not ft:
            raise ValueError(
                "no full-fidelity evaluation in the trace: this arm never "
                "confirmed any configuration, so it has no reportable score")
        return min(v for _, v in ft)

    @property
    def best_config(self) -> dict:
        ft = self.full_trace
        if not ft:
            raise ValueError("no full-fidelity evaluation in the trace")
        return min(ft, key=lambda t: t[1])[0]

    @property
    def n_full(self) -> int:
        return len(self.full_trace)
