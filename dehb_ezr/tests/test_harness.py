"""Guards for the three places this harness can be silently wrong.

    conda run -n dmoot python -m pytest dehb_ezr/tests -q

Run these BEFORE the sweep. Each one covers a failure mode that produces
plausible-looking numbers rather than an error, which is the only kind worth
a test here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from dehb_ezr.configspace import build_configspace, native      # noqa: E402
from dehb_ezr.fidelity_oracle import FidelityGate, FidelityOracle  # noqa: E402
from dehb_ezr.shared import BudgetExhausted, Dataset, Oracle, top  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent / "data" / "moot" / "optimize"
TASKS = ["config/SS-A.csv", "process/nasa93dem.csv", "binary_config/Scrum1k.csv"]


@pytest.fixture(scope="module", params=TASKS)
def loaded(request):
    ds = Dataset.load(str(ROOT / request.param))
    return ds, FidelityOracle(ds, seed=0)


# ---------------------------------------------------------------- fidelity
def test_full_fidelity_matches_stock_oracle(loaded):
    """The 84x speedup must be exact, not merely close.

    If tree_.predict ever diverges from rf.predict, every d2h in the study
    shifts and nothing raises. sklearn casts to float32 internally before
    reaching the tree, and so does tree_preds, so these should agree to
    floating-point noise on the mean of 100 values.
    """
    ds, fast = loaded
    slow = Oracle(ds, seed=0)                     # stock rf.predict path
    rows = ds.pool[:200]
    a = np.array([fast.d2h(r) for r in rows])
    b = np.array([slow.d2h(r) for r in rows])
    assert np.allclose(a, b, rtol=1e-9, atol=1e-10), np.abs(a - b).max()


def test_fidelity_is_monotone_in_information(loaded):
    """Low fidelity must be a noisy estimate of the same thing, not a
    different quantity: k = n_trees is the full value, and 1 <= k <= n_trees
    is enforced rather than silently wrapping."""
    ds, o = loaded
    row = ds.pool[0]
    preds = o.tree_preds(o.ds.encode(row))
    assert o.d2h_from_preds(preds, o.n_trees) == pytest.approx(o.d2h(row))
    assert o.d2h_from_preds(preds, 10**6) == pytest.approx(o.d2h(row))
    assert o.d2h_from_preds(preds, 0) == pytest.approx(o.d2h_from_preds(preds, 1))


# -------------------------------------------------------------------- gate
def test_gate_counts_every_call_regardless_of_fidelity(loaded):
    """A cached (promoted) config must still be charged: the cache exists to
    avoid recomputation, not to buy free evaluations."""
    ds, o = loaded
    gate = FidelityGate(o, budget=5)
    row = ds.pool[0]
    for k in (1, 3, 11, 33, 100):
        gate(row, fidelity=k)
    assert gate.used == 5
    with pytest.raises(BudgetExhausted):
        gate(row, fidelity=1)


def test_best_ignores_partial_fidelity(loaded):
    """A lucky 1-tree score must never become the reported result."""
    ds, o = loaded
    gate = FidelityGate(o, budget=10)
    for r in ds.pool[:8]:
        gate(r, fidelity=1)
    with pytest.raises(ValueError):        # nothing confirmed yet
        _ = gate.best
    gate(ds.pool[0], fidelity=o.n_trees)
    assert gate.n_full == 1
    assert gate.best == pytest.approx(o.d2h(ds.pool[0]))


# ------------------------------------------------------------- configspace
def test_configspace_covers_only_observed_values(loaded):
    """Every sampled value must exist in the column, or the RF is
    extrapolating and a DEHB win is unattributable."""
    ds, _ = loaded
    cs = build_configspace(ds, seed=0)
    assert set(cs.keys()) == set(ds.x_cols)
    for cfg in cs.sample_configuration(50):
        d = native(dict(cfg))
        for c in ds.x_cols:
            assert d[c] in set(ds.domain(c)), f"{c}={d[c]!r} not observed"
        ds.encode(d)                       # must not raise


def test_pool_rows_are_encodable_configs(loaded):
    """Table rows and optimizer proposals must live in one comparable space,
    otherwise best_on_table is meaningless."""
    ds, _ = loaded
    keys = ds.pool_keys
    for r in ds.pool[:50]:
        assert ds.key(r) in keys


# -------------------------------------------------------------------- top
def test_top_ties_identical_samples():
    x = [0.5 + 0.001 * i for i in range(10)]
    assert top({"a": list(x), "b": list(x)}, eps=0.01) == {"a", "b"}


def test_top_picks_the_lower_when_separated():
    lo = [0.10 + 0.001 * i for i in range(10)]
    hi = [0.90 + 0.001 * i for i in range(10)]
    # d2h is lower-better, so reverse stays False -- passing a higher-better
    # metric without reverse=True silently returns the WORST arm
    assert top({"lite": lo, "heavy": hi}, eps=0.01) == {"lite"}
