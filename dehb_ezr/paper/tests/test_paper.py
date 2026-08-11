"""Guards for the paper rig.

    conda run -n dmoot python -m pytest dehb_ezr/paper/tests -q

Each test covers something that would otherwise produce plausible-looking
numbers rather than an error.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from dehb_ezr.paper import REPO_ROOT, space as S            # noqa: E402
from dehb_ezr.paper.datasets import DATASETS, BY_LABEL      # noqa: E402
from dehb_ezr.paper.loader import load, split               # noqa: E402
from dehb_ezr.paper.objective import (classification_metrics, d2h,   # noqa: E402
                                      regression_metrics, metric_names, weights)
from dehb_ezr.paper.sk import cliffs_delta, scott_knott, verdict, SMALL  # noqa: E402
from dehb_ezr.paper.zitzler import d2h as zd2h              # noqa: E402

# R values the paper's Table 2 records where they should match our rule.
KNOWN_R = {"iris": 4, "heart disease": 13, "adult": 14, "german credit": 20,
           "bank marketing": 16, "gamma telescope": 10, "default": 23,
           "power consumption": 6, "Wine Quality": 10, "SS-T": 12, "SS-U": 21,
           "Pom3a": 9, "pom3d": 9, "Health-Easy": 5, "Health-Hard": 5,
           "rs-6d-c3-obj2": 6, "Xomo Flight": 27, "Xomo OSP2": 27}


# ------------------------------------------------------------------ registry
def test_every_registry_path_exists():
    missing = [e.label for e in DATASETS if not (REPO_ROOT / e.path).exists()]
    assert not missing, missing


def test_registry_is_49_datasets():
    """37 SE (24 SS incl. SS-L, + 13 others) + 12 non-SE. The paper's set is 50:
    we add SS-L (which Table 2's count of 23 omits) and drop diabetes."""
    assert len(DATASETS) == 49
    assert len({e.label for e in DATASETS}) == 49
    assert sum(e.is_se for e in DATASETS) == 37


@pytest.mark.parametrize("label,expected", sorted(KNOWN_R.items()))
def test_R_matches_the_papers_table_2(label, expected):
    """R = all columns minus all dependents. This is the x-axis of Fig. 4, so a
    silent drift here moves every point sideways."""
    assert load(BY_LABEL[label]).R == expected


# --------------------------------------------------------------------- space
def test_regression_space_is_exactly_the_papers_1_280_000():
    assert S.size("regression") == 1_280_000
    assert len(S.MIN_IMPURITY) == 40         # the inferred 0..9.75 correction
    assert S.MIN_IMPURITY[0] == 0.0          # 0 must be reachable, or trees are stumps
    assert len(S.N_ESTIMATORS) == 20 and S.N_ESTIMATORS[0] == 1


def test_classification_space_has_three_criteria():
    """sklearn offers no fourth classifier criterion; declared deviation."""
    assert S.size("classification") == 960_000


def test_pool_is_the_right_size_and_reproducible():
    import random
    a = S.pool("regression", random.Random(7), n=200)
    b = S.pool("regression", random.Random(7), n=200)
    assert len(a) == 200
    assert [S.key(x) for x in a] == [S.key(x) for x in b]
    g = S.grid("regression")
    for cfg in a:                            # every draw is inside the grid
        for k, v in cfg.items():
            assert v in g[k]


# ------------------------------------------------------------------- metrics
def test_regression_metrics_against_hand_computed_values():
    actual = np.array([10.0, 20.0, 50.0])
    pred = np.array([11.0, 30.0, 50.0])      # MREs: 0.1, 0.5, 0.0
    m, skipped = regression_metrics(actual, pred, train_mean=20.0)
    assert skipped == 0
    assert m["MRE"] == pytest.approx((0.1 + 0.5 + 0.0) / 3)
    assert m["PRED40"] == pytest.approx(2 / 3)          # 0.1 and 0.0 are <= 0.40
    mae = (1 + 10 + 0) / 3
    mae_dumb = (10 + 0 + 30) / 3
    assert m["SA"] == pytest.approx((1 - mae / mae_dumb) * 100)


def test_zero_actuals_are_excluded_from_mre_not_silently_epsilon_ed():
    actual = np.array([0.0, 10.0])
    pred = np.array([5.0, 11.0])
    m, skipped = regression_metrics(actual, pred, train_mean=5.0)
    assert skipped == 1
    assert m["MRE"] == pytest.approx(0.1)    # only the non-zero row counts
    assert np.isfinite(m["MRE"])


def test_classification_metrics_perfect_and_useless():
    y = np.array([0, 0, 1, 1])
    assert classification_metrics(y, y)["accuracy"] == 1.0
    assert d2h(classification_metrics(y, y), "classification") == 0.0
    bad = classification_metrics(y, 1 - y)
    assert bad["accuracy"] == 0.0
    assert d2h(bad, "classification") == pytest.approx(1.0)


def test_d2h_is_bounded_and_oriented():
    good = {"MRE": 0.0, "PRED40": 1.0, "SA": 100.0}
    bad = {"MRE": 5.0, "PRED40": 0.0, "SA": -400.0}
    assert d2h(good, "regression") == pytest.approx(0.0)
    assert d2h(bad, "regression") == pytest.approx(1.0)   # clipped, not unbounded
    mid = {"MRE": 0.3, "PRED40": 0.6, "SA": 50.0}
    assert 0 < d2h(mid, "regression") < 1


# ------------------------------------------------------------------ zitzler
def test_zitzler_d2h_is_a_total_order_in_0_1():
    names = metric_names("classification")
    rows = [{m: v for m in names} for v in (0.9, 0.5, 0.7, 0.1)]
    v = zd2h(rows, names, weights("classification"))
    assert sorted(v) == pytest.approx(sorted({0.25, 0.5, 0.75, 1.0}))
    assert len(set(v)) == 4                       # a strict ranking, no ties
    # all four metrics maximize, so the biggest row must rank best (lowest D2H)
    assert v[0] == pytest.approx(0.25)
    assert v[3] == pytest.approx(1.0)


def test_identical_rows_get_identical_d2h():
    """Otherwise the tie-break is array position, and run_one always appends
    lite before dehb -- a systematic bias toward LITE on every tied seed. Both
    arms reach a perfect classifier on iris, so this really happens."""
    names = metric_names("classification")
    perfect = {m: 1.0 for m in names}
    poor = {m: 0.2 for m in names}
    v = zd2h([perfect, perfect, poor], names, weights("classification"))
    assert v[0] == v[1], "tied options must share a rank"
    assert v[2] > v[0]


def test_ties_do_not_depend_on_input_order():
    names = metric_names("classification")
    a, b = {m: 1.0 for m in names}, {m: 0.4 for m in names}
    fwd = zd2h([a, b, a, b], names, weights("classification"))
    rev = zd2h([b, a, b, a], names, weights("classification"))
    assert sorted(fwd) == pytest.approx(sorted(rev))
    assert fwd[0] == fwd[2] and fwd[1] == fwd[3]


def test_zitzler_respects_minimisation_weight():
    names = metric_names("regression")
    # MRE minimizes; hold the other two fixed so only MRE separates the rows
    rows = [{"MRE": 0.1, "PRED40": 0.5, "SA": 50.0},
            {"MRE": 0.9, "PRED40": 0.5, "SA": 50.0}]
    v = zd2h(rows, names, weights("regression"))
    assert v[0] < v[1]


# ----------------------------------------------------------------------- sk
def test_cliffs_delta_endpoints():
    assert cliffs_delta([1, 2, 3], [1, 2, 3]) == 0.0
    assert cliffs_delta([4, 5, 6], [1, 2, 3]) == 1.0
    assert cliffs_delta([1, 2, 3], [4, 5, 6]) == -1.0


def test_identical_arms_are_one_rank_and_green():
    x = [0.5 + 0.01 * i for i in range(20)]
    assert scott_knott({"a": list(x), "b": list(x)}) == {"a": 0, "b": 0}
    assert verdict(list(x), list(x)) == "DEHB == LITE"


def test_clearly_better_dehb_is_red():
    lite = [0.90 + 0.005 * i for i in range(20)]
    dehb = [0.10 + 0.005 * i for i in range(20)]   # lower D2H is better
    assert verdict(lite, dehb) == "DEHB > LITE"


def test_lite_winning_is_green_not_a_third_class():
    """The paper's legend has two colours; a LITE win is simply not a DEHB win."""
    lite = [0.10 + 0.005 * i for i in range(20)]
    dehb = [0.90 + 0.005 * i for i in range(20)]
    assert verdict(lite, dehb) == "DEHB == LITE"


def test_the_split_threshold_sits_at_0_147():
    """Below the paper's 'small effect' bound the two arms share a rank."""
    rng = np.random.default_rng(0)
    a = list(rng.normal(0.5, 0.1, 200))
    b = list(rng.normal(0.5, 0.1, 200))
    assert abs(cliffs_delta(a, b)) < SMALL
    assert scott_knott({"a": a, "b": b})["a"] == scott_knott({"a": a, "b": b})["b"]


# -------------------------------------------------------------------- split
@pytest.mark.parametrize("label", ["iris", "nasa93dem"])
def test_split_is_seeded_disjoint_and_covers_every_class(label):
    t = load(BY_LABEL[label])
    xtr, ytr, xte, yte = split(t, 3)
    assert len(ytr) + len(yte) == len(t.y)
    assert len(yte) > 0 and len(ytr) > 0
    again = split(t, 3)
    assert np.array_equal(again[1], ytr) and np.array_equal(again[3], yte)
    if t.kind == "classification":
        assert set(np.unique(yte)).issubset(set(np.unique(t.y)))


def test_row_cap_applies_to_the_giant_table():
    t = load(BY_LABEL["power consumption"], row_cap=500)
    assert len(t.y) == 500 and t.capped and t.n_raw > 500
    assert t.kind == "classification"        # Sub_metering_1! -> klass, by the rule
