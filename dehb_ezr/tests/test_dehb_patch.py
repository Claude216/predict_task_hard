"""The DEHB vector->config patch: does it fix the crash without moving results?

    conda run -n dmoot python -m pytest dehb_ezr/tests/test_dehb_patch.py -q

Two things must both hold, or the patch is not safe to apply to a completed
sweep:

  1. It maps the coordinate that used to raise IndexError.
  2. Everywhere else it agrees with DEHB's own mapping -- otherwise the 121
     tasks that never hit the bug would need re-running too.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from ConfigSpace import Categorical, ConfigurationSpace, OrdinalHyperparameter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from dehb_ezr.DEHBOptimizer import _patch_vector_to_configspace   # noqa: E402

# 49 is the length that broke misc/Car_price_cleaned: np.arange(0, 1, 1/49)
# yields 50 edges, so a coordinate at 1.0 indexed sequence[49] of 49.
BAD_LENGTHS = [49, 98, 103, 107, 196]


def test_arange_really_is_the_culprit():
    """Documents the upstream defect this patch exists for."""
    assert len(np.arange(0, 1, 1 / 49)) == 50
    assert 49 * (1 / 49) < 1.0


def _de(cs):
    from dehb.optimizers.de import DEBase
    de = DEBase.__new__(DEBase)
    de.cs = cs
    return de


@pytest.mark.parametrize("n", BAD_LENGTHS)
def test_patch_handles_the_coordinate_that_used_to_raise(n):
    _patch_vector_to_configspace()
    cs = ConfigurationSpace()
    cs.add(OrdinalHyperparameter("a", list(range(n))))
    de = _de(cs)
    for v in (1.0, 0.9999999999999999, 1.0 - 1e-18):
        cfg = de.vector_to_configspace(np.array([v]))
        assert cfg["a"] == n - 1          # clamped to the last value, not IndexError


def test_patch_clamps_out_of_range_coordinates():
    _patch_vector_to_configspace()
    cs = ConfigurationSpace()
    cs.add(OrdinalHyperparameter("a", [10, 20, 30]))
    de = _de(cs)
    assert de.vector_to_configspace(np.array([-0.5]))["a"] == 10
    assert de.vector_to_configspace(np.array([1.5]))["a"] == 30


@pytest.mark.parametrize("n", [2, 3, 5, 13, 48, 50, 124, 205])
def test_patch_agrees_with_dehbs_own_mapping_elsewhere(n):
    """The reason the completed sweep did not need re-running wholesale."""
    _patch_vector_to_configspace()
    seq = list(range(n))
    cs = ConfigurationSpace()
    cs.add(OrdinalHyperparameter("a", seq))
    de = _de(cs)
    ranges = np.arange(start=0, stop=1, step=1 / n)
    for v in np.linspace(0, 1, 997):
        orig_idx = np.where((v < ranges) == False)[0][-1]   # noqa: E712
        if orig_idx >= n:
            continue                    # the broken sliver; covered above
        assert de.vector_to_configspace(np.array([v]))["a"] == seq[orig_idx]


@pytest.mark.parametrize("n", [2, 49, 205])
def test_round_trip_matches_whatever_dehb_itself_does(n):
    """configspace_to_vector maps index -> index/n.

    DEHB does NOT round-trip that perfectly: at some k, arange's edge for k is a
    last-bit above k/n and the index comes back k-1. That is upstream behaviour,
    not something this patch introduces or should silently 'improve' -- changing
    it would alter results on tasks that never hit the bug. So the assertion is
    agreement with DEHB, and the imperfect round-trips are counted and reported
    rather than asserted away.
    """
    _patch_vector_to_configspace()
    seq = list(range(n))
    cs = ConfigurationSpace()
    cs.add(OrdinalHyperparameter("a", seq))
    de = _de(cs)
    ranges = np.arange(start=0, stop=1, step=1 / n)
    off = 0
    for k in range(n):
        got = de.vector_to_configspace(np.array([k / n]))["a"]
        orig_idx = min(int(np.where((k / n < ranges) == False)[0][-1]), n - 1)  # noqa: E712
        assert got == seq[orig_idx], f"patch disagrees with DEHB at k={k}"
        off += got != seq[k]
    assert off <= 2, f"{off} of {n} indices drift; expected upstream noise only"


def test_categorical_is_handled_too():
    _patch_vector_to_configspace()
    cs = ConfigurationSpace()
    cs.add(Categorical("c", ["x", "y", "z"]))
    de = _de(cs)
    assert de.vector_to_configspace(np.array([0.0]))["c"] == "x"
    assert de.vector_to_configspace(np.array([1.0]))["c"] == "z"
