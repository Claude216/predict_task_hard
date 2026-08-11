"""The search space DEHB (and the ConfigSpace floor) sample from.

Reproduced from smac_ezr/optimizers/smac_opt.py rather than imported, because
importing that module pulls in `smac`, which is not installed in the `dmoot`
env. Keep the two in sync: if the space changes, a later DEHB-vs-SMAC
comparison is only valid while they agree.

WHY ORDINAL OVER OBSERVED VALUES, NOT A CONTINUOUS RANGE

Per MOOT's own framing the table carries no real-world ranges, so the values a
column is observed to take ARE its domain. SS-A's Spout_wait, for instance,
takes 13 values: 1..10, 100, 1000, 10000.

DEHB may still propose configurations that are not rows in the table -- a novel
combination of observed values -- and its final answer may be one. What it
cannot do is invent a value. That is deliberate:

  * The RF oracle is a step function. It splits on thresholds learned from the
    observed values, so Spout_wait=473.2 returns the identical prediction to
    Spout_wait=200. Most of the extra freedom of a continuous range is
    numerically inert.
  * Where it is not inert, it is extrapolation into a region the RF never saw.
    DEHB could then "win" on a surrogate artifact that EZR, being pool-based,
    can never chase -- and the win would be unattributable.

This is also exactly the "continuous index interpolation" reading: DEHB's DE
mutates a vector in [0,1]^d and recovers an Ordinal by binning that coordinate
into len(sequence) equal slices and indexing the sorted values
(dehb/optimizers/de.py, vector_to_configspace). Declaring Ordinal gets that
mapping natively, with no adapter layer and no divergence from DE's arithmetic.

KNOWN DEVIATION
The same de.py applies that index interpolation to CategoricalHyperparameter as
well. DE arithmetic therefore treats symbolic levels that are adjacent in the
choices list as "similar", imposing an order on unordered symbols. The order is
Dataset.levels, i.e. sorted() -- deterministic, but arbitrary. Not fixable
without forking DEHB; recorded in RESULTS.md.
"""

from __future__ import annotations

from ConfigSpace import Categorical, ConfigurationSpace, OrdinalHyperparameter

from .shared import Dataset


def build_configspace(ds: Dataset, seed: int = 0) -> ConfigurationSpace:
    """Ordinal for numeric columns, Categorical for symbolic ones."""
    cs = ConfigurationSpace(seed=seed)
    for c in ds.x_cols:
        dom = ds.domain(c)
        cs.add(Categorical(c, dom) if c in ds.levels else OrdinalHyperparameter(c, dom))
    return cs


def native(cfg: dict) -> dict:
    """numpy / ConfigSpace scalars -> plain python.

    Dataset.encode looks values up in `levels` lists built from the csv, so a
    np.str_ or np.int64 that compares equal but hashes differently would raise
    a ValueError deep inside encode(). Cheap to normalise here, painful to
    debug 40 minutes into a sweep.
    """
    return {k: (v.item() if hasattr(v, "item") else v) for k, v in cfg.items()}
