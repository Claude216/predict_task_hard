"""The RandomForest hyperparameter space the paper searches (§3.3.1).

    n_estimators           1..200 step 10          -> 20 values (1, 11, ..., 191)
    criterion              4 regression options    -> 4
    min_samples_leaf       1..20                   -> 20
    min_impurity_decrease  step 0.25               -> 40  (see below)
    max_depth              1..20                   -> 20

    20 * 4 * 20 * 40 * 20 = 1,280,000, exactly the paper's figure.

THE min_impurity_decrease RANGE IS AN INFERRED CORRECTION.
The paper's prose says "Varied from 1 to 10 (in steps of 0.25)", which is 37
values and contradicts the 40 its own product requires. 0 to 9.75 step 0.25 is
exactly 40. It is also the only reading that works: a classifier's weighted gini
impurity decrease can never reach 1.0, so a range starting at 1 would forbid
every split, make every tree a stump, and make all classification configurations
identical -- which would render half of Fig. 4 degenerate. Recorded in
RESULTS.md as a deviation from the printed text.

CLASSIFICATION HAS THREE CRITERIA, NOT FOUR.
The paper lists the regression criteria. scikit-learn's RandomForestClassifier
offers gini/entropy/log_loss only, so the classification space is 960,000.
"""

from __future__ import annotations

import random

import numpy as np

N_ESTIMATORS = list(range(1, 200, 10))                       # 20
CRITERION_REG = ["squared_error", "absolute_error", "friedman_mse", "poisson"]
CRITERION_CLF = ["gini", "entropy", "log_loss"]
MIN_SAMPLES_LEAF = list(range(1, 21))                        # 20
MIN_IMPURITY = [round(0.25 * i, 2) for i in range(40)]       # 0 .. 9.75  -> 40
MAX_DEPTH = list(range(1, 21))                               # 20

POOL_SIZE = 10_000        # the paper's per-dataset candidate pool
NAMES = ["n_estimators", "criterion", "min_samples_leaf",
         "min_impurity_decrease", "max_depth"]


def grid(kind: str) -> dict[str, list]:
    return {
        "n_estimators": N_ESTIMATORS,
        "criterion": CRITERION_REG if kind == "regression" else CRITERION_CLF,
        "min_samples_leaf": MIN_SAMPLES_LEAF,
        "min_impurity_decrease": MIN_IMPURITY,
        "max_depth": MAX_DEPTH,
    }


def size(kind: str) -> int:
    n = 1
    for v in grid(kind).values():
        n *= len(v)
    return n


def configspace(kind: str, seed: int = 0):
    """ConfigSpace for DEHB. Ordinal for the ordered numerics, Categorical for
    criterion -- the same convention dehb_ezr/configspace.py argues for, so DE's
    index interpolation stays monotone in each ordered parameter."""
    from ConfigSpace import Categorical, ConfigurationSpace, OrdinalHyperparameter

    g = grid(kind)
    cs = ConfigurationSpace(seed=seed)
    for name, values in g.items():
        cs.add(Categorical(name, values) if name == "criterion"
               else OrdinalHyperparameter(name, values))
    return cs


def pool(kind: str, rng: random.Random, n: int = POOL_SIZE) -> list[dict]:
    """The paper's pool: n configurations drawn at random from the space.

    Drawn fresh per (dataset, repeat) from that repeat's seed, so the 20 repeats
    are genuinely independent rather than differing only in RF seed and split.
    Duplicates are possible in principle (10,000 out of 1.28M) but are not removed:
    the paper says "10,000 randomly selected", and de-duplicating would quietly
    change the sampling distribution.
    """
    g = grid(kind)
    return [{k: rng.choice(v) for k, v in g.items()} for _ in range(n)]


def as_rf_kwargs(cfg: dict, kind: str, seed: int, n_jobs: int = 1) -> dict:
    """Config dict -> scikit-learn constructor kwargs."""
    return dict(
        n_estimators=int(cfg["n_estimators"]),
        criterion=str(cfg["criterion"]),
        min_samples_leaf=int(cfg["min_samples_leaf"]),
        min_impurity_decrease=float(cfg["min_impurity_decrease"]),
        max_depth=int(cfg["max_depth"]),
        random_state=seed,
        n_jobs=n_jobs,
    )


def key(cfg: dict) -> tuple:
    """Hashable, order-stable cache key."""
    return tuple(
        (v.item() if hasattr(v, "item") else v) for v in (cfg[n] for n in NAMES))


def native(cfg: dict) -> dict:
    """numpy / ConfigSpace scalars -> plain python, so cache keys and sklearn
    kwargs never depend on which library produced the value."""
    return {k: (v.item() if hasattr(v, "item") else v) for k, v in cfg.items()}


__all__ = ["grid", "size", "configspace", "pool", "as_rf_kwargs", "key", "native",
           "NAMES", "POOL_SIZE", "MIN_IMPURITY", "N_ESTIMATORS",
           "CRITERION_REG", "CRITERION_CLF", "MAX_DEPTH", "MIN_SAMPLES_LEAF",
           "np"]
