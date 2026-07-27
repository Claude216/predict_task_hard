"""SMAC adapter.  Table IV: RF surrogate + EI + init 10, otherwise defaults.

This module owns the ConfigurationSpace, because only membership-query methods
need one.  Per VI-D the domain of each decision variable is the set of values
observed in the table, so numeric columns become Ordinal and symbolic columns
become Categorical.  Integer(min, max) would let SMAC roam through values the
table never contains, where the RF has no data and its predictions are pure
extrapolation.
"""

from __future__ import annotations

import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

from ConfigSpace import Categorical, Configuration, ConfigurationSpace, OrdinalHyperparameter
from smac import HyperparameterOptimizationFacade as HPOFacade
from smac import Scenario

from data import Dataset
from oracle import BudgetExhausted, BudgetGate
from .base import register


INIT_CONFIGS = 10          # Table IV


def build_configspace(ds: Dataset, seed: int = 0) -> ConfigurationSpace:
    cs = ConfigurationSpace(seed=seed)
    for c in ds.x_cols:
        dom = ds.domain(c)
        cs.add(Categorical(c, dom) if c in ds.levels else OrdinalHyperparameter(c, dom))
    return cs


@register
class SMAC:
    name = "smac"

    def __init__(self, out_dir: str = "/tmp/smoot_smac"):
        self.out_dir = out_dir

    def run(self, ds: Dataset, gate: BudgetGate, seed: int) -> dict | None:
        cs = build_configspace(ds)

        def target(config: Configuration, seed: int = 0) -> float:
            try:
                return gate(dict(config))
            except BudgetExhausted:
                return float("inf")

        scenario = Scenario(
            cs,
            deterministic=True,        # the frozen oracle is noise-free
            n_trials=gate.budget,
            seed=seed,
            output_directory=f"{self.out_dir}/{Path(ds.path).stem}/{seed}_{gate.budget}",
        )
        smac = HPOFacade(
            scenario,
            target,
            initial_design=HPOFacade.get_initial_design(
                scenario,
                n_configs=INIT_CONFIGS,
                # without max_ratio=1.0 the default 0.25 silently cuts the
                # initial design to 7 configs at B=30
                max_ratio=1.0,
            ),
            overwrite=True,
            logging_level=40,
        )
        return dict(smac.optimize())