"""Random Search: the floor every other method must earn the right to beat.

Pool-based, like EZR/SWAY/LINE: it labels rows that exist in the table.
"""

from __future__ import annotations

import numpy as np

from data import Dataset
from oracle import BudgetGate
from optimizers.base import register


@register
class RandomSearch:
    name = "random"

    def run(self, ds: Dataset, gate: BudgetGate, seed: int) -> dict | None:
        rng = np.random.default_rng(seed)
        pool = ds.pool
        idx = rng.choice(len(pool), size=min(gate.budget, len(pool)), replace=False)
        for i in idx:
            gate(pool[i])
        return None          # runner falls back to gate.best_config