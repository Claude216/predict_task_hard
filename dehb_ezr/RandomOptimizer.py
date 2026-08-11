"""Two random floors, one per weight class.

One floor does not fit both arms, and using only a pool-based floor would make
the HEAVY comparison meaningless:

  random_pool @ 30   -- samples table rows without replacement. The right floor
                        for EZR, which is also pool-based.

  random_cs @ 3000   -- uniform samples from the SAME ConfigSpace DEHB searches.
                        Necessary because 45 of the 127 MOOT tasks (35%) have
                        fewer than 3000 rows: a pool-based floor at 3000 would
                        exhaust the table and report the global optimum by
                        construction, so every such task would show a tie and
                        the figure would be measuring table size, not search.

The DEHB floor is what separates "DEHB searched better" from "DEHB had 100x the
budget". A task where DEHB does not beat random_cs is marked in the figure
rather than counted as a DEHB win.
"""

from __future__ import annotations

import numpy as np

from .configspace import build_configspace, native
from .fidelity_oracle import FidelityGate
from .shared import BudgetExhausted, Dataset


class RandomPool:
    """Uniform over table rows, without replacement."""

    name = "random_pool"

    def run(self, ds: Dataset, gate: FidelityGate, seed: int) -> dict | None:
        rng = np.random.default_rng(seed)
        pool = ds.pool
        idx = rng.choice(len(pool), size=min(gate.budget, len(pool)), replace=False)
        for i in idx:
            gate(pool[i])
        return None


class RandomConfigSpace:
    """Uniform over the ConfigSpace, at full fidelity.

    Full fidelity throughout, deliberately. This floor answers "what does 3000
    blind draws from DEHB's own space get you?", so its draws must be scored on
    the same yardstick DEHB is finally judged by. Giving it a fidelity schedule
    would make it a second optimizer rather than a floor.
    """

    name = "random_cs"

    def run(self, ds: Dataset, gate: FidelityGate, seed: int) -> dict | None:
        cs = build_configspace(ds, seed=seed)
        cs.seed(seed)
        n = gate.budget
        # sample_configuration(n) returns a list for n > 1, a single config for n == 1
        configs = cs.sample_configuration(n) if n > 1 else [cs.sample_configuration()]
        for cfg in configs:
            try:
                gate(native(dict(cfg)))
            except BudgetExhausted:
                break
        return None
