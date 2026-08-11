"""EZR adapter: the LITE arm. 30 labels, pool-based.

This is a thin pin over smac_ezr/optimizers/ezr_opt.py, NOT a re-implementation.
That adapter already solves three non-obvious problems, each of which silently
corrupts the comparison if got wrong:

1. SCALARISATION. Stock EZR scores rows with its own disty(), a logistic
   normalisation about each goal's running mean -- not the Eq.1 min-max
   normalisation the oracle uses. Letting EZR keep disty() would judge EZR and
   DEHB on two different yardsticks. ezr_opt replaces the goal columns with one
   synthetic "D2h-" column carrying the shared oracle's score, so EZR's
   mechanism is untouched and only the yardstick is shared.

2. BUDGET. warm_start labels the.learn.start rows and the loop labels
   the.learn.budget more, so learn.budget is set to B - start to spend exactly
   B. Labelling is idempotent per row, so warm_start's sort does not double-charge.

3. WARM START. acquire() shuffles and clones the first `start` rows BEFORE
   calling label(), so under lazy labelling those rows still carry "?", the
   D2h column clones with sd=0, and the best/rest split degenerates. ezr_opt
   replays the same shuffle to pre-label them.

The verified md5 of smac_ezr/optimizers/ezr.py equals repo-root ezr.py, so the
frozen implementation is genuinely the one under test.

BUDGET NOTE: 30 comes from the FidelityGate, not from here. All of EZR's
evaluations are at full fidelity -- it never sees the tree-count axis, which is
the point: LITE spends 30 real labels while HEAVY spends 3000 mostly-cheap ones.
"""

from __future__ import annotations

from optimizers.ezr_opt import EZR as _EZR      # smac_ezr, via shared.py's sys.path

from .shared import Dataset  # noqa: F401  (import order: sets up sys.path)

BUDGET = 30
INIT_LABELS = 4          # the.learn.start; 26 acquisitions follow


class EZROptimizer(_EZR):
    """EZR with `few` pinned. Budget arrives via the gate."""

    name = "ezr"

    def __init__(self, start: int = INIT_LABELS, few: int = 128,
                 acquisition: str = "centroid"):
        # few=128 is ezr's own default and comfortably exceeds a 30-label
        # budget, so unlike the B=200 cell in the SMAC study there is no
        # truncation here and no deviation to declare.
        super().__init__(start=start, few=few, acquisition=acquisition)
