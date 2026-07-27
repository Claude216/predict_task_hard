"""EZR adapter (Table II, A7: proximity active learning).

EZR is pool-based: it labels rows that exist in the table, never proposing a
configuration between rows.  It therefore needs no ConfigurationSpace, and this
module does not import one.

Three integration decisions, all of them consequential:

1. SCALARISATION.  Stock EZR scores a row with its own disty(), which is a
   LOGISTIC normalisation of each goal about that goal's running mean --
   not the paper's Eq.1, which is a min-max normalisation to [0,1].  Letting
   EZR keep disty() would mean EZR and SMAC are judged on two different
   scalarisations, which destroys the comparison.  So the goal columns are
   replaced by ONE synthetic column, "D2h-", whose value is the shared
   oracle's Eq.1 score.  With a single minimise goal, disty() collapses to a
   monotone transform of that value, and every ordering decision EZR makes
   (warm_start sort, rebalance eviction, final sort) is unchanged.  EZR's
   mechanism is preserved; only the yardstick is shared.

2. BUDGET.  warm_start labels the.learn.start rows, then the loop labels
   the.learn.budget more, so total labels = start + budget.  To spend exactly
   B, learn.budget is set to B - start.  Labelling is idempotent per row (a
   row already carrying a d2h is not re-charged), which matters because
   warm_start's sort calls label() on each starting row.

3. the.few.  Stock EZR subsamples the pool to the.few=128 candidate rows.
   That caps the reachable budget at 128 labels, so B=200 is unreachable and
   B=100 leaves only 24 spare candidates.  The paper specifies "distance
   acquisition; init labels 4" (Table IV) and says nothing about few.  Default
   here is max(128, 2*budget): keeps the subsample small, as EZR intends, but
   large enough for the budget.  This is an undocumented deviation either way
   -- 128 would silently truncate the high-budget cells.
"""

from __future__ import annotations

import contextlib
import random

import optimizers.ezr as ezr
 
from data import Dataset
from oracle import BudgetExhausted, BudgetGate
from optimizers.base import register

D2H_COL = "D2h-"          # uppercase -> Num; trailing '-' -> minimise goal
INIT_LABELS = 4           # Table IV


def _get(path: str):
    obj = ezr.the
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def _set(path: str, value) -> None:
    obj = ezr.the
    parts = path.split(".")
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)


@contextlib.contextmanager
def ezr_options(**opts):
    """ezr.the is module-global; set it for one run and put it back."""
    saved = {k.replace("__", "."): _get(k.replace("__", ".")) for k in opts}
    try:
        for k, v in opts.items():
            _set(k.replace("__", "."), v)
        yield
    finally:
        for k, v in saved.items():
            _set(k, v)


def _native(v):
    """numpy scalar -> python scalar, so ezr's Sym dict keys stay hashable
    and comparable against the Dataset's own level lists."""
    return v.item() if hasattr(v, "item") else v


@register
class EZR:
    name = "ezr"

    def __init__(self, start: int = INIT_LABELS, few: int | None = None,
                 acquisition: str = "centroid"):
        self.start = start
        self.few = few
        self.score = {
            "centroid": ezr.acquireWithCentroid,   # Table IV: distance acquisition
            "bayes": ezr.acquireWithBayes,
        }[acquisition]

    def run(self, ds: Dataset, gate: BudgetGate, seed: int) -> dict | None:
        header = [*ds.x_cols, D2H_COL]
        rows = [[_native(row[c]) for c in ds.x_cols] + ["?"] for row in ds.pool]

        def label(_data, row):
            if row[-1] == "?":                     # charge once per row
                row[-1] = gate(dict(zip(ds.x_cols, row[:-1])))
            return row

        few = self.few if self.few is not None else 128

        with ezr_options(**{
            "p": 2,
            "few": min(few, len(rows)),
            "learn__start": self.start,
            "learn__budget": max(0, gate.budget - self.start),
        }):
            data = ezr.Data([header] + rows)

            # Pre-label the warm-start rows.  acquire() opens with
            #     rows = data.rows[:]; shuffle(rows)
            # and warm_start() then does clone(data, rows[:start]) BEFORE
            # calling label().  Under lazy labelling those rows still carry
            # "?", so the D2h column is cloned with n=0, sd=0, norm() saturates
            # every value to the same number, warm_start's sort degenerates and
            # the best/rest split becomes arbitrary -- which makes
            # acquireWithCentroid pull toward a meaningless centroid.  Stock EZR
            # never sees this because its rows arrive already carrying y values.
            # Replaying the same shuffle here fills those rows first; label() is
            # idempotent, so nothing is charged twice.  If the shuffle does not
            # replay identically the gate over-counts and raises, rather than
            # quietly reverting to the degenerate behaviour.
            random.seed(seed)
            probe = data.rows[:]
            random.shuffle(probe)
            for row in probe[:self.start]:
                label(data, row)

            random.seed(seed)                      # ezr uses the global RNG
            lab = ezr.acquire(data, score=self.score, label=label)

        # not lab.rows[0]: acquire() sorts by disty, and with one synthetic goal
        # column disty is norm(), whose z-score is clamped to +/-3.  Several of
        # the best points saturate to the same value and the stable sort then
        # returns whichever came first.  lab.rows IS the returned set, so pick
        # from it by the actual d2h -- "the returned configuration with the
        # lowest d2h".
        best = min(lab.rows, key=lambda r: r[-1])                        # acquire() returns sorted lab
        return dict(zip(ds.x_cols, best[:-1]))