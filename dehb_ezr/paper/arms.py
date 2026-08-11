"""The two arms: LITE (ezr, 30 evaluations) and DEHB (3000 evaluations).

Per the paper (§3.4): "LITE will explore up to 30 of these [10,000 configs].
DEHB, on the other hand, would sample 100 (at random) then go on to evolve its
own set of preferred configurations." So LITE is pool-restricted and DEHB is
not -- restricting DEHB to the pool would stop it being DEHB.

BUDGET GATE. Every evaluation goes through EvalGate, which counts. LITE spends
exactly 30, DEHB exactly 3000 (2999 plus one reserved to confirm its incumbent
at full fidelity). Costs are 1 per call whatever the fidelity, so the two
budgets are directly comparable, which is the whole point of the comparison.

BOTH ARMS SHARE ONE YARDSTICK. Stock ezr scores rows with its own disty(), a
logistic normalisation about each goal's running mean. Letting LITE keep that
while DEHB optimises the natural-bounds d2h would judge the two on different
scales. So the ezr table carries ONE synthetic goal column, "D2h-", holding the
shared scalar; with a single minimise goal, disty() collapses to a monotone
transform of it and every ordering decision ezr makes is unchanged. This is the
same device smac_ezr/optimizers/ezr_opt.py uses, and the reason is the same.
"""

from __future__ import annotations

import contextlib
import random
import shutil
import tempfile
import warnings
from pathlib import Path

import numpy as np

import ezr  # repo root, frozen -- imported, never modified

from . import space as S
from .objective import Objective

warnings.filterwarnings("ignore")

LITE_BUDGET = 30
LITE_START = 4          # ezr's the.learn.start; 26 acquisitions follow
DEHB_BUDGET = 3000
ETA = 3


class BudgetExhausted(Exception):
    pass


class EvalGate:
    """The only way either arm may score a configuration."""

    def __init__(self, objective: Objective, budget: int):
        self.obj = objective
        self.budget = budget
        self.used = 0
        self.trace: list[dict] = []      # {cfg, d2h, metrics, n_rows}

    def __call__(self, cfg: dict, n_rows: int | None = None) -> float:
        if self.used >= self.budget:
            raise BudgetExhausted(f"budget {self.budget} exhausted")
        self.used += 1
        r = self.obj.evaluate(cfg, n_rows)
        self.trace.append({"cfg": S.native(cfg), "d2h": r["d2h"],
                           "metrics": r["metrics"], "n_rows": r["n_rows"]})
        return r["d2h"]

    @property
    def remaining(self) -> int:
        return self.budget - self.used

    def full_trace(self) -> list[dict]:
        """Only evaluations that saw the whole training set. A configuration
        that looked good on 1.2% of the rows has not been confirmed."""
        return [t for t in self.trace if t["n_rows"] >= self.obj.n_train]

    def best(self) -> dict:
        ft = self.full_trace()
        if not ft:
            raise ValueError("no full-fidelity evaluation: nothing confirmed")
        return min(ft, key=lambda t: t["d2h"])


# ------------------------------------------------------------------- LITE
@contextlib.contextmanager
def _ezr_options(**opts):
    """ezr.the is module-global: set it for one run and put it back."""
    def get(p):
        o = ezr.the
        for part in p.split("."):
            o = getattr(o, part)
        return o

    def put(p, v):
        o = ezr.the
        parts = p.split(".")
        for part in parts[:-1]:
            o = getattr(o, part)
        setattr(o, parts[-1], v)

    saved = {k.replace("__", "."): get(k.replace("__", ".")) for k in opts}
    try:
        for k, v in opts.items():
            put(k.replace("__", "."), v)
        yield
    finally:
        for k, v in saved.items():
            put(k, v)


# ezr decides a column's type from its initial's case, so the numeric
# hyperparameters need uppercase headers and `criterion` must stay lowercase.
_EZR_HEADER = {"n_estimators": "Nestimators", "criterion": "criterion",
               "min_samples_leaf": "MinSamplesLeaf",
               "min_impurity_decrease": "MinImpurityDecrease",
               "max_depth": "MaxDepth"}
D2H_COL = "D2h-"        # uppercase -> Num, trailing '-' -> minimise


class Lite:
    name = "lite"

    def run(self, pool: list[dict], gate: EvalGate, seed: int) -> dict:
        header = [_EZR_HEADER[n] for n in S.NAMES] + [D2H_COL]
        rows = [[cfg[n] for n in S.NAMES] + ["?"] for cfg in pool]

        def label(_data, row):
            if row[-1] == "?":                      # charge once per row
                row[-1] = gate({n: row[i] for i, n in enumerate(S.NAMES)})
            return row

        with _ezr_options(**{
            "p": 2,
            # ezr's own default. It means LITE ranks 128 of the 10,000 pool
            # configs before spending its 30 labels -- stock behaviour, kept so
            # LITE is the algorithm the paper cites rather than a strengthened
            # variant. Recorded in RESULTS.md.
            "few": min(128, len(rows)),
            "learn__start": LITE_START,
            "learn__budget": max(0, gate.budget - LITE_START),
        }):
            data = ezr.Data([header] + rows)

            # acquire() shuffles, then warm_start() clones rows[:start] BEFORE
            # calling label(). Under lazy labelling those rows still carry "?",
            # so the D2h column clones with sd=0, norm() saturates, and the
            # best/rest split is arbitrary. Replaying the same shuffle fills
            # them first; label() is idempotent so nothing is charged twice.
            random.seed(seed)
            probe = data.rows[:]
            random.shuffle(probe)
            for row in probe[:LITE_START]:
                label(data, row)

            random.seed(seed)                       # ezr uses the global RNG
            ezr.acquire(data, score=ezr.acquireWithBayes, label=label)

        return gate.best()


# ------------------------------------------------------------------- DEHB
class Dehb:
    name = "dehb"

    def __init__(self, eta: int = ETA):
        self.eta = eta

    def run(self, kind: str, gate: EvalGate, seed: int) -> dict:
        from dehb import DEHB

        from ..DEHBOptimizer import _patch_vector_to_configspace
        _patch_vector_to_configspace()      # the de.py:190 clamp; see that module

        cs = S.configspace(kind, seed=seed)
        n_train = gate.obj.n_train
        # eta=3 with min = max/81 gives 5 rungs at 1/81, 1/27, 1/9, 1/3, 1 of the
        # training set -- the 100/33/11/3/1 schedule -- and rung sizes
        # [81, 27, 9, 3, 1]. A floor of 20 rows keeps the lowest rung trainable
        # on the small tables (nasa93dem has 93 rows, Player Statistics 81).
        min_fid = max(20, n_train // 81)
        max_fid = max(min_fid * self.eta, n_train)

        def target(config, fidelity, **kw):
            try:
                fitness = gate(S.native(dict(config)), n_rows=int(round(fidelity)))
            except BudgetExhausted:
                fitness = float("inf")
            return {"fitness": fitness, "cost": 1.0}

        out = Path(tempfile.mkdtemp(prefix="dehb_paper_"))
        try:
            d = DEHB(f=target, cs=cs, dimensions=len(S.NAMES),
                     min_fidelity=min_fid, max_fidelity=max_fid, eta=self.eta,
                     n_workers=1, output_path=str(out), save_freq="end", seed=seed)
            d.run(fevals=max(1, gate.budget - 1))
            # DEHB hides exceptions raised inside its own ask() behind loguru's
            # @logger.catch and returns as if it finished. Without this check a
            # truncated run is indistinguishable from a real one.
            if gate.used < gate.budget - 1:
                raise RuntimeError(
                    f"dehb stopped after {gate.used} of {gate.budget - 1} "
                    f"evaluations; it caught and hid an internal error")
        finally:
            shutil.rmtree(out, ignore_errors=True)

        self._confirm(gate)
        return gate.best()

    @staticmethod
    def _confirm(gate: EvalGate) -> None:
        """Spend the reserved evaluation confirming the most promising config
        that was never scored on the full training set. Comparing a 1.2%-fidelity
        score against a full one is apples to oranges, which is exactly why it
        has to be confirmed before it may count."""
        if gate.remaining <= 0:
            return
        full = {S.key(t["cfg"]) for t in gate.trace if t["n_rows"] >= gate.obj.n_train}
        best: dict[tuple, tuple[float, dict]] = {}
        for t in gate.trace:
            k = S.key(t["cfg"])
            if k in full:
                continue
            if k not in best or t["d2h"] < best[k][0]:
                best[k] = (t["d2h"], t["cfg"])
        if best:
            gate(min(best.values(), key=lambda t: t[0])[1], n_rows=gate.obj.n_train)
        elif not gate.full_trace() and gate.trace:
            gate(gate.trace[0]["cfg"], n_rows=gate.obj.n_train)
