"""DEHB adapter: the HEAVY arm. 3000 evaluations, tree-count fidelity.

BRACKET GEOMETRY
    min_fidelity=1, max_fidelity=100, eta=3
      -> s_max = floor(log_3(100)) = 4
      -> fidelities {1, 3, 11, 33, 100} trees
      -> first bracket 81 configs at 1 tree -> 27 -> 9 -> 3 -> 1 at 100 trees
3000 evaluations buys roughly 20 such brackets.

THE RESERVED EVALUATION
DEHB's incumbent may last have been evaluated below full fidelity, and
FidelityGate.best only counts full-fidelity scores (a 1-tree lucky draw is not
a result). So the run is given `budget - 1` evaluations and the final one is
spent confirming the incumbent at 100 trees. Total spend is exactly `budget`.
Without this the arm could, in principle, finish with nothing confirmed.

WHAT COUNTS AS THE INCUMBENT
Taken from the gate's own trace, not from DEHB's internal attributes. DEHB
0.1.x exposes the incumbent under names that have moved between releases
(`inc_config` as a vector, `get_incumbents()`, `traj`), and depending on them
would make this adapter version-fragile for no gain -- the gate has seen every
evaluation DEHB made, by construction. `_dehb_incumbent` is a best-effort
cross-check that logs a disagreement rather than failing the run.
"""

from __future__ import annotations

import shutil
import tempfile
import warnings
from pathlib import Path

import numpy as np

from .configspace import build_configspace, native
from .fidelity_oracle import FidelityGate, N_TREES
from .shared import BudgetExhausted, Dataset

warnings.filterwarnings("ignore")

MIN_FIDELITY = 1
MAX_FIDELITY = N_TREES
ETA = 3


def _patch_vector_to_configspace() -> None:
    """Fix an off-by-one in DEHB's own vector -> config mapping.

    THE BUG (dehb/optimizers/de.py:188-190):

        ranges = np.arange(start=0, stop=1, step=1/len(hyper.sequence))
        param_value = hyper.sequence[np.where((vector[i] < ranges) == False)[0][-1]]

    `np.arange(0, 1, 1/n)` does not always yield n elements. For n = 49,
    49 * (1/49) == 0.9999999999999999 < 1, so arange emits a 50th edge and a DE
    coordinate at ~1.0 selects index 49 of a 49-element tuple:

        IndexError: tuple index out of range

    328 of the first 5000 lengths are affected (49, 98, 103, 107, 196, ...), and
    11 of the 126 MOOT tasks carry at least one column with such a domain size.

    WHY IT MATTERED. DEHB decorates run() with loguru's @logger.catch, so the
    IndexError was logged and SWALLOWED -- dehb.run() returned normally and the
    arm was recorded as a completed 3000-evaluation run when it had actually
    stopped after 55. Silent truncation, not a crash. Caught by the integrity
    check on n_evals, not by anything DEHB reported.

    THE FIX. Reproduce DEHB's own selection exactly, then clamp -- the
    round-and-clamp index interpolation this study specified. DEHB's expression
    picks the last edge with ranges[j] <= v, which is
    `np.searchsorted(ranges, v, side="right") - 1` on the identical `ranges`
    array. Deriving the index from the same edges (rather than from floor(v*n),
    which disagrees in the last bit at some k/n boundaries -- the tests caught
    this) makes agreement true by construction, not by luck.

    So the ONLY behavioural change is the clamp, which fires exactly where the
    original raised. Runs that did not crash are bit-for-bit unaffected, which
    is why the completed sweep needed only the 2 broken tasks re-run;
    test_dehb_patch.py asserts that equivalence over a dense sample.

    Only Ordinal and Categorical are handled because configspace.py builds
    nothing else; anything else raises rather than being silently mishandled.
    """
    from ConfigSpace import Configuration
    from dehb.optimizers.de import DEBase

    if getattr(DEBase, "_moot_patched", False):
        return

    def vector_to_configspace(self, vector):
        values = {}
        for i, hp in enumerate(self.cs.get_hyperparameters()):
            seq = getattr(hp, "sequence", None)
            if seq is None:
                seq = getattr(hp, "choices", None)
            if seq is None:
                raise TypeError(
                    f"{hp.name}: {type(hp).__name__} is not Ordinal or Categorical; "
                    "configspace.py builds only those two")
            n = len(seq)
            # same edges DEHB builds, same selection rule, plus the clamp
            ranges = np.arange(start=0, stop=1, step=1 / n)
            idx = int(np.searchsorted(ranges, float(vector[i]), side="right")) - 1
            values[hp.name] = seq[min(max(idx, 0), n - 1)]
        return Configuration(self.cs, values=values)

    DEBase.vector_to_configspace = vector_to_configspace
    DEBase._moot_patched = True


class DEHBOptimizer:
    name = "dehb"

    def __init__(self, min_fidelity: int = MIN_FIDELITY,
                 max_fidelity: int = MAX_FIDELITY, eta: int = ETA):
        self.min_fidelity = min_fidelity
        self.max_fidelity = max_fidelity
        self.eta = eta

    # ------------------------------------------------------------------ #
    def run(self, ds: Dataset, gate: FidelityGate, seed: int) -> dict | None:
        from dehb import DEHB          # imported late: only this arm needs it

        _patch_vector_to_configspace()
        cs = build_configspace(ds, seed=seed)

        def target(config, fidelity, **kwargs) -> dict:
            """DEHB's contract: return a dict carrying 'fitness' and 'cost'.

            cost is fixed at 1.0 -- every oracle call is one label, whatever the
            fidelity. That is what makes 3000 comparable with EZR's 30.
            """
            try:
                fitness = gate(native(dict(config)), fidelity=int(round(fidelity)))
            except BudgetExhausted:
                fitness = float("inf")
            return {"fitness": fitness, "cost": 1.0}

        out_dir = Path(tempfile.mkdtemp(prefix="dehb_moot_"))
        try:
            dehb = DEHB(
                f=target,
                cs=cs,
                dimensions=len(ds.x_cols),
                min_fidelity=self.min_fidelity,
                max_fidelity=self.max_fidelity,
                eta=self.eta,
                # one worker per optimizer run: batch.py parallelises across
                # tasks instead, so a dask cluster per run would oversubscribe
                n_workers=1,
                output_path=str(out_dir),
                # default "incumbent" rewrites incumbent.json on every
                # improvement; 1270 runs x that I/O buys nothing here
                save_freq="end",
                seed=seed,
            )
            # budget - 1: the last evaluation is reserved to confirm the
            # incumbent at full fidelity (see module docstring)
            dehb.run(fevals=max(1, gate.budget - 1))
            # DEHB swallows exceptions raised inside its own ask() via loguru's
            # @logger.catch and returns as if the run completed. Without this
            # check a truncated run is indistinguishable from a real one in the
            # output, which is exactly how the de.py:190 IndexError went
            # unnoticed until an integrity pass on n_evals found it.
            if gate.used < gate.budget - 1:
                raise RuntimeError(
                    f"dehb stopped after {gate.used} of {gate.budget - 1} "
                    f"evaluations; it caught and hid an internal error. Check "
                    f"the loguru output for the traceback.")
            self._cross_check(gate, dehb)
        finally:
            shutil.rmtree(out_dir, ignore_errors=True)

        self._confirm(ds, gate)
        return None      # run_task falls back to gate.best_config

    # ------------------------------------------------------------------ #
    def _confirm(self, ds: Dataset, gate: FidelityGate) -> None:
        """Spend the reserved evaluation on the most promising unconfirmed config.

        "Most promising" is the lowest d2h seen at ANY fidelity among configs
        never scored at 100 trees. Comparing a 1-tree score against a 100-tree
        score is apples-to-oranges, which is precisely why the winner has to be
        confirmed before it may count.

        Single pass, reusing the keys the gate already computed: at 3000 entries
        x 1044 columns (FFM-1000) re-encoding here costs ~40 s per run.
        """
        if gate.remaining <= 0:
            return
        confirmed, best = set(), {}
        for (r, v, k), key in zip(gate.trace, gate.trace_keys):
            if k >= gate.oracle.n_trees:
                confirmed.add(key)
            elif key not in best or v < best[key][0]:
                best[key] = (v, r)
        candidates = [t for key, t in best.items() if key not in confirmed]
        if candidates:
            gate(min(candidates, key=lambda t: t[0])[1], fidelity=gate.oracle.n_trees)
        elif not gate.full_trace and gate.trace:
            # everything already seen was partial-fidelity: guarantee at least
            # one confirmed score so gate.best is defined
            gate(gate.trace[0][0], fidelity=gate.oracle.n_trees)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _cross_check(gate: FidelityGate, dehb) -> None:
        """Warn if DEHB's own incumbent is not the gate's best full-fidelity point.

        get_incumbents() returns (Configuration, score) with the score at
        whatever fidelity it was last evaluated on, so a mismatch is EXPECTED
        and is not an error -- a config that looked best at 3 trees need not be
        best at 100. This exists so the size of that gap is visible in the pilot
        rather than assumed. Guarded, because it must never fail a run.
        """
        try:
            cfg, score = dehb.get_incumbents()
            cfg = native(dict(cfg)) if not isinstance(cfg, np.ndarray) else None
            if cfg is None or not gate.full_trace:
                return
            if abs(score - gate.best) > 1e-9:
                warnings.warn(
                    f"dehb: incumbent score {score:.6f} (at its own fidelity) != "
                    f"best confirmed {gate.best:.6f} over {gate.n_full} "
                    f"full-fidelity evals", stacklevel=2)
        except Exception:
            return
