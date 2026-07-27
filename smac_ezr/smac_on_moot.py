"""
SMAC3 on MOOT, following Ganguly & Menzies (arXiv:2607.11705).

Protocol notes (paper section -> implementation):
  IV-A   evaluation = ONE tuned RF surrogate per task, frozen, shared by every
         optimizer.  Nearest-neighbour lookup is deliberately NOT used, because
         it penalises membership-query methods like SMAC that propose points
         between rows.
  IV-B   score = d2h, Eq.1, normalised Euclidean distance to the ideal point.
  VI-C   multi-objective tasks are collapsed to a single d2h BEFORE search;
         SMAC runs as a single-objective optimizer.  Native multi-objective
         mode is intentionally not used.
  VI-D   MOOT carries no real-world ranges for decision variables, so the
         domain of each variable is the set of values OBSERVED in the table.
         Hence Ordinal/Categorical, never Integer(min,max).
  V/Eq.2 rho = fraction of the random-to-optimal regret removed.
  Tab.IV SMAC = RF surrogate + EI + init 10, at originating-paper defaults.

Known deviation: the paper says "we tuned the surrogate" without giving the
search space, so the RF here uses scikit-learn defaults.  Absolute d2h values
are therefore NOT comparable with the paper's; relative rankings are.
"""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from ConfigSpace import (
    Categorical,
    Configuration,
    ConfigurationSpace,
    OrdinalHyperparameter,
)
from sklearn.ensemble import RandomForestRegressor
from smac import HyperparameterOptimizationFacade as HPOFacade
from smac import Scenario

BUDGETS = (30, 50, 100, 200)
N_SEEDS = 20
INIT_CONFIGS = 10          # Table IV


# --------------------------------------------------------------------------- #
# MOOT parsing
# --------------------------------------------------------------------------- #
def parse_moot(path: str):
    """MOOT header convention: '+' maximise, '-' minimise, 'X' ignore,
    anything else is a decision variable."""
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].astype(str).str.strip()

    x_cols, goals = [], {}
    for c in df.columns:
        if c.endswith("X"):
            continue
        if c.endswith("+"):
            goals[c] = "max"
        elif c.endswith("-"):
            goals[c] = "min"
        else:
            x_cols.append(c)

    df = df.dropna(subset=x_cols + list(goals)).reset_index(drop=True)
    return df, x_cols, goals


# --------------------------------------------------------------------------- #
# Task = frozen RF oracle + config space, shared by all optimizers
# --------------------------------------------------------------------------- #
class Task:
    def __init__(self, path: str, oracle_seed: int = 0):
        self.path = path
        self.df, self.x_cols, self.goals = parse_moot(path)
        self.y_cols = list(self.goals)

        # symbolic decisions -> integer codes for the RF only; the config space
        # keeps the original labels so configs stay readable.
        self.levels: dict[str, list] = {}
        Xenc = self.df[self.x_cols].copy()
        for c in self.x_cols:
            if not pd.api.types.is_numeric_dtype(Xenc[c]):
                lv = sorted(Xenc[c].unique())
                self.levels[c] = lv
                Xenc[c] = Xenc[c].map({v: i for i, v in enumerate(lv)})
        self.Xenc = Xenc.astype(float)

        # ONE RF per goal, fit on the full table, then frozen (IV-A, VI-E)
        self.models = {
            y: RandomForestRegressor(random_state=oracle_seed, n_jobs=-1).fit(
                self.Xenc.values, self.df[y].values
            )
            for y in self.y_cols
        }
        # normalisation bounds come from the TRUE table values and never change,
        # so d2h is comparable across methods, budgets and seeds.
        self.bounds = {y: (float(self.df[y].min()), float(self.df[y].max()))
                       for y in self.y_cols}

        self.cs = self._build_cs()
        self.pool = self.df[self.x_cols].to_dict("records")   # for pool methods

        self.pool_scores = np.array([self.d2h(r) for r in self.pool])
        # d2h* = best known so far.  Seeded with the on-table minimum, then
        # lowered by anything an optimizer actually reaches: methods that propose
        # off-table points (SMAC, TPE, DE, ...) routinely beat the table's own
        # best on the oracle scale, which would otherwise push rho above 1 or
        # divide by zero.  Mirrors how the paper pools the reference frontier
        # over all optimizers' results (VI-C).  MUST be recomputed over the full
        # optimizer set before any cross-method rho is reported.
        self.d2h_star = float(self.pool_scores.min())

    def observe(self, value: float) -> None:
        self.d2h_star = min(self.d2h_star, value)

    # ---------------- config space (VI-D: observed values only) ------------- #
    def _build_cs(self) -> ConfigurationSpace:
        cs = ConfigurationSpace(seed=0)
        for c in self.x_cols:
            if c in self.levels:                       # symbolic
                cs.add(Categorical(c, self.levels[c]))
            else:                                      # numeric -> ordinal
                vals = sorted(self.df[c].unique().tolist())
                cs.add(OrdinalHyperparameter(c, vals))
        return cs

    # ---------------- scoring ---------------------------------------------- #
    def _encode(self, row: dict) -> np.ndarray:
        out = []
        for c in self.x_cols:
            v = row[c]
            if c in self.levels:
                v = float(self.levels[c].index(v))
            out.append(float(v))
        return np.asarray(out, dtype=float).reshape(1, -1)

    def _oriented(self, y: str, raw: float) -> float:
        lo, hi = self.bounds[y]
        if hi == lo:
            return 0.0
        f = (raw - lo) / (hi - lo)
        return f if self.goals[y] == "min" else 1.0 - f     # 0 = best

    def d2h(self, row: dict) -> float:
        """Eq.1 on the oracle scale."""
        x = self._encode(row)
        vals = [self._oriented(y, float(self.models[y].predict(x)[0]))
                for y in self.y_cols]
        return float(np.sqrt(np.mean(np.square(vals))))

    def space_size(self) -> float:
        """sum log2 |Xi|, the paper's search-space estimate (VI-D)."""
        return float(sum(np.log2(self.df[c].nunique()) for c in self.x_cols))

    def input_shape(self) -> str:
        """binary/SAT, large-numeric or small-numeric (VI-D)."""
        n = len(self.x_cols)
        n_bin = sum(self.df[c].nunique() == 2 for c in self.x_cols)
        if n_bin / n >= 0.80:
            return "binary/SAT"
        return "large-numeric" if self.space_size() >= 40 else "small-numeric"


# --------------------------------------------------------------------------- #
# Budget gate: nothing may be scored without paying for it
# --------------------------------------------------------------------------- #
class BudgetExhausted(Exception):
    pass


@dataclass
class BudgetedOracle:
    task: Task
    budget: int
    used: int = 0
    trace: list = field(default_factory=list)   # (row, d2h) for GD/IGD/HV later

    def __call__(self, row: dict) -> float:
        if self.used >= self.budget:
            raise BudgetExhausted(f"budget {self.budget} exhausted")
        self.used += 1
        v = self.task.d2h(row)
        self.trace.append((row, v))
        return v

    @property
    def best(self) -> float:
        return min(v for _, v in self.trace)


# --------------------------------------------------------------------------- #
# Optimizers
# --------------------------------------------------------------------------- #
def run_smac(task: Task, budget: int, seed: int, out_dir: str = "/tmp/smac_moot"):
    """Table IV: RF surrogate + EI + init 10, otherwise SMAC defaults."""
    gate = BudgetedOracle(task, budget)

    def target(config: Configuration, seed: int = 0) -> float:
        try:
            return gate(dict(config))
        except BudgetExhausted:
            return float("inf")

    scenario = Scenario(
        task.cs,
        deterministic=True,          # the frozen oracle is noise-free
        n_trials=budget,
        seed=seed,
        output_directory=f"{out_dir}/{seed}_{budget}",
    )
    smac = HPOFacade(
        scenario,
        target,
        initial_design=HPOFacade.get_initial_design(
            scenario,
            n_configs=INIT_CONFIGS,
            max_ratio=0.25,   # else max_ratio=0.25 silently cuts 10 -> 7 at B=30
        ),
        overwrite=True,
        logging_level=40,
    )
    smac.optimize()
    return gate


def run_random(task: Task, budget: int, seed: int):
    """Random Search floor: label a random budget of rows, keep the best."""
    rng = np.random.default_rng(seed)
    gate = BudgetedOracle(task, budget)
    idx = rng.choice(len(task.pool), size=min(budget, len(task.pool)), replace=False)
    for i in idx:
        gate(task.pool[i])
    return gate


# --------------------------------------------------------------------------- #
# Eq.2 regret reduction
# --------------------------------------------------------------------------- #
def rho(d2h_m: float, d2h_rand: float, d2h_star: float) -> float:
    denom = d2h_rand - d2h_star
    return float("nan") if abs(denom) < 1e-12 else (d2h_rand - d2h_m) / denom


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--budgets", type=int, nargs="+", default=list(BUDGETS))
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    ap.add_argument("--json", default=None, help="dump per-run results here")
    args = ap.parse_args()

    task = Task(args.csv)
    print(f"task            : {args.csv}")
    print(f"decisions       : {len(task.x_cols)}   goals: {task.goals}")
    print(f"rows            : {len(task.df)}")
    print(f"sum log2|Xi|    : {task.space_size():.1f}  -> {task.input_shape()}")
    print(f"objective type  : {'multi' if len(task.y_cols) > 1 else 'single'}")
    print(f"oracle d2h/table: best={task.d2h_star:.4f} "
          f"median={np.median(task.pool_scores):.4f} "
          f"worst={task.pool_scores.max():.4f}")
    print()
    print(f"{'B':>5} {'SMAC d2h':>10} {'Rand d2h':>10} {'rho':>8}  (medians over "
          f"{args.seeds} seeds)")

    # pass 1: run everything, let d2h* fall to the pooled best-known
    records = []
    for B in args.budgets:
        for s in range(args.seeds):
            gs = run_smac(task, B, s)
            gr = run_random(task, B, s)
            task.observe(gs.best)
            task.observe(gr.best)
            records.append(dict(task=args.csv, budget=B, seed=s,
                                smac=gs.best, random=gr.best,
                                smac_evals=gs.used))

    # pass 2: score against the settled d2h*
    for B in args.budgets:
        rs = [r for r in records if r["budget"] == B]
        m_s = float(np.median([r["smac"] for r in rs]))
        m_r = float(np.median([r["random"] for r in rs]))
        print(f"{B:>5} {m_s:>10.4f} {m_r:>10.4f} "
              f"{rho(m_s, m_r, task.d2h_star):>8.3f}")
    print(f"\nd2h* used: {task.d2h_star:.4f} "
          f"(on-table best was {task.pool_scores.min():.4f}; "
          f"recompute over the full optimizer set before comparing methods)")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(records, fh, indent=1)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()