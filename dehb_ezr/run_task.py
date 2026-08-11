#!/usr/bin/env python3
"""One MOOT task, end to end: 4 arms x 10 seeds, then the verdicts.

    conda run -n dmoot python dehb_ezr/run_task.py data/moot/optimize/config/SS-A.csv

ARMS AND BUDGETS -- deliberately asymmetric, that is the experiment:

    ezr          30    LITE   pool-based active learning, all full fidelity
    random_pool  30    floor for EZR
    dehb       3000    HEAVY  DE + Hyperband over tree-count fidelity
    random_cs  3000    floor for DEHB (same ConfigSpace, blind draws)

VERDICTS
Each is stats.top() over the two arms' 10 best-d2h values, with
eps = 0.01 x d2h_spread (p90-p10 of the table's own d2h), unchanged from the
SMAC study. top() returns the subset of arms that are indistinguishable at the
top, so a 2-element result is a tie.

  lite_vs_heavy   ezr  vs dehb       <- the headline, plotted
  ezr_vs_floor    ezr  vs random_pool
  dehb_vs_floor   dehb vs random_cs  <- if DEHB cannot beat 3000 blind draws,
                                        its win over EZR is a budget artifact
                                        and the point is drawn hollow

d2h is lower-better, hence reverse=False inside match(): passing a
higher-better metric without reverse=True silently returns the WORST arm.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

# Runnable both as `python -m dehb_ezr.run_task` and as `python dehb_ezr/run_task.py`.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "dehb_ezr"

from .DEHBOptimizer import DEHBOptimizer
from .EZROptimizer import EZROptimizer
from .RandomOptimizer import RandomConfigSpace, RandomPool
from .fidelity_oracle import FidelityGate, FidelityOracle, N_TREES
from .shared import Dataset, EPS_BASE, match, rho, task_eps, task_id

SEEDS = 10                      # seeds 0..9
ARMS = {                        # name -> (factory, budget)
    "ezr":         (EZROptimizer,      30),
    "random_pool": (RandomPool,        30),
    "dehb":        (DEHBOptimizer,   3000),
    "random_cs":   (RandomConfigSpace, 3000),
}
PAIRS = {
    "lite_vs_heavy": ("ezr", "dehb"),
    "ezr_vs_floor":  ("ezr", "random_pool"),
    "dehb_vs_floor": ("dehb", "random_cs"),
}


@dataclass
class RunResult:
    task_id: str
    optimizer: str
    budget: int
    seed: int
    best_d2h: float
    n_evals: int
    n_full_fidelity: int
    best_on_table: bool
    trace_on_table_rate: float
    seconds: float
    best_config: dict = field(default_factory=dict)


def one_run(ds, oracle, tid, name, opt, budget, seed) -> RunResult:
    gate = FidelityGate(oracle, budget)
    t0 = time.time()
    declared = opt.run(ds, gate, seed)
    best_cfg = declared if declared is not None else gate.best_config

    if declared is not None:
        # The oracle is deterministic, so an arm's own incumbent must equal the
        # best full-fidelity point in its trajectory. Assert rather than assume:
        # a mismatch means the arm is bookkeeping differently and the numbers
        # are not what they look like.
        assert abs(oracle.d2h(declared) - gate.best) < 1e-9, (
            f"{name}: declared {oracle.d2h(declared):.6f} != trace best {gate.best:.6f}")

    keys = ds.pool_keys
    return RunResult(
        task_id=tid, optimizer=name, budget=budget, seed=seed,
        best_d2h=gate.best, n_evals=gate.used, n_full_fidelity=gate.n_full,
        best_on_table=ds.key(best_cfg) in keys,
        # gate.trace_keys, not ds.key(r): the gate already encoded every one of
        # these rows, and re-encoding 3000 of them costs more than the run
        trace_on_table_rate=round(
            sum(k in keys for k in gate.trace_keys) / max(1, gate.used), 4),
        seconds=round(time.time() - t0, 2),
        best_config={k: (v.item() if hasattr(v, "item") else v)
                     for k, v in best_cfg.items()},
    )


def run_task(csv_path: str, root: str = "data/moot/optimize",
             seeds: int = SEEDS, arms: list[str] | None = None,
             oracle_seed: int = 0) -> tuple[dict, list[RunResult], dict]:
    ds = Dataset.load(csv_path)
    tid = task_id(csv_path, root)
    oracle = FidelityOracle(ds, seed=oracle_seed, n_trees=N_TREES)

    table = oracle.d2h_many(ds.pool)
    spread = float(np.percentile(table, 90) - np.percentile(table, 10))
    eps = task_eps(spread)

    names = arms or list(ARMS)
    records = [one_run(ds, oracle, tid, n, ARMS[n][0](), ARMS[n][1], s)
               for n in names for s in range(seeds)]

    scores = {n: [r.best_d2h for r in records if r.optimizer == n] for n in names}
    verdicts = {label: (match({a: scores[a], b: scores[b]}, eps)
                        if a in scores and b in scores else None)
                for label, (a, b) in PAIRS.items()}

    # d2h* pooled over every arm AND the table, mirroring the SMAC study: arms
    # that propose off-table points routinely beat the table's own best on the
    # oracle scale, and seeding d2h* from the table alone pushes rho above 1.
    d2h_star = min([float(table.min())] + [r.best_d2h for r in records])
    med = {n: float(np.median(v)) for n, v in scores.items()}

    # Each arm is measured against ITS OWN floor: rho = 1 matches the best known
    # configuration, rho = 0 matches blind sampling of the same space. Comparing
    # DEHB to the pool floor would credit it for searching a larger space.
    floor_of = {"ezr": "random_pool", "random_pool": "random_pool",
                "dehb": "random_cs", "random_cs": "random_cs"}
    rhos = {n: (rho(med[n], med[floor_of[n]], d2h_star)
                if floor_of[n] in med else None) for n in med}

    meta = ds.describe() | dict(
        task_id=tid,
        d2h_spread=spread, eps=eps, eps_base=EPS_BASE,
        seeds=seeds, oracle_seed=oracle_seed, n_trees=N_TREES,
        table_best=float(table.min()), table_median=float(np.median(table)),
        d2h_star=d2h_star, median_d2h=med, rho=rhos,
    )
    return meta, records, verdicts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--root", default="data/moot/optimize")
    ap.add_argument("--seeds", type=int, default=SEEDS)
    ap.add_argument("--arms", nargs="+", default=None, choices=list(ARMS))
    ap.add_argument("--oracle-seed", type=int, default=0)
    ap.add_argument("--out-dir", default=None,
                    help="write <task_id>.json and <task_id>.runs.jsonl here")
    args = ap.parse_args()

    meta, records, verdicts = run_task(args.csv, root=args.root, seeds=args.seeds,
                                       arms=args.arms, oracle_seed=args.oracle_seed)

    print(" | ".join(f"{k}={v}" for k, v in meta.items()
                     if k not in ("median_d2h", "rho")))
    print()
    names = args.arms or list(ARMS)
    print(f"{'arm':<14}{'budget':>8}{'median d2h':>13}{'evals':>8}{'full-fid':>10}{'s/run':>8}")
    for n in names:
        rs = [r for r in records if r.optimizer == n]
        print(f"{n:<14}{ARMS[n][1]:>8}{meta['median_d2h'][n]:>13.4f}"
              f"{int(np.median([r.n_evals for r in rs])):>8}"
              f"{int(np.median([r.n_full_fidelity for r in rs])):>10}"
              f"{np.median([r.seconds for r in rs]):>8.1f}")
    print()
    for label, v in verdicts.items():
        print(f"  {label:<16} {v}")
    print(f"\neps={meta['eps']:.5f} (= {EPS_BASE} x spread {meta['d2h_spread']:.4f})")

    if args.out_dir:
        d = Path(args.out_dir)
        d.mkdir(parents=True, exist_ok=True)
        tid = meta["task_id"]
        (d / f"{tid}.json").write_text(json.dumps(
            dict(task_id=tid, path=args.csv, meta=meta, verdicts=verdicts),
            indent=1, default=str))
        with (d / f"{tid}.runs.jsonl").open("w") as fh:
            for r in records:
                fh.write(json.dumps(asdict(r), default=str) + "\n")
        print(f"wrote {d}/{tid}.json")


if __name__ == "__main__":
    main()
