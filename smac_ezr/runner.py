"""Experiment driver: one dataset, N optimizers x budgets x seeds.

Two passes are needed because d2h* (Eq.2's denominator) must be the best value
POOLED over every optimizer, mirroring how the paper pools the reference
frontier (VI-C).  Methods that propose off-table points routinely beat the
table's own best on the oracle scale, so seeding d2h* from the table alone
would push rho above 1.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field

import numpy as np

from data import Dataset
from metrics import match, rho, task_eps, EPS_BASE
from oracle import BudgetGate, Oracle
from optimizers import base

# importing the adapters is what populates base.REGISTRY
from optimizers import random_opt, smac_opt, ezr_opt   # noqa: F401


@dataclass
class RunResult:
    task: str
    optimizer: str
    budget: int
    seed: int
    best_d2h: float
    n_evals: int
    best_config: dict = field(default_factory=dict)
    trace: list | None = None                  # kept only if --save-traces
    best_on_table: bool = False
    trace_on_table_rate: float = 0.0
    


def one_run(ds, oracle, opt, budget, seed, keep_trace=False) -> RunResult:
    gate = BudgetGate(oracle, budget)
    declared = opt.run(ds, gate, seed)

    if declared is not None:
        # The frozen oracle is deterministic, so an optimizer's own incumbent
        # should equal the best point in its trajectory.  Assert rather than
        # assume: a mismatch means the optimizer is bookkeeping differently
        # (e.g. SMAC's inf fallback) and the numbers are not what they look like.
        assert abs(oracle.d2h(declared) - gate.best) < 1e-9, (
            f"{opt.name}: declared incumbent {oracle.d2h(declared):.6f} != "
            f"best in trace {gate.best:.6f}"
        )

    keys = ds.pool_keys
    best_cfg = declared or gate.best_config

    return RunResult(
        task=ds.path,
        optimizer=opt.name,
        budget=budget,
        seed=seed,
        best_d2h=gate.best,
        n_evals=gate.used,
        best_config=declared or gate.best_config,
        trace=[(r, v) for r, v in gate.trace] if keep_trace else None,
        best_on_table=ds.key(best_cfg) in keys,
        trace_on_table_rate=round(
            sum(ds.key(r) in keys for r, _ in gate.trace) / max(1, gate.used), 4),
    )

def run_task(csv_path, optimizers, budgets, seeds,
             oracle_seed=0, save_traces=False):
    """One task, end to end. Returns (meta, records, verdicts)."""
    ds = Dataset.load(csv_path)
    oracle = Oracle(ds, seed=oracle_seed)
    opts = [base.REGISTRY[n]() for n in optimizers]

    table = oracle.d2h_many(ds.pool)
    spread = float(np.percentile(table, 90) - np.percentile(table, 10))
    eps = task_eps(spread)
    meta = ds.describe() | dict(
        d2h_spread=spread, eps=eps, eps_base=EPS_BASE,
        seeds=seeds, oracle_seed=oracle_seed,
        table_best=float(table.min()), table_median=float(np.median(table)))

    records = [one_run(ds, oracle, opt, B, s, save_traces)
               for B in budgets for s in range(seeds) for opt in opts]
    d2h_star = min([float(table.min())] + [r.best_d2h for r in records])
    meta["d2h_star"] = d2h_star
    verdicts = {}
    for B in budgets:
        pair = {n: [r.best_d2h for r in records
                    if r.budget == B and r.optimizer == n]
                for n in ("smac", "ezr") if n in optimizers}
        verdicts[B] = match(pair, eps) if len(pair) == 2 else None

    return meta, records, verdicts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--optimizers", nargs="+", default=["smac", "ezr", "random"])
    ap.add_argument("--budgets", type=int, nargs="+", default=[30, 50, 100, 200])
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--oracle-seed", type=int, default=0)
    ap.add_argument("--save-traces", action="store_true",
                    help="keep full search trajectories (needed for GD/IGD/HV)")
    ap.add_argument("--out", default=None, help="write per-run results as JSONL")
    args = ap.parse_args()

    meta, records, verdicts = run_task(
        args.csv,
        optimizers=args.optimizers,
        budgets=args.budgets,
        seeds=args.seeds,
        oracle_seed=args.oracle_seed,
        save_traces=args.save_traces,
    )

    print(" | ".join(f"{k}={v}" for k, v in meta.items()))
    print()

    names = args.optimizers
    show_rho = "random" in names
    header = f"{'B':>5} " + " ".join(f"{n:>10}" for n in names) + f"{'winner':>10}"
    if show_rho:
        header += "   rho_vs_random"
    print(header)

    for B in args.budgets:
        med = {n: float(np.median([r.best_d2h for r in records
                                   if r.budget == B and r.optimizer == n]))
               for n in names}
        line = (f"{B:>5} " + " ".join(f"{med[n]:>10.4f}" for n in names)
                + f"{str(verdicts[B]):>10}")
        if show_rho:
            line += "   " + " ".join(
                f"{n}={rho(med[n], med['random'], meta['d2h_star']):.3f}"
                for n in names if n != "random")
        print(line)

    print(f"\neps={meta['eps']:.5f} (= {meta['eps_base']} x d2h_spread "
          f"{meta['d2h_spread']:.4f}); winner column is SMAC vs EZR only, "
          f"random is a floor check")
    print(f"d2h* = {meta['d2h_star']:.4f} (table best {meta['table_best']:.4f}); "
          f"pooled over {names} only -- recompute when adding optimizers")

    if args.out:
        with open(args.out, "w") as fh:
            for r in records:
                fh.write(json.dumps(asdict(r), default=str) + "\n")
        print(f"wrote {len(records)} runs -> {args.out}")


if __name__ == "__main__":
    main()