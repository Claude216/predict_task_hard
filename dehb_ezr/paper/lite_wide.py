#!/usr/bin/env python3
"""A second LITE arm that ranks the WHOLE 10,000-config pool, scored against the
DEHB runs we already have.

    conda run -n dmoot python -m dehb_ezr.paper.lite_wide            # all datasets
    conda run -n dmoot python -m dehb_ezr.paper.lite_wide --only iris

WHY. The main run gives LITE ezr's stock `the.few=128`, so it ranks 128 of the
10,000 pool configurations before spending its 30 labels -- a 23x disadvantage
in configurations *seen* on top of the 100x disadvantage in evaluations. The
paper's wording ("LITE will explore up to 30 of these [10,000]") can equally be
read as LITE ranking the whole pool. If the missing green in our Fig. 4 is an
artifact of the subsample cap rather than a real non-replication, this is what
shows it.

WHY DEHB DOES NOT NEED RE-RUNNING. Everything that defines a repeat is a
deterministic function of its seed:

    split(task, seed)          -> identical train/test partition
    Objective(..., seed)       -> identical RF random_state and fidelity order
    space.pool(kind, Random(seed)) -> the identical 10,000 configurations

so re-running only LITE reproduces exactly the conditions the stored DEHB run
faced. The single difference is `the.few`. As a guard against that reasoning
being wrong, the recomputed n_train/n_test are asserted against the values
stored in the DEHB record; a mismatch aborts rather than quietly comparing two
different experiments.

The verdict is recomputed the same way as the main run: Zitzler-rank the 40
incumbents (20 LITE-wide + 20 stored DEHB) against each other, then Scott-Knott
with Cliff's delta 0.147.
"""

from __future__ import annotations

import argparse
import json
import random as _random
import sys
import time
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    __package__ = "dehb_ezr.paper"

import ezr

from . import space as S
from .arms import D2H_COL, LITE_BUDGET, LITE_START, EvalGate, _EZR_HEADER, _ezr_options
from .datasets import BY_LABEL, DATASETS, task_id
from .loader import load, split
from .objective import Objective, metric_names, weights
from .run_one import OUT_DIR, REPEATS
from .sk import cliffs_delta, verdict
from .zitzler import d2h as zitzler_d2h

WIDE_OUT = Path(__file__).resolve().parent.parent / "paper_results_lite10k"


def run_lite(pool: list[dict], gate: EvalGate, seed: int, few: int) -> dict:
    """ezr LITE with a configurable `the.few`.

    Deliberately a copy of arms.Lite.run rather than an edit to it: the main
    batch was still running when this was written, and changing a module that
    live worker processes import is how you get two half-experiments in one
    output directory. Keep the two in step.
    """
    header = [_EZR_HEADER[n] for n in S.NAMES] + [D2H_COL]
    rows = [[cfg[n] for n in S.NAMES] + ["?"] for cfg in pool]

    def label(_data, row):
        if row[-1] == "?":
            row[-1] = gate({n: row[i] for i, n in enumerate(S.NAMES)})
        return row

    with _ezr_options(**{
        "p": 2,
        "few": min(few, len(rows)),
        "learn__start": LITE_START,
        "learn__budget": max(0, gate.budget - LITE_START),
    }):
        data = ezr.Data([header] + rows)
        # replay acquire()'s shuffle so the warm-start rows are labelled before
        # they are cloned; see arms.Lite for why this is load-bearing
        _random.seed(seed)
        probe = data.rows[:]
        _random.shuffle(probe)
        for row in probe[:LITE_START]:
            label(data, row)
        _random.seed(seed)
        ezr.acquire(data, score=ezr.acquireWithBayes, label=label)

    return gate.best()


def stored_dehb(results: Path, tid: str) -> list[dict]:
    f = results / f"{tid}.runs.jsonl"
    if not f.exists():
        raise FileNotFoundError(f"no stored run for {tid}: {f}")
    return [r for r in map(json.loads, f.open()) if r["arm"] == "dehb"]


def run_dataset(label: str, few: int, repeats: int, results: Path) -> dict:
    entry = BY_LABEL[label]
    tid = task_id(label)
    dehb = stored_dehb(results, tid)
    if len(dehb) != repeats:
        raise ValueError(f"{label}: stored DEHB has {len(dehb)} repeats, expected {repeats}")
    by_seed = {r["seed"]: r for r in dehb}

    task = load(entry, seed=0)
    records = []
    for seed in range(repeats):
        xtr, ytr, xte, yte = split(task, seed)
        ref = by_seed[seed]
        # The guard that makes reusing DEHB legitimate.
        assert len(ytr) == ref["n_train"] and len(yte) == ref["n_test"], (
            f"{label} seed {seed}: split is {len(ytr)}/{len(yte)} but the stored "
            f"DEHB run saw {ref['n_train']}/{ref['n_test']} -- not the same experiment")
        obj = Objective(xtr, ytr, xte, yte, task.kind, seed)
        gate = EvalGate(obj, LITE_BUDGET)
        t0 = time.time()
        best = run_lite(S.pool(task.kind, _random.Random(seed)), gate, seed, few)
        records.append(dict(arm="lite10k", seed=seed, n_evals=gate.used,
                            budget=LITE_BUDGET, few=few,
                            n_full_fidelity=len(gate.full_trace()),
                            seconds=round(time.time() - t0, 2), d2h=best["d2h"],
                            metrics=best["metrics"], cfg=best["cfg"],
                            n_train=len(ytr), n_test=len(yte)))

    # rank the 20 new LITE incumbents against the 20 stored DEHB ones
    names = metric_names(task.kind)
    allrec = records + [dict(arm="dehb", **{k: r[k] for k in ("seed", "d2h", "metrics")})
                        for r in dehb]
    D2H = zitzler_d2h([r["metrics"] for r in allrec], names, weights(task.kind))
    for r, v in zip(allrec, D2H):
        r["D2H"] = float(v)

    lite = [r["D2H"] for r in allrec if r["arm"] == "lite10k"]
    dh = [r["D2H"] for r in allrec if r["arm"] == "dehb"]
    meta = dict(
        label=label, task_id=tid, path=entry.path, is_se=entry.is_se, kind=task.kind,
        R=task.R, repeats=repeats, few=few,
        median_D2H={"lite10k": float(np.median(lite)), "dehb": float(np.median(dh))},
        median_d2h={"lite10k": float(np.median([r["d2h"] for r in records])),
                    "dehb": float(np.median([r["d2h"] for r in dehb]))},
        cliffs_delta_D2H=cliffs_delta(dh, lite),
        verdict=verdict(lite, dh),
        verdict_natural_d2h=verdict([r["d2h"] for r in records],
                                    [r["d2h"] for r in dehb]),
        seconds=round(sum(r["seconds"] for r in records), 1),
    )
    return dict(meta=meta, records=allrec)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--few", type=int, default=S.POOL_SIZE,
                    help="how many pool configs LITE ranks (default: the whole pool)")
    ap.add_argument("--repeats", type=int, default=REPEATS)
    ap.add_argument("--results", default=str(OUT_DIR), type=Path,
                    help="where the stored DEHB runs live")
    ap.add_argument("--out", default=str(WIDE_OUT), type=Path)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--redo", action="store_true")
    args = ap.parse_args()

    labels = args.only or [e.label for e in DATASETS]
    args.out.mkdir(parents=True, exist_ok=True)
    done = skipped = flipped = 0
    for label in labels:
        tid = task_id(label)
        if not args.redo and (args.out / f"{tid}.json").exists():
            continue
        try:
            blob = run_dataset(label, args.few, args.repeats, args.results)
        except FileNotFoundError:
            skipped += 1
            print(f"  {label:<22} skipped (no stored DEHB run yet)", flush=True)
            continue
        m = blob["meta"]
        (args.out / f"{tid}.json").write_text(json.dumps(
            dict(meta=m, verdict=m["verdict"]), indent=1, default=str))
        with (args.out / f"{tid}.runs.jsonl").open("w") as fh:
            for r in blob["records"]:
                fh.write(json.dumps(r, default=str) + "\n")
        done += 1
        print(f"  {label:<22} R={m['R']:<5} {m['verdict']:<14} "
              f"(lite10k D2H {m['median_D2H']['lite10k']:.3f} vs dehb "
              f"{m['median_D2H']['dehb']:.3f}, delta {m['cliffs_delta_D2H']:+.3f})  "
              f"{m['seconds']}s", flush=True)
    print(f"\n{done} written, {skipped} skipped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
