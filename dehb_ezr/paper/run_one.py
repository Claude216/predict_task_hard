#!/usr/bin/env python3
"""One Table 1 dataset, end to end: 2 arms x 20 repeats -> verdict.

    conda run -n dmoot python -m dehb_ezr.paper.run_one iris

Each repeat r (seed r) gets its own train/test split, its own RF seed, and its
own fresh 10,000-config pool -- so the 20 repeats are genuinely independent,
which is what the Cliff's delta test needs in order to be measuring anything.

The verdict is computed on the paper's D2H: the 40 run incumbents (20 LITE +
20 DEHB) are Zitzler-ranked against each other, D2H = i/40, then Scott-Knott
with Cliff's delta 0.147 decides "DEHB > LITE" or "DEHB == LITE". The
natural-bounds d2h is recorded alongside so the verdict's sensitivity to that
choice can be checked rather than assumed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    __package__ = "dehb_ezr.paper"

from . import REPO_ROOT, space as S
from .arms import DEHB_BUDGET, LITE_BUDGET, Dehb, EvalGate, Lite
from .datasets import BY_LABEL, task_id
from .loader import load, split
from .objective import Objective, metric_names, weights
from .sk import cliffs_delta, verdict
from .zitzler import d2h as zitzler_d2h

REPEATS = 20
OUT_DIR = Path(__file__).resolve().parent.parent / "paper_results"


def one_repeat(task, seed: int) -> list[dict]:
    xtr, ytr, xte, yte = split(task, seed)
    records = []
    for arm in ("lite", "dehb"):
        obj = Objective(xtr, ytr, xte, yte, task.kind, seed)
        budget = LITE_BUDGET if arm == "lite" else DEHB_BUDGET
        gate = EvalGate(obj, budget)
        t0 = time.time()
        if arm == "lite":
            import random as _r
            pool = S.pool(task.kind, _r.Random(seed))
            best = Lite().run(pool, gate, seed)
        else:
            best = Dehb().run(task.kind, gate, seed)
        records.append(dict(
            arm=arm, seed=seed, n_evals=gate.used, budget=budget,
            n_full_fidelity=len(gate.full_trace()), n_fits=obj.n_fits,
            n_failed_fits=obj.n_failed, seconds=round(time.time() - t0, 2),
            d2h=best["d2h"], metrics=best["metrics"], cfg=best["cfg"],
            n_train=obj.n_train, n_test=len(yte),
            n_skipped_zero_actual=obj.n_skipped_zero_actual))
    return records


def run(label: str, repeats: int = REPEATS, row_cap: int | None = None) -> dict:
    entry = BY_LABEL[label]
    task = load(entry, seed=0) if row_cap is None else load(entry, seed=0, row_cap=row_cap)

    records = []
    for r in range(repeats):
        records.extend(one_repeat(task, r))

    # Paper's D2H: Zitzler-rank the 40 incumbents against each other.
    names = metric_names(task.kind)
    D2H = zitzler_d2h([r["metrics"] for r in records], names, weights(task.kind))
    for rec, v in zip(records, D2H):
        rec["D2H"] = float(v)

    lite = [r["D2H"] for r in records if r["arm"] == "lite"]
    dehb = [r["D2H"] for r in records if r["arm"] == "dehb"]
    lite_d2h = [r["d2h"] for r in records if r["arm"] == "lite"]
    dehb_d2h = [r["d2h"] for r in records if r["arm"] == "dehb"]

    meta = dict(
        label=label, task_id=task_id(label), path=entry.path, is_se=entry.is_se,
        kind=task.kind, target=task.target, R=task.R, paper_R=entry.paper_R,
        n_raw=task.n_raw, n_rows=len(task.y), capped=task.capped,
        repeats=repeats,
        median_D2H={"lite": float(np.median(lite)), "dehb": float(np.median(dehb))},
        median_d2h={"lite": float(np.median(lite_d2h)), "dehb": float(np.median(dehb_d2h))},
        cliffs_delta_D2H=cliffs_delta(dehb, lite),
        verdict=verdict(lite, dehb),
        # the same call on the natural-bounds scalar: if these disagree, the
        # colour depends on the scoring choice and that must be said out loud
        verdict_natural_d2h=verdict(lite_d2h, dehb_d2h),
        total_failed_fits=sum(r["n_failed_fits"] for r in records),
        seconds=round(sum(r["seconds"] for r in records), 1),
    )
    return dict(meta=meta, records=records)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("label", help="dataset label, e.g. 'iris' or 'Xomo OSP2'")
    ap.add_argument("--repeats", type=int, default=REPEATS)
    ap.add_argument("--row-cap", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    blob = run(args.label, args.repeats, args.row_cap)
    m = blob["meta"]
    print(" | ".join(f"{k}={v}" for k, v in m.items()
                     if k not in ("median_D2H", "median_d2h")))
    print()
    print(f"{'arm':<8}{'budget':>8}{'evals':>8}{'full-fid':>10}"
          f"{'med D2H':>10}{'med d2h':>10}{'s/run':>8}")
    for arm in ("lite", "dehb"):
        rs = [r for r in blob["records"] if r["arm"] == arm]
        print(f"{arm:<8}{rs[0]['budget']:>8}"
              f"{int(np.median([r['n_evals'] for r in rs])):>8}"
              f"{int(np.median([r['n_full_fidelity'] for r in rs])):>10}"
              f"{m['median_D2H'][arm]:>10.3f}{m['median_d2h'][arm]:>10.4f}"
              f"{np.median([r['seconds'] for r in rs]):>8.1f}")
    print(f"\nverdict: {m['verdict']}   (Cliff's delta {m['cliffs_delta_D2H']:+.3f}, "
          f"threshold 0.147)")
    if m["verdict"] != m["verdict_natural_d2h"]:
        print(f"NOTE: on the natural-bounds d2h the verdict would be "
              f"{m['verdict_natural_d2h']}")

    out = Path(args.out) if args.out else OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    tid = m["task_id"]
    (out / f"{tid}.json").write_text(json.dumps(
        dict(meta=m, verdict=m["verdict"]), indent=1, default=str))
    with (out / f"{tid}.runs.jsonl").open("w") as fh:
        for r in blob["records"]:
            fh.write(json.dumps(r, default=str) + "\n")
    print(f"wrote {out}/{tid}.json")


if __name__ == "__main__":
    main()
