#!/usr/bin/env python3
"""Run every MOOT task under a root. Resumable, one process per task.

    conda run -n dmoot python -m dehb_ezr.batch \
        --root data/moot/optimize --out dehb_ezr/results -j 8

Tasks are independent -- each has its own frozen oracle, its own eps, its own
verdicts -- so this parallelises at the task level and shares no state across
processes. DEHB itself runs with n_workers=1 for exactly this reason: a dask
cluster per task on top of a process pool would oversubscribe the machine.

Results are one json + one jsonl per task and existing files are skipped, so an
interrupted sweep resumes where it stopped. Delete a task's two files to redo it.

Expected cost at 10 seeds, 127 tasks: roughly 2 core-hours, dominated by DEHB's
own bookkeeping rather than the oracle (see fidelity_oracle.py for why the
oracle is no longer the bottleneck). Run --pilot first.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "dehb_ezr"

from .run_task import ARMS, SEEDS, run_task
from .shared import task_id

# Three tasks spanning the range: few rows / high R, many rows / low R, and a
# wide binary space. Enough to shake out the fidelity oracle, DEHB's eval
# accounting and real per-task wall time before committing to 127.
PILOT = ["config/SS-A.csv", "process/nasa93dem.csv", "binary_config/Scrum1k.csv"]


def one(path: str, root: str, out: str, seeds: int, arms, oracle_seed: int) -> dict:
    tid = task_id(path, root)
    meta, records, verdicts = run_task(path, root=root, seeds=seeds,
                                       arms=arms, oracle_seed=oracle_seed)
    d = Path(out)
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{tid}.json").write_text(json.dumps(
        dict(task_id=tid, path=str(path), meta=meta, verdicts=verdicts),
        indent=1, default=str))
    with (d / f"{tid}.runs.jsonl").open("w") as fh:
        for r in records:
            fh.write(json.dumps(asdict(r), default=str) + "\n")
    return dict(task_id=tid, verdicts=verdicts,
                seconds=round(sum(r.seconds for r in records), 1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/moot/optimize")
    ap.add_argument("--out", default="dehb_ezr/results")
    ap.add_argument("--seeds", type=int, default=SEEDS)
    ap.add_argument("--arms", nargs="+", default=None, choices=list(ARMS))
    ap.add_argument("--oracle-seed", type=int, default=0)
    ap.add_argument("-j", "--jobs", type=int, default=8)
    ap.add_argument("--pilot", action="store_true",
                    help=f"only {', '.join(PILOT)}")
    ap.add_argument("--redo", action="store_true", help="ignore existing results")
    args = ap.parse_args()

    root = Path(args.root)
    if args.pilot:
        paths = [root / p for p in PILOT]
        missing = [p for p in paths if not p.exists()]
        if missing:
            sys.exit(f"pilot task(s) not found: {missing}")
    else:
        paths = sorted(root.rglob("*.csv"))
    if not paths:
        sys.exit(f"no csv under {root}")

    out = Path(args.out)
    todo = [p for p in paths
            if args.redo or not (out / f"{task_id(p, root)}.json").exists()]
    print(f"{len(paths)} task(s), {len(todo)} to run "
          f"({len(paths) - len(todo)} already done), jobs={args.jobs}, "
          f"seeds={args.seeds}\n")
    if not todo:
        return 0

    done, failed = [], []
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(one, str(p), str(root), str(out), args.seeds,
                          args.arms, args.oracle_seed): p for p in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            p = futs[fut]
            try:
                r = fut.result()
                done.append(r)
                v = r["verdicts"].get("lite_vs_heavy")
                floor = r["verdicts"].get("dehb_vs_floor")
                print(f"[{i}/{len(todo)}] {r['task_id']:<40} "
                      f"lite_vs_heavy={str(v):<12} dehb_vs_floor={str(floor):<12} "
                      f"{r['seconds']}s")
            except Exception as e:     # skip-and-log: one bad table must not kill the sweep
                failed.append((p, e))
                print(f"[{i}/{len(todo)}] {p} FAILED: {type(e).__name__}: {e}")

    print(f"\n{len(done)} ok, {len(failed)} failed")
    if failed:
        log = out / "errors.log"
        with log.open("a") as fh:
            for p, e in failed:
                fh.write(f"{p}: {type(e).__name__}: {e}\n"
                         f"{''.join(traceback.format_exception(type(e), e, e.__traceback__))}\n")
        print(f"see {log}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
