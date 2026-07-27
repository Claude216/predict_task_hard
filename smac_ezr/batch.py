"""Batch driver: run every MOOT task under a directory tree.

Tasks are independent -- each has its own frozen oracle, its own eps, its own
verdicts -- so this parallelises at the task level and never shares state
across processes.

Two things worth knowing before a long run:

  * The paper's 106 tasks are the SE core of MOOT; sales, finance and generic
    ML tables are excluded (Table I). A bare recursive glob will pick those up
    too, so use --exclude-dir or --tasks to match the paper's set.

  * Results are written one file per task and existing files are skipped, so an
    interrupted run resumes where it stopped. Delete a task's files to redo it.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

from runner import run_task

# Non-SE MOOT directories, excluded by the paper (Table I). Adjust to match the
# layout of your MOOT checkout.
DEFAULT_EXCLUDE = ()#"sales", "finance", "misc", "hpo")


def task_id(path: Path, root: Path) -> str:
    """Stable, filesystem-safe id that stays unique across subdirectories:
    config/SS-A.csv -> config__SS-A"""
    rel = path.relative_to(root).with_suffix("")
    return str(rel).replace("/", "__").replace("\\", "__")


def discover(root: Path, exclude_dirs: tuple[str, ...],
             tasks_file: Path | None) -> list[Path]:
    if tasks_file:
        wanted = {ln.strip() for ln in tasks_file.read_text().splitlines()
                  if ln.strip() and not ln.startswith("#")}
        found = [p for p in sorted(root.rglob("*.csv"))
                 if task_id(p, root) in wanted or p.name in wanted]
        missing = wanted - {task_id(p, root) for p in found} - {p.name for p in found}
        if missing:
            print(f"warning: {len(missing)} names in {tasks_file} not found: "
                  f"{sorted(missing)[:5]}...", file=sys.stderr)
        return found

    out = []
    for p in sorted(root.rglob("*.csv")):
        parts = {part.lower() for part in p.relative_to(root).parts[:-1]}
        if parts & {d.lower() for d in exclude_dirs}:
            continue
        out.append(p)
    return out


def _work(payload: dict) -> dict:
    """Runs in a worker process. Never raises: a bad table must not take the
    batch down with it."""
    path, out_dir, cfg = Path(payload["path"]), Path(payload["out"]), payload["cfg"]
    tid = payload["tid"]
    try:
        meta, records, verdicts = run_task(str(path), **cfg)
        (out_dir / f"{tid}.json").write_text(json.dumps(
            dict(task_id=tid, path=str(path), meta=meta,
                 verdicts={str(k): v for k, v in verdicts.items()}), indent=1))
        with (out_dir / f"{tid}.runs.jsonl").open("w") as fh:
            for r in records:
                fh.write(json.dumps(asdict(r), default=str) + "\n")
        return dict(tid=tid, ok=True, verdicts=verdicts,
                    shape=meta["input_shape"], obj=meta["objective"])
    except Exception as exc:
        (out_dir / f"{tid}.error.txt").write_text(traceback.format_exc())
        return dict(tid=tid, ok=False, error=f"{type(exc).__name__}: {exc}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="directory to search recursively for *.csv")
    ap.add_argument("--out", required=True, help="results directory")
    ap.add_argument("--optimizers", nargs="+", default=["smac", "ezr", "random"])
    ap.add_argument("--budgets", type=int, nargs="+", default=[30, 50, 100, 200])
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--oracle-seed", type=int, default=0)
    ap.add_argument("--save-traces", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--exclude-dir", nargs="*", default=list(DEFAULT_EXCLUDE))
    ap.add_argument("--tasks", type=Path, default=None,
                    help="file of task ids or filenames, one per line; "
                         "use this to pin the paper's exact task set")
    ap.add_argument("--redo", action="store_true", help="ignore existing results")
    ap.add_argument("--dry-run", action="store_true", help="list tasks and exit")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    files = discover(root, tuple(args.exclude_dir), args.tasks)
    jobs = []
    for p in files:
        tid = task_id(p, root)
        if not args.redo and (out_dir / f"{tid}.json").exists():
            continue
        jobs.append(dict(path=str(p), out=str(out_dir), tid=tid, cfg=dict(
            optimizers=args.optimizers, budgets=args.budgets, seeds=args.seeds,
            oracle_seed=args.oracle_seed, save_traces=args.save_traces)))

    print(f"found {len(files)} tables under {root}; "
          f"{len(files) - len(jobs)} already done; {len(jobs)} to run")
    if args.dry_run:
        for j in jobs:
            print("  ", j["tid"])
        return

    done, failed = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(_work, j): j["tid"] for j in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            res = fut.result()
            (done if res["ok"] else failed).append(res)
            tag = (" ".join(f"B{b}={v}" for b, v in res["verdicts"].items())
                   if res["ok"] else res["error"])
            print(f"[{i}/{len(jobs)}] {res['tid']}: {tag}", flush=True)

    print(f"\ndone={len(done)} failed={len(failed)}")
    if failed:
        print("failures (see *.error.txt):")
        for f in failed:
            print("  ", f["tid"], f["error"])

    # tally, the shape Figure 5 is built from
    tally: dict = {}
    for r in done:
        for b, v in r["verdicts"].items():
            tally.setdefault(b, {}).setdefault(v, 0)
            tally[b][v] += 1
    print("\nSMAC vs EZR, tasks won per budget:")
    for b in sorted(tally):
        row = tally[b]
        print(f"  B={b:<4} " + "  ".join(f"{k}={row[k]}" for k in sorted(row)))


if __name__ == "__main__":
    main()