#!/usr/bin/env python3
"""Run all 49 Table 1 datasets. Resumable, one process per dataset.

    conda run -n dmoot python -m dehb_ezr.paper.batch -j 6
    conda run -n dmoot python -m dehb_ezr.paper.batch --pilot

Datasets are independent -- own split, own pool, own verdict -- so this
parallelises at the dataset level. DEHB runs with n_workers=1 for the same
reason as in dehb_ezr/batch.py: a dask cluster per dataset on top of a process
pool would oversubscribe the machine.

One json + one jsonl per dataset; existing files are skipped, so an interrupted
sweep resumes. Delete a dataset's two files to redo it.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    __package__ = "dehb_ezr.paper"

from .datasets import DATASETS, task_id
from .run_one import OUT_DIR, REPEATS, run

# One tiny classification set, one tiny regression set, one mid-size SE table,
# and one that exercises the 5000-row cap. Enough to shake out both metric
# families, DEHB's eval accounting and real wall time before committing to 49.
PILOT = ["iris", "nasa93dem", "SS-A", "adult"]


def one(label: str, repeats: int, out: str) -> dict:
    blob = run(label, repeats)
    m = blob["meta"]
    d = Path(out)
    d.mkdir(parents=True, exist_ok=True)
    tid = m["task_id"]
    (d / f"{tid}.json").write_text(json.dumps(
        dict(meta=m, verdict=m["verdict"]), indent=1, default=str))
    with (d / f"{tid}.runs.jsonl").open("w") as fh:
        for r in blob["records"]:
            fh.write(json.dumps(r, default=str) + "\n")
    return dict(label=label, verdict=m["verdict"], R=m["R"], kind=m["kind"],
                seconds=m["seconds"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--repeats", type=int, default=REPEATS)
    ap.add_argument("-j", "--jobs", type=int, default=6)
    ap.add_argument("--pilot", action="store_true", help=f"only {', '.join(PILOT)}")
    ap.add_argument("--only", nargs="*", default=None, help="explicit labels")
    ap.add_argument("--redo", action="store_true")
    args = ap.parse_args()

    labels = args.only or (PILOT if args.pilot else [e.label for e in DATASETS])
    out = Path(args.out)
    todo = [x for x in labels
            if args.redo or not (out / f"{task_id(x)}.json").exists()]
    print(f"{len(labels)} dataset(s), {len(todo)} to run "
          f"({len(labels) - len(todo)} already done), jobs={args.jobs}, "
          f"repeats={args.repeats}\n", flush=True)
    if not todo:
        return 0

    done, failed = [], []
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(one, x, args.repeats, str(out)): x for x in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            x = futs[fut]
            try:
                r = fut.result()
                done.append(r)
                print(f"[{i}/{len(todo)}] {r['label']:<22} R={r['R']:<5} "
                      f"{r['kind']:<15} {r['verdict']:<14} {r['seconds']}s",
                      flush=True)
            except Exception as e:      # skip-and-log: one bad table must not stop 49
                failed.append((x, e))
                print(f"[{i}/{len(todo)}] {x:<22} FAILED: {type(e).__name__}: {e}",
                      flush=True)

    print(f"\n{len(done)} ok, {len(failed)} failed")
    if failed:
        log = out / "errors.log"
        out.mkdir(parents=True, exist_ok=True)
        with log.open("a") as fh:
            for x, e in failed:
                fh.write(f"{x}: {type(e).__name__}: {e}\n"
                         f"{''.join(traceback.format_exception(type(e), e, e.__traceback__))}\n")
        print(f"see {log}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
