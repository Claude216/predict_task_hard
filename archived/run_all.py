"""Driver: tasks x meta-features -> results/features.csv.

Usage:  python src/run_all.py [--groups A[,B[,C]]] [--merge]

--merge: compute only the requested groups and merge their columns into the
existing results/features.csv (keeping previously computed columns), instead
of rewriting it from scratch. labels_used_total accumulates.

For every MOOT csv under data/moot/optimize/ this loads the task once via
ezr.Data(csv(path)) and runs each registered feature function
f(data, rng) -> dict, merging the dicts into one row per task.

Reproducibility: GLOBAL_SEED below is the single driver seed. Each
(task, feature) pair gets its own random.Random seeded deterministically
from GLOBAL_SEED + task name + feature id (sha256-derived, so independent
of Python hash randomization). Per-repeat seeds inside Group C features
are handled explicitly in probes.py.

Failure policy: skip-and-log. A feature that raises writes NaN for all of
its output columns and appends one line to results/errors.log.
"""

import argparse
import csv as csvmod
import hashlib
import math
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import ezr  # frozen -- wrap, never edit

GLOBAL_SEED = 1
TASK_DIR = ROOT / "data" / "moot" / "optimize"
OUT_CSV = ROOT / "results" / "features.csv"
ERR_LOG = ROOT / "results" / "errors.log"


def feature_registry(groups):
  """Collect (feature_id, fn, out_keys, labels_used) for requested groups."""
  reg = []
  if "A" in groups:
    import static
    reg += static.FEATURES
  if "B" in groups:
    import landscape
    reg += landscape.FEATURES
  if "C" in groups:
    import probes
    reg += probes.FEATURES
  return reg


def seed_for(task, fid):
  """Deterministic per-(task, feature) seed from the global driver seed."""
  key = f"{GLOBAL_SEED}:{task}:{fid}".encode()
  return int.from_bytes(hashlib.sha256(key).digest()[:8], "big")


def log_error(errfh, task, fid, err):
  errfh.write(f"{task}\t{fid}\t{type(err).__name__}: {err}\n")
  errfh.flush()


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--groups", default="A",
                  help="comma-separated feature groups to run (A,B,C)")
  ap.add_argument("--merge", action="store_true",
                  help="merge new columns into existing features.csv")
  args = ap.parse_args()
  groups = [g.strip().upper() for g in args.groups.split(",")]

  registry = feature_registry(groups)
  all_keys = [k for _, _, keys, _ in registry for k in keys]

  old_rows, old_keys = {}, []
  if args.merge:
    with open(OUT_CSV, newline="") as fh:
      for r in csvmod.DictReader(fh):
        old_rows[r["task"]] = r
    old_keys = [k for k in next(iter(old_rows.values()))
                if k not in ("task", "labels_used_total") + tuple(all_keys)]

  tasks = sorted(TASK_DIR.rglob("*.csv"), key=lambda p: p.stem)
  rows, t_start = [], time.time()
  with open(ERR_LOG, "a") as errfh:
    errfh.write(f"# run groups={','.join(groups)} seed={GLOBAL_SEED} "
                f"at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    for i, path in enumerate(tasks, 1):
      task = path.stem
      t0 = time.time()
      row = {"task": task}
      labels_total = 0
      try:
        data = ezr.Data(ezr.csv(str(path)))
      except Exception as e:  # unreadable task: NaN everything
        log_error(errfh, task, "LOAD", e)
        for k in all_keys:
          row[k] = math.nan
        row["labels_used_total"] = math.nan
        rows.append(row)
        continue
      for fid, fn, keys, labels in registry:
        rng = random.Random(seed_for(task, fid))
        try:
          out = fn(data, rng)
          for k in keys:
            row[k] = out[k]
          labels_total += out.get("_labels_used", labels)
        except Exception as e:
          log_error(errfh, task, fid, e)
          for k in keys:
            row[k] = math.nan
      if args.merge and task in old_rows:
        old = old_rows[task]
        for k in old_keys:
          row.setdefault(k, old[k])
        prev = float(old.get("labels_used_total") or 0)
        labels_total += 0 if math.isnan(prev) else int(prev)
      row["labels_used_total"] = labels_total
      rows.append(row)
      print(f"[{i:3}/{len(tasks)}] {task:45s} {time.time()-t0:6.1f}s",
            flush=True)

  header = ["task"] + old_keys + all_keys + ["labels_used_total"]
  with open(OUT_CSV, "w", newline="") as fh:
    w = csvmod.DictWriter(fh, fieldnames=header)
    w.writeheader()
    w.writerows(rows)
  print(f"wrote {OUT_CSV} ({len(rows)} tasks, "
        f"{len(all_keys)} feature columns) in {time.time()-t_start:.0f}s")


if __name__ == "__main__":
  main()
