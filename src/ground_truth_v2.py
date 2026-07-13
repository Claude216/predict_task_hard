"""Regenerate EZR ground truth under the CURRENT frozen ezr.py defaults.

User-approved override (2026-07-13) of CLAUDE.md's "do not regenerate ground
truth": the original scores came from an OLDER ezr variant at its defaults
(Budget=30, likely()/Tree API). This reruns the SAME metric definitions with
this repo's ezr.py at its defaults:
  p=2, learn.budget=50, learn.start=4, learn.check=5, few=128,
  bayes.m=2, bayes.k=1

Port of src/per_dataset_meta.py's ezr_default_scores (itself 9.py's "ezr"
treatment), old API -> frozen ezr.py API:
  likely(train)          -> acquire(train).rows   (4 warm start + 50 budget)
  Tree(data)/leaf .mu    -> treeGrow(data, rows)/leaf .ynum.mu
  win formula UNCHANGED  -> int(100*(1 - (d2h-lo)/(mean-lo))), 100=best row,
                            0=average row (NOT ezr.wins, which clamps).

score (performance rmse, lower = better):
  20 train/holdout half-splits, seeds 0,10,...,190 (as the original run).
  Per split: label via acquire() on the train half; grow a tree on the
  labeled rows; take the the.learn.check=5 holdout rows with the best
  predicted leaf value; err = win(best of holdout+labels)
  - max(win(best of picks), win(best of labels)); rmse = sqrt(mean err^2).

stability (agreement %, higher = more stable):
  Fixed seed-42 half-split (part of the metric definition); 20 trees from
  20 independent acquire() labelings, seeds 0..19, tree target =
  disty w.r.t. the FULL data; agreement = % of <=100 test rows whose 20
  predicted win scores have sd < 0.35 * sd(win over all rows).

Skipped per user: "Comparison" metrics (need multiple treatments) and
stability_jaccard (needs the old repo's jaccard.py; supplementary anyway).

Output: results/ground_truth_v2.csv (task, score, stability).
"""

import csv as csvmod
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import ezr
from ezr import Data, acquire, adds, clone, csv, disty, the, treeGrow, treeLeaf

REPEATS = 20
OUT = ROOT / "results" / "ground_truth_v2.csv"

assert (the.learn.budget, the.learn.check, the.learn.start, the.few,
        the.p, the.bayes.m, the.bayes.k) == (50, 5, 4, 128, 2, 2, 1), \
       "ezr.py defaults changed?"


def scores(path):
  all_data = Data(csv(str(path)))
  ys = [disty(all_data, row) for row in all_data.rows]
  lo, mu = min(ys), sum(ys) / len(ys)
  win = lambda v: int(100 * (1 - (v - lo) / (mu - lo + 1e-32)))
  b4_wins = adds([win(v) for v in ys])

  # ---- performance rmse (old Part 1, seeds 0,10,...,190) ----
  mse = 0
  for rand_seed in range(REPEATS):
    random.seed(rand_seed * 10)
    the.seed = rand_seed * 10
    shuffled = random.sample(all_data.rows, len(all_data.rows))
    half = int(0.5 * len(all_data.rows))
    train = clone(all_data, shuffled[:half])
    holdout = shuffled[half:]
    labels = acquire(train).rows
    lab = clone(train, labels)
    tree = treeGrow(lab, lab.rows)
    top = sorted(((treeLeaf(tree, row).ynum.mu, row) for row in holdout),
                 key=lambda x: x[0])[:the.learn.check]
    trt = max(win(min(disty(all_data, row) for _, row in top)),
              win(min(disty(all_data, row) for row in labels)))
    ref = win(min(disty(all_data, row) for row in holdout + labels))
    mse += abs(ref - trt) ** 2
  rmse = (mse / REPEATS) ** 0.5

  # ---- stability agreement (old Part 2: seed-42 split, tree seeds 0..19) ----
  random.seed(42)
  rows = all_data.rows[:]
  random.shuffle(rows)
  half = len(rows) // 2
  train = clone(all_data, rows[:half])
  test_rows = rows[half:][:min(100, len(rows) - half)]
  full_y = lambda r: disty(all_data, r)

  trees = []
  for rand_seed in range(REPEATS):
    the.seed = rand_seed
    random.seed(rand_seed)
    labels = acquire(train).rows
    lab = clone(train, labels)
    trees.append(treeGrow(lab, lab.rows, klass=full_y))

  agreement = 0
  for row in test_rows:
    preds = [win(treeLeaf(tree, row).ynum.mu) for tree in trees]
    if adds(preds).sd < 0.35 * b4_wins.sd:
      agreement += 1
  agreement = agreement * 100 // len(test_rows)

  return rmse, agreement


def main():
  tasks = sorted((ROOT / "data" / "moot" / "optimize").rglob("*.csv"),
                 key=lambda p: p.stem)
  out, t0 = [], time.time()
  for i, path in enumerate(tasks, 1):
    t1 = time.time()
    rmse, agreement = scores(path)
    out.append({"task": path.stem, "score": round(rmse, 2),
                "stability": agreement})
    print(f"[{i:3}/{len(tasks)}] {path.stem:45s} score={rmse:7.2f} "
          f"stability={agreement:3d}  {time.time()-t1:6.1f}s", flush=True)
  with open(OUT, "w", newline="") as fh:
    w = csvmod.DictWriter(fh, fieldnames=["task", "score", "stability"])
    w.writeheader()
    w.writerows(out)
  print(f"wrote {OUT} ({len(out)} tasks) in {time.time()-t0:.0f}s")


if __name__ == "__main__":
  main()
