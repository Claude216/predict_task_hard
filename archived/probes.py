"""Group C meta-features: mechanism-specific probes.

Signature f(data: Data, rng: random.Random) -> dict. NOTE: per CLAUDE.md,
Group C runs 20 repeats per task with EXPLICIT seeds 1..20 (the rng passed
by run_all.py is not used for the repeats); each feature reports the median
and IQR over repeats as two columns.

C1 nb_auc: per repeat -- warm-start sample of 32 rows; split best/rest by
  disty at int(sqrt(32))=5 exactly as acquire()'s warm_start does; train NB
  (ezr.likes) on the two groups; rank a held-out sample by
  likes(best)-likes(rest) and report AUC of best-vs-rest.
  Held-out design (fixed with user 2026-07-12, spec was silent):
    held-out = seeded sample of min(500, n_rows-32) unseen rows (mirrors
    B1's min(500, n) cap); its true "best" rows are the top int(sqrt(|H|))
    by disty -- the same split rule the mechanism itself uses.
  Labels used: 32.

C2 landmark_10: per repeat -- run ezr.acquire() with the.learn.budget=10
  (restored afterwards), report the win score (ezr.wins on the full task,
  100=best) of the best row found. Labels used: 10 (+4 warm start) = 14.

labels_used_total records the PER-REPEAT deployment cost (32+14=46), not
the 20x benchmark total: repeats are a measurement device for median/IQR,
not part of the deployed probe (user decision 2026-07-12).
"""

import math
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import ezr  # frozen -- wrap, never edit

N_REPEATS = 20
REPEAT_SEEDS = range(1, N_REPEATS + 1)
WARM = 32
HELD_CAP = 500


def _auc(scores, labels):
  """Rank-based AUC (Mann-Whitney): P(score_pos > score_neg)."""
  from scipy.stats import rankdata
  scores, labels = np.asarray(scores), np.asarray(labels, bool)
  npos, nneg = labels.sum(), (~labels).sum()
  if npos == 0 or nneg == 0:
    raise ValueError("AUC undefined: one class empty in held-out")
  ranks = rankdata(scores)
  return (ranks[labels].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def _nb_auc_once(data, seed):
  rng = random.Random(seed)
  n = len(data.rows)
  if n < WARM + 2:
    raise ValueError(f"too few rows ({n}) for 32-row warm start")
  idx = list(range(n))
  train = rng.sample(idx, WARM)
  train_sorted = sorted(train, key=lambda i: ezr.disty(data, data.rows[i]))
  nbest = int(math.sqrt(WARM))  # as acquire()'s warm_start splits
  best = ezr.clone(data, [data.rows[i] for i in train_sorted[:nbest]])
  rest = ezr.clone(data, [data.rows[i] for i in train_sorted[nbest:]])
  unseen = [i for i in idx if i not in set(train)]
  held = rng.sample(unseen, min(HELD_CAP, len(unseen)))
  held_sorted = sorted(held, key=lambda i: ezr.disty(data, data.rows[i]))
  h_best = set(held_sorted[:int(math.sqrt(len(held)))])
  scores, labels = [], []
  for i in held:
    row = data.rows[i]
    scores.append(ezr.likes(best, row, WARM, 2) - ezr.likes(rest, row, WARM, 2))
    labels.append(i in h_best)
  return _auc(scores, labels)


def nb_auc(data: ezr.Data, rng: random.Random) -> dict:
  """C1: NB best-vs-rest AUC from a 32-row warm start; 20 seeded repeats."""
  vals = [_nb_auc_once(data, s) for s in REPEAT_SEEDS]
  return {"nb_auc_med": float(np.median(vals)),
          "nb_auc_iqr": float(np.percentile(vals, 75)
                              - np.percentile(vals, 25))}


def landmark_10(data: ezr.Data, rng: random.Random) -> dict:
  """C2: win score of best row found by acquire() with budget=10;
  20 seeded repeats. the.learn.budget restored afterwards."""
  win = ezr.wins(data)  # free on MOOT: objectives present in all rows
  old_budget = ezr.the.learn.budget
  old_state = random.getstate()
  vals = []
  try:
    ezr.the.learn.budget = 10
    for s in REPEAT_SEEDS:
      random.seed(s)  # acquire() uses the module-level random
      lab = ezr.acquire(data)
      vals.append(win(lab.rows[0]))  # lab is sorted best-first by disty
  finally:
    ezr.the.learn.budget = old_budget
    random.setstate(old_state)
  return {"landmark_10_med": float(np.median(vals)),
          "landmark_10_iqr": float(np.percentile(vals, 75)
                                   - np.percentile(vals, 25))}


# labels_used: per-repeat deployment cost (see module docstring).
FEATURES = [
  ("C1", nb_auc,      ["nb_auc_med", "nb_auc_iqr"],           32),
  ("C2", landmark_10, ["landmark_10_med", "landmark_10_iqr"], 14),
]
