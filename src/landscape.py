"""Group B meta-features: landscape geometry.

Every feature has signature f(data: Data, rng: random.Random) -> dict.

These use disty (distance-to-heaven over Y columns). On MOOT, objective
values are present in every csv row, so disty is FREE here (labels_used=0);
in deployment these features would cost labels. run_all.py records this via
the labels_used column of the registry.

B1 fdc        : Spearman rho between distx(row, best_row) and disty(row)
                over min(500, n_rows) sampled rows; best_row = argmin disty.
B2 smoothness : Spearman rho between distx(r1,r2) and |disty(r1)-disty(r2)|
                over 1000 sampled pairs.
B3 d2h_var    : variance of disty over all rows (confirmed baseline signal).
B4 d2h_skew   : skewness of the disty distribution (all rows).
B5 d2h_tail_gap: (median(d2h) - p10(d2h)) / IQR(d2h); heavy-left-tail flag.
"""

import random
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import ezr  # frozen -- wrap, never edit


def _d2h_all(data):
  """disty for every row (free on MOOT)."""
  return np.array([ezr.disty(data, row) for row in data.rows])


def fdc(data: ezr.Data, rng: random.Random) -> dict:
  """B1: fitness-distance correlation."""
  d2h = _d2h_all(data)
  best = data.rows[int(np.argmin(d2h))]
  n = min(500, len(data.rows))
  idx = rng.sample(range(len(data.rows)), n)
  dx = [ezr.distx(data, data.rows[i], best) for i in idx]
  dy = [d2h[i] for i in idx]
  rho, _ = stats.spearmanr(dx, dy)
  if np.isnan(rho):
    raise ValueError("fdc undefined (constant distances or d2h)")
  return {"fdc": float(rho)}


def smoothness(data: ezr.Data, rng: random.Random) -> dict:
  """B2: Spearman rho of x-distance vs |delta d2h| over 1000 pairs."""
  d2h = _d2h_all(data)
  n = len(data.rows)
  dx, dy = [], []
  for _ in range(1000):
    i = rng.randrange(n)
    j = rng.randrange(n)
    while j == i:
      j = rng.randrange(n)
    dx.append(ezr.distx(data, data.rows[i], data.rows[j]))
    dy.append(abs(d2h[i] - d2h[j]))
  rho, _ = stats.spearmanr(dx, dy)
  if np.isnan(rho):
    raise ValueError("smoothness undefined (constant distances or d2h)")
  return {"smoothness": float(rho)}


def d2h_var(data: ezr.Data, rng: random.Random) -> dict:
  """B3: variance of disty over all rows."""
  return {"d2h_var": float(np.var(_d2h_all(data)))}


def d2h_skew(data: ezr.Data, rng: random.Random) -> dict:
  """B4: skewness of the disty distribution."""
  return {"d2h_skew": float(stats.skew(_d2h_all(data)))}


def d2h_tail_gap(data: ezr.Data, rng: random.Random) -> dict:
  """B5: (median - p10) / IQR of d2h -- heavy-left-tail indicator."""
  d2h = _d2h_all(data)
  p10, p25, p50, p75 = np.percentile(d2h, [10, 25, 50, 75])
  iqr = p75 - p25
  if iqr == 0:
    raise ValueError("d2h_tail_gap undefined (IQR of d2h is 0)")
  return {"d2h_tail_gap": float((p50 - p10) / iqr)}


# labels_used = 0 on MOOT (objectives present in every row; see docstring).
FEATURES = [
  ("B1", fdc,          ["fdc"],          0),
  ("B2", smoothness,   ["smoothness"],   0),
  ("B3", d2h_var,      ["d2h_var"],      0),
  ("B4", d2h_skew,     ["d2h_skew"],     0),
  ("B5", d2h_tail_gap, ["d2h_tail_gap"], 0),
]
