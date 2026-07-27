"""Group A meta-features: static, label-free (X/Y structure).

Every feature has signature f(data: Data, rng: random.Random) -> dict
mapping {feature_name: value}. run_all.py merges the dicts per task.

Encoding rule for A5-A7 (fixed by CLAUDE.md -- do not change):
  - Sym columns are one-hot encoded (one indicator per observed category).
  - Num columns are standardized: (v - mean) / std over observed values.
  - Missing values "?" are imputed with the column mean (Num) / mode (Sym)
    BEFORE standardization / one-hot.
A8 uses signed Pearson correlation on raw Y values (pairwise-complete),
since objective conflict is a signed notion.
"""

import math
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import ezr  # frozen -- wrap, never edit

MISSING = "?"


# ---------------------------------------------------------------- encoding

def _num_stats(vals):
  """Mean/std of observed (non-'?') values in a raw column."""
  obs = [v for v in vals if v != MISSING]
  if not obs:
    return 0.0, 0.0
  mu = sum(obs) / len(obs)
  sd = math.sqrt(sum((v - mu) ** 2 for v in obs) / len(obs))
  return mu, sd


def encode_cols(data, cols):
  """Encode a list of ezr columns into a numeric matrix (n_rows x d).

  Per CLAUDE.md: one-hot Sym; standardize Num; impute '?' with mean/mode.
  Returns an (n, d) float array; d may be 0 if cols is empty.
  """
  n = len(data.rows)
  blocks = []
  for col in cols:
    vals = [row[col.at] for row in data.rows]
    if type(col) is ezr.Num:
      mu, sd = _num_stats(vals)
      filled = np.array([mu if v == MISSING else float(v) for v in vals])
      blocks.append(((filled - mu) / sd if sd > 0
                     else np.zeros(n)).reshape(n, 1))
    else:
      obs = [v for v in vals if v != MISSING]
      if not obs:
        continue  # column is all-missing: nothing to encode
      counts = {}
      for v in obs:
        counts[v] = counts.get(v, 0) + 1
      mode = max(counts, key=counts.get)
      filled = [mode if v == MISSING else v for v in vals]
      cats = sorted(counts, key=str)
      onehot = np.zeros((n, len(cats)))
      idx = {c: j for j, c in enumerate(cats)}
      for i, v in enumerate(filled):
        onehot[i, idx[v]] = 1.0
      blocks.append(onehot)
  if not blocks:
    return np.zeros((n, 0))
  return np.hstack(blocks)


def d2h_vector(data):
  """disty for every row (free on MOOT: objectives present in all rows)."""
  return np.array([ezr.disty(data, row) for row in data.rows])


# ---------------------------------------------------------------- features

def n_rows(data: ezr.Data, rng: random.Random) -> dict:
  """A1: row count."""
  return {"n_rows": len(data.rows)}


def n_xy(data: ezr.Data, rng: random.Random) -> dict:
  """A2: count of X and Y columns."""
  return {"n_x": len(data.cols.xs), "n_y": len(data.cols.ys)}


def sym_ratio(data: ezr.Data, rng: random.Random) -> dict:
  """A3: fraction of X columns that are Sym."""
  xs = data.cols.xs
  return {"sym_ratio":
          sum(type(c) is ezr.Sym for c in xs) / len(xs)}


def missing_density(data: ezr.Data, rng: random.Random) -> dict:
  """A4: fraction of '?' cells in X."""
  xs = data.cols.xs
  miss = sum(row[c.at] == MISSING for row in data.rows for c in xs)
  return {"missing_density": miss / (len(data.rows) * len(xs))}


def eff_rank_x(data: ezr.Data, rng: random.Random) -> dict:
  """A5: effective rank of X correlation matrix:
  exp(entropy of normalized eigenvalues)."""
  x = encode_cols(data, data.cols.xs)
  x = x[:, x.std(axis=0) > 0]  # constant columns have undefined correlation
  if x.shape[1] == 0:
    raise ValueError("no non-constant X columns after encoding")
  if x.shape[1] == 1:
    return {"eff_rank_x": 1.0}
  corr = np.corrcoef(x, rowvar=False)
  eig = np.clip(np.linalg.eigvalsh(corr), 0, None)
  p = eig / eig.sum()
  ent = -sum(pi * math.log(pi) for pi in p if pi > 0)
  return {"eff_rank_x": math.exp(ent)}


def cca_1(data: ezr.Data, rng: random.Random) -> dict:
  """A6: first canonical correlation between X and Y."""
  from sklearn.cross_decomposition import CCA
  x = encode_cols(data, data.cols.xs)
  y = encode_cols(data, data.cols.ys)
  x = x[:, x.std(axis=0) > 0]
  y = y[:, y.std(axis=0) > 0]
  if x.shape[1] == 0 or y.shape[1] == 0:
    raise ValueError("no non-constant columns for CCA")
  cca = CCA(n_components=1, max_iter=1000)
  xs, ys = cca.fit_transform(x, y)
  r = np.corrcoef(xs[:, 0], ys[:, 0])[0, 1]
  return {"cca_1": float(abs(r))}


def linear_r2(data: ezr.Data, rng: random.Random) -> dict:
  """A7: R^2 of ridge regression X -> d2h (5-fold CV)."""
  from sklearn.linear_model import Ridge
  from sklearn.model_selection import KFold, cross_val_score
  x = encode_cols(data, data.cols.xs)
  if x.shape[1] == 0:
    raise ValueError("no X columns after encoding")
  y = d2h_vector(data)
  kf = KFold(n_splits=5, shuffle=True,
             random_state=rng.randrange(2 ** 31))
  scores = cross_val_score(Ridge(alpha=1.0), x, y, cv=kf, scoring="r2")
  return {"linear_r2": float(scores.mean())}


def y_corr_mean(data: ezr.Data, rng: random.Random) -> dict:
  """A8: mean pairwise (signed Pearson) correlation among Y columns."""
  ys = [c for c in data.cols.ys if type(c) is ezr.Num]
  if len(ys) < 2:
    raise ValueError("fewer than 2 numeric Y columns: no pairs")
  rs = []
  for i in range(len(ys)):
    for j in range(i + 1, len(ys)):
      a, b = [], []
      for row in data.rows:  # pairwise-complete observations
        u, v = row[ys[i].at], row[ys[j].at]
        if u != MISSING and v != MISSING:
          a.append(float(u))
          b.append(float(v))
      if len(a) > 1 and np.std(a) > 0 and np.std(b) > 0:
        rs.append(float(np.corrcoef(a, b)[0, 1]))
  if not rs:
    raise ValueError("no defined Y-Y correlations")
  return {"y_corr_mean": float(np.mean(rs))}


# Registry: (feature_id, fn, output column names, labels used on MOOT).
# Group A is label-free everywhere (labels_used = 0).
FEATURES = [
  ("A1", n_rows,          ["n_rows"],        0),
  ("A2", n_xy,            ["n_x", "n_y"],    0),
  ("A3", sym_ratio,       ["sym_ratio"],     0),
  ("A4", missing_density, ["missing_density"], 0),
  ("A5", eff_rank_x,      ["eff_rank_x"],    0),
  ("A6", cca_1,           ["cca_1"],         0),
  ("A7", linear_r2,       ["linear_r2"],     0),
  ("A8", y_corr_mean,     ["y_corr_mean"],   0),
]
