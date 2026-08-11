"""Load a Table 1 csv into the (X, y) a RandomForest is trained on.

The paper (§3.4): "Each run optimizes a random forest performing regression (or
classification) for a single dependent variable in each of the datasets. The
selected dependent was always the first available in each dataset (and, for this
analysis, all other dependents are excluded from the data)."

So, per file:
  dependents = columns ending in + - !   (MOOT's convention)
  target     = the FIRST dependent, in header order
  features   = every column that is not a dependent  -> R
  dropped    = the remaining dependents

Note that MOOT's `X`-suffix "ignore" columns ARE features here. That is
deliberate: it is what makes R match the paper's Table 2 (xomo 27, adult 14),
and the paper fed whatever it counted. Recorded rather than silently chosen.

regression vs classification is read off the target's suffix: `!` is a klass,
`+`/`-` is numeric. One rule for all 49 datasets, including power consumption
whose `Sub_metering_1!` therefore makes it a classification task even though the
paper's prose describes predicting sub-metering values.

Encoding: MOOT's convention is that an UPPERCASE initial means numeric. That is
the primary rule (it is what ezr.Col does), with a fallback to symbolic when a
supposedly-numeric column will not convert -- external UCI files do not always
follow the convention, and a silent NaN column would be worse than a warning.
Missing values are the string "?" and are imputed with the column mean (numeric)
or mode (symbolic). Symbolic features become ordinal codes over sorted levels.
"""

from __future__ import annotations

import csv as _csv
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import REPO_ROOT
from .datasets import Entry

MISSING = "?"
ROW_CAP = 5000        # see module note in objective.py; also the DRR estimator's cap
TEST_FRAC = 0.20


@dataclass
class Task:
    label: str
    path: str
    is_se: bool
    kind: str                 # "regression" | "classification"
    target: str
    features: list[str]
    X: np.ndarray             # (n, R) float
    y: np.ndarray             # (n,) float (regression) or int (classification)
    n_raw: int                # rows in the file, before the cap
    capped: bool

    @property
    def R(self) -> int:
        return len(self.features)


def _read(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open(newline="", encoding="utf-8-sig") as fh:
        r = _csv.reader(fh)
        header = [c.strip() for c in next(r)]
        rows = [[c.strip() for c in row] for row in r if row and len(row) == len(header)]
    return header, rows


def _encode_column(name: str, raw: list[str]) -> np.ndarray:
    """One feature column -> float vector, imputed. See module docstring."""
    numeric = name[:1].isupper()
    if numeric:
        vals, ok = [], True
        for v in raw:
            if v == MISSING or v == "":
                vals.append(np.nan)
                continue
            try:
                vals.append(float(v))
            except ValueError:
                ok = False
                break
        if ok:
            a = np.array(vals, dtype=float)
            if np.isnan(a).all():
                return np.zeros(len(a))
            return np.where(np.isnan(a), np.nanmean(a), a)
        warnings.warn(f"column {name!r} has an uppercase initial (MOOT: numeric) "
                      f"but will not convert; treating as symbolic", stacklevel=2)
    # symbolic -> ordinal codes over sorted observed levels, mode-imputed
    obs = [v for v in raw if v not in (MISSING, "")]
    if not obs:
        return np.zeros(len(raw))
    levels = sorted(set(obs))
    idx = {v: i for i, v in enumerate(levels)}
    mode = max(levels, key=obs.count)
    return np.array([idx[v if v not in (MISSING, "") else mode] for v in raw], dtype=float)


def _encode_target(name: str, raw: list[str]) -> tuple[np.ndarray, str, np.ndarray]:
    """Target column -> (y, kind, keep_mask). Rows with a missing target are dropped."""
    keep = np.array([v not in (MISSING, "") for v in raw])
    vals = [v for v in raw if v not in (MISSING, "")]
    if name.endswith("!"):
        levels = sorted(set(vals))
        idx = {v: i for i, v in enumerate(levels)}
        return np.array([idx[v] for v in vals], dtype=int), "classification", keep
    return np.array([float(v) for v in vals], dtype=float), "regression", keep


def load(entry: Entry, seed: int = 0, row_cap: int = ROW_CAP) -> Task:
    """Read one Table 1 dataset. `seed` only affects the subsample of big tables."""
    path = REPO_ROOT / entry.path
    header, rows = _read(path)
    deps = [c for c in header if c.endswith(("+", "-", "!"))]
    if not deps:
        raise ValueError(f"{entry.label}: no dependent column (none ends in + - !)")
    target = deps[0]
    features = [c for c in header if c not in deps]
    if not features:
        raise ValueError(f"{entry.label}: no feature columns")

    col = {c: [row[i] for row in rows] for i, c in enumerate(header)}
    y, kind, keep = _encode_target(target, col[target])
    X = np.column_stack([_encode_column(c, col[c]) for c in features])[keep]

    n_raw = len(rows)
    capped = len(y) > row_cap
    if capped:
        # Seeded subsample. Needed at all for power consumption (2.08M rows) and
        # chosen at 5000 to match the cap the DRR estimator itself applies, so
        # the x-axis and the optimization see the same scale of data.
        pick = np.random.default_rng(seed).choice(len(y), size=row_cap, replace=False)
        pick.sort()
        X, y = X[pick], y[pick]

    return Task(label=entry.label, path=entry.path, is_se=entry.is_se, kind=kind,
                target=target, features=features, X=X, y=y,
                n_raw=n_raw, capped=capped)


def split(task: Task, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Seeded 80/20 split, stratified for classification.

    The paper mentions a test set ("N is the size of the test set") but never
    states the protocol, so this is a declared deviation. Stratifying keeps rare
    classes present in both halves -- heart disease's `num!` is very unbalanced,
    and an absent class makes macro precision/recall undefined.
    """
    rng = np.random.default_rng(seed)
    n = len(task.y)
    if task.kind == "classification":
        te = []
        for c in np.unique(task.y):
            idx = np.flatnonzero(task.y == c)
            rng.shuffle(idx)
            k = max(1, int(round(len(idx) * TEST_FRAC))) if len(idx) > 1 else 0
            te.append(idx[:k])
        te_idx = np.concatenate(te) if te else np.array([], dtype=int)
    else:
        idx = np.arange(n)
        rng.shuffle(idx)
        te_idx = idx[: max(1, int(round(n * TEST_FRAC)))]
    mask = np.zeros(n, dtype=bool)
    mask[te_idx] = True
    return task.X[~mask], task.y[~mask], task.X[mask], task.y[mask]
