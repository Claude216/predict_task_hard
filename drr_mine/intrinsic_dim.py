"""Intrinsic dimensionality (correlation dimension) and DRR for MOOT-style csvs.

Math is extracted from drr_mine/estimate_intrinsic_dim.py (Agrawal et al. /
XueqiYang's estimator, the code footnote 4 of the DRR paper says produced its
numbers): correlation integral C(r) over a log-spaced radius grid, then the
slope of ln C(r) vs ln r. Two estimates are reported:

  I_fit      (primary)   the scaling region 0.01 <= C(r) <= 0.2 is split into
                         100 equal pieces in ln r and the LARGEST piece slope
                         (dlnC/dlnr) is taken. No curve is fitted. Residual
                         underestimation at d >= ~8 is inherent to correlation
                         dimension at these sample sizes (a raw-euclidean
                         control reproduces it, so it is not our normalization).
  I_maxgrad  (secondary) Algorithm 1 of the DRR paper: smooth the finite
                         forward-difference gradients, take the max.

Why a max over pieces rather than a fitted curve. A polynomial fit forces one
smooth shape onto the whole window; where the data is a staircase (MOOT config
and binary_config tables have as few as 2-20 distinct distances) it smooths the
steps into a gentle curve and returns a plausible-looking slope from data that
has no scaling region at all. The binned reading cannot do that: an atomic
spectrum leaves most pieces empty, which is detectable (`n_live`, `flat_frac`)
and is reported as NaN + status=atomic_spectrum instead of a number.

Where the window sits. C ~ r^d only holds while the ball is small relative to
the data's extent; as C -> 1 the curve must flatten regardless of dimension, so
the top of it describes the bounding box, not the manifold. Under a MAX readout
the upper bound is nearly irrelevant -- the flattening tail only adds pieces
that never win -- and this is measured: on seeded Gaussians (d = 1..12, n=2000)
mean |error| moves only 0.99 -> 1.13 as the upper bound goes 0.1 -> 0.2 -> 0.35
-> 0.5 -> 1.0 at a fixed 0.01 floor. The LOWER bound is what matters: mean
|error| is 0.74 at 0.005, 1.02 at 0.01, 1.18 at 0.02, because every estimate is
biased low and a lower floor is closer to the r->0 limit that defines
correlation dimension. 0.01 is kept as the default floor because it is the
lowest at which MOOT's rougher tasks still hold up; the cost is a known
underestimate of roughly 25% at d >= 8.

ln C is taken WITHOUT an epsilon: C=0 gives -inf and the non-finite gradients
are dropped, so the jump off the zero-pairs plateau can never be mistaken for
signal (the bug that poisons external/drr_upstream).

Preprocessing follows external/drr_upstream/src/drr/data_processor.py in
structure only (csv load -> drop goal columns -> seeded row cap -> validate).
Encoding/normalization is our own, matching ezr.py's distx semantics:
  - column typing by MOOT header rule (uppercase-initial = Num, else Sym);
  - goal columns (+ - !) AND X-suffix ignored columns are excluded;
  - missing cells are the string "?" and are NEVER imputed; the distance layer
    applies ezr.aha's pessimistic rule instead;
  - Num values are normalized like ezr.norm (logistic squash of the z-score,
    clamped to +-3) by default; norm="minmax" is available as an alternative;
  - Sym distance is 0/1 mismatch; both-"?" is distance 1 (any column type);
  - per-row distance aggregates per-column distances with ezr.minkowski's
    mean form: (sum(d^p)/n_cols)^(1/p), p = ezr's the.p default of 2.

No fallback constants anywhere: when an estimate cannot be made the value is
NaN and the reason lands in the status field (skip-and-log, never crash).
"""

from __future__ import annotations

import math
import os

import numpy as np
import pandas as pd

MISSING = "?"


# ----------------------------------------------------------------- preprocessing

def load_moot_csv(path: str) -> pd.DataFrame:
    """Read a MOOT csv keeping "?" as a marker; strip header/cell whitespace."""
    df = pd.read_csv(path, dtype=str, skipinitialspace=True)
    df.columns = [str(c).strip() for c in df.columns]
    return df.apply(lambda s: s.str.strip() if s.dtype == "object" else s)


def split_columns(columns) -> tuple[list[str], list[str]]:
    """(feature_cols, dropped_cols) per MOOT conventions: drop goal columns
    ending + - ! and ignored columns ending X."""
    feats, dropped = [], []
    for c in columns:
        (dropped if str(c)[-1:] in "+-!X" else feats).append(c)
    return feats, dropped


def is_num(name: str) -> bool:
    """MOOT header rule: uppercase-initial column name = numeric."""
    return str(name)[:1].isupper()


def sample_rows(df: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(df), max_rows, replace=False)
    return df.iloc[np.sort(idx)].reset_index(drop=True)


def dedup_rows(df: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    """Drop rows that are duplicates ON THE FEATURE COLUMNS.

    Distances are computed from feats alone, so two rows agreeing on every
    feature are one point of the manifold however their Y columns differ: they
    contribute only zero-distance pairs, which carry no geometric information
    but do inflate the denominator of C(r) and hence shift every quantile of the
    scaling region. Deduplicating here (rather than dropping zero-distance pairs
    later) also means the row budget is spent on distinct points.
    """
    return df.drop_duplicates(subset=feats).reset_index(drop=True)


# ------------------------------------------------------------------- distances

def _norm_ezr(vals: np.ndarray) -> np.ndarray:
    """ezr.norm: logistic squash of the z-score, z clamped to [-3, 3].
    NaN (missing) stays NaN. Uses sd with ddof=1 like ezr's Welford sd."""
    ok = ~np.isnan(vals)
    mu = vals[ok].mean() if ok.any() else 0.0
    sd = vals[ok].std(ddof=1) if ok.sum() > 1 else 0.0
    z = np.clip((vals - mu) / (sd + 1e-32), -3, 3)
    return 1.0 / (1.0 + np.exp(-1.7 * z))


def _norm_minmax(vals: np.ndarray) -> np.ndarray:
    ok = ~np.isnan(vals)
    lo = vals[ok].min() if ok.any() else 0.0
    hi = vals[ok].max() if ok.any() else 0.0
    return (vals - lo) / (hi - lo + 1e-32)


def pairwise_distx(df: pd.DataFrame, feats: list[str], p: float = 2,
                   norm: str = "ezr") -> np.ndarray:
    """Condensed upper-triangle vector of ezr-distx distances (in [0, 1]).

    Per column d in [0,1] via ezr.aha:
      Num: |u - v| on normalized values; one side "?" -> max(x, 1-x) of the
           present value (aha's pessimistic substitution collapses to this);
           both "?" -> 1.
      Sym: 0 if equal else 1; both "?" -> 1 (aha checks this before symness).
    Rows aggregate as (mean of d^p)^(1/p), matching ezr.minkowski.
    """
    n = len(df)
    iu, ju = np.triu_indices(n, k=1)
    norm_fn = {"ezr": _norm_ezr, "minmax": _norm_minmax}[norm]
    acc = np.zeros(len(iu))

    for c in feats:
        col = df[c]
        if is_num(c):
            vals = pd.to_numeric(col.replace(MISSING, np.nan), errors="coerce").to_numpy(float)
            x = norm_fn(vals)
            u, v = x[iu], x[ju]
            d = np.abs(u - v)
            u_nan, v_nan = np.isnan(u), np.isnan(v)
            one = u_nan ^ v_nan
            present = np.where(u_nan, v, u)
            d[one] = np.maximum(present[one], 1.0 - present[one])
            d[u_nan & v_nan] = 1.0
        else:
            codes = col.fillna(MISSING).astype("category").cat.codes.to_numpy()
            miss = (col.fillna(MISSING) == MISSING).to_numpy()
            d = (codes[iu] != codes[ju]).astype(float)
            d[miss[iu] & miss[ju]] = 1.0
        acc += d ** p

    return (acc / (len(feats) + 1e-32)) ** (1.0 / p)


# ------------------------------------------------- correlation integral + slopes

def correlation_integral(distances: np.ndarray, num_radii: int = 100
                         ) -> tuple[np.ndarray, np.ndarray]:
    """C(r) = fraction of pairs with d < r, on a log-spaced grid from the min
    positive distance to just past the max. Callers pass positive distances
    only (duplicate rows carry no geometric information and are removed in
    estimate_intrinsic_dim); any zeros still present are kept in the
    denominator but excluded from the grid anchor."""
    pos = distances[distances > 0]
    if len(pos) == 0:
        raise ValueError("all pairwise distances are zero")
    radii = np.exp(np.linspace(np.log(pos.min()), np.log(distances.max() * 1.001),
                               num_radii))
    counts = np.searchsorted(np.sort(distances), radii, side="left")
    return radii, counts / len(distances)


def max_binned_slope(distances: np.ndarray, lo: float = 0.01, hi: float = 0.2,
                     n_pieces: int = 100, min_pairs: int = 100,
                     smooth: int = 1, min_live_frac: float = 0.5) -> dict:
    """Split the scaling region into `n_pieces` pieces, measure the log-log
    slope of each, and return the LARGEST as the intrinsic dimension.

    No curve is fitted. Each piece's slope is a straight finite difference of
    the empirical correlation integral:  dlnC / dlnr  across that piece.

    Region location vs piece placement (these are deliberately different):
      * the region is LOCATED by C bounds -- r_lo, r_hi are the lo- and
        hi-quantiles of the pairwise distances -- so it is always populated no
        matter where the distance mass sits (a fixed radius grid anchored at
        the min distance starves the window on e.g. MOOT's x264);
      * the pieces are PLACED equally in ln r inside [r_lo, r_hi], so every
        piece has the same dlnr and each slope is dlnC/const.
    Placing the pieces at equal dlnC instead (i.e. equal-count pieces) makes
    every slope const/dlnr, so "max slope" degenerates into "find the smallest
    radius gap" -- the exact division-by-a-vanishing-gap pathology that makes
    drr_upstream's max-gradient rule return hundreds. Measured on MOOT's SS-A:
    399 that way against 5.5 this way.

    A piece that gains no pairs has slope 0 and can never win the max, so empty
    pieces do not corrupt the value -- but they do say the spectrum is too
    atomic to measure. `min_live_frac` of the pieces must actually gain pairs;
    otherwise the ECDF is a step function whose "slope" is set by where the
    steps fall, and the result is NaN rather than a fabricated number. This is
    the guard that the old polynomial fit did NOT have: smoothing a 14-value
    staircase into a gentle curve produces a plausible-looking slope out of
    data that has no scaling region at all.

    `smooth` averages over w consecutive piece slopes before the max (w=1 = off,
    the raw reading). Raising it trades the max's upward bias -- the max of many
    noisy local slopes overshoots the true one -- against resolution.

    Returns {I, n_live, flat_frac, argmax_frac, slope_iqr}. argmax_frac locates
    the winning piece (0 = low-C edge, 1 = high-C edge); the slope should be
    non-increasing in r, so a max at the low edge is the expected shape and says
    the scaling region continues below `lo`. slope_iqr is the spread of the live
    pieces' slopes: large spread means the local slope never settled, i.e. there
    was no clean scaling region, and replaces the old fit's r2 as trust signal.
    """
    out = {"I": float("nan"), "n_live": 0, "flat_frac": float("nan"),
           "argmax_frac": float("nan"), "slope_iqr": float("nan")}
    d = np.sort(distances)
    if len(d) == 0:
        return out

    # never read below min_pairs of support: C=lo is a *fraction*, so on a small
    # table it can mean almost nothing (nasa93dem has 93 rows -> 4278 pairs, so
    # C=0.01 is 43 pairs). Binds only under ~10k pairs; above that this is a no-op.
    lo = max(lo, min_pairs / len(d))
    if not lo < hi:
        return out

    r_lo, r_hi = np.quantile(d, lo), np.quantile(d, hi)
    if not (r_lo > 0 and r_hi > r_lo):
        return out

    x = np.linspace(np.log(r_lo), np.log(r_hi), n_pieces + 1)
    counts = np.searchsorted(d, np.exp(x), side="left")
    gained = np.diff(counts)
    out["n_live"] = int((gained > 0).sum())
    out["flat_frac"] = float((gained == 0).mean())

    if out["n_live"] < min_live_frac * n_pieces:
        return out  # atomic spectrum: too few pieces carry any pairs

    with np.errstate(divide="ignore"):
        y = np.log(counts / len(d))
    slopes = np.diff(y) / np.diff(x)
    slopes = slopes[np.isfinite(slopes)]
    if len(slopes) == 0:
        return out

    live = slopes[slopes > 0]
    if len(live):
        out["slope_iqr"] = float(np.subtract(*np.percentile(live, [75, 25])))

    if smooth > 1 and len(slopes) >= smooth:
        slopes = np.convolve(slopes, np.ones(smooth) / smooth, mode="valid")
    k = int(np.argmax(slopes))
    out["I"] = float(slopes[k])
    out["argmax_frac"] = float(k / max(len(slopes) - 1, 1))
    return out


def max_smoothed_gradient(radii: np.ndarray, C: np.ndarray, window: int = 5):
    """Algorithm 1 of the DRR paper: forward-difference gradients of
    ln C vs ln r, drop non-finite (kills the C=0 plateau jump), moving-average
    smooth, return the max. NaN if nothing finite survives."""
    with np.errstate(divide="ignore"):
        lnC = np.log(C)
    grads = np.diff(lnC) / np.diff(np.log(radii))
    grads = grads[np.isfinite(grads)]
    if len(grads) == 0:
        return float("nan")
    w = min(window, len(grads))
    smoothed = np.convolve(grads, np.ones(w) / w, mode="valid")
    return float(smoothed.max())


# ------------------------------------------------------------------ public API

def estimate_intrinsic_dim(source, *, seed: int = 42, max_rows: int = 2000,
                           pool_rows: int = 5000, p: float = 2,
                           num_radii: int = 100, norm: str = "ezr") -> dict:
    """Estimate intrinsic dimension and DRR for a MOOT csv path, a DataFrame,
    or a numeric matrix (ndarray columns are treated as Num features).

    Row budget, in order: deduplicate on the feature columns, sample `pool_rows`
    (5000) from what remains, then sample `max_rows` (2000) from that pool for
    the distance matrix. The two-stage draw mirrors drr_upstream's own budget
    (DataProcessor caps at 5000, the estimator then subsamples to 2000) so the
    backends see comparably sized samples; both stages use `seed`."""
    if isinstance(source, str):
        df = load_moot_csv(source)
    elif isinstance(source, pd.DataFrame):
        df = source
    else:
        arr = np.asarray(source, dtype=float)
        df = pd.DataFrame(arr, columns=[f"N{j}" for j in range(arr.shape[1])])

    feats, _ = split_columns(df.columns)
    out = {"n_rows": len(df), "n_dedup": None, "dup_row_frac": float("nan"),
           "n_used": None, "R": len(feats),
           "I_fit": float("nan"), "n_live": 0, "flat_frac": float("nan"),
           "fit_argmax_frac": float("nan"), "slope_iqr": float("nan"),
           "I_maxgrad": float("nan"),
           "drr_fit": float("nan"), "drr_maxgrad": float("nan"),
           "dup_pair_frac": float("nan"), "seed": seed, "status": "ok"}

    if len(feats) == 0:
        out["status"] = "error: no feature columns"
        return out
    if len(df) < 2:
        out["status"] = "error: fewer than 2 rows"
        return out

    # dedup on features, then the two-stage row budget: N -> pool_rows -> max_rows
    df = dedup_rows(df, feats)
    out["n_dedup"] = len(df)
    out["dup_row_frac"] = 1.0 - len(df) / out["n_rows"] if out["n_rows"] else float("nan")
    if len(df) < 2:
        out["status"] = "error: fewer than 2 distinct rows"
        return out

    df = sample_rows(df, pool_rows, seed)
    df = sample_rows(df, max_rows, seed)
    out["n_used"] = len(df)

    try:
        distances = pairwise_distx(df, feats, p=p, norm=norm)
        # duplicate rows (zero-distance pairs) carry no geometric information:
        # drop them so C(r) is the ECDF of the positive distances only, matching
        # drr_modified. Keeping them would deflate every C level by a constant
        # factor and let heavy duplication push the fit window around.
        n_all = len(distances)
        distances = distances[distances > 0]
        out["dup_pair_frac"] = 1.0 - len(distances) / n_all if n_all else float("nan")
        if len(distances) == 0:
            raise ValueError("all pairwise distances are zero")
        radii, C = correlation_integral(distances, num_radii)
    except ValueError as e:
        out["status"] = f"error: {e}"
        return out

    f = max_binned_slope(distances)
    out.update(I_fit=f["I"], n_live=f["n_live"], flat_frac=f["flat_frac"],
               fit_argmax_frac=f["argmax_frac"], slope_iqr=f["slope_iqr"])
    out["I_maxgrad"] = max_smoothed_gradient(radii, C)

    R = out["R"]
    if not math.isfinite(out["I_fit"]):
        out["status"] = ("atomic_spectrum" if f["n_live"] else "no_scaling_region")
    elif out["I_fit"] > R:
        # a slope steeper than the embedding dimension is not a dimension;
        # keep I_fit as evidence but refuse to turn it into a DRR
        out["status"] = "warn: I_fit>R (no clean scaling region)"
    else:
        out["drr_fit"] = 1.0 - out["I_fit"] / R
    if math.isfinite(out["I_maxgrad"]):
        out["drr_maxgrad"] = 1.0 - out["I_maxgrad"] / R

    return out


if __name__ == "__main__":
    import json
    import sys

    for path in sys.argv[1:]:
        res = estimate_intrinsic_dim(path)
        print(os.path.basename(path), json.dumps(res, default=float))
