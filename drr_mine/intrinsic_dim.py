"""Intrinsic dimensionality (correlation dimension) and DRR for MOOT-style csvs.

Math is extracted from drr_mine/estimate_intrinsic_dim.py (Agrawal et al. /
XueqiYang's estimator, the code footnote 4 of the DRR paper says produced its
numbers): correlation integral C(r) over a log-spaced radius grid, then the
slope of ln C(r) vs ln r. Two estimates are reported:

  I_fit      (primary)   OLS slope over the scaling region 0.01 <= C(r) <= 0.2,
                         with the fit's R^2 as a trust diagnostic. Residual
                         underestimation at d >= ~8 is inherent to correlation
                         dimension at these sample sizes (a raw-euclidean
                         control reproduces it, so it is not our normalization).
  I_maxgrad  (secondary) Algorithm 1 of the DRR paper: smooth the finite
                         forward-difference gradients, take the max.

Why the fit window sits low. C ~ r^d only holds while the ball is small
relative to the data's extent; as C -> 1 the curve must flatten regardless of
dimension, so the top of it describes the bounding box, not the manifold.
Raising the upper bound therefore biases the slope down monotonically -- on
seeded Gaussians (n=2000, lower bound 0.01) true d=8 recovers as 6.17 / 5.91 /
5.63 / 5.39 / 4.57 at upper bounds 0.1 / 0.2 / 0.35 / 0.5 / 0.95, and median
fit r^2 over the 127 MOOT tasks falls from 0.998 at 0.2 to 0.985 at 0.95.
0.2 is 20% of *pairs* (~400k at n=2000), so fit support is never the binding
constraint -- the 0.01 floor is what protects support, by excluding the
sparse-count staircase where slopes are set by grid spacing rather than
geometry. The exact bound barely matters downstream anyway: Spearman of task
ranking against the 0.2 default is >= 0.978 for every window tried, with
median |dI| <= 0.28.

This is a fixed window, NOT a search for the steepest stretch -- the same C
levels are read on every dataset, so on some tasks it returns less than a
higher window would (dress-up: 12.5 at 0.01-0.2's low neighbour vs 16.6 at
0.05-0.5). Sliding the window down by 5x repeatedly makes the estimate
*converge* on the truth rather than diverge (true d=8 -> 5.89, 6.74, 7.40,
7.86, 8.05), which is the r->0 limit that defines correlation dimension; the
max-gradient rule by contrast runs away (4.5 on a 1-D line, 26.7 on pom3a).
So our default is deliberately conservative: on clean continuous data a lower
window is more accurate, but real MOOT data breaks below C=0.01 -- SS-A
collapses to 0.43 (r^2 0.89) then 0.49 (r^2 0.74), Wine_quality to 1.20
(r^2 0.76), accessories to NaN. 0.01 is the lowest floor at which all 127
tasks still yield a well-formed fit; the cost is a known underestimate of
roughly 25% at d >= 8.

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
    """C(r) = fraction of ALL pairs (zero-distance ones included) with d < r,
    on a log-spaced grid from the min positive distance to just past the max."""
    pos = distances[distances > 0]
    if len(pos) == 0:
        raise ValueError("all pairwise distances are zero")
    radii = np.exp(np.linspace(np.log(pos.min()), np.log(distances.max() * 1.001),
                               num_radii))
    counts = np.searchsorted(np.sort(distances), radii, side="left")
    return radii, counts / len(distances)


def fit_scaling_region(distances: np.ndarray, lo: float = 0.01, hi: float = 0.2,
                       n_levels: int = 60, deg: int = 3,
                       min_pairs: int = 100) -> dict:
    """Fit a CURVE to ln C vs ln r over the scaling region lo <= C <= hi and
    return the MAXIMUM of that curve's slope as the intrinsic dimension.

    Samples the curve by inverse CDF: for log-spaced levels c in [lo, hi] the
    radius is the c-quantile of the pairwise distances, so the fit region is
    always fully populated no matter how the distance mass is distributed (a
    fixed radius grid anchored at the min distance can starve the window when
    distances concentrate far from the minimum, e.g. MOOT's x264). Duplicate
    quantiles (atomic distance spectra, e.g. 17 distinct values across 6e5
    pairs) collapse to too few distinct points and yield NaN — a slope through
    an atomic spectrum is not a dimension.

    Max-of-a-curve rather than the slope of one straight line, because the local
    slope is not constant across the region: it decays as C(r) starts to
    saturate, so a single OLS line averages the dimension together with that
    tail and reads low (seeded Gaussians, true d=8: line 5.87, curve max 6.63).
    Taking the max of a SMOOTH fit is also what keeps this safe -- the derivative
    of a fitted polynomial cannot divide by a vanishing radius gap, which is
    precisely how the raw max-gradient rule (Algorithm 1) reaches 1e15 on
    discrete data.

    Degree adapts to the support available -- the largest d in 1..deg with
    n_points >= 3*(d+1) -- because a high-order fit through few points bends at
    the window edge and invents a maximum there (nasa93dem has 6 distinct radii;
    a cubic claims 4.02 at the high-C edge where the honest slope is 0.83).
    Degree 1 reproduces the plain OLS slope exactly.

    Returns {I, r2, n_points, degree, argmax_frac, I_linear}. argmax_frac locates
    the max in the window (0 = low-C edge, 1 = high-C edge). The slope should be
    non-increasing in r, so a max at the low-C edge is the expected shape and
    says the scaling region continues below `lo`; argmax_frac near 1 means the
    fit bent the wrong way and that row deserves suspicion.
    """
    # never fit below min_pairs of support: C=lo is a *fraction*, so on a small
    # table it can mean almost nothing (nasa93dem has 93 rows -> 4278 pairs, so
    # C=0.01 is 43 pairs). Binds only under ~142 rows; above that 0.01 already
    # means thousands of pairs and this is a no-op.
    lo = max(lo, min_pairs / max(len(distances), 1))
    if not lo < hi:
        return {"I": float("nan"), "r2": float("nan"), "n_points": 0,
                "degree": 0, "argmax_frac": float("nan"),
                "I_linear": float("nan")}

    levels = np.geomspace(lo, hi, n_levels)
    radii = np.quantile(distances, levels)
    keep = radii > 0
    radii, keep_levels = radii[keep], levels[keep]
    x, idx = np.unique(np.log(radii), return_index=True)
    y = np.log(keep_levels[idx])

    out = {"I": float("nan"), "r2": float("nan"), "n_points": int(len(x)),
           "degree": 0, "argmax_frac": float("nan"), "I_linear": float("nan")}
    if len(x) < 3 or np.ptp(x) < 1e-12:
        return out

    out["I_linear"] = float(np.polyfit(x, y, 1)[0])
    d = max((k for k in range(1, deg + 1) if len(x) >= 3 * (k + 1)), default=1)

    coef = np.polyfit(x, y, d)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = (1.0 - np.sum((y - np.polyval(coef, x)) ** 2) / ss_tot
          if ss_tot > 0 else float("nan"))

    grid = np.linspace(x.min(), x.max(), 400)
    slopes = np.polyval(np.polyder(coef), grid)
    k = int(np.argmax(slopes))
    out.update(I=float(slopes[k]), r2=float(r2), degree=int(d),
               argmax_frac=float((grid[k] - x.min()) / np.ptp(x)))
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
                           p: float = 2, num_radii: int = 100,
                           norm: str = "ezr") -> dict:
    """Estimate intrinsic dimension and DRR for a MOOT csv path, a DataFrame,
    or a numeric matrix (ndarray columns are treated as Num features)."""
    if isinstance(source, str):
        df = load_moot_csv(source)
    elif isinstance(source, pd.DataFrame):
        df = source
    else:
        arr = np.asarray(source, dtype=float)
        df = pd.DataFrame(arr, columns=[f"N{j}" for j in range(arr.shape[1])])

    feats, _ = split_columns(df.columns)
    out = {"n_rows": len(df), "n_used": None, "R": len(feats),
           "I_fit": float("nan"), "fit_r2": float("nan"), "fit_n_radii": 0,
           "fit_degree": 0, "fit_argmax_frac": float("nan"),
           "I_linear": float("nan"), "I_maxgrad": float("nan"),
           "drr_fit": float("nan"), "drr_maxgrad": float("nan"),
           "seed": seed, "status": "ok"}

    if len(feats) == 0:
        out["status"] = "error: no feature columns"
        return out
    if len(df) < 2:
        out["status"] = "error: fewer than 2 rows"
        return out

    df = sample_rows(df, max_rows, seed)
    out["n_used"] = len(df)

    try:
        distances = pairwise_distx(df, feats, p=p, norm=norm)
        radii, C = correlation_integral(distances, num_radii)
    except ValueError as e:
        out["status"] = f"error: {e}"
        return out

    f = fit_scaling_region(distances)
    out.update(I_fit=f["I"], fit_r2=f["r2"], fit_n_radii=f["n_points"],
               fit_degree=f["degree"], fit_argmax_frac=f["argmax_frac"],
               I_linear=f["I_linear"])
    out["I_maxgrad"] = max_smoothed_gradient(radii, C)

    R = out["R"]
    if not math.isfinite(out["I_fit"]):
        out["status"] = "no_scaling_region"
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
