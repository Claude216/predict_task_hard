"""
Per-dataset meta-features for all 127 MOOT CSVs under data/optimize/*/*.csv,
merged with each dataset's EZR performance and stability, into ONE data file.

Column roles follow ezr.py's naming convention (ezr.py:Cols, lines 54-63):
  - name[-1] == "X"    -> ignored column (not modeled)
  - name[-1] in "-+"   -> y / optimization-target column
  - otherwise          -> x / feature column
  - name[0].isupper()  -> numeric (Num); else symbolic (Sym)

Computed per dataset:

1. Variation of each x column (targets excluded), standardized to [0,1]:
     numeric : column min-max scaled to [0,1], variation = 2 * population std.
               (a [0,1]-valued variable has max std 0.5, at half the mass on
               each endpoint, so 2*std is exactly [0,1]; constant column = 0)
     symbolic: normalized Gini impurity = gini / (1 - 1/k), k = #distinct
               values (max gini for k categories is 1-1/k, so this is [0,1];
               gini matches ezr's default -I Impurity=gini). Constant = 0.

2. Correlation of each pair of x features, all on a [0,1] scale:
     numeric-numeric  : |Pearson r|          (pairwise-complete rows)
     symbolic-symbolic: Cramer's V           (chi-square based)
     numeric-symbolic : correlation ratio eta (sqrt(SS_between/SS_total))
   Pairs involving a constant column are undefined (NaN) and excluded from
   the aggregates.

3. Variation of the y / optimization-target columns: same standardized
   variation as (1).

Because datasets have different numbers of columns, the single wide file
stores per-dataset AGGREGATES (mean/std/min/max) of (1)-(3); the full
per-column / per-pair values are preserved in a long-format companion file.

Merged outcome columns (both sources cover all 127 datasets):
  - performance_rmse, ezr_stability_agreement: results/9/{dataset}.csv row
    trt=="ezr" (Table IV refined config: labels=50, near, min_leaf=3, gini) --
    the same performance target used by analysis/validation2_prediction.py.
  - stability_jaccard_mean/min/max: jaccard.csv (20-seed pairwise Jaccard of
    tree feature sets, from jaccard.py).

Outputs:
  analysis/meta_features.csv        (single file: 1 row per dataset)
  analysis/meta_features_detail.csv (long format: every column variation and
                                     every feature-pair correlation)

Run: /Users/claudeli/Tool/miniconda3/envs/caus26/bin/python analysis/compute_meta_features.py
"""
import glob
import os

import numpy as np
import pandas as pd

REPO = "/Users/claudeli/NCSU/Research/26sp_aise/stability_evaluation"
DATA = os.path.join(REPO, "data/optimize")


def col_role(name: str) -> str:
    if name[-1] == "X":
        return "ignore"
    if name[-1] in "-+":
        return "y"
    return "x"


def is_numeric_name(name: str) -> bool:
    return name[0].isupper()


def to_numeric(df, name):
    "Float array with NaN for missing ('?')."
    return pd.to_numeric(df[name].replace("?", np.nan), errors="coerce").values


def to_codes(df, name):
    "Integer category codes with -1 for missing ('?')."
    s = df[name].astype(str).str.strip()
    codes, _ = pd.factorize(s.where(s != "?"))
    return codes  # pd.factorize maps NaN to -1


def variation01_numeric(v):
    v = v[~np.isnan(v)]
    if len(v) < 2 or v.max() == v.min():
        return 0.0
    scaled = (v - v.min()) / (v.max() - v.min())
    return float(min(1.0, 2.0 * scaled.std()))


def variation01_symbolic(codes):
    codes = codes[codes >= 0]
    if len(codes) < 2:
        return 0.0
    counts = np.bincount(codes)
    counts = counts[counts > 0]
    k = len(counts)
    if k < 2:
        return 0.0
    p = counts / counts.sum()
    gini = 1.0 - np.sum(p ** 2)
    return float(gini / (1.0 - 1.0 / k))


def cramers_v(ca, cb):
    ok = (ca >= 0) & (cb >= 0)
    a, b = ca[ok], cb[ok]
    n = len(a)
    if n < 3:
        return np.nan
    _, a = np.unique(a, return_inverse=True)
    _, b = np.unique(b, return_inverse=True)
    r, c = a.max() + 1, b.max() + 1
    if min(r, c) < 2:
        return np.nan
    tab = np.bincount(a * c + b, minlength=r * c).reshape(r, c).astype(float)
    expected = tab.sum(1, keepdims=True) * tab.sum(0, keepdims=True) / n
    chi2 = ((tab - expected) ** 2 / expected).sum()
    return float(np.sqrt(chi2 / (n * (min(r, c) - 1))))


def corr_ratio(v, codes):
    ok = ~np.isnan(v) & (codes >= 0)
    v, g = v[ok], codes[ok]
    if len(v) < 3:
        return np.nan
    _, g = np.unique(g, return_inverse=True)
    if g.max() + 1 < 2:
        return np.nan
    ss_total = ((v - v.mean()) ** 2).sum()
    if ss_total == 0:
        return np.nan
    cnt = np.bincount(g)
    mu = np.bincount(g, weights=v) / cnt
    ss_between = (cnt * (mu - v.mean()) ** 2).sum()
    return float(np.sqrt(ss_between / ss_total))


def abs_pearson(va, vb):
    ok = ~np.isnan(va) & ~np.isnan(vb)
    a, b = va[ok], vb[ok]
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return np.nan
    return float(abs(np.corrcoef(a, b)[0, 1]))


def agg(prefix, vals):
    v = np.array([x for x in vals if not np.isnan(x)], dtype=float)
    if len(v) == 0:
        return {f"{prefix}_{s}": np.nan for s in ("mean", "std", "min", "max")}
    return {f"{prefix}_mean": v.mean(), f"{prefix}_std": v.std(),
            f"{prefix}_min": v.min(), f"{prefix}_max": v.max()}


def main():
    files = sorted(glob.glob(os.path.join(DATA, "*", "*.csv")))
    assert len(files) == 127, f"expected 127 files, got {len(files)}"

    wide_rows, detail_rows = [], []
    for path in files:
        dataset = os.path.basename(path)[:-4]
        df = pd.read_csv(path, dtype=str, skipinitialspace=True)
        df.columns = [c.strip() for c in df.columns]
        x_cols = [c for c in df.columns if col_role(c) == "x"]
        y_cols = [c for c in df.columns if col_role(c) == "y"]

        # convert every column exactly once
        arr = {c: (to_numeric(df, c) if is_numeric_name(c) else to_codes(df, c))
               for c in x_cols + y_cols}

        x_vars, y_vars, corrs = [], [], []
        for c in x_cols:
            v = (variation01_numeric if is_numeric_name(c)
                 else variation01_symbolic)(arr[c])
            x_vars.append(v)
            detail_rows.append((dataset, "x_variation", c, v))
        for c in y_cols:
            v = (variation01_numeric if is_numeric_name(c)
                 else variation01_symbolic)(arr[c])
            y_vars.append(v)
            detail_rows.append((dataset, "y_variation", c, v))

        # numeric-numeric pairs in one shot (pairwise-complete Pearson)
        num_x = [c for c in x_cols if is_numeric_name(c)]
        pearson = (pd.DataFrame({c: arr[c] for c in num_x}).corr(min_periods=3)
                   if len(num_x) >= 2 else None)

        for i, a in enumerate(x_cols):
            for b in x_cols[i + 1:]:
                na, nb = is_numeric_name(a), is_numeric_name(b)
                if na and nb:
                    r = abs(pearson.loc[a, b])
                    r = float(r) if not np.isnan(r) else np.nan
                    # a pairwise-constant column yields NaN from pandas already
                elif not na and not nb:
                    r = cramers_v(arr[a], arr[b])
                else:
                    num, sym = (a, b) if na else (b, a)
                    r = corr_ratio(arr[num], arr[sym])
                corrs.append(r)
                detail_rows.append((dataset, "correlation", f"{a}|{b}", r))

        row = {"dataset": dataset,
               "domain": os.path.basename(os.path.dirname(path)),
               "n_rows": len(df), "n_x": len(x_cols), "n_y": len(y_cols),
               "n_pairs": len(corrs)}
        row.update(agg("x_var", x_vars))
        row.update(agg("corr", corrs))
        row.update(agg("y_var", y_vars))
        wide_rows.append(row)
        print(f"{dataset}: x={len(x_cols)} y={len(y_cols)} pairs={len(corrs)} "
              f"x_var_mean={row['x_var_mean']:.3f} y_var_mean={row['y_var_mean']:.3f}",
              flush=True)

    out = pd.DataFrame(wide_rows)

    # performance (+ EZR's own stability score) from results/9, trt == "ezr"
    perf = []
    for dataset in out["dataset"]:
        t = pd.read_csv(os.path.join(REPO, f"results/9/{dataset}.csv"),
                        skipinitialspace=True)
        t.columns = [c.strip() for c in t.columns]
        r = t[t["trt"].str.strip() == "ezr"].iloc[0]
        perf.append({"dataset": dataset,
                     "performance_rmse": float(r["performance_error"]),
                     "ezr_stability_agreement": float(r["stability_agreement"])})
    out = out.merge(pd.DataFrame(perf), on="dataset", validate="1:1")

    # structural stability from jaccard.csv (20 seeds, pairwise Jaccard)
    jac = pd.read_csv(os.path.join(REPO, "jaccard.csv")).rename(columns={
        "mean_jaccard": "stability_jaccard_mean",
        "min_jaccard": "stability_jaccard_min",
        "max_jaccard": "stability_jaccard_max"})
    out = out.merge(jac[["dataset", "stability_jaccard_mean",
                         "stability_jaccard_min", "stability_jaccard_max"]],
                    on="dataset", validate="1:1")

    out_path = os.path.join(REPO, "analysis/meta_features.csv")
    out.to_csv(out_path, index=False)
    detail = pd.DataFrame(detail_rows, columns=["dataset", "kind", "name", "value"])
    detail_path = os.path.join(REPO, "analysis/meta_features_detail.csv")
    detail.to_csv(detail_path, index=False)

    print(f"\nwrote {len(out)} datasets x {len(out.columns)} columns to {out_path}")
    print(f"wrote {len(detail)} rows to {detail_path}")
    nan_summary = out.isna().sum()
    nan_summary = nan_summary[nan_summary > 0]
    print("\nNaN counts in wide file:")
    print(nan_summary.to_string() if len(nan_summary) else "  (none)")


if __name__ == "__main__":
    main()
