#!/usr/bin/env python3
"""Roll every run_full_vs_paper.py result up into one comparison table.

Each row is one estimator configuration, scored against the paper's Fig. 3
values in drr_cal/paper_ground_truth.csv. Rows are sorted best-first by total
absolute error, so the table reads directly as "which configs improve DRR".

CAVEAT recorded in the csv itself: every run is a SINGLE run at seed=42. No
config has been tested against sampling variation, so differences of 1-2 in
sum_abs_err are not established as real. Treat the ordering as a ranking of
point estimates, not a significance test.

    conda run -n drr python drr_cal/summarize_configs.py
"""

import csv
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

# (config id, result csv, estimate column, metric, radius range, *2 present,
#  branch logic). Order here is the order the runs were made.
RUNS = [
    ("baseline",   "full_vs_paper_id.csv",    "full_id",     "l1", "full min..max", "yes", "as-found"),
    ("pct_25_75",  "partial_vs_paper_id.csv", "partial_id",  "l1", "25-75 pct",     "yes", "as-found"),
    ("pct_20_80",  "20vspaper.csv",           "20_id",       "l1", "20-80 pct",     "yes", "as-found"),
    ("pct_30_70",  "30vspaper.csv",           "30_id",       "l1", "30-70 pct",     "yes", "as-found"),
    ("pct_35_65",  "35vspaper.csv",           "35_id",       "l1", "35-65 pct",     "yes", "as-found"),
    ("pct_40_60",  "40vspaper.csv",           "40_id",       "l1", "40-60 pct",     "yes", "as-found"),
    ("l2_full",    "l2vspaper.csv",           "l2_id",       "l2", "full min..max", "yes", "as-found"),
    ("nox2_full",  "nox2vspaper.csv",         "nox2_id",     "l1", "full min..max", "no",  "as-found"),
    ("nox2_30_70", "30nox2vspaper.csv",       "30nox2_id",   "l1", "30-70 pct",     "no",  "as-found"),
    ("nox2_30_70_l2", "l230nox2vspaper.csv",  "l230nox2_id", "l2", "30-70 pct",     "no",  "as-found"),
    ("med_30_70",  "medvspaper.csv",          "med_id",      "l1", "30-70 pct",     "no",  "medium reworked + guards tightened"),
    ("med_30_70_l2", "medl2vspaper.csv",      "medl2_id",    "l2", "30-70 pct",     "no",  "medium reworked + guards tightened"),
]

BASELINE = "baseline"

SE = {
    "SS-B", "SS-D", "SS-M", "SS-T", "SS-U", "rs-6d-c3-obj2", "Wine Quality",
    "nasa93dem", "Pom3a", "pom3d", "Xomo Flight", "Xomo Ground", "Xomo OSP",
    "Xomo OSP2", "SCRUM", "FFM-250", "Health-Easy",
}

FIELDS = [
    "config", "run_order", "metric", "radius_range", "x2_in_behavior_branch",
    "branch_logic", "sum_abs_err", "mean_abs_err", "median_abs_err", "rmse",
    "max_abs_err", "within_1", "within_2", "exact", "se_mean_abs_err",
    "nonse_mean_abs_err", "mean_signed_err", "better_than_baseline",
    "worse_than_baseline", "same_as_baseline", "source_csv",
]


def load(fname, col):
    """dataset -> (estimate, paper_id). Skips rows whose estimate is blank."""
    with open(os.path.join(RESULTS, fname), newline="") as f:
        return {
            r["dataset"]: (float(r[col]), int(r["paper_id"]))
            for r in csv.DictReader(f)
            if r[col] != ""
        }


def score(data, baseline):
    err = {k: abs(v - g) for k, (v, g) in data.items()}
    e = list(err.values())
    n = len(e)
    se = [err[k] for k in err if k in SE]
    nonse = [err[k] for k in err if k not in SE]
    signed = [v - g for v, g in data.values()]

    better = worse = same = ""
    if baseline is not None:
        b = {k: abs(v - g) for k, (v, g) in baseline.items()}
        shared = set(err) & set(b)
        better = sum(1 for k in shared if err[k] < b[k])
        worse = sum(1 for k in shared if err[k] > b[k])
        same = sum(1 for k in shared if err[k] == b[k])

    return {
        "sum_abs_err": round(sum(e), 2),
        "mean_abs_err": round(sum(e) / n, 3),
        "median_abs_err": round(st.median(e), 3),
        "rmse": round((sum(x * x for x in e) / n) ** 0.5, 3),
        "max_abs_err": round(max(e), 2),
        "within_1": sum(1 for x in e if x <= 1),
        "within_2": sum(1 for x in e if x <= 2),
        "exact": sum(1 for x in e if x == 0),
        "se_mean_abs_err": round(sum(se) / len(se), 3) if se else "",
        "nonse_mean_abs_err": round(sum(nonse) / len(nonse), 3) if nonse else "",
        "mean_signed_err": round(sum(signed) / n, 3),
        "better_than_baseline": better,
        "worse_than_baseline": worse,
        "same_as_baseline": same,
    }


def main():
    loaded = {}
    missing = []
    for cfg, fname, col, *_ in RUNS:
        try:
            loaded[cfg] = load(fname, col)
        except (FileNotFoundError, KeyError) as e:
            missing.append(f"{cfg} ({fname}): {e}")

    if missing:
        print("skipped:")
        for m in missing:
            print(f"  {m}")

    base = loaded.get(BASELINE)
    rows = []
    for order, (cfg, fname, col, metric, rng, x2, logic) in enumerate(RUNS, 1):
        if cfg not in loaded:
            continue
        row = {
            "config": cfg,
            "run_order": order,
            "metric": metric,
            "radius_range": rng,
            "x2_in_behavior_branch": x2,
            "branch_logic": logic,
            "source_csv": fname,
        }
        row.update(score(loaded[cfg], None if cfg == BASELINE else base))
        rows.append(row)

    rows.sort(key=lambda r: r["sum_abs_err"])

    out = os.path.join(RESULTS, "config_comparison.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    print(f"\nwrote {os.path.relpath(out, os.path.dirname(HERE))} ({len(rows)} configs, best first)")
    print("all runs: seed=42, max_rows=5000, max_samples=2000, 24 datasets\n")
    hdr = f"{'config':<16}{'metric':>7}{'range':>16}{'*2':>4}{'sum':>7}{'mean':>7}{'RMSE':>7}{'SE':>7}{'nonSE':>7}"
    print(hdr)
    for r in rows:
        print(
            f"{r['config']:<16}{r['metric']:>7}{r['radius_range']:>16}"
            f"{r['x2_in_behavior_branch']:>4}{r['sum_abs_err']:>7.0f}"
            f"{r['mean_abs_err']:>7.2f}{r['rmse']:>7.2f}"
            f"{r['se_mean_abs_err']:>7.2f}{r['nonse_mean_abs_err']:>7.2f}"
        )


if __name__ == "__main__":
    main()
