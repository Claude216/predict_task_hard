#!/usr/bin/env python3
"""Estimate intrinsic dimensionality for every dataset in the DRR paper's Fig. 3
and chart it against the paper's own reported values.

The open question is which radius range the estimator should use in
intrinsic_dimension_estimator.py: the full min..max of the pairwise distances,
or their 25th-75th percentile. That is a source-level switch in the drr
package, so this script does not select it -- `--label` just records which one
is currently live and keeps the two runs' outputs apart:

    --label full     -> full_id,    results/full_vs_paper_id.{csv,png}
    --label partial  -> partial_id, results/partial_vs_paper_id.{csv,png}

Everything else is held fixed between runs, which is what makes them
comparable. `--label` is also the ONLY thing that distinguishes them in the
logs: both runs come from the same git commit, differing only in uncommitted
working-tree edits.

`paper_id` is read off Fig. 3 and lives in drr_cal/paper_ground_truth.csv.
Closer to `paper_id` wins.

The `drr` package is imported plainly, so it resolves to whatever the active
environment has installed -- currently an editable install of
dimensionality_reduction_ratio. The resolved path, version and git state are
printed at startup, because results depend on the working-tree state of that
checkout.

Usage:

    conda run -n drr python drr_cal/run_full_vs_paper.py --label partial
"""

import argparse
import csv
import os
import subprocess
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)

# Fig. 3's datasets, resolved to files under external/. SE names come from
# drr_cal/datasets.txt; a few differ in spelling from the labels in the figure
# (rs-6d-c3_obj2 vs rs-6d-c3-obj2, Scrum1k vs SCRUM), so the mapping is
# explicit rather than derived from the name.
#
# heart disease and diabetes are in the figure but deliberately not processed.
# Health-Hard (hpo/Health-ClosedIssues0000.csv) is skipped: it has no paper_id.
SE_ROOT = "external/moot/optimize"
DATASETS = [
    # (label in Fig. 3, path relative to REPO_ROOT, is_se)
    ("SS-B", f"{SE_ROOT}/config/SS-B.csv", True),
    ("SS-D", f"{SE_ROOT}/config/SS-D.csv", True),
    ("SS-M", f"{SE_ROOT}/config/SS-M.csv", True),
    ("SS-T", f"{SE_ROOT}/config/SS-T.csv", True),
    ("SS-U", f"{SE_ROOT}/config/SS-U.csv", True),
    ("rs-6d-c3-obj2", f"{SE_ROOT}/config/rs-6d-c3_obj2.csv", True),
    # blue in Fig. 3 and shipped under optimize/, so SE here even though
    # calculate_drr/data_info.py grouped it with the non-SE sets
    ("Wine Quality", f"{SE_ROOT}/misc/Wine_quality.csv", True),
    ("nasa93dem", f"{SE_ROOT}/process/nasa93dem.csv", True),
    ("Pom3a", f"{SE_ROOT}/process/pom3a.csv", True),
    ("pom3d", f"{SE_ROOT}/process/pom3d.csv", True),
    ("Xomo Flight", f"{SE_ROOT}/process/xomo_flight.csv", True),
    ("Xomo Ground", f"{SE_ROOT}/process/xomo_ground.csv", True),
    ("Xomo OSP", f"{SE_ROOT}/process/xomo_osp.csv", True),
    ("Xomo OSP2", f"{SE_ROOT}/process/xomo_osp2.csv", True),
    ("SCRUM", f"{SE_ROOT}/binary_config/Scrum1k.csv", True),
    ("FFM-250", f"{SE_ROOT}/binary_config/FFM-250-50-0.50-SAT-1.csv", True),
    ("Health-Easy", f"{SE_ROOT}/hpo/Health-Commits0000.csv", True),
    ("iris", "external/moot/classify/iris.csv", False),
    ("german credit", "external/moot/fairness/german.csv", False),
    ("adult", "external/moot/fairness/adult.csv", False),
    ("bank marketing", "external/moot/fairness/bank.csv", False),
    ("gamma telescope", "external/uci/gamma_telescope.csv", False),
    ("default", "external/uci/default.csv", False),
    ("power consumption", "external/uci/power_consumption.csv", False),
]

GROUND_TRUTH = os.path.join(HERE, "paper_ground_truth.csv")

# --label names the run: it picks the estimate column ("<label>_id") and the
# output basenames, so switching the estimator's radius range and re-running
# does not overwrite the previous run's results.
LABELS = {
    "full": "full min..max radius range",
    "partial": "25th-75th percentile radius range",
    "20": "20th-80th percentile radius range",
    "30": "30th-70th percentile radius range",
    "40": "40th-60th percentile radius range",
    "35": "35th-65th percentile radius range",
    # not a radius range: same full min..max as `full`, but L2 distances
    "l2": "full min..max radius range, L2 distance",
    "nox2": "full min..max radius range, L1, no *2 in the behavior branch",
    "30nox2": "30th-70th percentile radius range, L1, no *2 in the behavior branch",
    "l230nox2": "30th-70th percentile radius range, L2, no *2 in the behavior branch",
    "med": "30-70 range, L1, no *2, reworked medium branch + tightened branch guards",
    "medl2": "30-70 range, L2, no *2, reworked medium branch + tightened branch guards",
}


def load_paper_ids():
    with open(GROUND_TRUTH, newline="") as f:
        return {r["dataset"]: int(r["paper_id"]) for r in csv.DictReader(f)}


def drr_provenance(module):
    """Where did `drr` come from, and is that checkout dirty? Results are not
    reproducible without this."""
    pkg = os.path.dirname(os.path.abspath(module.__file__))
    repo = os.path.dirname(os.path.dirname(pkg))  # src/drr -> repo root

    def git(*args):
        try:
            return subprocess.run(
                ["git", "-C", repo, *args], capture_output=True, text=True, check=True
            ).stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return "?"

    sha = git("rev-parse", "--short", "HEAD")
    dirty = " (dirty)" if git("status", "--porcelain") else ""
    return f"{pkg}\n  version {module.__version__}, git {sha}{dirty}"


def estimate(classes, path, max_rows, max_samples, metric, seed):
    """Return (R, I). Raises on failure; the caller logs and moves on."""
    DataProcessor, IntrinsicDimensionEstimator = classes
    processor = DataProcessor(max_rows_for_processing=max_rows, random_seed=seed)
    estimator = IntrinsicDimensionEstimator(max_samples=max_samples, distance_metric=metric)

    data, _meta = processor.process_dataset(path)
    if not processor.validate_processed_data(data):
        raise ValueError("DataProcessor.validate_processed_data failed")

    R, I, _drr = estimator.estimate(data)
    return R, I


def chart_order(rows, paper_ids, se_flags):
    """Fig. 3 reads left to right by intrinsic dimensionality, so order by
    paper_id. Ties are broken SE before non-SE, then case-insensitively by
    name, which keeps the two colour families clustered."""
    return sorted(
        rows,
        key=lambda r: (paper_ids[r["dataset"]], not se_flags[r["dataset"]], r["dataset"].lower()),
    )


def plot(rows, se_flags, out_png, label):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    id_col = f"{label}_id"
    names = [r["dataset"] for r in rows]
    xs = range(len(names))
    # Both series index off the same x, so a dataset's dot and cross always
    # share a vertical line.
    blue, red = "#3b6ef5", "#e8392f"
    colors = [blue if se_flags[n] else red for n in names]

    green = "#1a9e4b"
    fig, ax = plt.subplots(figsize=(13, 6))
    for x, r, c in zip(xs, rows, colors):
        # R is the raw dimensionality, the ceiling the estimate sits under
        if r["R"] != "":
            ax.scatter(x, r["R"], color=green, marker="^", s=28, zorder=2)
        ax.scatter(x, r["paper_id"], color=c, marker="o", s=55, zorder=3)
        if r[id_col] != "":
            ax.scatter(x, r[id_col], color=c, marker="x", s=70, linewidths=2, zorder=3)

    # R reaches 256 while the ids stay under 20, so a linear axis would flatten
    # every id into a band along the bottom.
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_yticks([1, 2, 3, 5, 10, 20, 50, 100, 250])

    ax.set_xticks(list(xs))
    ax.set_xticklabels(names, rotation=45, ha="right")
    for x, c in zip(xs, colors):
        ax.get_xticklabels()[x].set_color(c)
    ax.set_xlabel("Dataset (ordered as in Fig. 3)")
    ax.set_ylabel("Dimensionality (log)")
    ax.set_title(f"Paper (dot) vs {LABELS[label]} (cross)")
    ax.grid(axis="y", alpha=0.3)
    ax.margins(x=0.02)

    handles = [
        plt.Line2D([], [], color="k", marker="o", ls="", label="paper_id"),
        plt.Line2D([], [], color="k", marker="x", ls="", label=id_col),
        plt.Line2D([], [], color=green, marker="^", ls="", ms=5, label="R (raw dims)"),
        plt.Line2D([], [], color=blue, marker="s", ls="", label="SE"),
        plt.Line2D([], [], color=red, marker="s", ls="", label="non-SE"),
    ]
    ax.legend(handles=handles, loc="upper left", frameon=False)

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"\nwrote {os.path.relpath(out_png, REPO_ROOT)}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--label",
        default="full",
        choices=sorted(LABELS),
        help="which radius range the installed estimator is currently using; names the "
        "estimate column and the output files (default: full)",
    )
    ap.add_argument("--out", default=None, help="output csv (default: results/<label>_vs_paper_id.csv)")
    ap.add_argument("--png", default=None, help="output chart (default: results/<label>_vs_paper_id.png)")
    ap.add_argument("--max-rows", type=int, default=5000, help="DataProcessor row cap before sampling")
    ap.add_argument("--max-samples", type=int, default=2000, help="estimator sample cap for pdist")
    ap.add_argument(
        "--metric",
        default="l1",
        choices=["l1", "l2", "euclidean", "manhattan", "cosine"],
        help="distance metric (drr default is l1)",
    )
    ap.add_argument("--seed", type=int, default=42, help="DataProcessor sampling seed")
    args = ap.parse_args()

    id_col = f"{args.label}_id"
    fields = ["dataset", id_col, "paper_id", "R"]
    out = args.out or os.path.join(HERE, "results", f"{args.label}_vs_paper_id.csv")
    png = args.png or os.path.join(HERE, "results", f"{args.label}_vs_paper_id.png")

    import logging

    logging.basicConfig(level=logging.ERROR, format="%(levelname)s %(name)s: %(message)s")

    try:
        import drr
    except ModuleNotFoundError:
        sys.exit("no `drr` package on the path -- run under the 'drr' conda env")

    # The git SHA does NOT distinguish a full-range run from a percentile run --
    # both come from the same commit with different working-tree edits. The label
    # is the only discriminator, so it goes in the log next to the provenance.
    print(f"drr from {drr_provenance(drr)}")
    print(f"  label={args.label} ({LABELS[args.label]})")
    print(f"  seed={args.seed} metric={args.metric} max_rows={args.max_rows} max_samples={args.max_samples}\n")

    classes = (drr.DataProcessor, drr.IntrinsicDimensionEstimator)
    paper_ids = load_paper_ids()
    se_flags = {name: is_se for name, _path, is_se in DATASETS}

    rows, failed = [], []
    for name, rel, _is_se in DATASETS:
        path = os.path.join(REPO_ROOT, rel)
        row = {"dataset": name, id_col: "", "paper_id": paper_ids[name], "R": ""}
        try:
            R, est = estimate(classes, path, args.max_rows, args.max_samples, args.metric, args.seed)
            row["R"], row[id_col] = R, round(float(est), 4)
            print(f"  {name:<18} R={row['R']:<5} {id_col}={row[id_col]:<8} paper_id={row['paper_id']}")
        except Exception as e:  # skip-and-log: one bad dataset must not stop the run
            failed.append((name, rel, e))
            print(f"  {name:<18} FAILED: {e}")
        rows.append(row)

    rows = chart_order(rows, paper_ids, se_flags)

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {os.path.relpath(out, REPO_ROOT)} ({len(rows)} datasets)")

    plot(rows, se_flags, png, args.label)

    if failed:
        print(f"\n{len(failed)} dataset(s) had no {id_col}:", file=sys.stderr)
        for name, rel, e in failed:
            print(f"\n--- {name} ({rel}) ---", file=sys.stderr)
            traceback.print_exception(type(e), e, e.__traceback__)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
