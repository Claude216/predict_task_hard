#!/usr/bin/env python3
"""Run the current DRR implementation over every MOOT optimize dataset.

Walks data/moot/optimize/**/*.csv (127 tasks) and, for each, records:

    dataset  file stem (unique across the tree -- checked, no collisions)
    group    the subdirectory it lives in (config, process, hpo, ...)
    R        raw dimensionality: feature columns left after the DataProcessor
             drops goal variables (headers ending +, -, !)
    I        estimated intrinsic dimensionality
    DRR      1 - I/R, straight from the estimator (not recomputed here)

"Current" means whatever `drr` the active environment resolves to, including
uncommitted working-tree edits -- the estimator's radius range and branch logic
are source-level switches, so the checkout's git state is part of the result.
The resolved path, version and dirty flag are printed at startup for exactly
that reason. Same provenance discipline as run_full_vs_paper.py.

Settings are held at run_full_vs_paper.py's defaults (seed 42, l1, max_rows
5000, max_samples 2000) so numbers here line up with the Fig. 3 comparison
runs. Note the DataProcessor only drops +/-/! columns; MOOT's "ignore" columns
(headers ending X) stay in and count toward R. That is the implementation as it
stands, and this script does not second-guess it.

Skip-and-log: a dataset that raises gets blank R/I/DRR and a traceback at the
end, and the run continues.

Usage:

    conda run -n drr python drr_cal/run_all_moot.py
"""

import argparse
import csv
import os
import subprocess
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)

MOOT_ROOT = os.path.join(REPO_ROOT, "data", "moot", "optimize")


def find_datasets(root):
    """(dataset, group, path) for every csv under root, sorted by group then name."""
    found = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in sorted(filenames):
            if not fn.endswith(".csv"):
                continue
            path = os.path.join(dirpath, fn)
            group = os.path.relpath(dirpath, root)
            found.append((fn[: -len(".csv")], "" if group == "." else group, path))
    return sorted(found, key=lambda t: (t[1], t[0].lower()))


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
    """Return (R, I, DRR). Raises on failure; the caller logs and moves on."""
    DataProcessor, IntrinsicDimensionEstimator = classes
    processor = DataProcessor(max_rows_for_processing=max_rows, random_seed=seed)
    estimator = IntrinsicDimensionEstimator(max_samples=max_samples, distance_metric=metric)

    data, _meta = processor.process_dataset(path)
    if not processor.validate_processed_data(data):
        raise ValueError("DataProcessor.validate_processed_data failed")

    return estimator.estimate(data)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", default=MOOT_ROOT, help="directory to walk for csvs")
    ap.add_argument("--out", default=os.path.join(HERE, "re_DRR.csv"), help="output csv")
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

    import logging

    logging.basicConfig(level=logging.ERROR, format="%(levelname)s %(name)s: %(message)s")

    try:
        import drr
    except ModuleNotFoundError:
        sys.exit("no `drr` package on the path -- run under the 'drr' conda env")

    print(f"drr from {drr_provenance(drr)}")
    print(f"  seed={args.seed} metric={args.metric} max_rows={args.max_rows} max_samples={args.max_samples}")

    datasets = find_datasets(args.root)
    print(f"  {len(datasets)} datasets under {os.path.relpath(args.root, REPO_ROOT)}\n")

    classes = (drr.DataProcessor, drr.IntrinsicDimensionEstimator)
    rows, failed = [], []
    for name, group, path in datasets:
        row = {"dataset": name, "group": group, "R": "", "I": "", "DRR": ""}
        try:
            R, I, drr_val = estimate(classes, path, args.max_rows, args.max_samples, args.metric, args.seed)
            row["R"], row["I"], row["DRR"] = R, I, round(float(drr_val), 4)
            print(f"  {group + '/' + name:<45} R={row['R']:<5} I={row['I']:<5} DRR={row['DRR']}")
        except Exception as e:  # skip-and-log: one bad dataset must not stop the run
            failed.append((name, path, e))
            print(f"  {group + '/' + name:<45} FAILED: {e}")
        rows.append(row)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "group", "R", "I", "DRR"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {os.path.relpath(args.out, REPO_ROOT)} ({len(rows)} datasets)")

    if failed:
        print(f"\n{len(failed)} dataset(s) had no estimate:", file=sys.stderr)
        for name, path, e in failed:
            print(f"\n--- {name} ({os.path.relpath(path, REPO_ROOT)}) ---", file=sys.stderr)
            traceback.print_exception(type(e), e, e.__traceback__)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
