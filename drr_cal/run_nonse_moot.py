#!/usr/bin/env python3
"""Run the current DRR implementation over the NON-SE datasets in MOOT.

Companion to run_all_moot.py, which covers the SE side (data/moot/optimize).
This one walks the general-ML trees that also ship inside external/moot:

    classify        73  classic UCI (glass, colic, hypothyroid, mushroom, letter, mnist_1)
    regression      61  classic UCI (abalone, housing, sensory, cal.housing, mv, fried)
    fairness         6  adult, bank, compas, german, law, communities (+ fairness/etc)
    optimize/misc    4  auto93, Car_price_cleaned, Wine_quality, single_wine_quality
                   ---
                   144

Everything else under optimize/ is treated as SE and left to run_all_moot.py;
text_mining/, re/ and old/ are out of scope.

Per dataset it records:

    dataset  file stem -- NOT unique here, unlike the optimize tree: auto93 and
             german each appear twice. The key is (group, dataset).
    group    directory relative to external/moot (classify, regression,
             fairness, fairness/etc, optimize/misc)
    R        raw dimensionality: feature columns left after the DataProcessor
             drops goal variables (headers ending +, -, !)
    I        estimated intrinsic dimensionality
    DRR      1 - I/R, straight from the estimator (not recomputed here)

"Current" means whatever `drr` the active environment resolves to, including
uncommitted working-tree edits -- the estimator's radius range and branch logic
are source-level switches, so the checkout's git state is part of the result.
The resolved path, version and dirty flag are printed at startup for exactly
that reason. Same provenance discipline as run_all_moot.py.

Settings match run_all_moot.py's defaults (seed 42, l1, max_rows 5000,
max_samples 2000) so the non-SE numbers line up with the SE sweep.

Known estimator behaviours this run does NOT work around -- kept as-is so the
numbers stay comparable to the SE sweep, but do not over-read the CSV:

  * Silent fallbacks. _select_intrinsic_dimension branches on R alone (R<=6
    config / R>15 behavior / else medium). When the middle-quartile log-gradient
    median lands outside [1, R), the estimator substitutes a constant 0.3R /
    0.5R / max(0.7R, R-5) and returns it indistinguishably from a real estimate.
    Nothing in the output marks those rows.
  * classify/breastcancer.csv's last header is "class! " WITH A TRAILING SPACE,
    so _remove_goal_variables does not drop it: the class label survives as a
    feature and R is one too high. Only file of the 144 with no droppable goal.
  * "X" (ignore) columns are never dropped and count toward R -- HpX in
    optimize/misc/auto93.csv, FnlwgtX in fairness/adult.csv.
  * "~" sensitive-attribute columns in fairness/ (Age~, race~, sex~) are not in
    the drop list either, and stay in as features.
  * Symbolic columns are label-encoded to arbitrary integers and then compared
    under L1, so nominal categories get fabricated ordinal spacing. That bites
    harder here than on SE config data: classify/ and fairness/ are far more
    symbolic-heavy.
  * "?" is not treated as missing for symbolic columns; it becomes its own
    encoded category (colic has 1927, hypothyroid 6064).

Skip-and-log: a dataset that raises gets blank R/I/DRR and a traceback at the
end, and the run continues.

Usage:

    conda run -n drr python drr_cal/run_nonse_moot.py
"""

import argparse
import csv
import os
import subprocess
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)

MOOT_ROOT = os.path.join(REPO_ROOT, "external", "moot")

# The non-SE trees, relative to MOOT_ROOT. Walked recursively (fairness has an
# etc/ subdir), and `group` is taken from where each csv actually lives.
NON_SE_TREES = ["classify", "regression", "fairness", os.path.join("optimize", "misc")]


def find_datasets(root, trees):
    """(dataset, group, path) for every csv under each tree, sorted by group then name.

    `group` is the csv's directory relative to `root`, so stems that collide
    across trees (auto93, german) stay distinguishable.
    """
    found = []
    for tree in trees:
        for dirpath, _dirnames, filenames in os.walk(os.path.join(root, tree)):
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
    ap.add_argument("--root", default=MOOT_ROOT, help="moot checkout to walk")
    ap.add_argument(
        "--trees",
        nargs="+",
        default=NON_SE_TREES,
        help="subtrees of --root to treat as non-SE",
    )
    ap.add_argument("--out", default=os.path.join(HERE, "re_Non_SE_DRR.csv"), help="output csv")
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

    datasets = find_datasets(args.root, args.trees)
    print(f"  {len(datasets)} datasets under {os.path.relpath(args.root, REPO_ROOT)}/{{{','.join(args.trees)}}}\n")

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
