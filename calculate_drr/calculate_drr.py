#!/usr/bin/env python3
"""Compute DRR for the non-SE datasets listed in data_info.NONSE_PATHS.

Wraps a vendored copy of Lustosa's `drr` package without modifying it:
DataProcessor does the preprocessing (drops goal columns ending in + - !,
label-encodes categoricals, samples large frames) and IntrinsicDimensionEstimator
returns (R, I, DRR) with DRR = 1 - I/R.

Two vendored copies exist under external/ and --repo picks between them. They
disagree about how I is estimated, so their DRRs are NOT comparable row-by-row:

  upstream  external/drr_upstream  -- the original release. I comes from
            _select_intrinsic_dimension: a heuristic over the C(r) gradients
            that branches on R (<=6, 7-15, >15) and falls back to a fixed
            fraction of R when the gradients look implausible. I is an integer.

  modified  external/drr_modified  -- upstream plus exactly two bug fixes:
            C(r)=0 points are dropped from the log-log gradients (no 1e-15
            epsilon, so the plateau-jump artifact that inflated I past R is
            gone) and the R-banded acceptance windows are widened to the pure
            sanity bound 0 < I <= R. Statistics, banding, and fallback
            constants are otherwise upstream's; I is still an integer.

fit_r2/fit_n_radii are blank for both repos (neither exposes a fit).

Usage (needs the `drr` conda env, which has pandas/numpy/scipy):

    conda run -n drr python calculate_drr/calculate_drr.py --repo modified
    conda run -n drr python calculate_drr/calculate_drr.py --repo upstream
    conda run -n drr python calculate_drr/calculate_drr.py --metric l2 --out /tmp/x.csv

Paths in NONSE_PATHS are relative to the repo root; this script resolves them
from its own location, so it can be run from any working directory.
"""

import argparse
import csv
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
DEFAULT_REPO = "modified"
REPO_SRCS = {
    "modified": os.path.join(REPO_ROOT, "external", "drr_modified", "src"),
    "upstream": os.path.join(REPO_ROOT, "external", "drr_upstream", "src"),
}

sys.path.insert(0, HERE)  # data_info.py

from data_info import NONSE_PATHS, PREPROCESSING  # noqa: E402


def load_drr(repo):
    """Import the chosen vendored copy. Both are packages named `drr`, so the
    selected src/ must go on sys.path before the first import in this process."""
    src = REPO_SRCS[repo]
    if not os.path.isdir(src):
        sys.exit(f"no drr package at {src} (--repo {repo})")
    sys.path.insert(0, src)
    try:
        import drr as drr_pkg
    except ModuleNotFoundError as e:
        sys.exit(
            f"cannot import the vendored drr package ({e}).\n"
            f"expected it at {src}; run under the 'drr' conda env, e.g.\n"
            f"  conda run -n drr python {os.path.relpath(__file__, os.getcwd())}"
        )
    loaded = os.path.dirname(os.path.abspath(drr_pkg.__file__))
    if os.path.abspath(src) not in loaded:
        sys.exit(f"imported the wrong drr: wanted {src}, got {loaded}")
    return drr_pkg.DataProcessor, drr_pkg.IntrinsicDimensionEstimator, drr_pkg.__version__


FIELDS = [
    "dataset",
    "repo",
    "path",
    "rows_raw",
    "cols_raw",
    "goal_cols",
    "R",
    "I",
    "DRR",
    "fit_r2",
    "fit_n_radii",
    "row_sampling_applied",
    "preprocessing_note",
    "status",
]


def dataset_name(path):
    return os.path.splitext(os.path.basename(path))[0]


def preprocessing_note(name):
    """PREPROCESSING in data_info.py is keyed by prose names ('power consumption')."""
    key = name.lower().replace("_", " ")
    return PREPROCESSING.get(key, "")


def compute_one(classes, repo, path, max_rows, max_samples, metric, seed):
    """Return a result dict for one dataset. Raises on failure; caller logs."""
    DataProcessor, IntrinsicDimensionEstimator = classes
    processor = DataProcessor(max_rows_for_processing=max_rows, random_seed=seed)
    estimator = IntrinsicDimensionEstimator(max_samples=max_samples, distance_metric=metric)

    data, meta = processor.process_dataset(path)
    if not processor.validate_processed_data(data):
        raise ValueError("DataProcessor.validate_processed_data failed")

    R, I, drr_value = estimator.estimate(data)

    # only the modified repo records the log-log fit diagnostics
    r2 = getattr(estimator, "last_r2", None)
    n_radii = getattr(estimator, "last_n_kept", None)

    name = dataset_name(path)
    rows_raw, cols_raw = meta["original_shape"]
    return {
        "dataset": name,
        "repo": repo,
        "path": os.path.relpath(path, REPO_ROOT),
        "rows_raw": rows_raw,
        "cols_raw": cols_raw,
        "goal_cols": ";".join(str(c) for c in meta["goal_variables_removed"]),
        "R": R,
        "I": round(float(I), 4),
        "DRR": round(float(drr_value), 4),
        "fit_r2": "" if r2 is None else round(float(r2), 4),
        "fit_n_radii": "" if n_radii is None else n_radii,
        "row_sampling_applied": int(meta["sampling_applied"]),
        "preprocessing_note": preprocessing_note(name),
        "status": "ok",
    }


def blank_row(repo, path, status):
    """Skip-and-log row: keep the dataset in the output with empty measurements."""
    row = {f: "" for f in FIELDS}
    row["dataset"] = dataset_name(path)
    row["repo"] = repo
    row["path"] = path
    row["status"] = status
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--repo",
        default=DEFAULT_REPO,
        choices=sorted(REPO_SRCS),
        help=f"which vendored drr copy to run (default: {DEFAULT_REPO})",
    )
    ap.add_argument("--out", default=None, help="output csv (default: results/nonse_drr_<repo>.csv)")
    ap.add_argument("--errors", default=os.path.join(HERE, "results", "errors.log"), help="error log")
    ap.add_argument("--max-rows", type=int, default=5000, help="DataProcessor row cap before sampling")
    ap.add_argument("--max-samples", type=int, default=2000, help="estimator sample cap for pdist")
    ap.add_argument(
        "--metric",
        default="l1",
        choices=["l1", "l2", "euclidean", "manhattan", "cosine"],
        help="distance metric (drr default is l1)",
    )
    ap.add_argument("--seed", type=int, default=42, help="DataProcessor sampling seed")
    ap.add_argument("--quiet", action="store_true", help="silence the drr package's INFO logging")
    args = ap.parse_args()

    out = args.out or os.path.join(HERE, "results", f"nonse_drr_{args.repo}.csv")

    import logging

    logging.basicConfig(
        level=logging.ERROR if args.quiet else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    DataProcessor, IntrinsicDimensionEstimator, drr_version = load_drr(args.repo)
    classes = (DataProcessor, IntrinsicDimensionEstimator)
    print(f"repo={args.repo} ({os.path.relpath(REPO_SRCS[args.repo], REPO_ROOT)}, drr {drr_version})")

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.errors)), exist_ok=True)

    rows, errors = [], []
    for rel in NONSE_PATHS:
        path = os.path.join(REPO_ROOT, rel)
        if not os.path.exists(path):
            print(f"  MISSING {rel}")
            rows.append(blank_row(args.repo, rel, "missing_file"))
            errors.append(f"[{args.repo}] {rel}: file not found")
            continue
        try:
            row = compute_one(classes, args.repo, path, args.max_rows, args.max_samples, args.metric, args.seed)
            rows.append(row)
            r2 = row["fit_r2"] if row["fit_r2"] != "" else "n/a"
            print(f"  {row['dataset']:<20} R={row['R']:<5} I={row['I']:<8} DRR={row['DRR']:<8} r2={r2}")
        except Exception as e:  # skip-and-log, never crash the sweep
            print(f"  FAILED  {rel}: {e}")
            rows.append(blank_row(args.repo, rel, f"error: {type(e).__name__}"))
            errors.append(f"[{args.repo}] {rel}: {type(e).__name__}: {e}\n{traceback.format_exc()}")

    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    if errors:
        with open(args.errors, "a") as f:
            for line in errors:
                f.write(line + "\n")

    ok = sum(1 for r in rows if r["status"] == "ok")
    print(f"\n{ok}/{len(rows)} datasets processed -> {os.path.relpath(out, os.getcwd())}")
    print(
        f"settings: repo={args.repo} metric={args.metric} max_rows={args.max_rows} "
        f"max_samples={args.max_samples} seed={args.seed}"
    )
    if errors:
        print(f"{len(errors)} problem(s) logged to {os.path.relpath(args.errors, os.getcwd())}")


if __name__ == "__main__":
    main()
