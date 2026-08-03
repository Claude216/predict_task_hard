#!/usr/bin/env python3
"""Redraw the DRR paper's Figure 4 (DRR per dataset, SE vs non-SE) with each of
our estimators, holding every dataset at its OWN x position from the paper so
the plots can be read vertically against the original.

Datasets: the 18 SE + 9 non-SE points of the paper's Fig. 4 (see PAPER_FIG4,
which also carries the paper's own DRR per point, recovered from the PDF).

Backends:
  drr_mine     drr_mine/intrinsic_dim.py -- I_fit is the MAX slope of a curve
               fitted to ln C(r) vs ln r over the scaling region 0.01<=C<=0.2,
               sampled by inverse CDF. DRR = 1 - I_fit/R; fit_r2 is a trust
               diagnostic.
  drr_modified external/drr_modified with method="fit". Same family of estimate,
               but its own preprocessing (label-encodes categoricals, imputes,
               and does NOT know the MOOT "X"-suffix / "~" column conventions),
               so its R can differ from drr_mine's on the same csv.
  drr_upstream external/drr_upstream, the released package. I comes from an
               R-banded heuristic over the C(r) gradients that substitutes a
               constant fraction of R whenever its acceptance windows reject the
               measured value; the branch actually taken is logged per row in
               code_path, and its DRR can be negative. No r2 is available.

All backends: seed 42, and drr_upstream's own sampling budget -- rows capped
at 5000 by the data processor, then subsampled to 2000 for the distance matrix.

Outputs (one set per backend):
  results/<backend>/fig4_drr.csv
  results/<backend>/fig4_drr.png

    conda run -n drr python calculate_drr/run_fig4.py
    conda run -n drr python calculate_drr/run_fig4.py --backend drr_upstream
"""

from __future__ import annotations

import argparse
import csv
import io
import logging
import os
import sys
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

SEED = 42
# mirror drr_upstream's own defaults: its DataProcessor caps rows at 5000, then
# its estimator subsamples to max_samples=2000, so 2000 is what actually reaches
# the pairwise-distance computation in every backend.
MAX_ROWS = 5000
MAX_SAMPLES = 2000

MOOT = os.path.join(REPO_ROOT, "external", "drr_modified", "data", "optimize")

# ---------------------------------------------------------------------------
# The paper's Figure 4, recovered from the PDF rather than eyeballed:
#   * marker centres were located by colour in a 300-dpi render of page 4;
#   * the SE row has exactly 18 markers on an even 21.24pt grid (x 147.5-508.5)
#     and the non-SE row 9 markers on the same grid (x 147.5-317.4);
#   * the % tick labels calibrate y exactly (55.24pt per 25%), giving the
#     paper's own DRR for each point -- recorded here as paper_drr.
# Columns: (paper label, group, paper x in PDF points, paper DRR, our csv).
# csv=None means the paper plots a dataset we do not have locally.
#
# Two callers' decisions are baked in:
#   * Wine_quality is SE -- Fig. 4 colours it blue at slot 10, even though the
#     paper's own Table 1 lists it under Non-SE data.
#   * Health-Hard/Health-Easy have no file of that name (36 hpo/Health-* files
#     exist: ClosedIssues/ClosedPRs/Commits x 12 seeds), so seed-0000
#     representatives stand in. This mapping is an assumption, not a fact
#     recoverable from the paper, and is flagged in the note column.
# ---------------------------------------------------------------------------
PAPER_FIG4 = [
    ("FFM-250",        "SE", 147.51, 0.974, f"{MOOT}/binary_config/FFM-250-50-0.50-SAT-1.csv"),
    ("SCRUM",          "SE", 168.73, 0.948, f"{MOOT}/binary_config/Scrum1k.csv"),
    ("nasa93dem",      "SE", 189.97, 0.842, f"{MOOT}/process/nasa93dem.csv"),
    ("Xomo Flight",    "SE", 211.21, 0.743, f"{MOOT}/process/xomo_flight.csv"),
    ("Xomo Ground",    "SE", 232.45, 0.743, f"{MOOT}/process/xomo_ground.csv"),
    ("Xomo OSP",       "SE", 253.68, 0.743, f"{MOOT}/process/xomo_osp.csv"),
    ("Xomo OSP2",      "SE", 274.92, 0.632, f"{MOOT}/process/xomo_osp2.csv"),
    ("SS-U",           "SE", 296.15, 0.574, f"{MOOT}/config/SS-U.csv"),
    ("SS-M",           "SE", 317.37, 0.815, f"{MOOT}/config/SS-M.csv"),
    ("Wine Quality",   "SE", 338.63, 0.602, f"{MOOT}/misc/Wine_quality.csv"),
    ("pom3d",          "SE", 359.86, 0.447, f"{MOOT}/process/pom3d.csv"),
    ("Pom3a",          "SE", 381.10, 0.447, f"{MOOT}/process/pom3a.csv"),
    ("Health-Hard",    "SE", 402.32, 0.602, f"{MOOT}/hpo/Health-ClosedIssues0000.csv"),
    ("Health-Easy",    "SE", 423.57, 0.602, f"{MOOT}/hpo/Health-Commits0000.csv"),
    ("SS-D",           "SE", 444.79, 0.669, f"{MOOT}/config/SS-D.csv"),
    ("SS-B",           "SE", 466.04, 0.669, f"{MOOT}/config/SS-B.csv"),
    ("rs-6d-c3-obj2",  "SE", 487.26, 0.336, f"{MOOT}/config/rs-6d-c3_obj2.csv"),
    ("SS-T",           "SE", 508.53, 0.336, f"{MOOT}/config/SS-T.csv"),

    ("adult",             "Non-SE", 147.48, 0.288, f"{REPO_ROOT}/external/moot/fairness/adult.csv"),
    ("iris",              "Non-SE", 168.72, 0.752, f"{REPO_ROOT}/external/moot/classify/iris.csv"),
    ("default",           "Non-SE", 189.96, 0.176, f"{REPO_ROOT}/external/uci/default.csv"),
    ("diabetes",          "Non-SE", 211.20, 0.252, f"{REPO_ROOT}/external/moot/classify/diabetes.csv"),
    ("german credit",     "Non-SE", 232.44, 0.252, f"{REPO_ROOT}/external/moot/fairness/german.csv"),
    ("bank marketing",    "Non-SE", 253.68, 0.252, f"{REPO_ROOT}/external/moot/fairness/bank.csv"),
    ("heart disease",     "Non-SE", 274.92, 0.233, f"{REPO_ROOT}/external/moot/classify/heart.c.csv"),
    ("gamma telescope",   "Non-SE", 296.18, 0.202, f"{REPO_ROOT}/external/uci/gamma_telescope.csv"),
    ("power consumption", "Non-SE", 317.40, 0.169, f"{REPO_ROOT}/external/uci/power_consumption.csv"),
]

NOTES = {
    "Wine Quality": "SE per the paper's Fig. 4 (blue); its Table 1 lists it as Non-SE",
    "Health-Hard": "paper gives no file name; stand-in = hpo/Health-ClosedIssues0000",
    "Health-Easy": "paper gives no file name; stand-in = hpo/Health-Commits0000",
}

FIELDS = ["dataset", "group", "paper_x", "paper_DRR", "path", "R", "I_fit", "DRR",
          "delta_vs_paper", "fit_r2", "code_path", "n_rows", "n_used", "seed",
          "max_rows", "note", "status"]


def build_tasks():
    """(paper label, group, abspath|None, paper_x, paper_drr, note)."""
    return [(name, grp, csv, x, pdrr, NOTES.get(name, ""))
            for name, grp, x, pdrr, csv in PAPER_FIG4]


# --------------------------------------------------------------- backends

MIN_DISTINCT_FOR_NUM = 10


def coerce_moot_headers(df, split_columns, is_num):
    """Some inputs here are not MOOT-native (external/uci/*.csv keeps UCI's own
    column names), so the MOOT rule "uppercase-initial = Num" mistypes their
    continuous columns as Sym -- e.g. gamma_telescope's fLength becomes 18,643
    distinct *symbols*, collapsing the distance spectrum to 5 values and leaving
    no scaling region at all.

    Fix the input, not the estimator: uppercase the initial letter of any
    feature column that is lowercase-initial, fully numeric, and has more than
    MIN_DISTINCT_FOR_NUM distinct values. The cardinality guard is what keeps
    genuinely symbolic numeric codes as Sym -- heart.c's sex (2 values), cp (4),
    thal (3); binary_config's 0/1 flags; nasa93dem's COCOMO ratings (~6).

    Returns (df, renamed_columns).
    """
    import pandas as pd

    feats, _ = split_columns(df.columns)
    rename = {}
    for c in feats:
        if is_num(c):
            continue
        vals = df[c].replace("?", np.nan).dropna()
        if vals.empty or vals.nunique() <= MIN_DISTINCT_FOR_NUM:
            continue
        if pd.to_numeric(vals, errors="coerce").notna().all():
            rename[c] = c[0].upper() + c[1:]
    return (df.rename(columns=rename), list(rename))


def run_drr_mine(path):
    sys.path.insert(0, os.path.join(REPO_ROOT, "drr_mine"))
    from intrinsic_dim import (estimate_intrinsic_dim, is_num, load_moot_csv,
                               split_columns)

    df, renamed = coerce_moot_headers(load_moot_csv(path), split_columns, is_num)
    r = estimate_intrinsic_dim(df, seed=SEED, max_rows=MAX_SAMPLES)
    return {"R": r["R"], "I_fit": r["I_fit"], "DRR": r["drr_fit"],
            "fit_r2": r["fit_r2"], "n_rows": r["n_rows"], "n_used": r["n_used"],
            "status": r["status"], "_renamed": renamed}


VENDORED = {"drr_modified": "drr_modified", "drr_upstream": "drr_upstream"}


def load_vendored(which):
    """Both vendored copies are packages literally named `drr`, so only one can
    be live in a process. Purge the cached one (and its src path) before
    importing the other, then assert we got the copy we asked for."""
    src = os.path.join(REPO_ROOT, "external", VENDORED[which], "src")
    if not os.path.isdir(src):
        raise RuntimeError(f"no drr package at {src}")
    for m in [m for m in sys.modules if m == "drr" or m.startswith("drr.")]:
        del sys.modules[m]
    other = os.path.join(REPO_ROOT, "external", VENDORED["drr_upstream" if
                         which == "drr_modified" else "drr_modified"], "src")
    sys.path[:] = [p for p in sys.path if os.path.abspath(p) != os.path.abspath(other)]
    sys.path.insert(0, src)
    import drr as drr_pkg

    loaded = os.path.dirname(os.path.abspath(drr_pkg.__file__))
    if os.path.abspath(src) not in loaded:
        raise RuntimeError(f"imported the wrong drr: wanted {src}, got {loaded}")
    return drr_pkg


def _run_vendored(which, path, **est_kwargs):
    drr_pkg = load_vendored(which)
    proc = drr_pkg.DataProcessor(max_rows_for_processing=MAX_ROWS, random_seed=SEED)
    est = drr_pkg.IntrinsicDimensionEstimator(max_samples=MAX_SAMPLES,
                                              distance_metric="l1", **est_kwargs)
    # capture the estimator's own DEBUG chatter so we can report which branch
    # produced the number -- upstream silently substitutes a constant fraction
    # of R whenever its acceptance windows reject the measured gradient
    buf = io.StringIO()
    lg = logging.getLogger("drr.intrinsic_dimension_estimator")
    handler = logging.StreamHandler(buf)
    handler.setLevel(logging.DEBUG)
    prev_level, prev_prop, prev_disable = lg.level, lg.propagate, logging.root.manager.disable
    logging.disable(logging.NOTSET)
    lg.addHandler(handler); lg.setLevel(logging.DEBUG); lg.propagate = False
    try:
        data, meta = proc.process_dataset(path)
        if not proc.validate_processed_data(data):
            raise ValueError("validate_processed_data failed")
        R, I, drr = est.estimate(data)
    finally:
        lg.removeHandler(handler)
        lg.setLevel(prev_level); lg.propagate = prev_prop
        logging.disable(prev_disable)

    log = buf.getvalue()
    code_path = ("FALLBACK" if "using fallback" in log else
                 "log-median" if "using log median" in log else
                 "log-max" if "using log max" in log else
                 "fit" if which == "drr_modified" else "other")
    return {"R": R, "I_fit": float(I), "DRR": float(drr),
            "fit_r2": getattr(est, "last_r2", float("nan")),
            "n_rows": meta["original_shape"][0],
            # the estimator subsamples again internally and never returns the
            # subsample, so report what actually reaches the distance matrix
            "n_used": min(data.shape[0], MAX_SAMPLES),
            "code_path": code_path, "status": "ok"}


def run_drr_modified(path):
    return _run_vendored("drr_modified", path, method="fit")


def run_drr_upstream(path):
    # upstream has no method= switch and no r2: I comes from its R-banded
    # heuristic over the C(r) gradients, with constant fallbacks
    return _run_vendored("drr_upstream", path)


BACKENDS = {"drr_mine": run_drr_mine, "drr_modified": run_drr_modified,
            "drr_upstream": run_drr_upstream}


# ------------------------------------------------------------------ plot

def plot_fig4(rows, out_png, backend):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    style = {"SE": ("tab:green", "SE"), "Non-SE": ("tab:red", "Non-SE")}
    fig, ax = plt.subplots(figsize=(14, 7.5))

    # x is the dataset's OWN x position in the paper's Fig. 4 (PDF points),
    # so this plot can be laid beside the paper's figure and read vertically.
    # SE and non-SE share the same x grid there, so labels are pinned above
    # (SE) and below (non-SE) to keep colliding pairs legible.
    drawn = [r for r in rows if r["DRR"] == r["DRR"]]

    def crowded(r):
        """True when a point of the OTHER group sits on top of this one. Both
        series share the paper's x grid, so wherever our DRRs happen to agree
        the two markers coincide and the default label offsets overlap."""
        return any(q["group"] != r["group"]
                   and abs(q["paper_x"] - r["paper_x"]) < 15
                   and abs(q["DRR"] - r["DRR"]) < 0.05
                   for q in drawn)

    for group, (colour, label) in style.items():
        pts = [r for r in drawn if r["group"] == group]
        if not pts:
            continue
        xs = [r["paper_x"] for r in pts]
        ys = [r["DRR"] for r in pts]
        ax.scatter(xs, ys, marker="x", s=150 if group == "SE" else 90,
                   linewidths=2.6 if group == "SE" else 2.0, color=colour,
                   label=label, zorder=3 if group == "SE" else 4)
        for x, y, r in zip(xs, ys, pts):
            up = group == "SE"
            off = (11 if up else -18) + (13 if up else -13) * crowded(r)
            ax.annotate(r["dataset"], (x, y), textcoords="offset points",
                        xytext=(0, off), ha="center", fontsize=7, color=colour)

    ax.axhline(1/3, ls="--", lw=1, color="0.45", zorder=1)
    ax.text(142, 1/3 + 0.015, "DRR = 1/3  (paper's Eq. 1 threshold)",
            ha="left", va="bottom", fontsize=8, color="0.35")

    lo = min([r["DRR"] for r in drawn] + [0.0])
    ax.set_ylim(min(-0.05, lo - 0.08), 1.05)
    ax.set_xlim(138, 522)
    ax.set_xticks([])
    ax.set_yticks(np.arange(0, 1.01, 0.25))
    ax.set_yticklabels([f"{v:.0%}" for v in np.arange(0, 1.01, 0.25)])
    ax.set_ylabel("DRR = 1 - I/R")
    ax.set_xlabel("datasets, held at their own x position in the paper's Fig. 4 "
                  "(so this plot can be read vertically against it)")
    est = "I heuristic + constant fallbacks" if backend == "drr_upstream" else "I_fit"
    ax.set_title(f"DRR per dataset - {backend} ({est}, seed {SEED}, <={MAX_SAMPLES} rows)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower left", frameon=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", default="all",
                    choices=["all", "drr_mine", "drr_modified", "drr_upstream"])
    ap.add_argument("--outdir", default=os.path.join(REPO_ROOT, "results"))
    args = ap.parse_args()

    logging.basicConfig(level=logging.ERROR, format="%(levelname)s %(name)s: %(message)s")
    logging.disable(logging.WARNING)

    tasks = build_tasks()
    backends = list(BACKENDS) if args.backend == "all" else [args.backend]

    for backend in backends:
        fn = BACKENDS[backend]
        outdir = os.path.join(args.outdir, backend)
        os.makedirs(outdir, exist_ok=True)
        rows, errors = [], []
        print(f"\n=== {backend} ===")
        print(f"{'dataset':<20} {'group':<7} {'R':>4} {'I_fit':>7} {'DRR':>7} "
              f"{'paper':>7} {'delta':>7} {'r2':>6}")
        for name, group, path, paper_x, paper_drr, note in tasks:
            base = {f: "" for f in FIELDS}
            base.update({"dataset": name, "group": group, "paper_x": paper_x,
                         "paper_DRR": paper_drr, "seed": SEED,
                         "max_rows": MAX_SAMPLES, "note": note,
                         "path": "" if path is None else os.path.relpath(path, REPO_ROOT)})
            if path is None or not os.path.exists(path):
                base["status"] = "not_available" if path is None else "missing_file"
                rows.append(base)
                if path is not None:
                    errors.append(f"[{backend}] {name}: file not found at {path}")
                print(f"{name:<20} {group:<7}   -- {base['status']}")
                continue
            try:
                base.update(fn(path))
                renamed = base.pop("_renamed", None)
                if renamed:
                    tag = f"headers coerced to Num: {','.join(renamed)}"
                    base["note"] = f"{base['note']}; {tag}" if base["note"] else tag
            except Exception as e:  # skip-and-log, never crash the sweep
                base["status"] = f"error: {type(e).__name__}: {e}"
                base["DRR"] = float("nan")
                errors.append(f"[{backend}] {name}: {traceback.format_exc()}")
            if isinstance(base.get("DRR"), float) and base["DRR"] == base["DRR"]:
                base["delta_vs_paper"] = base["DRR"] - paper_drr
            rows.append(base)
            d, i, r2 = base.get("DRR"), base.get("I_fit"), base.get("fit_r2")
            fmt = lambda v: f"{v:.3f}" if isinstance(v, float) and v == v else "nan"
            print(f"{name:<20} {group:<7} {str(base.get('R','')):>4} "
                  f"{fmt(i):>7} {fmt(d):>7} {paper_drr:>7.3f} "
                  f"{fmt(base.get('delta_vs_paper')):>7} {fmt(r2):>6}"
                  + ("" if base["status"] == "ok" else f"   <- {base['status']}"))

        csv_path = os.path.join(outdir, "fig4_drr.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow({k: (round(v, 4) if isinstance(v, float) else v)
                            for k, v in r.items()})

        png_path = os.path.join(outdir, "fig4_drr.png")
        plot_fig4([r for r in rows if isinstance(r.get("DRR"), float)], png_path, backend)

        if errors:
            with open(os.path.join(outdir, "errors.log"), "w") as f:
                f.write("\n".join(errors))
        ok = sum(1 for r in rows if r["status"] == "ok")
        print(f"{ok}/{len(rows)} ok -> {os.path.relpath(csv_path, os.getcwd())}, "
              f"{os.path.relpath(png_path, os.getcwd())}")


if __name__ == "__main__":
    main()
