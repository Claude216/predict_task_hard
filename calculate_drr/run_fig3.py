#!/usr/bin/env python3
"""Redraw the DRR paper's Figure 3 -- intrinsic dimension (x) against original
dimension (y) -- with each of our three estimators.

Datasets are exactly the ones behind our Fig. 4 redraw (run_fig4.PAPER_FIG4:
the paper's 18 SE + 9 non-SE points), so the two figures describe the same
points and can be read against each other. SE / non-SE keep Fig. 4's colours.

Two reference lines carry the reading:
  I = R        the identity. A correlation dimension cannot exceed the
               embedding dimension, so anything on the right of this line is an
               artifact, not a measurement. drr_upstream's released estimator
               crosses it; its bug-fixed sibling cannot, by construction.
  I = (2/3) R  the paper's Eq. 1 threshold, DRR = 1/3. Points above this line
               are the "high DRR" datasets the paper's argument rests on.

The paper's own (I, R) pairs from external/moot/drr/DRR.csv are drawn behind
ours in grey wherever the dataset name matches, so a point that moved can be
read directly.

Backends are the same three as run_fig4.py and are imported from it, so both
figures are produced by one definition of each estimator:
  drr_mine      max slope over 100 pieces of the scaling region; NaN (no point
                plotted) when the spectrum is too atomic to carry a slope
  drr_modified  bug-fixed upstream (C=0 dropped, windows widened to 0 < I <= R)
  drr_upstream  the released package, tight windows and constant fallbacks

    conda run -n drr python calculate_drr/run_fig3.py
    conda run -n drr python calculate_drr/run_fig3.py --backend drr_mine

Outputs (one set per backend):
  results/<backend>/fig3_dims.csv
  results/<backend>/fig3_dims.png
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

# same estimator definitions AND same dataset list as the Fig. 4 redraw
from run_fig4 import BACKENDS, build_tasks  # noqa: E402

PAPER_CSV = os.path.join(REPO_ROOT, "external", "moot", "drr", "DRR.csv")

FIELDS = ["dataset", "group", "path", "R", "I", "DRR", "paper_R", "paper_I",
          "paper_DRR", "n_rows", "n_used", "note", "status"]


def load_paper_points():
    """{lowercased dataset name: (original dims, intrinsic dims, drr)} from the
    paper's own table. Names there are bare (no .csv, no directory)."""
    out = {}
    if not os.path.exists(PAPER_CSV):
        return out
    with open(PAPER_CSV, newline="") as f:
        for row in csv.DictReader(f):
            try:
                out[row["Dataset"].strip().lower()] = (
                    float(row["Original Dimensions"]),
                    float(row["Intrinsic Dimensions"]),
                    float(row["DRR"]))
            except (KeyError, ValueError):
                continue
    return out


def plot_fig3(rows, out_png, backend):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    pts = [r for r in rows if isinstance(r["I"], float) and r["I"] == r["I"]
           and isinstance(r["R"], (int, float)) and r["R"]]
    fig, ax = plt.subplots(figsize=(10, 8.5))

    hi = max([r["R"] for r in pts] + [r["I"] for r in pts] + [10]) * 1.6
    lo = 0.7
    line = np.array([lo, hi])

    # impossible region: I > R
    ax.fill_between(line, line, lo, color="tab:red", alpha=0.06, zorder=0)
    ax.plot(line, line, ls="-", lw=1.2, color="0.35", zorder=1)
    ax.plot(line, line * 1.5, ls="--", lw=1.1, color="tab:blue", zorder=1)
    ax.text(hi * 0.95, hi * 0.95, "I = R", ha="right", va="bottom", fontsize=9,
            color="0.35", rotation=45, rotation_mode="anchor")
    ax.text(hi / 1.5 * 0.95, hi * 0.93, "DRR = 1/3", ha="right", va="bottom",
            fontsize=9, color="tab:blue", rotation=45, rotation_mode="anchor")
    ax.text(hi * 0.9, lo * 1.6, "I > R: impossible", ha="right", va="bottom",
            fontsize=8.5, color="tab:red")

    # the paper's own points, behind ours
    pap = [(r["paper_I"], r["paper_R"]) for r in rows
           if isinstance(r.get("paper_I"), float) and r["paper_I"] == r["paper_I"]]
    if pap:
        ax.scatter([p[0] for p in pap], [p[1] for p in pap], s=70, marker="o",
                   facecolors="none", edgecolors="0.6", linewidths=1.2, zorder=2,
                   label=f"paper's own values (n={len(pap)})")

    # SE / non-SE keep Fig. 4's colours; a ring marks a point in the impossible
    # region so it stays visible whichever group it belongs to
    for group, colour in (("SE", "tab:green"), ("Non-SE", "tab:red")):
        sub = [r for r in pts if r["group"] == group]
        if not sub:
            continue
        ax.scatter([r["I"] for r in sub], [r["R"] for r in sub], marker="x",
                   s=110, linewidths=2.2, color=colour, zorder=4,
                   label=f"{group} (n={len(sub)})")
    bad = [r for r in pts if r["I"] > r["R"]]
    if bad:
        ax.scatter([r["I"] for r in bad], [r["R"] for r in bad], s=190,
                   facecolors="none", edgecolors="tab:red", linewidths=1.6,
                   zorder=3, label=f"I > R: artifact (n={len(bad)})")
    # stack labels of coincident points instead of overprinting them: several
    # SE tasks land on the same (I, R) to within a percent (Xomo/Osp variants,
    # pom3a/pom3d, Health-Easy/Hard), so a fixed offset would draw them on top
    # of each other
    placed = []
    for r in sorted(pts, key=lambda q: (-q["R"], q["I"])):
        near = sum(1 for q in placed
                   if abs(np.log(q[0] / r["I"])) < 0.06
                   and abs(np.log(q[1] / r["R"])) < 0.06)
        ax.annotate(r["dataset"], (r["I"], r["R"]), textcoords="offset points",
                    xytext=(0, 9 + 9 * near), ha="center", fontsize=7,
                    color="tab:green" if r["group"] == "SE" else "tab:red")
        placed.append((r["I"], r["R"]))

    missing = [r["dataset"] for r in rows
               if not (isinstance(r["I"], float) and r["I"] == r["I"])]
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("intrinsic dimension  I")
    ax.set_ylabel("original dimension  R")
    ax.set_title(f"Fig. 3 redraw -- {backend}: intrinsic vs original dimension\n"
                 f"{len(pts)} of {len(rows)} paper Fig. 3/4 datasets plotted"
                 + (f"   (no value: {', '.join(missing)})" if missing else ""),
                 fontsize=10)
    ax.grid(alpha=0.25, which="both")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", default="all",
                    choices=["all", *BACKENDS])
    ap.add_argument("--outdir", default=os.path.join(REPO_ROOT, "results"))
    args = ap.parse_args()

    tasks = [t for t in build_tasks() if t[2]]  # csv=None: paper point we lack
    if not tasks:
        sys.exit("no datasets resolved from run_fig4.PAPER_FIG4")
    paper = load_paper_points()
    backends = list(BACKENDS) if args.backend == "all" else [args.backend]

    for backend in backends:
        outdir = os.path.join(args.outdir, backend)
        os.makedirs(outdir, exist_ok=True)
        rows, errors = [], []
        print(f"\n=== {backend} ({len(tasks)} datasets) ===")
        print(f"{'dataset':<20} {'group':<7} {'R':>5} {'I':>8} {'DRR':>7}")
        for name, group, path, _x, _pdrr, note in tasks:
            row = {"dataset": name, "group": group, "note": note,
                   "path": os.path.relpath(path, REPO_ROOT),
                   "R": "", "I": float("nan"), "DRR": float("nan"),
                   "n_rows": "", "n_used": "", "status": "ok"}
            try:
                r = BACKENDS[backend](path)
                row.update(R=r["R"], I=r["I_fit"], DRR=r["DRR"],
                           n_rows=r["n_rows"], n_used=r["n_used"],
                           status=r["status"])
            except Exception as e:  # skip-and-log, never crash the sweep
                row["status"] = f"error: {type(e).__name__}: {e}"
                errors.append(f"{name}: {traceback.format_exc()}")
            # the paper's own numbers, matched on either its Fig. 4 label or the
            # csv's basename (its table uses bare dataset names)
            key = os.path.splitext(os.path.basename(path))[0].lower()
            p = paper.get(name.strip().lower()) or paper.get(key)
            row["paper_R"], row["paper_I"], row["paper_DRR"] = p if p else (
                float("nan"), float("nan"), float("nan"))
            rows.append(row)
            fmt = lambda v: f"{v:.2f}" if isinstance(v, float) and v == v else "nan"
            print(f"{name:<20} {group:<7} {str(row['R']):>5} {fmt(row['I']):>8} "
                  f"{fmt(row['DRR']):>7}"
                  + ("" if row["status"] == "ok" else f"   <- {row['status']}"))

        csv_path = os.path.join(outdir, "fig3_dims.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow({k: (round(v, 4) if isinstance(v, float) else v)
                            for k, v in r.items()})

        png_path = os.path.join(outdir, "fig3_dims.png")
        plot_fig3(rows, png_path, backend)
        if errors:
            with open(os.path.join(outdir, "fig3_errors.log"), "w") as f:
                f.write("\n".join(errors))

        plotted = sum(1 for r in rows if isinstance(r["I"], float) and r["I"] == r["I"])
        over = sum(1 for r in rows if isinstance(r["I"], float) and r["I"] == r["I"]
                   and r["R"] and r["I"] > r["R"])
        matched = sum(1 for r in rows if r["paper_I"] == r["paper_I"])
        print(f"{plotted}/{len(rows)} plotted ({over} with I>R, "
              f"{matched} matched to the paper's table) -> "
              f"{os.path.relpath(csv_path, os.getcwd())}, "
              f"{os.path.relpath(png_path, os.getcwd())}")


if __name__ == "__main__":
    main()
