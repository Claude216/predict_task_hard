#!/usr/bin/env python3
"""Fig. 4 (and Fig. 3) of the paper, from our own runs.

    conda run -n dmoot python -m dehb_ezr.paper.fig4

    fig4.png   x = Raw (R), y = DRR = 1 - I/R
    fig3.png   x = Raw (R), y = Intrinsic (I)

Two colours only, matching the paper's legend exactly:

    red    DEHB > LITE     complex optimization (3000 samples) defeated the
                           simple method (30 samples)
    green  DEHB == LITE    everything else, INCLUDING LITE winning -- the paper
                           has no third class, so a LITE win is simply not a
                           DEHB win

The paper's own Fig. 4 draws a reference at DRR = 0.35 ("above this threshold,
100% of the data sets are easy to optimize"), so it is drawn here too: it is the
claim the figure exists to test, and its position must be visible.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    __package__ = "dehb_ezr.paper"

from .datasets import DROPPED

RED, GREEN = "#d1495b", "#2e9e4f"
LABEL = {"DEHB > LITE": "DEHB > LITE", "DEHB == LITE": "DEHB == LITE"}
FIGS = Path(__file__).resolve().parent.parent / "paper_figs"
RESULTS = Path(__file__).resolve().parent.parent / "paper_results"
DRR_THRESHOLD = 0.35


def load(results: Path, drr_csv: Path) -> pd.DataFrame:
    rows = []
    for jf in sorted(results.glob("*.json")):
        m = json.loads(jf.read_text())["meta"]
        rows.append({k: m[k] for k in
                     ("label", "task_id", "is_se", "kind", "R", "paper_R",
                      "verdict", "verdict_natural_d2h", "cliffs_delta_D2H",
                      "n_rows", "capped")}
                    | {"med_D2H_lite": m["median_D2H"]["lite"],
                       "med_D2H_dehb": m["median_D2H"]["dehb"]})
    df = pd.DataFrame(rows)
    if df.empty:
        sys.exit(f"no results under {results}")

    drr = pd.read_csv(drr_csv)
    drr = drr[drr.status == "ok"][["task_id", "I", "drr", "drr_commit", "drr_dirty"]]
    before = len(df)
    df = df.merge(drr, on="task_id", how="inner")
    if len(df) < before:
        print(f"warning: {before - len(df)} dataset(s) have a verdict but no DRR; "
              f"dropped\n")
    return df


def scatter(df, ycol, ylabel, title, png, logy, hline=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6.5))
    df = df.copy()

    # R and I are integers, so points collide; fan co-located datasets apart
    # horizontally (never vertically) so none is hidden. Capped at +/-8% of R
    # however many share a cell.
    x = df.R.astype(float).copy()
    for _, idx in df.groupby([df.R, df[ycol].round(6)]).groups.items():
        idx = list(idx)
        if len(idx) > 1:
            step = min(0.030, 0.16 / (len(idx) - 1))
            for j, i in enumerate(idx):
                x.loc[i] = df.R.loc[i] * (1 + step * (j - (len(idx) - 1) / 2))
    df["_x"] = x

    if hline is not None:
        ax.axhline(hline, color="#888", lw=1, ls="--", zorder=1)
        ax.annotate(f"DRR = {hline}  (the paper's threshold)", xy=(1.0, hline),
                    xycoords=("axes fraction", "data"), xytext=(-4, 4),
                    textcoords="offset points", ha="right", va="bottom",
                    fontsize=8, color="#777")

    # draw the larger class first so the rarer one is never painted over
    order = sorted(["DEHB == LITE", "DEHB > LITE"],
                   key=lambda v: -(df.verdict == v).sum())
    for v in order:
        sub = df[df.verdict == v]
        if sub.empty:
            continue
        ax.scatter(sub._x, sub[ycol], c=RED if v == "DEHB > LITE" else GREEN,
                   s=66, zorder=3, edgecolors="#333", linewidths=0.6,
                   label=f"{LABEL[v]} (n={len(sub)})")

    ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
        ax.set_ylim(df[ycol].min() * 0.75, df[ycol].max() * 1.5)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        axis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xticks([t for t in (3, 5, 10, 20, 50, 100, 250)
                   if df.R.min() * 0.8 <= t <= df.R.max() * 1.25])
    if logy:
        ax.set_yticks([t for t in (1, 2, 3, 5, 8, 10, 15, 20, 30, 50)
                       if df[ycol].min() * 0.8 <= t <= df[ycol].max() * 1.4])

    ax.set_xlabel("Raw  —  R, feature columns (log)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", frameon=True, framealpha=0.9, edgecolor="none",
              fontsize=9)
    fig.tight_layout()
    fig.savefig(png, dpi=150)
    print(f"wrote {png}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(RESULTS), type=Path)
    ap.add_argument("--drr", default=str(FIGS / "drr.csv"), type=Path)
    ap.add_argument("--out", default=str(FIGS), type=Path)
    args = ap.parse_args()

    df = load(args.results, args.drr)
    args.out.mkdir(parents=True, exist_ok=True)

    n = len(df)
    red = int((df.verdict == "DEHB > LITE").sum())
    print(f"datasets={n}   DEHB > LITE={red} ({red/n:.0%})   "
          f"DEHB == LITE={n-red} ({(n-red)/n:.0%})")
    for label, why in DROPPED.items():
        print(f"dropped: {label} -- {why}")
    if df.drr_dirty.any():
        print(f"NOTE: DRR from a dirty estimator checkout ({df.drr_commit.iloc[0]}); "
              f"see drr_estimator.patch")

    flip = df[df.verdict != df.verdict_natural_d2h]
    print(f"\nverdict changes if scored on natural-bounds d2h instead of the "
          f"paper's Zitzler D2H: {len(flip)}/{n}")
    for _, r in flip.iterrows():
        print(f"   {r.label:<22} {r.verdict}  ->  {r.verdict_natural_d2h}")

    above = df[df.drr > DRR_THRESHOLD]
    below = df[df.drr <= DRR_THRESHOLD]
    print(f"\nthe paper's claim: above DRR {DRR_THRESHOLD}, simple methods suffice")
    for name, sub in (("DRR >  0.35", above), ("DRR <= 0.35", below)):
        if len(sub):
            r = int((sub.verdict == "DEHB > LITE").sum())
            print(f"  {name}: {len(sub):>3} datasets, DEHB won {r} ({r/len(sub):.0%})")
    for name, sub in (("SE", df[df.is_se]), ("non-SE", df[~df.is_se])):
        if len(sub):
            hi = int((sub.drr > DRR_THRESHOLD).sum())
            r = int((sub.verdict == "DEHB > LITE").sum())
            print(f"  {name:<7}: {len(sub):>3} datasets, {hi} above threshold "
                  f"({hi/len(sub):.0%}), DEHB won {r}")

    scatter(df, "drr", "DRR = 1 − (Intrinsic / Raw)",
            "Fig. 4 reproduction — DRR vs raw dimensionality",
            args.out / "fig4.png", logy=False, hline=DRR_THRESHOLD)
    scatter(df, "I", "Intrinsic (log)",
            "Fig. 3 reproduction — intrinsic vs raw dimensionality",
            args.out / "fig3.png", logy=True)

    df.to_csv(args.out / "verdicts.csv", index=False)
    print(f"wrote {args.out}/verdicts.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
