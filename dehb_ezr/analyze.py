#!/usr/bin/env python3
"""The two figures: where does LITE keep up with HEAVY, in dimensionality terms?

    conda run -n dmoot python -m dehb_ezr.analyze \
        --results dehb_ezr/results --drr results/dehb_ezr/drr.csv \
        --out results/dehb_ezr

One point per task, coloured by the ezr-vs-dehb verdict:

    yellow  EZR (LITE, 30 labels) wins
    green   tie
    red     DEHB (HEAVY, 3000 evals) wins

Figure 1  dim_intrinsic_vs_raw.png   x = R (raw decision dims), y = I (intrinsic)
Figure 2  drr_vs_raw.png             x = R,                     y = DRR = 1 - I/R

The pairing is deliberate: figure 2 is a DETERMINISTIC TRANSFORM of figure 1 --
same x, y divided by x and flipped. Figure 1 shows absolute intrinsic
dimension, which is what "low intrinsic dimensionality" literally claims;
figure 2 shows relative compressibility. A task with I=3 sits at DRR 0.4 when
R=5 and 0.988 when R=256, so a colour boundary that is clean in one panel and
smeared in the other is itself the result.

Ties are their own class and are never folded into either side. In the SMAC
study ties were the largest class at 41%; a two-colour plot would have faked
half the data.

Tasks where DEHB failed to beat its own 3000-draw random floor are drawn
hollow. There, DEHB's position says nothing about search quality, so counting
it as a HEAVY win would overstate the effect.

Scatter, not a binned grid. Cutting R into five bands first would manufacture
structure: fifteen cells show some pattern whether or not one exists. The
quantile bands printed at the end summarise the scatter; they do not replace it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "dehb_ezr"

COLOUR = {"ezr": "#f0c000", "tie": "#2e9e4f", "dehb": "#d1495b"}
LABEL = {"ezr": "EZR wins (LITE, 30)", "tie": "tie",
         "dehb": "DEHB wins (HEAVY, 3000)"}
ORDER = ["ezr", "tie", "dehb"]


def load(results: Path) -> pd.DataFrame:
    """One row per task, from batch.py's per-task json."""
    out = []
    for jf in sorted(results.glob("*.json")):
        blob = json.loads(jf.read_text())
        meta, v = blob["meta"], blob["verdicts"]
        w = v.get("lite_vs_heavy")
        if w is None:
            continue
        out.append(dict(
            task_id=blob["task_id"],
            group=Path(blob["path"]).parent.name,
            winner=w,
            dehb_beat_floor=(v.get("dehb_vs_floor") == "dehb"),
            ezr_beat_floor=(v.get("ezr_vs_floor") == "ezr"),
            n_decisions=meta["n_decisions"],
            rows=meta["rows"],
            objective=meta["objective"],
            coverage=meta.get("coverage"),
            x_dup_rate=meta.get("x_dup_rate"),
            med_ezr=meta["median_d2h"].get("ezr"),
            med_dehb=meta["median_d2h"].get("dehb"),
        ))
    return pd.DataFrame(out)


def _spread(df, ycol):
    """Fan co-located tasks apart horizontally so none is hidden.

    R and I are integers, so DRR = 1 - I/R is discrete too and many tasks land
    on EXACTLY the same coordinate -- 126 tasks occupy far fewer distinct
    points. Overplotting here is not cosmetic: a rare EZR win sitting under a
    DEHB point disappears entirely, and the figure then understates the very
    class it exists to show.

    Members of a cell are fanned symmetrically about the true x by a
    multiplicative offset, so the fan is even on a log axis. The TOTAL fan is
    capped at +/-8% of R however many tasks share the cell -- the biggest cell
    here holds 35 tasks, and a fixed per-point step would have spread it over
    +/-50% and thrown points into neighbouring R values.

    y is never displaced. x is nudged only for legibility, so read x off the
    axis as approximate; verdicts.csv carries the exact values.
    """
    span = 0.08
    x = df.R.astype(float).copy()
    for _, idx in df.groupby([df.R, df[ycol].round(6)]).groups.items():
        idx = list(idx)
        n = len(idx)
        if n < 2:
            continue
        step = min(0.030, 2 * span / (n - 1))
        for j, i in enumerate(idx):
            x.loc[i] = df.R.loc[i] * (1.0 + step * (j - (n - 1) / 2))
    return x


def scatter(df, ycol, ylabel, title, out_png, logy: bool, ref_diag: bool):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6.5))
    df = df.copy()
    df["_x"] = _spread(df, ycol)

    if ref_diag:
        # y = x is "did not compress at all": I can never exceed R, so every
        # point must sit on or below this line. Drawn only across the range the
        # data occupies -- extending it to R=1044 would force the y-axis up to
        # 1000 and squash every real I (which never exceeds 15) into a band
        # along the bottom.
        lim = [max(1, df.R.min() * 0.9), df[ycol].max() * 1.6]
        ax.plot(lim, lim, color="#999", lw=1, ls="--", zorder=1)
        ax.annotate("I = R  (no compression)", xy=(lim[1], lim[1]),
                    xytext=(4, -2), textcoords="offset points",
                    ha="left", va="top", fontsize=8, color="#777")

    # Draw the LARGEST class first so the rare ones end up on top. Drawing in
    # ORDER would paint 102 DEHB points over the 7 EZR wins.
    for w in sorted(ORDER, key=lambda k: -(df.winner == k).sum()):
        sub = df[df.winner == w]
        if sub.empty:
            continue
        hollow = sub[(~sub.dehb_beat_floor) & (sub.winner == "dehb")]
        solid = sub.drop(hollow.index)
        z = 3 + ORDER.index(w)
        # thin dark edge: the specified yellow is otherwise near-invisible on white
        ax.scatter(solid._x, solid[ycol], c=COLOUR[w], s=64, zorder=z,
                   edgecolors="#333", linewidths=0.6,
                   label=f"{LABEL[w]} (n={len(sub)})")
        if len(hollow):
            ax.scatter(hollow._x, hollow[ycol], facecolors="none", s=64, zorder=z,
                       edgecolors=COLOUR[w], linewidths=1.5)

    ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
        ax.set_ylim(df[ycol].min() * 0.75, df[ycol].max() * 1.5)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        axis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xticks([t for t in (3, 5, 10, 20, 50, 100, 250, 1000)
                   if df.R.min() * 0.8 <= t <= df.R.max() * 1.25])
    if logy:
        ax.set_yticks([t for t in (1, 2, 3, 5, 8, 10, 15, 20)
                       if df[ycol].min() * 0.8 <= t <= df[ycol].max() * 1.4])

    ax.set_xlabel("R  —  raw decision dimensions (log)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    handles, _ = ax.get_legend_handles_labels()
    handles = sorted(handles, key=lambda h: ORDER.index(
        next(k for k in ORDER if LABEL[k] in h.get_label())))
    if any((~df.dehb_beat_floor) & (df.winner == "dehb")):
        handles.append(plt.Line2D([], [], marker="o", ls="", markerfacecolor="none",
                                  markeredgecolor=COLOUR["dehb"], markeredgewidth=1.5,
                                  label="DEHB win that did NOT beat its own floor"))
    ax.legend(handles=handles, loc="lower right", frameon=True, framealpha=0.9,
              edgecolor="none", fontsize=9)
    ax.text(0.005, -0.13, "126 tasks share only 64 distinct (R, I) cells, so "
            "co-located tasks are fanned horizontally (±8% of R) for legibility; "
            "y is never displaced", transform=ax.transAxes,
            fontsize=7.5, color="#777")

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"wrote {out_png}")


def bands(df, col, n=5) -> pd.DataFrame:
    """Quantile summary OF the scatter, not a substitute for it."""
    edges = np.quantile(df[col].unique(), np.linspace(0, 1, n + 1))
    edges[-1] += 1e-9
    rows = []
    for i in range(n):
        lo, hi = edges[i], edges[i + 1]
        sub = df[(df[col] >= lo) & (df[col] < hi)]
        if sub.empty:
            continue
        c = sub.winner.value_counts()
        rows.append(dict(band=f"[{lo:.3g},{hi:.3g})", tasks=len(sub),
                         **{k: int(c.get(k, 0)) for k in ORDER}))
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="dehb_ezr/results", type=Path)
    ap.add_argument("--drr", default="results/dehb_ezr/drr.csv", type=Path)
    ap.add_argument("--out", default="results/dehb_ezr", type=Path)
    ap.add_argument("--bands", type=int, default=5)
    args = ap.parse_args()

    df = load(args.results)
    if df.empty:
        sys.exit(f"no verdicts under {args.results}")

    drr = pd.read_csv(args.drr)
    drr = drr[drr.status == "ok"][["task_id", "R", "I", "drr", "drr_commit", "drr_dirty"]]
    before = len(df)
    df = df.merge(drr, on="task_id", how="inner")
    if len(df) < before:
        print(f"warning: {before - len(df)} task(s) have verdicts but no usable "
              f"DRR and were dropped. task_id is built from the path relative "
              f"to --root; check that batch.py and drr_interface.py used the "
              f"same one.\n")

    # R from the estimator should equal the decision-column count. A mismatch
    # means the estimator dropped columns, so DRR's denominator is not simply
    # the decision count -- worth knowing before reading the x-axis.
    gap = df[df.R != df.n_decisions]
    if len(gap):
        print(f"note: R != n_decisions on {len(gap)} task(s):")
        for _, r in gap.head(5).iterrows():
            print(f"  {r.task_id:<40} R={r.R} n_decisions={r.n_decisions}")
        print()

    if df.drr_dirty.any():
        print(f"NOTE: DRR came from a dirty checkout of the estimator "
              f"(commit {df.drr_commit.iloc[0]}); see drr_estimator.patch.\n")

    args.out.mkdir(parents=True, exist_ok=True)
    counts = df.winner.value_counts()
    print(f"tasks={len(df)}   " + "  ".join(
        f"{LABEL[k]}={int(counts.get(k, 0))} ({counts.get(k, 0)/len(df):.0%})"
        for k in ORDER))
    n_hollow = int(((~df.dehb_beat_floor) & (df.winner == "dehb")).sum())
    print(f"DEHB wins that did NOT beat their own 3000-draw floor: {n_hollow}")
    print(f"EZR wins that did NOT beat their own 30-draw floor:    "
          f"{int(((~df.ezr_beat_floor) & (df.winner == 'ezr')).sum())}")
    print(f"\nR   {df.R.min()}..{df.R.max()}   "
          f"I {df.I.min():.2f}..{df.I.max():.2f}   "
          f"DRR {df.drr.min():.3f}..{df.drr.max():.3f}\n")

    scatter(df, "I", "I  —  intrinsic dimension (log)",
            "Intrinsic vs raw dimensionality, by LITE/HEAVY verdict",
            args.out / "dim_intrinsic_vs_raw.png", logy=True, ref_diag=True)
    scatter(df, "drr", "DRR = 1 − I/R  (higher = collapses further)",
            "Dimensionality reduction ratio vs raw dimensionality, by verdict",
            args.out / "drr_vs_raw.png", logy=False, ref_diag=False)

    df.to_csv(args.out / "verdicts.csv", index=False)
    allb = []
    for col in ("R", "I", "drr"):
        b = bands(df, col, args.bands)
        b.insert(0, "axis", col)
        allb.append(b)
        print(f"\n{col} quantile bands, counts ezr/tie/dehb:")
        print(b.to_string(index=False))
    pd.concat(allb).to_csv(args.out / "bands.csv", index=False)
    print(f"\nwrote {args.out}/verdicts.csv and bands.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
