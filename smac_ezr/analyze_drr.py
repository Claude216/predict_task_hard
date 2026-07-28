"""SMAC vs EZR against DRR and budget.

    python analyze_drr.py results_b30 results_b50 results_b100 \
        --drr drr.csv --out drr_fig

A scatter rather than a binned grid, and deliberately so. Binning first would
manufacture structure: cut DRR into five ranges and fifteen cells will show
some pattern whether or not one exists. The scatter shows whether there is a
boundary to bin around, and only then are bin edges chosen with a reason. The
quantile bands printed at the end are a summary OF the scatter, not a substitute
for it.

The scatter also keeps something a grid cannot: each task's three points share
an x, so a task that changes winner as the budget grows is visible as a colour
change up one vertical line. Binning discards task identity and that migration
is lost -- and it is the most interesting quantity here.

Ties are grey and are not folded into either side. At B=30 over the full task
set the split was smac=35 ezr=31 tie=52: ties are the largest class, and a
two-colour plot would either drop 41% of the points or fake them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

COLOUR = {"smac": "#d1495b", "ezr": "#3b6ea5", "tie": "#b8b8b8"}
LABEL = {"smac": "SMAC", "ezr": "EZR", "tie": "tie"}


def load_verdicts(dirs: list[Path]) -> pd.DataFrame:
    """One row per (task, budget)."""
    out = []
    for d in dirs:
        for jf in sorted(d.glob("*.json")):
            blob = json.loads(jf.read_text())
            meta, tid = blob["meta"], blob["task_id"]

            on_table: dict[tuple, float] = {}
            runs = d / f"{tid}.runs.jsonl"
            if runs.exists():
                acc: dict[tuple, list] = {}
                for line in runs.open():
                    r = json.loads(line)
                    acc.setdefault((r["budget"], r["optimizer"]), []).append(
                        r.get("trace_on_table_rate", np.nan))
                on_table = {k: float(np.median(v)) for k, v in acc.items()}

            for b, winner in blob["verdicts"].items():
                if winner is None:
                    continue
                b = int(b)
                out.append(dict(
                    task_id=tid, dir=Path(blob["path"]).parent.name,
                    budget=b, winner=winner,
                    objective=meta["objective"],
                    input_shape=meta["input_shape"],
                    rows=meta["rows"], n_decisions=meta["n_decisions"],
                    coverage=meta.get("coverage"),
                    x_dup_rate=meta.get("x_dup_rate"),
                    # carried, not plotted: SMAC's advantage grows exactly where
                    # the surrogate extrapolates, so this is what will separate
                    # "searched better" from "extrapolated better" later
                    smac_on_table=on_table.get((b, "smac"), np.nan),
                ))
    return pd.DataFrame(out)


def migrations(df: pd.DataFrame, budgets: list[int]) -> pd.DataFrame:
    """Per task: did the winner change as the budget grew?"""
    rows = []
    for tid, g in df.groupby("task_id"):
        seq = [g.loc[g.budget == b, "winner"].iloc[0]
               for b in budgets if (g.budget == b).any()]
        decided = [w for w in seq if w != "tie"]
        rows.append(dict(
            task_id=tid,
            seq="/".join(seq),
            changed=len(set(seq)) > 1,
            # a tie turning into a win is a change of verdict but not a change
            # of side; flipping between smac and ezr is the stronger claim
            flipped=len(set(decided)) > 1,
        ))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", type=Path)
    ap.add_argument("--drr", default="drr.csv")
    ap.add_argument("--x", default="drr", choices=["drr", "I", "R", "coverage",
                                                   "x_dup_rate", "n_decisions"],
                    help="drr = 1 - I/R, relative compressibility; "
                         "I = absolute intrinsic dimension, which is what A7's "
                         "'low intrinsic dimensionality' literally claims; "
                         "R = decision columns. Two tasks with I=3 sit at "
                         "DRR 0.4 and 0.997 if R is 5 and 1000, so the two "
                         "axes can tell different stories.")
    ap.add_argument("--logx", action="store_true",
                    help="log x-axis; useful for I and R, which span orders "
                         "of magnitude, unlike drr which is bounded in [0,1)")
    ap.add_argument("--out", default="drr_fig")
    ap.add_argument("--only-dirs", nargs="*", default=None)
    ap.add_argument("--bands", type=int, default=5,
                    help="quantile bands for the printed summary")
    args = ap.parse_args()

    df = load_verdicts(args.dirs)
    if df.empty:
        raise SystemExit(f"no verdicts found under {args.dirs}")

    drr = pd.read_csv(args.drr)[["task_id", "drr", "R", "I", "drr_spread"]]
    before = df.task_id.nunique()
    df = df.merge(drr, on="task_id", how="inner")
    after = df.task_id.nunique()
    if after < before:
        missing = before - after
        print(f"warning: {missing} task(s) have verdicts but no DRR; dropped. "
              f"Check that compute_drr.py and batch.py used the same root, "
              f"since task_id is built from the path relative to it.\n")

    if args.only_dirs:
        df = df[df.dir.isin(args.only_dirs)]
        print(f"filtered to {sorted(args.only_dirs)}: "
              f"{df.task_id.nunique()} tasks\n")

    budgets = sorted(df.budget.unique())
    X = args.x
    XNAME = {"drr": "DRR  (1 - I/R;  higher = collapses into fewer dimensions)",
             "I": "intrinsic dimension I  (Levina-Bickel)",
             "R": "R  (decision columns fed to the estimator)",
             "coverage": "coverage  (log2 distinct rows - sum log2|Xi|)",
             "x_dup_rate": "fraction of rows repeating in x-space",
             "n_decisions": "number of decision columns"}[X]

    # R is the column count the estimator saw. It should equal n_decisions;
    # a mismatch means the patched estimate() dropped columns (the constant
    # column fix), which changes the denominator of DRR and is worth knowing.
    gap = df[df.R != df.n_decisions]
    if len(gap):
        print(f"note: R != n_decisions on {gap.task_id.nunique()} task(s) -- "
              f"columns were dropped before estimation, so DRR's denominator "
              f"is not simply the decision count:")
        for _, r in gap.drop_duplicates("task_id").head(5).iterrows():
            print(f"  {r.task_id:<34} R={r.R} n_decisions={r.n_decisions}")
        print()

    print(f"tasks={df.task_id.nunique()}  budgets={budgets}  points={len(df)}")
    print(f"x = {X}: {df[X].min():.3f} to {df[X].max():.3f}\n")

    # ---- overall counts ---------------------------------------------------
    print("verdicts per budget:")
    for b in budgets:
        vc = df[df.budget == b].winner.value_counts()
        n = int(vc.sum())
        print(f"  B={b:<5} " + "  ".join(
            f"{LABEL[k]}={int(vc.get(k, 0))} ({vc.get(k, 0)/n:.0%})"
            for k in ("smac", "ezr", "tie")))

    # ---- quantile bands ---------------------------------------------------
    print(f"\n{X} quantile bands ({args.bands}), counts smac/ezr/tie:")
    edges = np.quantile(df[X].unique(), np.linspace(0, 1, args.bands + 1))
    edges[-1] += 1e-9
    hdr = f"{X + ' band':<18}{'tasks':>7}" + "".join(
        f"{'B=' + str(b):>18}" for b in budgets)
    print(hdr)
    band_rows = []
    for i in range(args.bands):
        lo, hi = edges[i], edges[i + 1]
        sub = df[(df[X] >= lo) & (df[X] < hi)]
        if sub.empty:
            continue
        line = f"[{lo:.2f},{hi:.2f})".ljust(18) + f"{sub.task_id.nunique():>7}"
        for b in budgets:
            s = sub[sub.budget == b].winner.value_counts()
            line += f"{int(s.get('smac',0))}/{int(s.get('ezr',0))}/{int(s.get('tie',0))}".rjust(18)
            band_rows.append(dict(band=f"[{lo:.2f},{hi:.2f})", budget=b,
                                  smac=int(s.get("smac", 0)),
                                  ezr=int(s.get("ezr", 0)),
                                  tie=int(s.get("tie", 0))))
        print(line)

    # ---- where does each side win? ---------------------------------------
    print(f"\nmedian {X} by verdict:")
    for b in budgets:
        sub = df[df.budget == b]
        parts = []
        for k in ("smac", "ezr", "tie"):
            v = sub[sub.winner == k][X]
            parts.append(f"{LABEL[k]}={v.median():.3f} (n={len(v)})"
                         if len(v) else f"{LABEL[k]}=-")
        print(f"  B={b:<5} " + "   ".join(parts))
    print("A gap between the SMAC and EZR rows that widens with budget is the "
          "effect the plot is looking for.")

    # ---- migration --------------------------------------------------------
    mig = migrations(df, budgets)
    print(f"\ntasks whose verdict changed with budget: "
          f"{mig.changed.sum()}/{len(mig)} ({mig.changed.mean():.0%})")
    print(f"tasks that flipped side (smac <-> ezr): "
          f"{mig.flipped.sum()}/{len(mig)} ({mig.flipped.mean():.0%})")
    if mig.flipped.any():
        flip = mig[mig.flipped].merge(
            df[["task_id", "drr", "I", "R"]].drop_duplicates("task_id"),
            on="task_id")
        print("  flipped tasks:")
        for _, r in flip.sort_values(X if X in flip else "drr").iterrows():
            print(f"    {r.task_id:<34} DRR={r.drr:.3f} I={r.I:.2f} "
                  f"R={int(r.R):<5} {r.seq}")

    # ---- outputs ----------------------------------------------------------
    df.to_csv(f"{args.out}_points.csv", index=False)
    pd.DataFrame(band_rows).to_csv(f"{args.out}_bands.csv", index=False)
    mig.to_csv(f"{args.out}_migration.csv", index=False)
    print(f"\nwrote {args.out}_points.csv, _bands.csv, _migration.csv")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipped the figure "
              "(pip install matplotlib)")
        return

    ypos = {b: i for i, b in enumerate(budgets)}      # budgets are categorical:
    # 30/50/100/200 on a linear axis would crowd the low budgets and strand 200
    rng = np.random.default_rng(0)

    fig, ax = plt.subplots(figsize=(11, 4.2))

    # one faint line per task joining its budgets; darker where the side flipped
    flipped = set(mig.loc[mig.flipped, "task_id"])
    for tid, g in df.groupby("task_id"):
        g = g.sort_values("budget")
        if len(g) < 2:
            continue
        hot = tid in flipped
        ax.plot([g[X].iloc[0]] * len(g), [ypos[b] for b in g.budget],
                color="#333333" if hot else "#dddddd",
                lw=1.1 if hot else .6, alpha=.85 if hot else .5, zorder=1)

    for k in ("tie", "ezr", "smac"):          # ties underneath, decided on top
        sub = df[df.winner == k]
        jitter = rng.uniform(-.16, .16, len(sub))
        ax.scatter(sub[X], [ypos[b] for b in sub.budget] + jitter,
                   s=34, c=COLOUR[k], alpha=.75, linewidths=.4,
                   edgecolors="white", label=f"{LABEL[k]} (n={len(sub)})",
                   zorder=3)

    if args.logx:
        ax.set_xscale("log")

    ax.set_yticks(list(ypos.values()))
    ax.set_yticklabels([f"B={b}" for b in budgets])
    ax.set_ylim(-.5, len(budgets) - .5)
    ax.set_xlabel(XNAME)
    ax.set_title(f"SMAC vs EZR by {X} and labelling budget\n"
                 "one column per task; dark line = the winner flipped side",
                 fontsize=10)
    ax.grid(axis="x", color="#eeeeee", zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.22), ncol=3,
              frameon=False)
    fig.tight_layout()
    fig.savefig(f"{args.out}.png", dpi=200, bbox_inches="tight")
    print(f"wrote {args.out}.png")


if __name__ == "__main__":
    main()