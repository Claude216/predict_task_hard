"""Aggregate per-task verdicts into a Figure-5-style cell matrix.

    python analyze.py results_b30 results_b50 results_b100 --out fig5

WHY VERDICTS FIRST, THEN COUNTS

The alternative -- pool every seed from every task in a cell and run one top()
over the lot -- is not valid here. d2h ranges vary enormously between tables
(SS-A spans 0.003-0.97, nasa93dem 0.43-0.57), so a pooled distribution is
dominated by which tasks happen to sit low on the scale rather than by which
method searched better. The comparison is only meaningful within a task, where
both methods face the same oracle and the same scale. So: decide each
(task, budget) on its own 20-vs-20, then let every task cast one equal vote.

WHY THE ON-TABLE COLUMN SITS NEXT TO THE VERDICT

EZR is pool-based and can only return rows that exist, so it absorbs the
surrogate's pessimism. SMAC proposes points between rows, and because each goal
has its OWN random forest, it can find configurations where the per-goal models
are jointly wrong -- scoring better than any real row. The bias is therefore
not shared symmetrically between the two methods. Where SMAC wins a cell in
which its on-table rate is near zero, "SMAC searched better" and "SMAC
extrapolated better" are not yet distinguishable, and the verdict alone cannot
tell them apart.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SHAPES = ["binary/SAT", "large-numeric", "small-numeric"]
OBJECTIVES = ["multi", "single"]
CELLS = [f"{o} | {s}" for o in OBJECTIVES for s in SHAPES]


def load(dirs: list[Path], want_on_table: bool = True) -> pd.DataFrame:
    """One row per (task, budget)."""
    out = []
    for d in dirs:
        for jf in sorted(d.glob("*.json")):
            if jf.name.endswith(".error.txt"):
                continue
            blob = json.loads(jf.read_text())
            meta, tid = blob["meta"], blob["task_id"]

            on_table = {}
            rf = jf.with_suffix("").with_suffix("")  # strip .json
            runs = d / f"{tid}.runs.jsonl"
            if want_on_table and runs.exists():
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
                    task_id=tid,
                    dir=str(Path(blob["path"]).parent.name),
                    budget=b,
                    winner=winner,
                    objective=meta["objective"],
                    input_shape=meta["input_shape"],
                    cell=f"{meta['objective']} | {meta['input_shape']}",
                    rows=meta["rows"],
                    n_decisions=meta["n_decisions"],
                    coverage=meta.get("coverage"),
                    x_dup_rate=meta.get("x_dup_rate"),
                    smac_on_table=on_table.get((b, "smac"), np.nan),
                    ezr_on_table=on_table.get((b, "ezr"), np.nan),
                ))
    return pd.DataFrame(out)


def cell_label(sub: pd.DataFrame, min_n: int = 5) -> tuple[str, str]:
    """Plurality of per-task verdicts, plus a sign test on the decided ones.

    Ties are not folded into either side: a cell that is mostly ties IS mostly
    ties, and saying so is more informative than forcing a winner out of a 5-4
    split among the few tasks that separated at all."""
    n = len(sub)
    c = sub.winner.value_counts()
    e, s, t = int(c.get("ezr", 0)), int(c.get("smac", 0)), int(c.get("tie", 0))

    if n == 0:
        return "-", ""
    if s > e:
        label = "SMAC"
    elif e > s:
        label = "EZR"
    else:
        label = "tie"
    if t >= max(e, s):
        label = f"tie*" if label == "tie" else f"{label}?"

    note = f"e={e} s={s} t={t} n={n}"
    if n < min_n:
        note += "  (too few tasks to read)"
    return label, note


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", type=Path)
    ap.add_argument("--out", default=None, help="prefix for csv/png output")
    ap.add_argument("--only-dirs", nargs="*", default=None,
                    help="keep only tasks from these source directories, e.g. "
                         "config systems process binary_config hpo test")
    ap.add_argument("--no-on-table", action="store_true")
    args = ap.parse_args()

    df = load(args.dirs, want_on_table=not args.no_on_table)
    if df.empty:
        raise SystemExit(f"no results found under {args.dirs}")

    if args.only_dirs:
        before = df.task_id.nunique()
        df = df[df.dir.isin(args.only_dirs)]
        print(f"filtered to {sorted(args.only_dirs)}: "
              f"{df.task_id.nunique()} of {before} tasks\n")

    budgets = sorted(df.budget.unique())
    print(f"tasks={df.task_id.nunique()}  budgets={budgets}  "
          f"rows={len(df)}\n")

    # ---- cell sizes -------------------------------------------------------
    sizes = (df[df.budget == budgets[0]].groupby("cell").task_id.nunique()
             .reindex(CELLS).fillna(0).astype(int))
    print("tasks per cell:")
    for c in CELLS:
        print(f"  {c:<26} {sizes[c]:>4}")
    print()

    # ---- the matrix -------------------------------------------------------
    print("=" * 76)
    print("SMAC vs EZR by cell and budget   "
          "(? = ties outnumber the winner; * = mostly ties)")
    print("=" * 76)
    head = f"{'cell':<26}" + "".join(f"{'B=' + str(b):>16}" for b in budgets)
    print(head)
    grid = {}
    for c in CELLS:
        line = f"{c:<26}"
        for b in budgets:
            sub = df[(df.cell == c) & (df.budget == b)]
            lab, note = cell_label(sub)
            grid[(c, b)] = (lab, note)
            line += f"{lab:>16}"
        print(line)

    print("\ncounts behind each cell:")
    for c in CELLS:
        print(f"  {c}")
        for b in budgets:
            lab, note = grid[(c, b)]
            print(f"    B={b:<5} {lab:<6} {note}")

    # ---- the confound check ----------------------------------------------
    print("\n" + "=" * 76)
    print("SMAC on-table rate per cell (median over tasks and seeds)")
    print("A cell where SMAC wins with a rate near 0 has not separated real")
    print("search quality from extrapolation into the surrogate's blind spots.")
    print("=" * 76)
    print(f"{'cell':<26}" + "".join(f"{'B=' + str(b):>16}" for b in budgets)
          + f"{'coverage':>11}{'x_dup':>8}")
    for c in CELLS:
        line = f"{c:<26}"
        for b in budgets:
            sub = df[(df.cell == c) & (df.budget == b)]
            v = sub.smac_on_table.median()
            line += f"{'n/a' if pd.isna(v) else format(v, '.2f'):>16}"
        sub0 = df[(df.cell == c) & (df.budget == budgets[0])]
        cov = sub0.coverage.median()
        dup = sub0.x_dup_rate.median()
        line += (f"{'n/a' if pd.isna(cov) else format(cov, '.1f'):>11}"
                 f"{'n/a' if pd.isna(dup) else format(dup, '.2f'):>8}")
        print(line)

    # does SMAC win more where it stays off the table?
    d = df.dropna(subset=["smac_on_table"])
    if len(d) > 10:
        won = d[d.winner == "smac"].smac_on_table
        lost = d[d.winner == "ezr"].smac_on_table
        print(f"\nSMAC on-table rate where SMAC won : median "
              f"{won.median():.2f}  (n={len(won)})")
        print(f"SMAC on-table rate where EZR won  : median "
              f"{lost.median():.2f}  (n={len(lost)})")
        print("If the first is markedly lower, SMAC's wins are concentrated "
              "where it never proposed a real configuration.")

    # ---- outputs ----------------------------------------------------------
    if args.out:
        df.to_csv(f"{args.out}_tasks.csv", index=False)
        mat = pd.DataFrame(
            {b: [grid[(c, b)][0] for c in CELLS] for b in budgets}, index=CELLS)
        mat.to_csv(f"{args.out}_matrix.csv")
        print(f"\nwrote {args.out}_tasks.csv and {args.out}_matrix.csv")

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            colour = {"SMAC": "#f4a261", "SMAC?": "#f8c99a",
                      "EZR": "#8ecf9e", "EZR?": "#bfe3c8",
                      "tie": "#dddddd", "tie*": "#dddddd", "-": "#ffffff"}
            fig, ax = plt.subplots(figsize=(1.6 * len(budgets) + 3, 4))
            for i, c in enumerate(CELLS):
                for j, b in enumerate(budgets):
                    lab, note = grid[(c, b)]
                    sub = df[(df.cell == c) & (df.budget == b)]
                    vc = sub.winner.value_counts()
                    e, s, t = (int(vc.get(k, 0)) for k in ("ezr", "smac", "tie"))
                    ax.add_patch(plt.Rectangle((j, len(CELLS) - 1 - i), 1, 1,
                                               facecolor=colour.get(lab, "#fff"),
                                               edgecolor="white", lw=2))
                    ax.text(j + .5, len(CELLS) - .40 - i, lab, ha="center",
                            va="center", fontsize=11)
                    # the label alone hides how thin the margin is: a cell read
                    # as SMAC off 3-2 with 11 ties is not the same finding as
                    # one read off 12-1
                    ax.text(j + .5, len(CELLS) - .72 - i,
                            f"{s}/{e}/{t}", ha="center", va="center",
                            fontsize=7, color="#444")
            ax.set_xlim(0, len(budgets)); ax.set_ylim(0, len(CELLS))
            ax.set_xticks([j + .5 for j in range(len(budgets))])
            ax.set_xticklabels([f"B={b}" for b in budgets])
            ax.set_yticks([len(CELLS) - .5 - i for i in range(len(CELLS))])
            ax.set_yticklabels(CELLS)
            ax.xaxis.tick_top(); ax.tick_params(length=0)
            ax.set_title("SMAC vs EZR by task class and budget\n"
                         "small numbers = smac / ezr / tie task counts",
                         fontsize=9, pad=24)
            for sp in ax.spines.values():
                sp.set_visible(False)
            fig.tight_layout()
            fig.savefig(f"{args.out}.png", dpi=200)
            print(f"wrote {args.out}.png")
        except ImportError:
            print("matplotlib not installed; skipped the figure "
                  "(pip install matplotlib)")


if __name__ == "__main__":
    main()