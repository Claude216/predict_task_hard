"""Print the sweep rows where an estimated dimension exceeds the column count.

I > R is not a dimension: it means the slope was read off a stretch of the
log-log curve that is not a scaling region. For I_fit that is rare (a fit over
0.01<=C<=0.2 is always well populated); for I_maxgrad it is common, because
Algorithm 1's max() lands in the sparse-count end of the curve where the slope
is set by grid spacing rather than geometry.

    conda run -n drr python drr_mine/show_over_r.py
    conda run -n drr python drr_mine/show_over_r.py --which fit --csv path.csv
"""

import argparse
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default=os.path.join(HERE, "results", "moot_drr.csv"))
    ap.add_argument("--which", default="both", choices=["fit", "maxgrad", "both"])
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    for which in (["fit", "maxgrad"] if args.which == "both" else [args.which]):
        icol, dcol = f"I_{which}", f"drr_{which}"
        over = df[df[icol] > df.R].sort_values(icol, ascending=False)
        print(f"\n=== I_{which} > R : {len(over)} of {len(df)} tasks ===")
        if over.empty:
            print("  (none)")
            continue
        print(f"{'task':<40} {'R':>4} {icol:>10} {dcol:>11} {'slope_iqr':>10}")
        for _, r in over.iterrows():
            drr = "nan (guarded)" if pd.isna(r[dcol]) else f"{r[dcol]:>11.3f}"
            iqr = r.get("slope_iqr", float("nan"))
            iqr = "nan" if pd.isna(iqr) else f"{iqr:.3f}"
            print(f"{r.task:<40} {r.R:>4} {r[icol]:>10.2f} {drr:>11} {iqr:>10}")
        print(f"  worst overshoot: {(over[icol] / over.R).max():.1f}x R")


if __name__ == "__main__":
    main()
