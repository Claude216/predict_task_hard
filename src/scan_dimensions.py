#!/usr/bin/env python3
"""Table of per-task dimensionality for every MOOT csv under a root.

Column roles come from ezr itself (ezr.Cols on the header row), so the `+ - ! X`
naming convention is never re-derived here:

  x    independent columns -- the dimensionality asked for, objectives excluded
  y    objectives (name ends + - or !)
  ign  ignored columns (name ends X); excluded from x, but note that DRR's own
       R counts them, since drr_upstream only strips + and - columns
  x+ign  what drr_upstream reports as R, for comparison with dataset_results.csv

Usage:
    python src/scan_dimensions.py [--root data/moot/optimize] [--csv out.csv]
                                  [--sort name|x|rows]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import ezr  # noqa: E402  (needs the repo root on sys.path first)


def scan(path, root):
    """Row/column counts for one csv, using ezr's own header parsing."""
    src = ezr.csv(path)
    header = next(src)
    cols = ezr.Cols(header)
    n_rows = sum(1 for _ in src)

    xs = cols.xs
    return {
        "task": path.stem,
        "group": path.parent.relative_to(root).as_posix(),
        "rows": n_rows,
        "cols": len(cols.all),
        "x": len(xs),
        "y": len(cols.ys),
        "ign": len(cols.all) - len(xs) - len(cols.ys),
        "x+ign": len(cols.all) - len(cols.ys),
        "x_num": sum(1 for c in xs if isinstance(c, ezr.Num)),
        "x_sym": sum(1 for c in xs if isinstance(c, ezr.Sym)),
    }


FIELDS = ["group", "task", "rows", "cols", "x", "y", "ign", "x+ign", "x_num", "x_sym"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/moot/optimize")
    ap.add_argument("--csv", help="also write the table to this path")
    ap.add_argument("--sort", choices=["name", "x", "rows"], default="name")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        sys.exit(f"root not found: {root}")

    rows, failures = [], []
    for path in sorted(root.rglob("*.csv")):
        try:
            rows.append(scan(path, root))
        except Exception as e:  # skip-and-log: one bad csv must not kill the scan
            failures.append(f"{path}: {type(e).__name__}: {e}")

    if not rows:
        sys.exit(f"no csv files under {root}")

    key = {"name": lambda r: (r["group"], r["task"]),
           "x": lambda r: (-r["x"], r["group"], r["task"]),
           "rows": lambda r: (-r["rows"], r["group"], r["task"])}[args.sort]
    rows.sort(key=key)

    widths = {f: max(len(f), max(len(str(r[f])) for r in rows)) for f in FIELDS}
    print("  ".join(f.ljust(widths[f]) if f in ("group", "task") else f.rjust(widths[f])
                    for f in FIELDS))
    print("  ".join("-" * widths[f] for f in FIELDS))
    for r in rows:
        print("  ".join(str(r[f]).ljust(widths[f]) if f in ("group", "task")
                        else str(r[f]).rjust(widths[f]) for f in FIELDS))

    xs = [r["x"] for r in rows]
    print(f"\n{len(rows)} tasks | x ranges {min(xs)}..{max(xs)}, "
          f"median {sorted(xs)[len(xs) // 2]} | "
          f"{sum(1 for r in rows if r['ign'])} tasks have ignored (X) columns")

    if args.csv:
        import csv as csv_mod
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            writer = csv_mod.DictWriter(fh, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {args.csv}")

    for f in failures:
        print(f"FAILED: {f}", file=sys.stderr)


if __name__ == "__main__":
    main()
