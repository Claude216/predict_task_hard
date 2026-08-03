#!/usr/bin/env python3
"""Apply MOOT column-name conventions to the UCI csvs under external/uci/.

MOOT encodes a column's type in its NAME: an uppercase initial letter means
numeric (Num), anything else means symbolic (Sym); trailing + - ! mark goals and
trailing X marks an ignored column. The csvs collected straight from the UCI
repository kept UCI's own names, so their types were being read wrongly:

  gamma_telescope  every feature (fLength, fWidth, ...) is a continuous float
                   but is lowercase-initial, so all 10 were typed as Sym --
                   18,643 distinct floats became 18,643 distinct *symbols*,
                   collapsing the distance spectrum to 5 values.
  default          every column is X1..X23 (uppercase), so all 23 were typed
                   Num, including UCI's categorical X2=SEX, X3=EDUCATION,
                   X4=MARRIAGE. X6-X11 (PAY_*) are ordinal with a meaningful
                   order, so those correctly stay Num.
  power_consumption already conformant -- no change.

Only the header line of each file is rewritten; no data row is touched.

    conda run -n drr python calculate_drr/fix_uci_headers.py --dry-run
    conda run -n drr python calculate_drr/fix_uci_headers.py
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
UCI = os.path.join(REPO_ROOT, "external", "uci")

# old header -> new header, per file. Values not listed are left untouched.
RENAMES = {
    "gamma_telescope.csv": {
        # continuous floats -> uppercase initial so MOOT reads them as Num.
        # "class!" stays lowercase: it is a genuine symbolic klass (g / h).
        "fLength": "FLength", "fWidth": "FWidth", "fSize": "FSize",
        "fConc": "FConc", "fConc1": "FConc1", "fAsym": "FAsym",
        "fM3Long": "FM3Long", "fM3Trans": "FM3Trans", "fAlpha": "FAlpha",
        "fDist": "FDist",
    },
    "default.csv": {
        # UCI "default of credit card clients": X2=SEX, X3=EDUCATION,
        # X4=MARRIAGE are categorical codes, not quantities -> Sym.
        "X2": "x2_sex", "X3": "x3_education", "X4": "x4_marriage",
        # binary klass -> lowercase, matching moot's iris "class!" / heart "num!"
        "Y!": "y!",
    },
    "power_consumption.csv": {},
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="show the new headers, write nothing")
    ap.add_argument("--backup", action="store_true", help="keep a .orig copy beside each file")
    args = ap.parse_args()

    for fname, mapping in RENAMES.items():
        path = os.path.join(UCI, fname)
        if not os.path.exists(path):
            print(f"MISSING {path}")
            continue

        with open(path) as f:
            header = f.readline().rstrip("\n")
        cols = [c.strip() for c in header.split(",")]
        new_cols = [mapping.get(c, c) for c in cols]
        changed = [(a, b) for a, b in zip(cols, new_cols) if a != b]

        if not changed:
            print(f"{fname:<26} already conformant, no change")
            continue

        print(f"{fname:<26} {len(changed)} column(s) renamed: "
              + ", ".join(f"{a}->{b}" for a, b in changed[:6])
              + (" ..." if len(changed) > 6 else ""))
        if args.dry_run:
            continue

        if args.backup:
            shutil.copy2(path, path + ".orig")

        # rewrite the header line only, streaming the rest of the file through
        tmp = path + ".tmp"
        with open(path) as src, open(tmp, "w") as dst:
            src.readline()
            dst.write(",".join(new_cols) + "\n")
            shutil.copyfileobj(src, dst)
        os.replace(tmp, path)

    if args.dry_run:
        print("\ndry run: nothing written")


if __name__ == "__main__":
    sys.exit(main())
