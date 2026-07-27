"""MOOT table parsing and encoding.  Knows nothing about oracles or optimizers.

Invariant used everywhere downstream: a *row* is a plain dict mapping decision
column name -> raw value (the value as it appears in the CSV).  Encoding to a
float vector happens only in Dataset.encode(), which is the oracle's business.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import sys


@dataclass
class Dataset:
    """One MOOT table: columns, rows, and the paper's cheap structural attributes."""

    path: str
    df: pd.DataFrame
    x_cols: list[str]
    goals: dict[str, str]                    # goal column -> "min" | "max"
    levels: dict[str, list] = field(default_factory=dict)   # symbolic col -> ordered levels

    # ------------------------------------------------------------------ #
    @classmethod
    def load(cls, path: str) -> "Dataset":
        """MOOT header convention: '+' maximise, '-' minimise, 'X' ignore,
        anything else is a decision variable."""
        df = pd.read_csv(path)



        df.columns = [c.strip() for c in df.columns]
        for c in df.columns:
            if not pd.api.types.is_numeric_dtype(df[c]):
                df[c] = df[c].astype(str).str.strip()

        x_cols, goals = [], {}
        for c in df.columns:
            if c.endswith("X"):
                continue
            if c.endswith("+"):
                goals[c] = "max"
            elif c.endswith("-"):
                goals[c] = "min"
            else:
                x_cols.append(c)

        # MOOT convention, the same rule ezr.Col() applies: an uppercase initial
        # means numeric, lowercase means symbolic.  The header is authoritative;
        # dtype inference is not, because one "?" missing marker is enough to
        # make pandas read a numeric column as strings.  Goals are always
        # numeric -- d2h arithmetic requires it.
        symbolic = [c for c in x_cols if not c[0].isupper()]
        numeric = [c for c in x_cols if c[0].isupper()] + list(goals)
        for c in goals:
            if not c[0].isupper():
                print(f"warning: goal {c!r} has a lowercase initial; "
                      f"coercing to numeric anyway", file=sys.stderr)

        for c in numeric:
            df[c] = pd.to_numeric(df[c].replace("?", pd.NA), errors="coerce")
        for c in symbolic:
            # astype("string") not astype(str): the latter turns NaN into the
            # literal "nan" and it survives dropna as a level.
            df[c] = df[c].astype("string").str.strip().replace("?", pd.NA)

        df = df.dropna(subset=x_cols + list(goals)).reset_index(drop=True)
        if not goals:
            raise ValueError("no goal columns: nothing ends in '+' or '-'")
        if not x_cols:
            raise ValueError("no decision columns")
        if len(df) == 0:
            raise ValueError("no complete rows left after dropping missing values")

        levels = {c: sorted(df[c].unique().tolist()) for c in symbolic}
        return cls(path=path, df=df, x_cols=x_cols, goals=goals, levels=levels)

    # ------------------------------------------------------------------ #
    @property
    def y_cols(self) -> list[str]:
        return list(self.goals)

    @property
    def is_multi(self) -> bool:
        return len(self.goals) > 1

    @property
    def pool(self) -> list[dict]:
        if getattr(self, "_pool", None) is None:
            self._pool = self.df[self.x_cols].to_dict("records")
        return self._pool

    def domain(self, col: str) -> list:
        """The values OBSERVED for this column.  Per VI-D, MOOT carries no
        real-world ranges, so this set *is* the variable's domain."""
        return self.levels.get(col) or sorted(self.df[col].unique().tolist())

    # ------------------------------------------------------------------ #
    def encode(self, rows: list[dict] | dict) -> np.ndarray:
        """Raw row dict(s) -> float matrix for the RF.  Symbolic values become
        their index in the ordered level list."""
        if isinstance(rows, dict):
            rows = [rows]
        out = np.empty((len(rows), len(self.x_cols)), dtype=float)
        for i, row in enumerate(rows):
            for j, c in enumerate(self.x_cols):
                v = row[c]
                out[i, j] = self.levels[c].index(v) if c in self.levels else float(v)
        return out

    # ---------------- paper's cheap structural attributes (VI-D) -------- #
    def space_size(self) -> float:
        """sum log2 |Xi|."""
        return float(sum(np.log2(self.df[c].nunique()) for c in self.x_cols))

    def input_shape(self) -> str:
        """binary/SAT, large-numeric, or small-numeric."""
        n_bin = sum(self.df[c].nunique() == 2 for c in self.x_cols)
        if n_bin / len(self.x_cols) >= 0.80:
            return "binary/SAT"
        return "large-numeric" if self.space_size() >= 40 else "small-numeric"

    def describe(self) -> dict:
        n_distinct = len(self.df[self.x_cols].drop_duplicates())
        return dict(
            task=self.path,
            rows=len(self.df),
            n_decisions=len(self.x_cols),
            n_goals=len(self.goals),
            objective="multi" if self.is_multi else "single",
            space_size=round(self.space_size(), 2),
            input_shape=self.input_shape(),
            # 41% of nasa93dem's rows repeat in x-space: the same 22 COCOMO
            # ratings with different effort. Where this is high, y is not a
            # function of x, so NO oracle can be accurate -- not the RF, not
            # nearest-neighbour lookup. Irreducible, not a tuning problem.
            x_dup_rate=round(1 - n_distinct / len(self.df), 4),
            # log2(distinct rows) - sum log2|Xi|.  0 means the table enumerates
            # the whole space; large negative means it barely samples it, and
            # the RF is extrapolating almost everywhere.
            coverage=round(float(np.log2(n_distinct)) - self.space_size(), 2),
        )
    
    def key(self, row: dict) -> tuple:
            """Encoded tuple, comparable across pool rows and optimizer proposals."""
            return tuple(self.encode(row)[0])

    @property
    def pool_keys(self) -> set[tuple]:
        if getattr(self, "_pool_keys", None) is None:
            self._pool_keys = {tuple(r) for r in self.encode(self.pool)}
        return self._pool_keys