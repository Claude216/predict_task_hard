"""Put `smac_ezr/` on sys.path and re-export the pieces this study reuses.

smac_ezr's modules import each other with bare names (`from data import
Dataset`, `from optimizers.base import register`), so the directory itself has
to be a sys.path entry -- importing it as a package would not resolve those.
That is done here, once, so no other module in dehb_ezr has to think about it.

NOTHING under smac_ezr/ is modified by this study. The tree-count fidelity
layer lives in fidelity_oracle.py as a subclass, precisely so that the SMAC
study stays reproducible.

Deliberately NOT re-exported: optimizers.smac_opt. Importing it pulls in `smac`,
which is not installed in the `dmoot` env (and does not need to be). Its
`build_configspace` is 8 lines and is reproduced in configspace.py with a
pointer back to the original.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SMAC_EZR = REPO_ROOT / "smac_ezr"

for _p in (str(SMAC_EZR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ruff: noqa: E402
from data import Dataset                                    # MOOT table parsing
from oracle import BudgetExhausted, Oracle                   # frozen RF surrogate
from metrics import match, rho, task_eps, EPS_BASE           # verdict + eps
from stats import top, same                                  # Cliff's delta + KS

__all__ = [
    "REPO_ROOT", "SMAC_EZR",
    "Dataset", "Oracle", "BudgetExhausted",
    "match", "rho", "task_eps", "EPS_BASE",
    "top", "same",
]


def task_id(path, root) -> str:
    """`config/SS-A.csv` -> `config__SS-A`.

    Identical to smac_ezr/batch.py's rule and to what drr_interface.py writes,
    because the DRR join in analyze.py is on this string. A mismatch here shows
    up as tasks silently dropped from the figure.
    """
    rel = Path(path).resolve().relative_to(Path(root).resolve()).with_suffix("")
    return str(rel).replace("/", "__").replace("\\", "__")
