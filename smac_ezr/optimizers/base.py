"""The plug-in contract.

An optimizer receives:
  ds    - the Dataset (column metadata; .pool for pool-based methods)
  gate  - the BudgetGate; the ONLY way to score a row, and it counts
  seed  - the run's random seed

and returns the row dict it declares as its incumbent, or None to let the
runner fall back to gate.best_config.

Deliberately absent from this contract: the ConfigurationSpace.  It is a
SMAC/TPE concern and is built inside those adapters, so pool-based methods
(EZR, SWAY, LINE, Random) never import ConfigSpace at all.
"""

from __future__ import annotations

from typing import Protocol

from data import Dataset
from oracle import BudgetGate


class Optimizer(Protocol):
    name: str

    def run(self, ds: Dataset, gate: BudgetGate, seed: int) -> dict | None:
        ...


REGISTRY: dict[str, type] = {}


def register(cls):
    """Class decorator: make an optimizer selectable by name from the CLI."""
    REGISTRY[cls.name] = cls
    return cls