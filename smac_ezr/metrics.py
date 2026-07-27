"""Scoring metrics.  GD / IGD / HV (RQ3) belong here too, once needed."""

from __future__ import annotations

import math

from stats import top          # 你贴的模块

EPS_BASE = 0.01

def task_eps(spread: float, base: float = EPS_BASE) -> float:
    """Task-level and budget-invariant.  If eps moved with budget, a 'the
    winner changed at B=100' finding could be an artifact of the gate."""
    return base * spread


def match(scores: dict[str, list[float]], eps: float) -> str:
    """One (task, budget) cell.  d2h is lower-better, hence reverse=False --
    passing a higher-better metric without reverse=True silently returns the
    worst method."""
    w = top(scores, reverse=False, eps=eps)
    return "tie" if len(w) > 1 else next(iter(w))

def rho(d2h_method: float, d2h_random: float, d2h_star: float) -> float:
    """Eq.2: fraction of the random-to-optimal regret removed.

    rho = 1 matches the best known configuration, rho = 0 matches blind
    sampling.  d2h_star must be pooled over ALL optimizers before this is
    reported, otherwise methods that find points better than the table's own
    best drive rho above 1 or blow up the denominator.
    """
    denom = d2h_random - d2h_star
    return math.nan if abs(denom) < 1e-12 else (d2h_random - d2h_method) / denom
