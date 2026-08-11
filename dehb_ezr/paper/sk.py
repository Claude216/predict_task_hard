"""The paper's winner rule (§3.6): Scott-Knott ranking with Cliff's delta.

    "Scott-Knott sorts the list of experiments ... by their median score. After
     the sorting, it then splits the list into two sub-lists. The goal for such
     a split is to maximize the expected value of differences in the observed
     performances before and after division ...
        E(D) = |l1|/|l| * abs(mean1 - mean)^2 + |l2|/|l| * abs(mean2 - mean)^2
     ... Scott-Knott then implements some statistical hypothesis tests to check
     whether the division is useful or not ... hypothesis test H is the Cliff's
     delta non-parametric effect size measure ... The division passes the
     hypothesis test if it is not a 'small' effect (Delta >= 0.147)."

WHY NOT REUSE stats.top(). The repo's top() is the same Scott-Knott skeleton but
strictly more conservative: it requires Cliff's delta AND a Kolmogorov-Smirnov
test, uses delta=0.195, sorts by mean, and adds an `eps` minimum-difference
floor. Every one of those makes ties more likely, so it would report more green
than the paper does. This module implements the paper's rule as printed, and
stats.py is left untouched.

WITH TWO TREATMENTS Scott-Knott collapses to a single test: sort the two by
median, ask whether Cliff's delta reaches 0.147. The general recursive form is
implemented anyway so a third arm can be added without rewriting the rule.
"""

from __future__ import annotations

import numpy as np

SMALL = 0.147          # the paper's threshold: below this is a "small" effect


def cliffs_delta(a, b) -> float:
    """(#a>b - #a<b) / (|a||b|). Symmetric in magnitude, sign follows a vs b."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    gt = int((a[:, None] > b[None, :]).sum())
    lt = int((a[:, None] < b[None, :]).sum())
    return (gt - lt) / (len(a) * len(b))


def _useful(left: list[float], right: list[float]) -> bool:
    """The paper's hypothesis test H: the split is real if the effect is not small."""
    return abs(cliffs_delta(left, right)) >= SMALL


def _best_split(groups: list[tuple[str, list[float]]]) -> int:
    """Index that maximises E(D). 0 means no split was evaluated."""
    flat = [v for _, vs in groups for v in vs]
    mu = float(np.mean(flat))
    best_i, best_score = 0, -1.0
    for i in range(1, len(groups)):
        l1 = [v for _, vs in groups[:i] for v in vs]
        l2 = [v for _, vs in groups[i:] for v in vs]
        s = (len(l1) * (np.mean(l1) - mu) ** 2
             + len(l2) * (np.mean(l2) - mu) ** 2) / len(flat)
        if s > best_score:
            best_i, best_score = i, s
    return best_i


def scott_knott(groups: dict[str, list[float]]) -> dict[str, int]:
    """name -> rank, 0 = best. Lower scores are better (D2H is a distance)."""
    ordered = sorted(groups.items(), key=lambda kv: float(np.median(kv[1])))

    ranks: dict[str, int] = {}

    def recurse(sub: list[tuple[str, list[float]]], rank: int) -> int:
        if len(sub) == 1:
            ranks[sub[0][0]] = rank
            return rank + 1
        i = _best_split(sub)
        left = [v for _, vs in sub[:i] for v in vs]
        right = [v for _, vs in sub[i:] for v in vs]
        if i == 0 or not _useful(left, right):
            for name, _ in sub:            # indistinguishable: one shared rank
                ranks[name] = rank
            return rank + 1
        nxt = recurse(sub[:i], rank)
        return recurse(sub[i:], nxt)

    recurse(ordered, 0)
    return ranks


def verdict(lite: list[float], dehb: list[float]) -> str:
    """Fig. 4's two classes.

    'DEHB > LITE'  -- DEHB lands in a strictly better Scott-Knott rank
    'DEHB == LITE' -- anything else, INCLUDING LITE winning. The paper's legend
                      has only these two colours, so a LITE win is not a third
                      class; it is simply not a DEHB win.
    """
    r = scott_knott({"lite": lite, "dehb": dehb})
    return "DEHB > LITE" if r["dehb"] < r["lite"] else "DEHB == LITE"
