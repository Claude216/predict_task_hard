"""The paper's D2H (§3.5, Eqs. 8 and the Zitzler indicator above it).

    worse(A,B) = loss(A,B) > loss(B,A)
    loss(A,B)  = sum_j -exp(delta(j,A,B,n)) / n
    delta(j,A,B,n) = w_j * (o_{j,A} - o_{j,B}) / n
    w_j in {-1,+1} for minimize / maximize
    D2H(s) = i / |Z|,  Z = all options ranked best-to-worst, i = index of s

"jumping from B to A loses more than jumping from A to B" is a continuous
(rather than binary) dominance test: it weighs *how much* each objective moves,
so a candidate that is slightly worse on one goal and far better on another is
not discarded the way strict Pareto dominance would discard it.

OBJECTIVES ARE NORMALISED TO [0,1] FIRST, over the set Z being ranked. Without
that, exp() is dominated by whichever metric happens to have the largest raw
scale -- SA is a percentage that can reach -1e3 while accuracy sits in [0,1] --
and the indicator would be reporting units, not preference.

WHAT GOES IN Z. The paper says "all evaluated examples from all techniques".
Ranking all 20x(30+3000) evaluations per dataset is O(n^2) over ~60,000 points
and is not the comparison of interest anyway. Z here is the set of RUN
INCUMBENTS -- the 20 LITE answers and the 20 DEHB answers -- which is precisely
the set the Scott-Knott test then operates on, and is what "how close is each
example to the best available in the set" means once each run has produced one
answer.
"""

from __future__ import annotations

import numpy as np


def _normalise(mat: np.ndarray) -> np.ndarray:
    """Column-wise min-max over Z. Constant columns collapse to 0 (no signal)."""
    lo = mat.min(axis=0)
    hi = mat.max(axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    out = (mat - lo) / span
    return np.where(hi > lo, out, 0.0)


def losses(mat: np.ndarray, w: np.ndarray) -> np.ndarray:
    """L[a, b] = loss(a, b), vectorised over every pair."""
    n = mat.shape[1]
    # delta[a,b,j] = w_j * (o_ja - o_jb) / n
    delta = (mat[:, None, :] - mat[None, :, :]) * w[None, None, :] / n
    return (-np.exp(delta)).sum(axis=2) / n


def rank(metric_rows: list[dict], names: list[str], weights: dict[str, int]) -> np.ndarray:
    """Order the options best-to-worst. Returns positions (0 = best).

    Score each option by how often it is `worse` than the others -- a Copeland
    count over the continuous-domination relation. That is a total order even
    when the relation itself has cycles, which strict Pareto sorting is not.

    TIES GET THE AVERAGE POSITION, and that matters. Two runs that produce
    byte-identical metrics must receive identical D2H; breaking such ties by
    array position would hand the better rank to whichever arm happens to be
    appended first, and since run_one always appends lite before dehb that bias
    would run one way every time. On iris both arms reach a perfect classifier
    on some seeds, so this is not a hypothetical.
    """
    mat = np.array([[float(r[k]) for k in names] for r in metric_rows], dtype=float)
    # inf shows up when MRE is undefined for a failed config; clamp so exp() and
    # the min-max both stay finite. It is already the worst possible value.
    finite = mat[np.isfinite(mat)]
    ceiling = (finite.max() if finite.size else 1.0) * 10 + 1.0
    mat = np.where(np.isfinite(mat), mat, ceiling)

    w = np.array([weights[k] for k in names], dtype=float)
    L = losses(_normalise(mat), w)
    worse_count = (L > L.T).sum(axis=1)          # times this option loses

    order = np.argsort(worse_count, kind="stable")
    pos = np.empty(len(mat), dtype=float)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and worse_count[order[j + 1]] == worse_count[order[i]]:
            j += 1
        pos[order[i:j + 1]] = (i + j) / 2.0      # average position over the tie
        i = j + 1
    return pos


def d2h(metric_rows: list[dict], names: list[str], weights: dict[str, int]) -> np.ndarray:
    """Eq. 8: D2H = i/|Z|, in (0, 1]. Lower is better."""
    pos = rank(metric_rows, names, weights)
    return (pos + 1) / len(metric_rows)
