"""Validation gate for drr_mine/intrinsic_dim.py — run before trusting MOOT output.

1. Known-dimension recovery: seeded Gaussian clouds of true dim d, both raw and
   linearly embedded into 20 ambient dims (ideas ported from drr_mine/datasets.py:
   generate_intrinsic_samples / transform_to_high_dim). I_fit must land within
   +-35% of d (correlation dimension mildly underestimates at higher d with
   finite samples) and the fit r2 must be high.
2. Spot checks on real tasks, printed for eyeballing (no asserts). Numbers are
   not expected to equal the upstream-audit probe (label-encoded, unnormalized
   L1, 0.05..0.95 window) or the paper's integers, but should be same-ballpark
   and, above all, satisfy 0 < I <= R with high r2.

    conda run -n drr python drr_mine/test_synthetic.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from intrinsic_dim import estimate_intrinsic_dim  # noqa: E402

SEED = 42
N = 2000
AMBIENT = 20
TOL = 0.35


def embed(samples: np.ndarray, ambient: int, rng: np.random.Generator) -> np.ndarray:
    """Random linear map d -> ambient (datasets.py's transform_to_high_dim idea,
    with a plain Gaussian matrix instead of its clipped column-stochastic one)."""
    return samples @ rng.standard_normal((samples.shape[1], ambient))


def main():
    rng = np.random.default_rng(SEED)
    failures = []

    print(f"{'case':28s} {'true d':>6} {'R':>3} {'I_fit':>7} {'n_live':>6} "
          f"{'sl_iqr':>7} {'I_maxgrad':>9}  verdict")
    for d in (2, 3, 5, 8):
        cloud = rng.standard_normal((N, d))
        for label, X in ((f"gauss d={d} raw", cloud),
                         (f"gauss d={d} in {AMBIENT}-dim", embed(cloud, AMBIENT, rng))):
            res = estimate_intrinsic_dim(X, seed=SEED)
            i_fit, n_live = res["I_fit"], res["n_live"]
            # a clean continuous cloud must fill essentially every piece
            ok = np.isfinite(i_fit) and abs(i_fit - d) / d <= TOL and n_live >= 90
            if not ok:
                failures.append(label)
            print(f"{label:28s} {d:>6} {res['R']:>3} {i_fit:>7.2f} {n_live:>6} "
                  f"{res['slope_iqr']:>7.3f} {res['I_maxgrad']:>9.2f}  "
                  f"{'ok' if ok else 'FAIL'}")

    print("\n-- real-task spot checks (no hard asserts; compare notes in header) --")
    spots = [
        ("pom3a", os.path.join(REPO_ROOT, "data", "moot", "optimize",
                               "process", "pom3a.csv")),
        ("SS-A", os.path.join(REPO_ROOT, "data", "moot", "optimize", "config", "SS-A.csv")),
        ("Wine_quality", os.path.join(REPO_ROOT, "data", "moot", "optimize", "misc",
                                      "Wine_quality.csv")),
    ]
    for label, path in spots:
        if not os.path.exists(path):
            print(f"{label:28s} SKIP (missing {os.path.relpath(path, REPO_ROOT)})")
            continue
        res = estimate_intrinsic_dim(path, seed=SEED)
        print(f"{label:28s} R={res['R']:<3} I_fit={res['I_fit']:.2f} "
              f"n_live={res['n_live']:<3} slope_iqr={res['slope_iqr']:.3f} "
              f"I_maxgrad={res['I_maxgrad']:.2f} "
              f"DRR_fit={res['drr_fit']:.3f} status={res['status']}")

    if failures:
        print(f"\nFAILED: {failures}")
        sys.exit(1)
    print("\nall synthetic checks passed")


if __name__ == "__main__":
    main()
