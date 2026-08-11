"""Reproduction of Fig. 4 of Lustosa & Menzies (2026), "Less Noise, More Signal".

WHAT THE PAPER ACTUALLY DOES, and why this is a separate rig from dehb_ezr/.

The sibling study in dehb_ezr/ optimizes MOOT's decision columns against MOOT's
objective columns. The paper does something else (§3.3.1, §3.4): it tunes a
RandomForest's five hyperparameters -- 1,280,000 combinations -- where each
dataset supplies `X -> first dependent column` as the forest's training data and
every other dependent is dropped. The score is d2h over PREDICTION-QUALITY
metrics, not over the table's own objectives. Only the budgets coincide
(LITE 30 evaluations, DEHB 3000).

That difference is load-bearing: smac_ezr's Dataset.load requires a `+`/`-`
goal, so it raises on iris, heart, gamma and default -- the whole non-SE half of
Fig. 4 is unreachable without adopting the paper's framing.

Everything here is new code under dehb_ezr/. ezr.py, smac_ezr/ and the repo-root
stats.py are imported or consulted, never modified.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# ezr.py lives at the repo root and is imported (never edited) for LITE.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
