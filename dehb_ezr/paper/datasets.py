"""The Table 1 dataset registry: 49 datasets — 37 SE + 12 non-SE.

The paper studies 50. We locate all of them except diabetes, and add SS-L
(present in MOOT, absent from Table 2's count of 23 SS datasets and from
Fig. 6.C), which lands us at 49.


R (the x-axis of Fig. 3 and Fig. 4) is: every column MINUS every dependent
column, where a dependent is one whose name ends in `+`, `-` or `!`. MOOT's
`X`-suffix "ignore" columns are therefore KEPT and counted, which is what
reproduces the paper's Table 2 for xomo (24 features + 3 ignored = 27) and adult
(13 + 1 = 14).

`paper_R` records Table 2's value so the loader can assert against it. Where it
disagrees the reason is known and recorded:

  nasa93dem   26 vs 25   one column off; MOOT's idX/centerX/YearX/MonthsX
  SCRUM      124 vs 128  "128" is the generator's nominal name, not a count
  FFM-250    256 vs 250  likewise "250"
  behavioural  -1..-3    the paper counted the raw pre-MOOT Kaggle column sets

Membership calls, per the study decisions:
  * diabetes is DROPPED -- nothing on disk has its R=20 (moot's diabetes.csv is
    Pima with 8 features; external/uci/diabetes is Kahn's 4-field time series)
  * heart disease = heart.c.csv (Cleveland, 303 rows); its 13 features match
    Table 2 exactly. heart.statlog.csv and heart.h.csv are the other variants
  * SS-L is INCLUDED, though Table 2 counts 23 SS datasets and Fig. 6.C omits it
  * Wine Quality is coloured SE: Table 1 lists it under Non-SE, but Figs 6.A/6.C
    treat it as SE and it ships under moot/optimize/
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Entry:
    label: str
    path: str            # relative to the repo root
    is_se: bool
    paper_R: int | None  # Table 2's value, None where the paper lists no number


_SS = [Entry(f"SS-{c}", f"data/moot/optimize/config/SS-{c}.csv", True, None)
       for c in "ABCDEFGHIJKLMNOPQRSTUVWX"]

DATASETS: list[Entry] = _SS + [
    # ---------------------------------------------------------------- SE
    Entry("rs-6d-c3-obj2", "data/moot/optimize/config/rs-6d-c3_obj2.csv", True, 6),
    Entry("Pom3a", "data/moot/optimize/process/pom3a.csv", True, 9),
    Entry("pom3d", "data/moot/optimize/process/pom3d.csv", True, 9),
    Entry("Xomo Flight", "data/moot/optimize/process/xomo_flight.csv", True, 27),
    Entry("Xomo Ground", "data/moot/optimize/process/xomo_ground.csv", True, 27),
    Entry("Xomo OSP", "data/moot/optimize/process/xomo_osp.csv", True, 27),
    Entry("Xomo OSP2", "data/moot/optimize/process/xomo_osp2.csv", True, 27),
    Entry("nasa93dem", "data/moot/optimize/process/nasa93dem.csv", True, 25),
    Entry("Health-Easy", "data/moot/optimize/hpo/Health-Commits0000.csv", True, 5),
    Entry("Health-Hard", "data/moot/optimize/hpo/Health-ClosedIssues0000.csv", True, 5),
    Entry("SCRUM", "data/moot/optimize/binary_config/Scrum1k.csv", True, 128),
    Entry("FFM-250", "data/moot/optimize/binary_config/FFM-250-50-0.50-SAT-1.csv", True, 250),
    Entry("Wine Quality", "data/moot/optimize/misc/Wine_quality.csv", True, 10),
    # ------------------------------------------------------------ non-SE
    Entry("iris", "external/moot/classify/iris.csv", False, 4),
    Entry("heart disease", "external/moot/classify/heart.c.csv", False, 13),
    Entry("adult", "external/moot/fairness/adult.csv", False, 14),
    Entry("german credit", "external/moot/fairness/german.csv", False, 20),
    Entry("bank marketing", "external/moot/fairness/bank.csv", False, 16),
    Entry("gamma telescope", "external/uci/gamma_telescope.csv", False, 10),
    Entry("default", "external/uci/default.csv", False, 23),
    Entry("power consumption", "external/uci/power_consumption.csv", False, 6),
    Entry("Player Statistics",
          "data/moot/optimize/behavior_data/player_statistics_cleaned_final.csv", False, 27),
    Entry("Student Dropout",
          "data/moot/optimize/behavior_data/student_dropout.csv", False, 34),
    Entry("Employee Attrition",
          "data/moot/optimize/behavior_data/WA_Fn-UseC_-HR-Employee-Attrition.csv", False, 35),
    Entry("All Players",
          "data/moot/optimize/behavior_data/all_players.csv", False, 57),
]

# Deliberately absent. Kept as data so RESULTS.md can state it rather than
# leaving a silent gap in a 49-point figure.
DROPPED = {"diabetes": "paper R=20; moot diabetes.csv is Pima (8 features) and "
                       "external/uci/diabetes is Kahn's 4-field time series"}

BY_LABEL = {e.label: e for e in DATASETS}


def task_id(label: str) -> str:
    """Filesystem-safe id: 'Xomo OSP2' -> 'Xomo_OSP2'."""
    return label.replace(" ", "_").replace("/", "_")
