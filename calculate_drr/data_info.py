# Non-SE datasets from Lustosa & Menzies (2025), Table 1 / Figs 3-4.
# Paths relative to repo root. Verified against moot @ v0.9.1 (Oct 2025).
# R = raw dimension count reported in the paper's Table 2.

NONSE_PATHS = [
    "external/moot/fairness/adult.csv",
    "external/moot/fairness/bank.csv",
    "external/moot/fairness/german.csv",
    "external/moot/classify/iris.csv",
    "external/moot/classify/heart.c.csv",
    "external/moot/optimize/misc/Wine_quality.csv",
    "external/uci/gamma_telescope.csv",
    "external/uci/default.csv",
    "external/uci/power_consumption.csv",
]

UCI_IDS = {"gamma telescope": 159, "power consumption": 235, "default": 350}

PREPROCESSING = {
    # applied before computing intrinsic dimensionality
    "power consumption": "drop Date, Time; target = Sub_metering_1; '?' = missing",
    "default":           "drop ID column (else 24 x-cols, not 23)",
}