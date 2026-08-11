"""DEHB (HEAVY, 3000 evals) vs EZR (LITE, 30 labels) on MOOT, read against DRR.

Nothing here re-implements the harness. `smac_ezr/` already owns MOOT parsing
(`data.Dataset`), the frozen-RF evaluation surface (`oracle.Oracle`), the
budget gate, the debugged EZR adapter, and the statistical verdict
(`stats.top` via `metrics.match`). This package adds only what is new: a
tree-count fidelity layer, a DEHB adapter, a ConfigSpace-uniform floor, a DRR
interface, and the two figures.

See shared.py for how smac_ezr is put on the path.
"""
