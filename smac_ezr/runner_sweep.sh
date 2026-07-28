#!/bin/bash
#SBATCH --job-name=smac_ezr
#SBATCH --partition=rtx4060ti8g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --exclusive=user
#SBATCH --output=./logs/smac_ezr_%j.out
#SBATCH --error=./logs/smac_ezr_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=lli66@ncsu.edu
#
# One budget per job, one output directory per budget:
#   sbatch --job-name=smac_ezr_b30  --export=ALL,BUD=30  run_sweep.sh
#
# The budgets are independent jobs but stay comparable, because the oracle is
# deterministic (fixed seed, full-table fit), the d2h normalisation bounds come
# from the table's true y values, and eps is task-level and budget-invariant.
# That only holds if every job uses the SAME conda env and sklearn version.

BUD="${BUD:-30}"

# ---- Paths on the REMOTE server ----
PROJECT_DIR="/home/lli66/aise26/predict_task_hard/smac_ezr"
MOOT_ROOT="/home/lli66/aise26/predict_task_hard/data/moot/optimize"
OUT_ROOT="${PROJECT_DIR}/results_b${BUD}"

mkdir -p "${PROJECT_DIR}/logs" "${OUT_ROOT}"
cd "${PROJECT_DIR}" || exit 1

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# ---- Environment ----
# smac and ConfigSpace are not in the system python, so the conda env is
# mandatory. Build it on the LOGIN node: compute nodes usually have no network.
# source "${HOME}/miniconda3/etc/profile.d/conda.sh"

# ezr.py is loaded by file rather than by adding its directory to sys.path:
# ezr's own repo has top-level modules (stats.py, data.py) whose names collide
# with this project's, and a shadowed import would fail silently, not loudly.
export EZR_PATH="${PROJECT_DIR}/optimizers/ezr.py"

# SMAC writes one directory per run -- thousands across a sweep. Keep that churn
# on node-local disk, never on the shared filesystem, where it is slow and can
# exhaust an inode quota. Nothing reads it: BudgetGate is the record of truth.
export SMAC_OUT="/tmp/smac_${SLURM_JOB_ID}"
mkdir -p "${SMAC_OUT}"

# Parallelism lives at the task level -- batch.py runs one process per dataset.
# Pin every numeric library to one thread, or ${N_JOBS} workers each spawning
# ${N_JOBS} threads oversubscribes the node by a factor of ${N_JOBS}.
N_JOBS="${SLURM_CPUS_PER_TASK:-16}"
export SMOOT_RF_JOBS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

log "Project:  ${PROJECT_DIR}"
log "MOOT:     ${MOOT_ROOT}"
log "Output:   ${OUT_ROOT}"
log "Budget:   ${BUD}"
log "Workers:  ${N_JOBS}"
log "Commit:   $(git rev-parse --short HEAD 2>/dev/null || echo 'not a git repo')"
python -c "import sys, smac, sklearn, pandas; \
print('python ', sys.version.split()[0]); \
print('smac   ', smac.__file__); \
print('sklearn', sklearn.__version__); \
print('pandas ', pandas.__version__)"

if [[ ! -f "${EZR_PATH}" ]]; then
  log "FATAL: EZR_PATH not found: ${EZR_PATH}"
  exit 1
fi

# ---- Run ----
# batch.py discovers the tables itself, pools across ${N_JOBS} processes, keeps
# a single failing table from taking the batch down, and skips any task whose
# {tid}.json already exists -- so resubmitting this script resumes.
log "Starting sweep at B=${BUD} ..."
python batch.py "${MOOT_ROOT}" \
  --out "${OUT_ROOT}" \
  --exclude-dir \
  --optimizers smac ezr random \
  --seeds 20 \
  --budgets "${BUD}" \
  --workers "${N_JOBS}"
STATUS=$?

rm -rf "${SMAC_OUT}"

DONE=$(ls -1 "${OUT_ROOT}"/*.json 2>/dev/null | wc -l)
ERRS=$(ls -1 "${OUT_ROOT}"/*.error.txt 2>/dev/null | wc -l)
log "Completed tasks: ${DONE}   error files: ${ERRS}"
if [[ ${ERRS} -gt 0 ]]; then
  log "Errors (2 are expected: the COVID and wallpaper tables lose every row"
  log "to missing values, and both are outside the SE task families anyway):"
  for f in "${OUT_ROOT}"/*.error.txt; do
    log "  $(basename "${f}"): $(tail -1 "${f}")"
  done
fi

if [[ ${STATUS} -ne 0 ]]; then
  log "batch.py exited ${STATUS}"
  exit ${STATUS}
fi
log "Done."