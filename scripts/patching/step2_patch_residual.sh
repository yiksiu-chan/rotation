#!/usr/bin/env bash
#SBATCH --job-name=patch-step2-residual
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --output=logs/patch-step2-%j.out

# Step 2: mean residual-stream patching at each prompt's frozen best
# position (from step 1), swept independently across all layers.
# Validates, tabulates (CSV) and plots automatically at the end -- no
# separate summarize/plot command needed.
#
# Requires a completed step1_find_best_position.sh run at --baseline.
# Override cluster resources at submission, e.g.
#   sbatch --partition=mypartition --account=myaccount scripts/patching/step2_patch_residual.sh
# On a workstation with a GPU (no SLURM), set PATCHING_ALLOW_LOCAL=1.
set -euo pipefail

ROOT="${PATCHING_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}}"
cd "$ROOT"
if [[ -z "${SLURM_JOB_ID:-}" && "${PATCHING_ALLOW_LOCAL:-0}" != 1 ]]; then
  echo 'Run inside a SLURM allocation or submit with sbatch. On a workstation only, set PATCHING_ALLOW_LOCAL=1.' >&2
  exit 1
fi
PATCHING_PYTHON="${PATCHING_PYTHON:-python}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MPLBACKEND=Agg HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1

BASELINE="${BASELINE:-results/patching/unpatched-half-20prompts-100}"
OUTPUT="${OUTPUT:-results/patching/mean-residual-best-position-filtered}"
BATCH_SIZE="${BATCH_SIZE:-8}"
VARIANTS="${VARIANTS:-}"

exec "$PATCHING_PYTHON" -u src/activation_patching/step2_patch_residual.py \
  --baseline "$BASELINE" --output "$OUTPUT" --batch-size "$BATCH_SIZE" \
  ${VARIANTS:+--variants "$VARIANTS"} "$@"
