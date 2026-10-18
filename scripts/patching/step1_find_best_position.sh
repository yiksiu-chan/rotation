#!/usr/bin/env bash
#SBATCH --job-name=patch-step1-best-position
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=logs/patch-step1-%j.out

# Step 1: unpatched rotation baseline. Finds, per curated system-prompt
# variant, the token position with the strongest control-vs-experiment
# rotation, and freezes it for steps 2-4. Plots the per-prompt position
# heatmap automatically at the end.
#
# Override cluster resources at submission, e.g.
#   sbatch --partition=mypartition --account=myaccount scripts/patching/step1_find_best_position.sh
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

# All of these are optional; leave MODEL_PATH/VARIANTS unset to use
# config/default.yaml's model_path and config/patching_lying.yaml's
# default_patching_variants.
MODEL_PATH="${MODEL_PATH:-}"
VARIANTS="${VARIANTS:-}"
OUTPUT="${OUTPUT:-results/patching/unpatched-half-20prompts-100}"
N_ROWS="${N_ROWS:-100}"
BATCH_SIZE="${BATCH_SIZE:-8}"
SEED="${SEED:-0}"

exec "$PATCHING_PYTHON" -u src/activation_patching/step1_find_best_position.py \
  --output "$OUTPUT" \
  ${MODEL_PATH:+--model-path "$MODEL_PATH"} \
  ${VARIANTS:+--variants "$VARIANTS"} \
  --n-rows "$N_ROWS" --batch-size "$BATCH_SIZE" --seed "$SEED" "$@"
