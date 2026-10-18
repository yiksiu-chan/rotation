#!/usr/bin/env bash
#SBATCH --job-name=patch-step3-head-screen
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --output=logs/patch-step3-%j.out

# Step 3: fast full attention-head screen (all layers x all heads, patched
# one at a time) at each prompt's frozen position from step 1. Tabulates
# (CSV) and plots automatically at the end, into --output itself -- no
# separate summarize/plot command needed.
#
# Requires a completed step2_patch_residual.sh run at POSITIONS_FROM.
# Override cluster resources at submission, e.g.
#   sbatch --partition=mypartition --account=myaccount scripts/patching/step3_head_sweep.sh
# On a workstation with a GPU (no SLURM), set PATCHING_ALLOW_LOCAL=1.
set -euo pipefail

ROOT="${PATCHING_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}}"
cd "$ROOT"
if [[ -z "${SLURM_JOB_ID:-}" && "${PATCHING_ALLOW_LOCAL:-0}" != 1 ]]; then
  echo 'Run inside a SLURM allocation or submit with sbatch. On a workstation only, set PATCHING_ALLOW_LOCAL=1.' >&2
  exit 1
fi
PATCHING_PYTHON="${PATCHING_PYTHON:-python}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MPLBACKEND=Agg HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1

MODEL_PATH="${MODEL_PATH:-}"
VARIANTS="${VARIANTS:-}"
# best_positions.json is written by step2_patch_residual.py (derived from
# step1's results.json), NOT by step1 itself.
POSITIONS_FROM="${POSITIONS_FROM:-results/patching/mean-residual-best-position-filtered/best_positions.json}"
OUTPUT="${OUTPUT:-results/patching/mean-head-best-position-20prompts-100-fast}"
N_ROWS="${N_ROWS:-100}"
BATCH_SIZE="${BATCH_SIZE:-20}"
HEAD_BATCH_SIZE="${HEAD_BATCH_SIZE:-24}"
SEED="${SEED:-0}"

exec "$PATCHING_PYTHON" -u src/activation_patching/step3_head_sweep.py \
  --output "$OUTPUT" ${MODEL_PATH:+--model-path "$MODEL_PATH"} \
  --component head --sweep head --layers all --heads all \
  --positions-from "$POSITIONS_FROM" \
  --patch-source mean --mean-reference evaluation --metrics behavior \
  --direction both ${VARIANTS:+--variants "$VARIANTS"} \
  --n-rows "$N_ROWS" --batch-size "$BATCH_SIZE" --head-batch-size "$HEAD_BATCH_SIZE" \
  --seed "$SEED" --filter none --verify-top-heads 10 "$@"
