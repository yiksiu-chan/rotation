#!/usr/bin/env bash
#SBATCH --job-name=patch-step4-heads-cumulative
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=logs/patch-step4-%j.out

# Step 4: cumulative top-k mean head patching (k=1..TOP_K) at each prompt's
# frozen position from step 1, using step 3's head ranking (SELECTION=ranked)
# or a uniform random draw (SELECTION=random, the control). Validates,
# tabulates (CSV) and plots this run's own diagnostic figures automatically
# at the end -- run both SELECTION=ranked and SELECTION=random, then use
# step4_plot_heads_cumulative.py separately to combine them into the paper
# figure (the one plot that genuinely needs both runs at once).
#
# Requires completed step1_find_best_position.sh (BASELINE) and, for
# SELECTION=ranked, step3_head_sweep.sh (HEAD_SCREEN) runs.
# Override cluster resources at submission, e.g.
#   sbatch --partition=mypartition --account=myaccount scripts/patching/step4_patch_heads_cumulative.sh
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

SELECTION="${SELECTION:-ranked}"   # ranked | random
BASELINE="${BASELINE:-results/patching/unpatched-half-20prompts-100}"
HEAD_SCREEN="${HEAD_SCREEN:-results/patching/mean-head-best-position-20prompts-100-fast/results.json}"
TOP_K="${TOP_K:-30}"
BATCH_SIZE="${BATCH_SIZE:-8}"
VARIANTS="${VARIANTS:-}"
RANDOM_SEED="${RANDOM_SEED:-0}"
if [[ "$SELECTION" == ranked ]]; then
  OUTPUT="${OUTPUT:-results/patching/mean-head-best-position-noise-ranked-k${TOP_K}}"
  SELECTION_FLAGS=(--head-screen "$HEAD_SCREEN")
else
  OUTPUT="${OUTPUT:-results/patching/mean-head-best-position-random-baseline-k${TOP_K}}"
  SELECTION_FLAGS=(--random-seed "$RANDOM_SEED")
fi

exec "$PATCHING_PYTHON" -u src/activation_patching/step4_patch_heads_cumulative.py \
  --baseline "$BASELINE" --selection "$SELECTION" "${SELECTION_FLAGS[@]}" \
  --top-k "$TOP_K" --output "$OUTPUT" --batch-size "$BATCH_SIZE" \
  ${VARIANTS:+--variants "$VARIANTS"} "$@"
