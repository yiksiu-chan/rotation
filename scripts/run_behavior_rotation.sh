#!/usr/bin/env bash
#SBATCH --job-name=behavior-rotation
#SBATCH --partition=batch
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/behavior_rotation_%j.out
#SBATCH --error=logs/behavior_rotation_%j.err
# Combine behavior bar/boxplot PNGs with cos-sim curve SVGs into side-by-side
# panels under results/plots/<MODEL>/behavior_rotation/.
#
# Usage:
#   sbatch scripts/run_behavior_rotation.sh                            # default model
#   sbatch scripts/run_behavior_rotation.sh Llama-3.1-8B-Instruct      # one model
#   sbatch scripts/run_behavior_rotation.sh model_a model_b ...        # several
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PLOTS_BASE="${PLOTS_BASE:-$REPO_ROOT/results/plots}"
# PLOTS_BASE="/users/ychan19/data_sbach/ychan19/projects/deception_and_jailbreak/spatial_easy_all/results_spatial_easy_v8/plots"
SEED="${SEED:-0}"
SCALE="${SCALE:-1.0}"
TARGET_HEIGHT="${TARGET_HEIGHT:-1920}"
QUANTIZE="${QUANTIZE:-1}"  # set to 0 to disable palette quantization
# Optional: redirect output to a writable dir (use when PLOTS_BASE is read-only).
# If unset, outputs land under $PLOT_ROOT/behavior_rotation.
# When multiple MODELS are given, $MODEL is appended to OUT_DIR for each.
OUT_DIR="${OUT_DIR:-}"

if [[ $# -eq 0 ]]; then
  MODELS=("Llama-3.1-8B-Instruct")
else
  MODELS=("$@")
fi

for MODEL in "${MODELS[@]}"; do
  PLOT_ROOT="$PLOTS_BASE/$MODEL"
  if [[ ! -d "$PLOT_ROOT" ]]; then
    echo "[skip] $PLOT_ROOT does not exist" >&2
    continue
  fi
  echo "=== $MODEL ==="
  CMD=(python "$REPO_ROOT/src/make_behavior_rotation_panels.py"
       --plot-root "$PLOT_ROOT"
       --seed "$SEED"
       --target-height "$TARGET_HEIGHT"
       --scale "$SCALE")
  if [[ "$QUANTIZE" != "0" ]]; then
    CMD+=(--quantize)
  fi
  if [[ -n "$OUT_DIR" ]]; then
    if [[ "${#MODELS[@]}" -gt 1 ]]; then
      CMD+=(--out-dir "$OUT_DIR/$MODEL")
    else
      CMD+=(--out-dir "$OUT_DIR")
    fi
  fi
  "${CMD[@]}"
done
