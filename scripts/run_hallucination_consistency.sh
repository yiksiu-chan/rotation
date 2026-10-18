#!/usr/bin/env bash
#SBATCH --job-name=hallu-consistency
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/hallu_consistency_%j.out
#SBATCH --error=logs/hallu_consistency_%j.err

# Generate per-variant hallucination labels for one model. This is the first
# stage before `scripts/run_pipeline.sh --task hallucination --full ...`.
# Example:
#   sbatch scripts/run_hallucination_consistency.sh \
#     --model allenai/Olmo-3-7B-Instruct --alias Olmo-3-7B-Instruct \
#     --batch-size 256

set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  sed -n '11,17p' "$0"
  exit 0
fi
ROOT="${PATCHING_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
cd "$ROOT"
# Defaults to the .venv created by the README's setup instructions
# (python -m venv .venv && source .venv/bin/activate); override PATCHING_ENV
# to use a different environment.
source "${PATCHING_ENV:-$ROOT/.venv/bin/activate}"

MODEL=""; ALIAS=""; BATCH_SIZE=128; INPUT="datasets/triviaqa_tf/triviaqa_tf_dataset.json"; PASSES=10
EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model|--model-path) MODEL="$2"; shift 2;;
    --alias|--model-alias) ALIAS="$2"; shift 2;;
    --batch-size) BATCH_SIZE="$2"; shift 2;;
    --input) INPUT="$2"; shift 2;;
    --passes|--n-passes) PASSES="$2"; shift 2;;
    --) shift; EXTRA+=("$@"); break;;
    *) EXTRA+=("$1"); shift;;
  esac
done
[[ -n "$MODEL" && -n "$ALIAS" ]] || { echo "Usage: $0 --model MODEL --alias MODEL_ALIAS [options]" >&2; exit 2; }

# HF_HUB_CACHE: leave unset to use huggingface_hub's own default
# (~/.cache/huggingface/hub) unless the caller overrides it.
[[ -n "${HF_HUB_CACHE:-}" ]] && export HF_HUB_CACHE
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
# Gated models (e.g. meta-llama/*) need a token; fall back to the standard
# `huggingface-cli login` token cache location if HF_TOKEN isn't already set.
if [[ "$MODEL" == meta-llama/* && -z "${HF_TOKEN:-}" && -f "$HOME/.cache/huggingface/token" ]]; then
  export HF_TOKEN="$(< "$HOME/.cache/huggingface/token")"
fi

echo "host=$(hostname) model=$MODEL alias=$ALIAS"
nvidia-smi || true
python -u src/precompute_per_variant_consistency.py \
  --config_override config/hallucination.yaml \
  --input "$INPUT" --model_path "$MODEL" --model_alias "$ALIAS" \
  --n_passes "$PASSES" --batch_size "$BATCH_SIZE" "${EXTRA[@]}"
