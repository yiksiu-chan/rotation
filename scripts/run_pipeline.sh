#!/usr/bin/env bash
#SBATCH --job-name=deception-pipeline
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/pipeline_%j.out
#SBATCH --error=logs/pipeline_%j.err

# One configurable launcher for the behavior/cache/analysis/plot pipeline.
# Examples:
#   sbatch scripts/run_pipeline.sh --task lying --model allenai/Olmo-3-7B-Instruct
#   sbatch scripts/run_pipeline.sh --task spatial-easy --model Qwen/Qwen3-4B-Instruct-2507
#   sbatch scripts/run_pipeline.sh --task hallucination --model allenai/Olmo-3-7B-Instruct --full
# Adjust cluster resources at submission time, for example:
#   sbatch --mem=96G --time=12:00:00 scripts/run_pipeline.sh ...

set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  sed -n '11,22p' "$0"
  exit 0
fi

ROOT="${PATCHING_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
cd "$ROOT"
# Defaults to the .venv created by the README's setup instructions
# (python -m venv .venv && source .venv/bin/activate); override PATCHING_ENV
# to use a different environment.
source "${PATCHING_ENV:-$ROOT/.venv/bin/activate}"

TASK=""
MODEL=""
CONFIG=""
DATASET=""
STEPS="behavior,cache,analysis,plots"
GEN_BATCH=""
CACHE_BATCH=""
FULL=0
EXTRA=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --task) TASK="$2"; shift 2;;
    --model|--model-path) MODEL="$2"; shift 2;;
    --config) CONFIG="$2"; shift 2;;
    --dataset) DATASET="$2"; shift 2;;
    --steps) STEPS="$2"; shift 2;;
    --batch-generation) GEN_BATCH="$2"; shift 2;;
    --batch-cache) CACHE_BATCH="$2"; shift 2;;
    --full) FULL=1; shift;;
    --) shift; EXTRA+=("$@"); break;;
    *) EXTRA+=("$1"); shift;;
  esac
done

[[ -n "$TASK" ]] || { echo "Missing --task (lying, spatial-easy, sycophancy, or hallucination)" >&2; exit 2; }
case "$TASK" in
  lying) CONFIG="${CONFIG:-config/lying.yaml}"; DATASET="${DATASET:-azaria-mitchell}";;
  spatial-easy) CONFIG="${CONFIG:-config/spatial_easy.yaml}"; DATASET="${DATASET:-spatial-easy}";;
  sycophancy) CONFIG="${CONFIG:-config/sycophancy.yaml}"; DATASET="${DATASET:-azaria-mitchell}";;
  hallucination) CONFIG="${CONFIG:-config/hallucination.yaml}";;
  *) echo "Unknown task: $TASK" >&2; exit 2;;
esac

if [[ -z "$MODEL" && ( "$FULL" -eq 1 || "$TASK" != hallucination ) ]]; then
  echo "Missing --model (required for this task/mode)" >&2; exit 2;
fi

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

if [[ "$TASK" == hallucination && "$FULL" -eq 1 ]]; then
  ALIAS="${MODEL##*/}"
  echo "===== STAGE 1: per-variant consistency ($ALIAS) ====="
  FILTER_BATCH="${GEN_BATCH:-128}"
  python -u src/precompute_per_variant_consistency.py \
    --config_override "$CONFIG" --input datasets/triviaqa_tf/triviaqa_tf_dataset.json \
    --model_path "$MODEL" --model_alias "$ALIAS" --n_passes 10 --batch_size "$FILTER_BATCH"
fi

ARGS=(src/run_pipeline.py --steps "$STEPS" --config_override "$CONFIG")
[[ -n "$MODEL" ]] && ARGS+=(--model_path "$MODEL")
[[ -n "$DATASET" ]] && ARGS+=(--dataset "$DATASET")
[[ -n "$GEN_BATCH" ]] && ARGS+=(--batch_size_generation "$GEN_BATCH")
[[ -n "$CACHE_BATCH" ]] && ARGS+=(--batch_size_cache "$CACHE_BATCH")
ARGS+=("${EXTRA[@]}")

echo "host=$(hostname) task=$TASK model=${MODEL:-config-default} steps=$STEPS"
nvidia-smi || true
python -u "${ARGS[@]}"
