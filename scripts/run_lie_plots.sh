#!/bin/bash
#SBATCH --job-name=lie_analysis
#SBATCH --partition=batch
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00

set -euo pipefail

PROJECT_ROOT=/oscar/data/superlab/projects/Rotation
cd "${PROJECT_ROOT}"


echo "host: $(hostname)  cwd: $(pwd)  python: $(which python)"

python -u src/run_pipeline.py \
  --steps analysis,plots \
  --config_override config/lying.yaml