#!/usr/bin/env bash
#SBATCH --job-name=activation-patching
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --output=logs/patching-custom-%j.out
set -euo pipefail

# General-purpose launcher for run_custom_sweep.py / plot_custom_sweep.py --
# the flexible tool for sweeps the numbered pipeline steps (step1-4, see
# scripts/patching/step{1,2,3,4}_*.sh) don't cover. Override SBATCH resources
# at submission. Activate an environment beforehand or export
# PATCHING_PYTHON=/path/to/python.
PATCHING_ROOT="${PATCHING_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}}"
cd "${PATCHING_ROOT}"
if [[ -z "${SLURM_JOB_ID:-}" && "${PATCHING_ALLOW_LOCAL:-0}" != 1 ]]; then
  echo 'Run inside a SLURM allocation or submit with sbatch. On a workstation only, set PATCHING_ALLOW_LOCAL=1.' >&2
  exit 1
fi
PATCHING_PYTHON="${PATCHING_PYTHON:-python}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MPLBACKEND=Agg
mode="${1:-run}"
if [[ $# -gt 0 ]]; then shift; fi
case "${mode}" in
  run) exec "${PATCHING_PYTHON}" src/activation_patching/run_custom_sweep.py "$@" ;;
  plot) exec "${PATCHING_PYTHON}" src/activation_patching/plot_custom_sweep.py "$@" ;;
  test) exec "${PATCHING_PYTHON}" -m unittest discover -s tests -p test_patching_consolidation.py "$@" ;;
  *) echo 'Usage: run_custom_sweep.sh {run|plot|test} [arguments]' >&2; exit 2 ;;
esac
