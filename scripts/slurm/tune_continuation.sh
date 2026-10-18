#!/usr/bin/env bash
#SBATCH --job-name=grid-fairness-tuning
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=5-00:30:00
#SBATCH --gres=gpu:1
#SBATCH --output=output/grid_fairness_tuning_%j.out

set -euo pipefail
release_root="${FAIRDIVERSE_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}"
source "$release_root/scripts/slurm/_common.sh"
grid_python="${GRID_CONTINUATION_PYTHON:-$(command -v python)}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
"$grid_python" -u -m additional_baselines.tune_continuation "$@"
