#!/usr/bin/env bash
#SBATCH --job-name=grid-joint-fairness
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=5-00:30:00
#SBATCH --gres=gpu:1
#SBATCH --array=0-191%6
#SBATCH --output=output/grid_joint_fairness_%A_%a.out

set -euo pipefail
release_root="${FAIRDIVERSE_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}"
source "$release_root/scripts/slurm/_common.sh"
grid_python="${GRID_CONTINUATION_PYTHON:-$release_python}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/grid-joint-fairness-${SLURM_ARRAY_JOB_ID:-local}-${SLURM_ARRAY_TASK_ID}"
mkdir -p "$MPLCONFIGDIR"
"$grid_python" -u scripts/run_joint_fairness_trials.py "$2" \
    --manifest "$1" --index "$SLURM_ARRAY_TASK_ID"
