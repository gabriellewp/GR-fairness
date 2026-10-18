#!/usr/bin/env bash
#SBATCH --job-name=grid-deadline-v2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=08:45:00
#SBATCH --gres=gpu:1
#SBATCH --output=output/grid_deadline_v2_%A_%a.out
set -euo pipefail
release_root="${FAIRDIVERSE_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}"
source "$release_root/scripts/slurm/_common.sh"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
exec "$release_python" -u scripts/run_deadline_pilot_v2.py \
    --manifest "$1" --index "$SLURM_ARRAY_TASK_ID"
