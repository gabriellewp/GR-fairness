#!/usr/bin/env bash
#SBATCH --job-name=grid-deadline-pilot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=09:30:00
#SBATCH --gres=gpu:1
#SBATCH --array=0-11%6
#SBATCH --output=output/grid_deadline_pilot_%A_%a.out
set -euo pipefail
release_root="${FAIRDIVERSE_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}"
source "$release_root/scripts/slurm/_common.sh"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
exec "$release_python" -u scripts/run_deadline_pilot.py case \
    --manifest "$1" --index "$SLURM_ARRAY_TASK_ID"
