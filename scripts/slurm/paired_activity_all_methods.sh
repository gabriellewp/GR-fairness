#!/usr/bin/env bash
#SBATCH --job-name=activity-all-trainers
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --array=0-11%4
#SBATCH --output=output/paired_activity_all_trainers_%A_%a.out
set -euo pipefail
release_root="${FAIRDIVERSE_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}"
source "$release_root/scripts/slurm/_common.sh"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
analysis_python="${PAIRED_ACTIVITY_PYTHON:-$release_python}"
"$analysis_python" -u -m additional_baselines.paired_activity_all_methods run --manifest "$1" --case "$SLURM_ARRAY_TASK_ID"
