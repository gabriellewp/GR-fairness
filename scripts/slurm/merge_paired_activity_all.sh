#!/usr/bin/env bash
#SBATCH --job-name=activity-all-methods
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:45:00
#SBATCH --output=output/paired_activity_all_methods_%j.out
set -euo pipefail
release_root="${FAIRDIVERSE_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}"
source "$release_root/scripts/slurm/_common.sh"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/paired_activity_matplotlib_${SLURM_JOB_ID:-local}"
analysis_python="${PAIRED_ACTIVITY_PYTHON:-$release_python}"
"$analysis_python" -u -m additional_baselines.paired_activity_fairdiverse merge --manifest "$1"
