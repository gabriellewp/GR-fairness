#!/bin/bash
#SBATCH --job-name=rerank-tune-merge
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=output/rerank-tune-merge-%j.out

set -euo pipefail
release_root="${FAIRDIVERSE_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
source "$release_root/scripts/slurm/_common.sh"
grid_python="${GRID_RERANK_PYTHON:-$release_python}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
"$grid_python" -m additional_baselines.merge_rerank_tuning \
  --task-dir "${1:?Pass tasks directory}" --output-dir "${2:?Pass merged output directory}"
