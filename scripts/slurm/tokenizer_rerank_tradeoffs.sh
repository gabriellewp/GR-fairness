#!/usr/bin/env bash
#SBATCH --job-name=tokenizer-calibration
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=output/tokenizer_calibration_%j.out
set -euo pipefail
release_root="${FAIRDIVERSE_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}"
source "$release_root/scripts/slurm/_common.sh"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
grid_python="${GRID_RERANK_PYTHON:-$release_python}"
"$grid_python" -u scripts/compute_tokenizer_rerank_tradeoffs.py --workers 4
"$grid_python" -u scripts/build_tokenizer_reranking_tables.py
