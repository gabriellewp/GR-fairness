#!/usr/bin/env bash
#SBATCH --job-name=grid-preprocess
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=output/preprocess-%j.out
set -euo pipefail
release_root="${FAIRDIVERSE_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}"
source "$release_root/scripts/slurm/_common.sh"
export FAIRDIVERSE_GRID_ROOT="$release_root/GRID"
export GRID_PYTHON="${GRID_PYTHON:-$release_python}"
exec bash GRID/run_independent_pipeline.sh --preprocess-only "$@"
