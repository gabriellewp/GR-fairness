#!/usr/bin/env bash
#SBATCH --job-name=fairdiverse
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=output/experiment-%A_%a.out
set -euo pipefail
release_root="${FAIRDIVERSE_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}"
source "$release_root/scripts/slurm/_common.sh"
config="${1:?Usage: experiment.sh CONFIG [run_experiment options]}"
shift
exec "$release_python" -u run_experiment.py --config_file "$config" \
    --exp_id "${SLURM_ARRAY_TASK_ID:-0}" --slurm_job_id "${SLURM_JOB_ID:-local}" \
    --slurm_array_task_id "${SLURM_ARRAY_TASK_ID:-0}" "$@"
