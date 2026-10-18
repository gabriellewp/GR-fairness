#!/bin/bash
#SBATCH --job-name=beauty-rerank-tune
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=2-00:00:00
#SBATCH --output=output/beauty-rerank-tune-%A_%a.out

set -euo pipefail
release_root="${FAIRDIVERSE_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
source "$release_root/scripts/slurm/_common.sh"
grid_python="${GRID_RERANK_PYTHON:-$release_python}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
# Usage: ...sh METHOD [DATASET ARTIFACT_DIR] [additional tuner options]

method="${1:?Pass cpfair, pmmf, or rr}"
case "$method" in
  cpfair|pmmf|rr) ;;
  *) echo "Unknown method: $method" >&2; exit 2 ;;
esac
shift
dataset=amazon-beauty
artifact_dir=additional_baselines/artifacts/beauty_ifairlrs_rr_popularity/original
if [[ $# -ge 2 && "$1" != --* ]]; then
  dataset="$1"
  artifact_dir="$2"
  shift 2
fi
case "$dataset" in
  amazon-beauty|amazon-sports|amazon-toys|amazon-pet-supplies) ;;
  *) echo "Unsupported dataset: $dataset" >&2; exit 2 ;;
esac

task_options=()
output_dir="additional_baselines/artifacts/${dataset#amazon-}_${method}_validation_tuning_${SLURM_JOB_ID:-local}"
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  output_dir="additional_baselines/artifacts/${dataset#amazon-}_${method}_validation_tuning_array_${SLURM_ARRAY_JOB_ID}/tasks/task_${SLURM_ARRAY_TASK_ID}"
  task_options=(--task-index "$SLURM_ARRAY_TASK_ID")
fi

backend_options=(-m additional_baselines.tune_beauty_reranking --method "$method" --histories "$artifact_dir/validation_histories.json")
if [[ "$method" == pmmf && "${GRID_PMMF_BACKEND:-legacy}" == exact ]]; then
  : "${SLURM_ARRAY_TASK_ID:?Exact P-MMF tuning requires an array task}"
  backend_options=(-m additional_baselines.tune_fast_pmmf)
fi
"$grid_python" "${backend_options[@]}" \
  --candidates "$artifact_dir/validation_candidates.json" \
  --split "recommendation/processed_dataset/$dataset/$dataset.valid.ranking" \
  --activity "recommendation/processed_dataset/$dataset/user_activity.csv" \
  --output-dir "$output_dir" "${task_options[@]}" "$@"
