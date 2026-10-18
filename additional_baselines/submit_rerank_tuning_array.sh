#!/bin/bash
# Usage: bash ...sh METHOD DATASET ARTIFACT_DIR [CONCURRENCY=1] [extra sbatch options]
# Extra sbatch options apply to the array, e.g. --dependency=afterok:JOB_ID or --mem=96G.
set -euo pipefail
release_root="${FAIRDIVERSE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "$release_root/scripts/slurm/_common.sh"
mkdir -p output
method="${1:?Pass cpfair, pmmf, or rr}"
dataset="${2:?Pass amazon-beauty, amazon-sports, amazon-toys, or amazon-pet-supplies}"
artifact_dir="${3:?Pass candidate directory}"
shift 3
concurrency=1
if [[ $# -gt 0 && "$1" != --* ]]; then
  concurrency="$1"
  shift
fi
if [[ ! "$concurrency" =~ ^[1-9][0-9]*$ ]]; then
  echo "Concurrency must be a positive integer" >&2
  exit 2
fi
case "$dataset" in
  amazon-beauty|amazon-sports|amazon-toys) memory=32G ;;
  amazon-pet-supplies) memory=96G ;;
  *) echo "Unsupported dataset: $dataset" >&2; exit 2 ;;
esac
grid_python="${GRID_RERANK_PYTHON:-$release_python}"
count=$("$grid_python" -m additional_baselines.tune_beauty_reranking --method "$method" --task-count)
array_job=$(sbatch --parsable --chdir="$release_root" --export=ALL --mem="$memory" --array="0-$((count - 1))%$concurrency" "$@" \
  additional_baselines/sbatch_beauty_rerank_tuning.sh "$method" "$dataset" "$artifact_dir")
array_job="${array_job%%;*}"
root="additional_baselines/artifacts/${dataset#amazon-}_${method}_validation_tuning_array_${array_job}"
merge_job=$(sbatch --parsable --chdir="$release_root" --export=ALL --dependency="afterok:$array_job" \
  additional_baselines/sbatch_merge_rerank_tuning.sh "$root/tasks" "$root/merged")
echo "Array: $array_job ($count tasks, concurrency $concurrency); merge: ${merge_job%%;*}"
echo "Merged results: $root/merged"
