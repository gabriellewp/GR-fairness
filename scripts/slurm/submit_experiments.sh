#!/usr/bin/env bash
# Submit IDs from a JSON experiment list; forward explicit resource overrides.
set -euo pipefail
release_root="${FAIRDIVERSE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
source "$release_root/scripts/slurm/_common.sh"
config=""; selected=""; concurrency=2; dry_run=false
resources=(); run_options=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config_file) config="${2:?Missing config path}"; shift 2 ;;
        --exp_ids) selected="${2:?Missing experiment IDs}"; shift 2 ;;
        --max_concurrent) concurrency="${2:?Missing concurrency}"; shift 2 ;;
        --partition|--mem|--time|--gres|--cpus-per-task|--dependency|--job-name)
            resources+=("$1=${2:?Missing Slurm option value}"); shift 2 ;;
        --dry-run) dry_run=true; shift ;;
        --) shift; run_options=("$@"); break ;;
        --help|-h)
            echo "Usage: $0 --config_file FILE [--exp_ids '0 2'] [--max_concurrent 2] [Slurm resource options] [--dry-run] [-- runner options]"
            exit 0 ;;
        *) echo "Unknown argument: $1. Use --help." >&2; exit 2 ;;
    esac
done
[[ -n "$config" ]] || { echo "--config_file is required" >&2; exit 2; }
[[ "$concurrency" =~ ^[1-9][0-9]*$ ]] || { echo "Concurrency must be positive" >&2; exit 2; }
array_ids=$("$release_python" - "$config" "$selected" <<'PYCODE'
import json, sys
entries = json.load(open(sys.argv[1]))
ids = [e.get("exp_id", i) for i, e in enumerate(entries)]
if not ids or any(type(i) is not int or i < 0 for i in ids) or len(set(ids)) != len(ids):
    raise SystemExit("Experiment IDs must be unique nonnegative integers")
chosen = [int(i) for i in sys.argv[2].replace(",", " ").split()] if sys.argv[2] else ids
if not chosen or len(set(chosen)) != len(chosen) or not set(chosen) <= set(ids):
    raise SystemExit("Selected IDs must be unique and exist in the config")
print(",".join(map(str, chosen)))
PYCODE
)
command=(sbatch --parsable --chdir="$release_root" --export=ALL --array="${array_ids}%${concurrency}"
         "${resources[@]}" "$release_root/scripts/slurm/experiment.sh" "$config" "${run_options[@]}")
if [[ "$dry_run" == true ]]; then
    printf '%q ' "${command[@]}"; printf '\n'
else
    mkdir -p output
    "${command[@]}"
fi
