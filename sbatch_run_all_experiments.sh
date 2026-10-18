#!/usr/bin/env bash
# Compatibility entry point. See docs/slurm.md for portable submission options.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$root/scripts/slurm/submit_experiments.sh" "$@"
