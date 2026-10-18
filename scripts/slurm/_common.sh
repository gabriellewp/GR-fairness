#!/usr/bin/env bash
# Sourced by job wrappers after they resolve the submission directory.
set -euo pipefail
: "${release_root:?Job wrapper must resolve release_root}"
if [[ ! -f "$release_root/run_experiment.py" ]]; then
    echo "Set FAIRDIVERSE_ROOT to the repository root before submitting." >&2
    exit 2
fi
cd "$release_root"
release_python="${FAIRDIVERSE_PYTHON:-python}"
export FAIRDIVERSE_ROOT="$release_root"
export PYTHONPATH="$release_root${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
# GRID uses PyTorch; avoid loading the optional Transformers TensorFlow backend.
export USE_TF=0
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-4}}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$OMP_NUM_THREADS}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export GRID_USE_WANDB="${GRID_USE_WANDB:-false}"
export HF_HOME="${HF_HOME:-$release_root/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$release_root/.cache/torch}"
