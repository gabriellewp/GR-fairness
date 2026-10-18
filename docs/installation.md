# Installation

## Supported workflow

The generative fairness release is developed for Linux, Python 3.10 and an
editable source checkout. Some code uses `fcntl` and Slurm is a Linux/HPC tool.
Python 3.10 is the validation target; `setup.py` rejects older Python versions.
GRID is loaded from the adjacent `GRID/` source tree, so a standalone FairDiverse
wheel without that checkout is not sufficient for the model/trainer workflow.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Install the torch build appropriate for your CUDA driver or CPU first.
python -m pip install -e '.[grid,rerank,analysis,test]'
python -m pip check
python scripts/check_release.py
```

The `grid` extra supplies the native model runtime, Hydra/Lightning,
Transformers 4.57.6, TFRecord preprocessing, Arrow, and logging utilities. The
`rerank` extra supplies CVXPY and transport/linear-programming dependencies.
CPFair and P-MMF do not require a commercial solver. `test` supplies pytest.
The simpler artifact-only workflow can use `pip install -e '.[rerank,test]'`.

GPU training needs an appropriate PyTorch/CUDA installation and enough memory
for the chosen embedding model. The default GRID pipeline uses Flan-T5-XL;
embedding extraction is substantially heavier than artifact evaluation. Resource
values in Slurm examples are starting points, not guarantees for every dataset.

`GRID/requirements_core.txt` and `GRID/requirements.txt` preserve older frozen
GRID environments. They include many packages outside this release's minimum
workflow and should not be mixed blindly with a different Python/CUDA stack.
Record the actual environment used with `python -m pip freeze`.

## Authentication and cache locations

The documented workflows disable W&B by default. To enable it, authenticate in
your environment and set `WANDB_MODE=online` and `GRID_USE_WANDB=true`. Provide
`WANDB_API_KEY` through your shell or secret manager if needed. New-run configs
also set `use_wandb=false`; change that config field to enable FairDiverse logging. Hugging Face
models requiring access use `HF_TOKEN` from your environment. No token belongs
in an experiment config, batch script or tracked file.

`.env.example` lists available settings and contains no credentials. The Slurm
wrappers inherit exported variables; they do not source this template or
activate a named Conda environment. Activate your environment before submission,
or export `FAIRDIVERSE_PYTHON=/absolute/path/to/python` and `GRID_PYTHON`.
Set `HF_HOME`, `TORCH_HOME` and `WANDB_CACHE_DIR` for your storage policy.

## Existing optional upstream components

The original search/LLM modules remain in source. Their datasets and external
runtimes are not included in the core example. LLM inference may additionally
require vLLM, a service client, or gated models; CIFRank requires R/rpy2; other
search/reranking methods may require their own solver packages. Install those
only for the corresponding method. The complete tutorial here focuses on GRID,
classic recommendation, and the supported fairness adaptations.

## Troubleshooting

- Missing embedded GRID: check that the bundled GRID source is present before importing a
  FairDiverse trainer. Do not add another GRID checkout to `PYTHONPATH`.
- Missing solver: install the `rerank` extra and inspect `cvxpy.installed_solvers()`.
- Transformer import/checkpoint mismatch: use the documented dependency range
  and the exact native checkpoint architecture.
- Offline model download failure: prepopulate your cache or provide the required
  model files; generated artifacts are not distributed in Git.
- Package imports work only in one shell: export the environment's Python path
  explicitly before submitting Slurm jobs.
