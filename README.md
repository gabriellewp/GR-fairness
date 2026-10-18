# GR-fairness: Fairness in Generative Recommendation

FairDiverse runs fairness and diversity experiments with GRID generative
recommendation, SASRec, GRU4Rec, and candidate-based reranking. This repository
packages reusable implementations, continuation tuning, generation parity checks,
and paired activity analyses. GRID is included as ordinary source files. The original
FairDiverse recommendation and search implementations remain available.

The released GRID experiments use **complete_metadata only**: title, categories,
brand/store and description. The recipes cover all four datasets and three
tokenizers; description-only checkpoints and results are excluded. See the
[complete-metadata protocol](docs/complete-metadata.md).

## Install

Use Linux and Python 3.10. Train GRID on a CUDA GPU; artifact evaluation and most
reranking run on CPU. Install PyTorch for your CUDA runtime before installing the
remaining dependencies.

```bash
# Download/extract the repository archive, then work from its root directory.
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[grid,rerank,analysis,test]'
```

No submodule initialization or second repository is required. See
[installation](docs/installation.md) for dependencies, optional components,
authentication, and source-installation requirements.

## Run a small local example

This synthetic example checks the artifact/evaluation/reranking workflow without
raw datasets, model downloads, a GPU, or service credentials. Its values are
fixtures, not research results.

```bash
python scripts/smoke_test.py --output-dir artifacts/smoke
python -m additional_baselines evaluate artifacts/smoke/inputs.json \
  --topk 1 2 --output artifacts/smoke/evaluated.json
python -m additional_baselines rerank artifacts/smoke/inputs.json \
  --config configs/examples/cpfair-smoke.json \
  --output artifacts/smoke/cpfair.json
```

Output paths must be new: artifact commands deliberately reject overwrites.

## Run real experiments

1. Acquire the supported Amazon data and build temporal leave-one-out TFRecords.
2. Run GRID embedding, tokenizer, SID, and TIGER training stages.
3. Export/evaluate the trained model through FairDiverse, or train classic
   recommenders on the exact same processed splits.
4. Generate validation and test candidate artifacts, tune on validation, freeze
   the selection, and evaluate on test.

Follow [data preparation](docs/data.md), then [experiments](docs/experiments.md).
[Slurm instructions](docs/slurm.md) cover CPU/GPU jobs, arrays, and dependencies.
[Fairness methods](docs/fairness.md) cover CPFair, P-MMF, IFairLRS RR/RW, and SPRINT.
[Continuation tuning and analysis](docs/continuation-and-analysis.md) cover the
validation selection and paired comparison workflows. [Metrics and reproduction](docs/reproducibility.md) explain comparability and
provenance. [Development](docs/development.md) describes validation and testing.

## Repository layout

| Path | Purpose |
| --- | --- |
| `fairdiverse/` | FairDiverse models, trainers, preprocessing and metrics |
| `GRID/` | Bundled native GRID implementation and pipeline |
| `additional_baselines/` | Candidate artifacts, reranking and adaptations |
| `configs/reproduction/` | Preserved research protocols; local artifacts required |
| `scripts/` | Fresh-run configuration, data bridge, validation and utilities |
| `scripts/slurm/` | Portable Slurm wrappers |
| `tests/`, `additional_baselines/tests/` | Behavioral and regression checks |
| `docs/` | Maintained usage and release documentation |
| `doc/source/` | Original FairDiverse API documentation sources |

Data, checkpoints, embeddings, logs, temporary files and dated analysis drafts
are excluded from the release tree. The curation policy is recorded in
[release-file-decisions.json](docs/release-file-decisions.json).
See [release notes](docs/release-notes.md) for migration and validation limits.

Regenerate the historical paper tables and plots from bundled aggregate results:
`python -m scripts.paper.reproduce --output-dir artifacts/paper`.
See [paper reproduction and archived audits](docs/reproducibility.md#regenerate-the-paper-outputs).

The project retains the [MIT license](LICENSE). Upstream FairDiverse was authored
by Chen Xu and contributors; existing source attribution is preserved. GRID has
its own [license](GRID/LICENSE).
