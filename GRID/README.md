# GRID standalone pipeline

This release contains the GRID model and tokenizer code used by FairDiverse.
Raw data, TFRecords, embeddings, checkpoints, and logs are generated locally.

Use Python 3.10. The parent FairDiverse installation guide describes a shared
runtime for both components. `requirements_core.txt` records the historical
GRID environment; `requirements.txt` is its fuller frozen dependency snapshot.
GPU/CUDA packages must match your machine.

## Run

Run from the bundled `GRID/` directory, with your environment activated:

```bash
export GRID_PYTHON="$(command -v python)"
export RAW_DATA_DIR=/path/to/amazon-mcauley-2014
bash run_independent_pipeline.sh --dataset beauty --tokenizer rkmeans \
  --metadata-text-mode complete_metadata --preprocess-only
bash run_independent_pipeline.sh --dataset beauty --tokenizer rkmeans \
  --metadata-text-mode complete_metadata
```

Supported datasets: `beauty`, `sports`, `toys`, `pet-supplies`. Supported tokenizers:
`rkmeans`, `rqvae`, `rvq`. For Pet Supplies, point `RAW_DATA_DIR` to the Amazon 2023
input files. Run `bash run_independent_pipeline.sh --help` for schedule and path
options. `--tiger-max-steps` and `--tiger-run-tag` support separately named runs.
W&B is disabled by default; enable it with `GRID_USE_WANDB=true` after authenticating.

The pipeline performs preprocessing, text embedding, tokenizer training, SID
inference, TIGER training, and TIGER inference. It uses temporal leave-one-out
splits, three learned SID levels plus a collision suffix, and 40 history items
encoded as 160 SID tokens. Metadata modes have separate data/output namespaces.

Default outputs are `data/amazon_<dataset>_5core_tfrecords/temporal_loo/<mode>` and
`outputs/<dataset>_5core/temporal_loo/<mode>/<tokenizer>`. Set `GRID_DATA_ROOT`,
`GRID_OUTPUT_ROOT`, or `GRID_LOG_DIR` to change those roots. FairDiverse requires
its model artifacts under the embedded `GRID/` directory.

For Slurm, use the parent repository's `scripts/slurm/grid.sh`, or create `logs/`
and submit this script from this checkout. Override partition, time, memory and
GPU resources to suit your cluster. No launcher installs dependencies at runtime.

Generated datasets and historical experiment snapshots are not distributed here.

Third-party license notices are preserved in `notices.txt`.
