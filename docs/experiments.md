# Running experiments

All commands below run from the repository root with the installed environment.
Complete [data preparation](data.md) before training. Artifact files are generated
locally and ignored by Git.

## Train native GRID

```bash
export GRID_PYTHON="$(command -v python)"
export RAW_DATA_DIR=/path/to/amazon-mcauley-2014
bash GRID/run_independent_pipeline.sh --dataset beauty --tokenizer rkmeans \
  --metadata-text-mode complete_metadata
```

Repeat with `--tokenizer rqvae` or `--tokenizer rvq`. The pipeline performs text
embedding, three-level tokenizer training, SID inference with a collision suffix,
TIGER training, and evaluation. Existing completed stage artifacts are reused;
use a new output namespace when changing a representation or schedule.

Default per-tokenizer output root:
`GRID/outputs/beauty_5core/temporal_loo/complete_metadata/<tokenizer>/`.
SID output is `step2b_sid_inference/pickle/merged_predictions_tensor_suffix4.pt`.
Checkpoints live under `step3_training_suffix4/checkpoints/`. Training log and
checkpoint metadata determine the selected checkpoint; pin that exact path in
reported runs. `grid_checkpoint_path=auto` is available in the model adapter,
but chooses by modification time and is unsuitable as a frozen research identity.

For shorter development runs, use `--tiger-max-steps`,
`--tiger-val-check-interval`, and `--tiger-run-tag` to create a separate training
namespace. This limits TIGER training; it does not skip embedding/tokenizer stages.
Full runs use the native configured schedule. See the pipeline's `--help`.

## Export a new trained GRID model through FairDiverse

Create a config with your actual validation-selected checkpoint:

```bash
python scripts/configure_experiment.py --dataset amazon-beauty \
  --model GRID --tokenizer rkmeans \
  --checkpoint GRID/outputs/beauty_5core/temporal_loo/complete_metadata/rkmeans/step3_training_suffix4/checkpoints/YOUR_SELECTED_CHECKPOINT.ckpt \
  --output artifacts/configs/beauty-grid.json
python run_experiment.py --config_file artifacts/configs/beauty-grid.json --exp_id 0
```

Replace `YOUR_SELECTED_CHECKPOINT.ckpt` with a real filename. If using a named
representation/output run, pass `--sid-path` and `--grid-data-dir` too. Native
checkpoints and SID files must remain under this checkout's embedded `GRID/`.
Strict loading rejects architecture mismatches and external-checkout artifacts.

The generated config disables W&B and fairness fine-tuning, keeps four SID levels
and 40 history items, evaluates at K=5/10/20, and exports scores. It writes a
uniquely configured run below `fairdiverse/recommendation/log/`; inspect the
reported path. `config.yaml`, `process_config.yaml`, `ranking_scores.npz`, and
`ground_truth.npz` are prerequisites for fresh candidate preparation.

## Classic baselines on the same splits

Build the bridge tables first. Then:

```bash
python scripts/configure_experiment.py --dataset amazon-beauty --model SASRec \
  --output artifacts/configs/beauty-sasrec.json
python run_experiment.py --config_file artifacts/configs/beauty-sasrec.json --exp_id 0
```

Use `--model GRU4Rec` for the recurrent baseline. These generated starter configs
check the expected GRID source identity and use negative padding. They are new-run
examples, not claims to reproduce every archived tuned baseline hyperparameter.
Use `--epochs`, `--device`, and `--seed` as appropriate. Core model defaults are
under `fairdiverse/recommendation/properties/models/`.

## Existing fairness in-processing recipes

`configs/reproduction/` preserves the archived GRID tokenizer/fairness configs
and classic recipes. IDs are explicit and may be noncontiguous. Paths to dated
checkpoints must be replaced with your matching artifacts. The preserved recipes
include applicable APR, FairDual, IPS, Minmax_SGD, SDRO and Reg branches. For GRID
fine-tuning, keep `max_steps` equal to `grid_fair_finetune_steps` and initialize
from the correct native checkpoint. Do not silently change optimizer schedules
while presenting an exact reproduction.

```bash
python run_experiment.py \
  --config_file configs/reproduction/experiments_config_amazon-beauty_grid_tokenizers.json \
  --exp_id 57
```

Inspect that entry and resolve its artifact paths before execution. Its presence
in Git does not imply the corresponding checkpoint is distributed.

## Generate candidate artifacts

```bash
python -m additional_baselines.prepare_rerank_candidates \
  --dataset amazon-beauty --tokenizer rkmeans \
  --export-dir /path/to/your/fairdiverse/score-export \
  --data-dir recommendation/processed_dataset/amazon-beauty \
  --generation-beam-width 300 --require-min-candidates 20 \
  --output-dir artifacts/beauty-candidates
```

The export must match the checkpoint, SID mapping, item catalog and processed
split. This regenerates both validation and test candidates, history files and
provenance. Beam width can exceed the SID token vocabulary; it grows over levels.
`--limit-requests` is for smoke tests only. If a minimum-candidate gate fails,
preparation is incomplete and downstream jobs must not run; `complete.json` is
written only after both splits finish. Larger beams can still have fewer eligible
items because of invalid IDs, collisions or eligibility restrictions.

Candidate-based evaluation/reranking and training adaptations are documented in
[fairness.md](fairness.md). Batch execution is documented in [slurm.md](slurm.md).
