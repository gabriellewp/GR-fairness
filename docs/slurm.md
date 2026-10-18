# Slurm usage

Activate the installed environment before submission. From the repository root:

```bash
export FAIRDIVERSE_ROOT="$PWD"
export FAIRDIVERSE_PYTHON="$(command -v python)"
export GRID_PYTHON="$FAIRDIVERSE_PYTHON"
mkdir -p output
```

All wrappers resolve the repository through `FAIRDIVERSE_ROOT`, then
`SLURM_SUBMIT_DIR`. They work when Slurm runs a spool copy of the script. Set the
root explicitly if submitting from elsewhere. No script embeds credentials,
activates a personal Conda environment, or installs packages inside the job.

## Available jobs

| Script | Work | Default resources |
| --- | --- | --- |
| `scripts/slurm/preprocess.sh` | GRID TFRecord preprocessing only | CPU, 4 cores, 32 GB |
| `scripts/slurm/grid.sh` | Full native GRID pipeline | 1 GPU, 8 cores, 100 GB |
| `scripts/slurm/experiment.sh` | One experiment ID, including classic evaluation/training | 4 cores, 32 GB; request GPU explicitly |
| `scripts/slurm/candidates.sh` | Fresh validation/test candidates | 1 GPU, 4 cores, 48 GB |
| `scripts/slurm/rerank.sh` | Artifact evaluation/reranking/IFairLRS CLI | CPU, 4 cores, 24 GB |

Override partition, memory, time, CPUs and GPU resources for your cluster.
For training adaptations submitted through `rerank.sh`, request a GPU explicitly.
Create `output/` before `sbatch`: Slurm opens log files before the script starts.

```bash
export RAW_DATA_DIR=/path/to/amazon-mcauley-2014
preprocess_job=$(sbatch --parsable --partition=cpu scripts/slurm/preprocess.sh \
  --dataset beauty --tokenizer rkmeans --metadata-text-mode complete_metadata)
preprocess_job="${preprocess_job%%;*}"
sbatch --partition=gpu --dependency="afterok:$preprocess_job" scripts/slurm/grid.sh \
  --dataset beauty --tokenizer rkmeans --metadata-text-mode complete_metadata
```

## Experiment arrays

Inspect submission without launching jobs:

```bash
bash scripts/slurm/submit_experiments.sh \
  --config_file artifacts/configs/beauty-grid.json \
  --partition gpu --gres gpu:1 --max_concurrent 2 --dry-run
```

Remove `--dry-run` to submit. The helper reads explicit IDs from the JSON list,
validates selections and concurrency, creates `output/`, and submits an array.
`--exp_ids '57 58'` selects existing IDs; defaults cover all entries. Supply runner
arguments after `--`, for example `-- --seed 42`. Resource options supported by
the helper include `--partition`, `--mem`, `--time`, `--gres`,
`--cpus-per-task`, `--dependency` and `--job-name`. The root
`sbatch_run_all_experiments.sh` forwards to this helper for compatibility.

## Candidate and reranking jobs

```bash
candidate_job=$(sbatch --parsable --partition=gpu scripts/slurm/candidates.sh \
  --dataset amazon-beauty --tokenizer rkmeans --export-dir /path/to/score-export \
  --output-dir artifacts/beauty-candidates --generation-beam-width 300 \
  --require-min-candidates 20)
candidate_job="${candidate_job%%;*}"
sbatch --partition=cpu --dependency="afterok:$candidate_job" scripts/slurm/rerank.sh \
  evaluate artifacts/beauty-candidates/test_candidates.json --topk 5 10 20 \
  --output artifacts/beauty-control.json
```

For validation tuning arrays and dependent merge, the retained portable helper is
`additional_baselines/submit_rerank_tuning_array.sh METHOD DATASET CANDIDATE_DIR`.
It uses complete validation artifacts and source activity files. Set
`GRID_RERANK_PYTHON` to your interpreter; extra `sbatch` options follow the optional
concurrency argument. Override its partition/resources for your cluster.

Use `squeue -u "$USER"` to monitor jobs and `sacct -j JOB_ID` to inspect their
exit state. `afterok` dependencies ensure a failed candidate generation job blocks
its consumers. Preserve configs, logs, selected checkpoints and output manifests
outside Git. Use new output directories for retries to keep provenance clear.

## Continuation and analysis launchers

Recent workflows also have portable wrappers in `scripts/slurm/`:

- `tune_continuation.sh` runs the explicit validation-only continuation search.
- `joint_fairness_matrix.sh` runs a prepared trial/finalization array.
- `deadline_pilot.sh` and `deadline_pilot_v2.sh` run prepared pilot cases.
- `paired_activity.sh`, `paired_activity_all_methods.sh`, and
  `paired_activity_fairdiverse.sh` process analysis cases.
- `collect_paired_activity.sh` and `merge_paired_activity_all.sh` collect reports.
- `retry_activity_compact_coo.sh` and `tokenizer_rerank_tradeoffs.sh` support
  their corresponding repair and reporting utilities.

Each wrapper inherits the shared environment configuration. Supply your own
partition, GPU resource, array bounds and manifest paths. Array jobs require
`SLURM_ARRAY_TASK_ID`; launching an array-only wrapper as an ordinary local shell
command does not provide it. See [continuation and analysis](continuation-and-analysis.md)
for input requirements, selection policy and limitations.
