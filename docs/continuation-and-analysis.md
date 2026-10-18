# Continuation tuning and paired analysis

This snapshot includes the current continuation-selection, generation, pilot and
paired-analysis code. Generate the checkpoint exports, processed tables and
matched ordinary controls described in [experiments.md](experiments.md) and
[fairness.md](fairness.md) before running these workflows. No historical result
files are required for the synthetic example, and none are shipped in Git.

## Validation selection

`continue-beauty` supports four `--selection-objective` values: `recall`,
`popularity`, `brand`, and `both`. An ordinary control selects the greatest
validation Recall@5, with earlier updates breaking ties and update zero eligible.
A fairness objective compares against that matched ordinary control. The default
relative recall-loss budget is 0.02; selected fairness axes cannot worsen and at
least one must improve. Joint selection first minimizes the worst relative
fairness ratio, then their sum, then prefers higher recall and an earlier update.
Test measurements are excluded from the selection rule.

RW also supports `--rw-target history` and the explicitly labeled `uniform`
variant. `--rw-strength` controls interpolation with ordinary loss; strength zero
recovers the unweighted control. Do not conflate a changed weighting target with
a new result from an unchanged baseline definition.

For a new ordinary run, use `--validation-only` to defer testing. After freezing a
selection, evaluate saved selected weights with:

```bash
python -m additional_baselines evaluate-continuation \
  --run-dir artifacts/beauty-ordinary \
  --output-dir artifacts/beauty-ordinary-test --device cuda
```

The selected checkpoint can be restored and tested without repeating training.
As with candidate preparation, destination folders must be new.

## Tune matched RW and SPRINT variants

The default search contains nine RW settings (history/popularity,
uniform/popularity and uniform/brand targets at strengths 0.25, 0.5 and 1.0),
four SPRINT scales, and three SPRINT term ablations. It keeps the ordinary
control's seed, update budget, validation interval, beam width, learning rate,
optimizer-state policy, checkpoint and SID identity fixed.

```bash
python -m additional_baselines.tune_continuation \
  --control-report artifacts/beauty-ordinary/continuation.json \
  --export-dir /path/to/score-export \
  --data-dir recommendation/processed_dataset/amazon-beauty \
  --dataset amazon-beauty --tokenizer rkmeans \
  --objective both --max-relative-recall-loss 0.02 \
  --output-dir artifacts/beauty-fairness-search --dry-run
```

The dry run validates the named artifacts and prints the plan without training
or writing outputs. Remove `--dry-run` to execute. `--trials` accepts a JSON list
of explicit trial settings. Test evaluation remains deferred unless
`--evaluate-winners` is requested; only frozen per-method winners are tested.
A method with no feasible improvement retains the matched ordinary control.

Submit the same arguments through `scripts/slurm/tune_continuation.sh` for Slurm.
The matrix helpers `scripts/run_joint_fairness_matrix.py` and
`scripts/run_joint_fairness_trials.py` provide prepare/initialize/trial/finalize
phases. Their twelve-case preparation convention covers four Amazon datasets by
three tokenizers, using the research helper artifact names in
`prepare_rerank_candidates.py`. Resolve those local control/export locations or
use the explicit single-case tuner above. A prepared manifest pins inputs and
code; regenerate it if paths, inputs, or implementation change. Incomplete trial
sets cannot select a winner or evaluate test results. Set array bounds from the
prepared trial count rather than assuming every custom search has 192 trials.

## Batched generation and deadline pilots

`batched_generation.py` generates requests in batches while preserving request
order, targets and eligibility. `generation_parity.py` compares candidate sets,
item-aligned probabilities and reported top-K rankings against single-request
inference. Differences confined to lower ranks are recorded separately. Faster
execution is not evidence of equivalent candidates; inspect the parity report
before comparing results.

The deadline pilot modules and `scripts/run_deadline_pilot*.py` retain stratified,
paired validation sampling and manifest checks for bounded development runs.
Their Slurm wrappers are `scripts/slurm/deadline_pilot.sh` and
`scripts/slurm/deadline_pilot_v2.sh`. Read `--help` for the applicable manifest
schema. Label sampled/deadline runs as pilots; do not report them as full runs.

## Paired activity and popularity analysis

The analysis modules consume saved, verified predictions. They do not train a
model or choose a method using test performance. `paired_activity.py` aligns
users/requests and computes paired recall/NDCG differences, user-bootstrap
intervals, activity contrasts and multiple-comparison adjustments.
`paired_activity_all_methods.py` includes all training branches;
`paired_activity_fairdiverse.py` checks FairDiverse evaluation membership;
`popularity_fairness_analysis.py` reconstructs membership under the historical
dense evaluator's float32 and tie-order conventions.

Install the `analysis` extra for streaming JSON and plotting support. For an
already prepared manifest:

```bash
python -m additional_baselines.paired_activity run \
  --manifest artifacts/activity/manifest.json --case 0
python -m additional_baselines.paired_activity collect \
  --manifest artifacts/activity/manifest.json --no-plots
```

Use `--help` on each module to prepare its own manifest or convert the preceding
analysis manifest. Preparation helpers retain their documented research naming
conventions; supply locally generated inputs rather than expecting bundled
results. Some compute-audit and report-builder utilities also expect generated
result inventories. W&B audits read caller-supplied project paths from
`WANDB_RECOMMENDATION_PROJECT_PATH` and `WANDB_GRID_PROJECT_PATH`.

Use the portable `paired_activity*` and collection wrappers under
`scripts/slurm/` for the corresponding modules. Keep run/merge manifests matched
and preserve their fingerprints. The analyses condition on saved checkpoints
and the recorded training seed; user-bootstrap intervals do not estimate
training-seed uncertainty. Report candidate coverage, list-fill effects, source
split identity and the treatment/control definition alongside paired effects.
