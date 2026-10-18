# Metrics and reproduction

## Freeze the experiment identity

Record dataset release and raw file hashes, filtering and split strategy,
metadata text mode, item mapping, SID mapping, tokenizer/embedding identity,
checkpoint fingerprint, model/source revisions, seed, generation settings,
training schedule, optimizer-state policy and selection metric. Preserve the
actual environment (`pip freeze`), config and output manifests externally.

The four GRID tokenizer files in `configs/reproduction/` are refreshed from the
saved complete-metadata configurations behind the completed comparison tables:
12 original cases and 84 FairDiverse training recipes. Their case inventory is
`complete_metadata_manifest.json`. Other files preserve supporting configurations.
Some depend on historical checkpoint filenames and prior score exports that are
deliberately not in the release. Use `configure_experiment.py`
for a new run, or resolve all archived paths before reproducing a specific entry.
The release also retains GRID's original-RQ-VAE architecture/schedule presets.

## Comparable evaluations

The historical cohort has two evaluation paths. Native GRID and saved-candidate
reranking rank generated eligible items, with absent targets counted as misses.
The recorded FairDiverse training evaluations rank a target-first full catalog
after adding one to float32 probabilities, using descending NumPy quicksort.
Rounding-induced ties and catalog fillers can affect those orderings. The
historical recovery code preserves this behavior; those results have not been
freshly harmonized with native candidate evaluation. Replacing that evaluator
requires recomputing the affected results rather than retaining the old tables.

Keep metadata representation, train/validation/test identity, item catalog,
history length, beam width, eligibility, group definitions and cutoffs constant
when comparing methods. Native GRID uses three learned SID levels and a fourth
collision suffix. Item zero is real; padding is negative. A wider beam changes the
candidate opportunity set, so it is not an isolated reranker improvement.

For shared artifacts, accuracy is averaged over all requests, including empty
candidate sets and targets missed by generation. Hit rate means at least one hit;
recall divides hits by all held-out targets; NDCG's ideal ranking includes those
targets; MRR uses the first relevant rank. Short lists are not padded with catalog
fallbacks. Candidate coverage, list fill and empty/short-list rates are separate
diagnostics. Legacy FairDiverse HR has a different multi-target normalization;
do not compare it directly without checking definitions.

Group exposure is one count per recommended item rather than position-discounted
exposure. Gini, entropy and min/max ratios include catalog groups with zero
exposure. Undefined quantities for no exposure are `null`. Popularity and user
activity groups derive from training data. State resets, allocation horizon and
request order influence CPFair/P-MMF results and must be held fixed or reported.
The artifact API's `entropy` is in nats for backward compatibility;
`entropy_bits` and the paper tables use base two. Divide nats by `ln(2)` when
reading older artifacts. Relative entropy changes are invariant to this conversion.

## Regenerate the paper outputs

From a source checkout with the `analysis` dependencies installed:

```bash
python -m scripts.paper.reproduce --output-dir artifacts/paper
```

Choose a new output directory for each run. This command needs no dataset,
checkpoint, GPU, credentials or LaTeX installation. It checks the frozen input
hashes, renders the four dataset comparison tables, the local/published GRID and
classic-baseline tables, and the reranking summary. It regenerates the training,
popularity, combined intervention, and activity-loss figures as PDF and PNG,
plus plotting data, captions, provenance and checksums.

The inputs in `paper/inputs/` contain only aggregate measurements, group shares,
and summary statistics. They contain no raw item text, per-user predictions,
credentials or personal artifact locations. `paper/source_manifest.json` records
the source branch commit and original file hashes. Rendering these aggregates
reproduces the reported outputs; it does not independently rerun the experiments.
The manually authored integration diagram and manuscript layout are separate.

`configs/reproduction/paper_protocol.json` freezes the historical case identities,
36 matched continuation settings, 36 selected reranker settings, native schedules,
and distinct evaluation protocols. Later fairness-selected searches remain
available as explicitly separate extensions. For classic models, use
`experiments_config_paper_classics.json`, which preserves the eight saved
configurations, including Pet SASRec's embedding size of 100. The generic
`configure_experiment.py` helper remains a starter for new runs.

The historical combined figure deliberately preserves its original precision:
Recall/brand changes use the four-decimal table inputs, while popularity metrics
use unrounded recovered evaluations. `tables/unrounded_manuscript_summaries.json`
also reports unrounded RW/SPRINT changes against matched FT controls and the
four-dataset RKMeans RR range at K=10. Use this output when correcting manuscript
claims that explicitly promise unrounded calculations; do not mix its values
with the legacy figure's rounded-input summaries.

The individual builders under `scripts/paper/` accept input and output arguments.
Training/popularity builders optionally accept `--latex-preview`; only that
option requires `pdflatex`.

## Rebuild audits from the saved experiment workspace

The restored archive tools read externally stored native configs, run logs,
prediction exports, selection records and reports. They do not ship those large
or private artifacts in Git. Pass the historical workspace explicitly and choose
an output directory outside it:

```bash
python -m scripts.paper.run_archive audit-training \
  --source-root /path/to/experiment-workspace \
  --output-dir /path/to/new-training-audit
```

Available tools are `assemble-comparison`, `assemble-rkmeans`, `audit-training`,
`audit-continuation`, `audit-reranking`, `audit-datasets`, `audit-environment`,
`audit-rq1`, `parameter-inventory`, `compare-source-parameters`, and `compute-cost`.
These retain the historical cohort's record layout and fingerprint checks.
Keep referenced artifacts resolvable; a partial copy containing only code cannot
run a full archive audit. Some tools read large prediction files. Compute-cost
network and scheduler queries require explicit `--online`, `--native-online`
or `--slurm` options; the default only reads local records.

`audit-rq1` produces the inventory accepted by
`scripts/check_rq1_grid_agreement.py --inventory /path/to/inventory.json`.
To regenerate popularity recovery from external frozen manifests, use
`python -m additional_baselines.popularity_fairness_analysis --case 0
--source-manifest /path/to/original/manifest.json
--fairdiverse-manifest /path/to/all-methods/manifest.json
--output-dir /path/to/new-recovery` (one shell command).

The seven FairDiverse branches validated once at their recorded update cap
(100 updates, or 50 for FairDual), with no original-checkpoint fallback.
The matched FT/RW/SPRINT branches validated at updates 0/8/16/24/32. RW shares
count pooled nonpadding history and target occurrences over the actual augmented
replay, including repeats. SPRINT regularizes encoder self-attention and FFN
blocks plus the shared SID embedding, with coefficients 1/0.001/0.1,
temperature 4 and one power iteration. RR fits corrections across cutoffs
1/5/10/20; parameter selection uses validation K=5.

## Selection and artifacts

Tune on validation, select/freeze parameters and checkpoints, then test once.
RW/SPRINT comparisons require matched ordinary controls. Preserve optimizer state
when the selected protocol requires it. Artifact validation checks metadata shape
and fingerprints at production/verification steps; arbitrary remote paths are not
automatically re-read on every load. Keep the referenced files available.

Store datasets, checkpoints, score matrices, candidate artifacts and logs in an
artifact store or an explicitly shared external directory. If distributing final
result tables, include their protocol/config and source provenance. Dated slides,
research drafts and intermediate audit snapshots remain on the research branch.

For fairness-selected continuations, freeze the stated recall-loss budget and
fairness axes before search. Batched generation must pass the reported-rank
parity checks. Paired user-bootstrap analysis conditions on a fixed checkpoint
and seed; it does not establish robustness across training seeds.
