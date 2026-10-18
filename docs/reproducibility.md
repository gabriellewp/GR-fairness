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
