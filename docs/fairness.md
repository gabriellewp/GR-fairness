# Fairness methods and candidate workflows

## Shared artifacts

A candidate artifact contains ordered requests with unique request IDs,
`candidate_ids`, finite scores, held-out targets, catalog group mappings, and
provenance. Metadata records checkpoint/SID references and SHA-256 fingerprints,
the split, seed, generation settings, score convention, candidate eligibility and
short-list policy. A user may have multiple requests: do not reorder by user ID.

Group mappings include the complete evaluation catalog and zero-exposure groups.
Unknown brands must be explicitly labeled. Probability, log probability, negative
log probability, and arbitrary utility have different conventions; do not infer
candidate membership from a score threshold or targets. Generated zero-probability
beams remain candidates. Targets absent from candidates remain evaluation misses.

```bash
python -m additional_baselines evaluate artifacts/beauty-candidates/test_candidates.json \
  --topk 5 10 20 --output artifacts/beauty-control.json
python -m additional_baselines rerank artifacts/beauty-candidates/test_candidates.json \
  --config additional_baselines/configs/cpfair_fairdiverse.json \
  --output artifacts/beauty-cpfair.json
```

The FairDiverse CPFair/P-MMF adapters require at least K eligible candidates per
request and fail before allocation when a list is shorter; reduce K or regenerate
candidates. Shared control evaluation still supports short lists without padding.

The CPFair example's parameters are illustrative. Select strength on validation
before claiming test performance. P-MMF uses
`additional_baselines/configs/pmmf_fairdiverse.example.json`; replace its example
provider weights with positive weights for your actual group IDs. It needs CVXPY
and a QP solver. Dense memory guards/horizon options are explicit: changing
allocation horizon changes the method's state-reset policy and must be reported.
The fast P-MMF implementation offers lookup and exact-projection variants; its
backend identity belongs in the run manifest.

## Validation selection

```bash
python -m additional_baselines.tune_beauty_reranking --method cpfair \
  --candidates artifacts/beauty-candidates/validation_candidates.json \
  --histories artifacts/beauty-candidates/validation_histories.json \
  --split recommendation/processed_dataset/amazon-beauty/amazon-beauty.valid.ranking \
  --activity recommendation/processed_dataset/amazon-beauty/user_activity.csv \
  --output-dir artifacts/beauty-cpfair-validation
```

The tuner checks complete validation request identity and the split fingerprint.
It writes metric rows, a Pareto frontier and a manifest. `--method pmmf` tunes
P-MMF; `--method rr` tunes IFairLRS RR. Choose the utility/fairness tradeoff using
validation only, copy those frozen parameters into your test config, and preserve
the validation selection manifest alongside test results. `--limit-requests`
creates a smoke-prefix scope that cannot be represented as a full result.

## IFairLRS RR

RR fits corrections from validation histories and applies them without refitting:

```bash
python -m additional_baselines ifairlrs-fit-rr \
  artifacts/beauty-candidates/validation_candidates.json \
  --histories artifacts/beauty-candidates/validation_histories.json \
  --group popularity --alpha 0.05 --output artifacts/beauty-rr-correction.json
python -m additional_baselines ifairlrs-apply-rr \
  artifacts/beauty-candidates/test_candidates.json \
  --correction artifacts/beauty-rr-correction.json \
  --histories artifacts/beauty-candidates/test_histories.json \
  --topk 5 10 20 --output artifacts/beauty-rr-test.json
```

Here `0.05` is an example strength. Fit group definitions/popularity on training
interactions, choose correction strength on validation, and freeze before test.

## Matched ordinary, RW and SPRINT continuations

Provide your source export explicitly. Example ordinary control:

```bash
python -m additional_baselines continue-beauty --mode ordinary \
  --dataset amazon-beauty --tokenizer rkmeans --export-dir /path/to/score-export \
  --data-dir recommendation/processed_dataset/amazon-beauty --device cuda \
  --max-updates 32 --validation-interval 8 --learning-rate 0.001 \
  --optimizer-state require --output-dir artifacts/beauty-ordinary
```

Then run `--mode rw` or `--mode sprint` with the same arguments, a new output
folder, and `--control-report artifacts/beauty-ordinary/continuation.json`.
RW reweights loss using training group histories. SPRINT adds encoder regularizers.
Use validation-selected weights for test; a zero-update winner restores the source
checkpoint. Optimizer-state policy is explicit. Control-improvement gates prevent
unmatched fairness comparisons unless deliberately overridden and documented.

`--train-users` and `--validation-requests` are smoke limits. Do not report those
runs as full-dataset results. Standalone paired IFairLRS and SPRINT entrypoints
also remain available; inspect `python -m additional_baselines --help`.

## Legacy sparse score exports

`import-legacy` supports the verified FairDiverse probability-plus-one convention.
Compact-grid exports represent stored positive-probability membership, which can
lose beams through underflow; they are not complete native beam membership.
Ambiguous scores or unknown request order require regeneration. Dense exports
need an explicit aligned candidate mask. Fresh native candidate artifacts are
preferred when score magnitude or complete membership matters.

The current validation selection policies, RW variants, continuation sweeps and
paired analysis workflows are documented in [continuation-and-analysis.md](continuation-and-analysis.md).
