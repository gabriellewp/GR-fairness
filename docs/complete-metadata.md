# Complete-metadata experiment protocol

The released GRID workflow uses `complete_metadata` for all four Amazon datasets
and all three tokenizers: RKMeans, RVQ and RQ-VAE. Item text combines title,
categories, brand (or store) and description. Description-only experiments,
checkpoints and result records are outside this release's comparison cohort.

The completed comparison covers all twelve dataset–tokenizer cases. Its main
result inventory has 564 populated cutoff rows, including the classic baselines;
switching the published recipes to the correct metadata cohort requires no new
training or changes to those recorded results.

The four `experiments_config_<dataset>_grid_tokenizers.json` files under
`configs/reproduction/` contain 12 original configurations and 84 FairDiverse
training configurations, refreshed from the saved run settings used by the
completed tables. `complete_metadata_manifest.json` maps each case to its
experiment IDs and native artifact paths. Personal execution paths and W&B
settings are replaced with portable paths and disabled logging.

These are recipes for existing artifacts; checkpoints, SIDs and datasets must be
obtained or regenerated separately. Keep the checkpoint, tokenizer SID mapping,
item mapping and data namespace together. Relabeling a description checkpoint
does not convert it to complete metadata. Match its training provenance as well.

The maintained preprocessing CLI defaults to complete metadata. The native
pipeline rejects description mode, the bridge verifies the actual GRID
`split_config.json`, and model/configuration entrypoints reject description
artifact paths. Bridge version 4 records the metadata mode in its manifest and
rebuilds older bridge caches from verified complete-metadata inputs.

For a fresh run, follow [data preparation](data.md) and
[experiment preparation](experiments.md). For an existing research workspace with
the completed table inputs and saved configurations, recipes can be refreshed by:

```bash
python scripts/curate_complete_metadata_configs.py --source-root /path/to/experiment-workspace
```

The source workspace is read only. The command reads the comparison CSV and its
provenance, verifies saved configuration fingerprints, selects complete-metadata
run configs and updates the four release
recipe files plus their case manifest. It validates every case before writing.
`sync_grid_independent_experiments.py` preserves these curated settings instead
of recreating older unscoped Beauty recipes.

The paper's historical Recall-selected RW/SPRINT runs and the later joint-fairness
screening remain distinct experiments. Both must use complete metadata; metadata
agreement does not make their selection policies interchangeable.
