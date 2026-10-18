# Data preparation

## Dataset identity

The maintained GRID pipeline supports these inputs:

| FairDiverse dataset | GRID selector | Source release | Default input filenames |
| --- | --- | --- | --- |
| `amazon-beauty` | `beauty` | Amazon 2014 | `meta_Beauty_5core.json.gz`, `reviews_Beauty_5.json.gz` |
| `amazon-sports` | `sports` | Amazon 2014 | `meta_Sports_and_Outdoors_5core.json.gz`, `reviews_Sports_and_Outdoors_5.json.gz` |
| `amazon-toys` | `toys` | Amazon 2014 | `meta_Toys_and_Games_5core.json.gz`, `reviews_Toys_and_Games_5.json.gz` |
| `amazon-pet-supplies` | `pet-supplies` | Amazon 2023 | `meta_Pet_Supplies_5core.jsonl.gz`, `Pet_Supplies.csv.gz` |

Acquire the applicable McAuley release and obey its access/redistribution terms.
`scripts/prepare_amazon.py sports toys` downloads/filters the supported 2014
inputs into `data/raw/amazon-mcauley-2014`. Pet Supplies uses the 2023 source;
run `python scripts/prepare_amazon.py pet-supplies`. Use `--data-dir` to change
the destination. Beauty inputs should be supplied locally as listed above.
Metadata may need filtering to the retained interaction catalog. Set
`METADATA_FILE` and `INTERACTIONS_FILE` to your actual filenames when they differ.
Do not substitute Amazon releases while claiming a matched reproduction.

## Build GRID data

From the FairDiverse repository root:

```bash
export GRID_PYTHON="$(command -v python)"
export RAW_DATA_DIR=/path/to/amazon-mcauley-2014
export GRID_USE_WANDB=false
bash GRID/run_independent_pipeline.sh --dataset beauty --tokenizer rkmeans \
  --metadata-text-mode complete_metadata --preprocess-only
```

Use the other selectors in the table for additional datasets. The default split
is `temporal_loo`: final item is test, preceding item is validation, earlier
items form training. Default complete metadata text combines title, categories,
brand and description. `description` is a separate explicit compatibility mode.
Do not reuse its outputs for complete-metadata experiments.

TFRecords, `item_id_mapping.json`, `split_config.json`, and preprocessing stats
are written beneath:

```text
GRID/data/amazon_beauty_5core_tfrecords/temporal_loo/complete_metadata/
  training/  validation/  testing/  items/
```

The minimum sequence length is four for temporal leave-one-out preprocessing.
The preprocessing module also exposes random splits, but those are not matched
to this release's temporal-LOO comparison protocol.

## Build matching FairDiverse tables

The GRID evaluation adapter normally builds these automatically. For classic
baselines, build them explicitly first:

```bash
python scripts/prepare_fairdiverse.py --dataset amazon-beauty \
  --grid-data-dir GRID/data/amazon_beauty_5core_tfrecords/temporal_loo/complete_metadata
```

The bridge reads the existing GRID TFRecord splits without resplitting them.
It writes `recommendation/processed_dataset/<dataset>/` containing training,
validation/test CTR and ranking tables, item/provider/text mappings, dataset
stats, process config, a source-fingerprinted manifest, and user activity groups.
A matching manifest permits reuse. A partial/corrupt source fails rather than
silently falling back to another dataset.

Item ID zero is a real item in this aligned catalog. Sequence padding is `-1`.
Never apply legacy item-zero padding rules to these data. The native comparison
uses 40 history items, represented as 160 SID tokens across four levels.
Popularity/activity definitions are derived from training interactions.

Only one canonical processed mapping per dataset lives in the default directory.
Switching metadata modes rebuilds that mapping. Finish dependent experiments
before switching, or preserve and explicitly reference separate processed data.
