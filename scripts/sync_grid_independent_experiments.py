#!/usr/bin/env python3
"""Validate and order the curated complete-metadata Beauty experiment recipes."""
import json
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/reproduction/experiments_config_amazon-beauty_grid_tokenizers.json"
FIRST_EXPERIMENT_ID = 57
FAIR_RANKERS = ("APR", "FairDual", "IPS", "Minmax_SGD", "SDRO", "FOCF", "Reg")


def canonical_config(tokenizer, *, fair_ranker):
    entries = json.loads(CONFIG_PATH.read_text())
    matches = [entry["config"] for entry in entries if entry["config"].get("grid_tokenizer") == tokenizer
               and (entry["config"].get("rank_model") if entry["config"].get("fair-rank") else None) == fair_ranker]
    if len(matches) != 1:
        raise ValueError("Expected a single curated recipe; refresh from saved run configurations first")
    config = deepcopy(matches[0])
    if config.get("metadata_text_mode") != "complete_metadata":
        raise ValueError("Description-only recipes are excluded; refresh from complete_metadata runs")
    for key in ("grid_data_dir", "grid_item_mapping_path", "sid_path", "grid_checkpoint_path"):
        if "complete_metadata" not in Path(config[key]).parts:
            raise ValueError(f"{key} must point to complete_metadata artifacts")
    return config


def experiment(exp_id, tokenizer, fair_ranker):
    return {"task": "recommendation", "stage": "in-processing", "dataset": "amazon-beauty",
            "config": canonical_config(tokenizer, fair_ranker=fair_ranker), "exp_id": exp_id}


def desired_experiments():
    return [experiment(FIRST_EXPERIMENT_ID + index, tokenizer, ranker)
            for index, (tokenizer, ranker) in enumerate(
                (tokenizer, ranker) for tokenizer in ("rkmeans", "rqvae", "rvq")
                for ranker in (None, *FAIR_RANKERS))]


def main():
    experiments = desired_experiments()
    CONFIG_PATH.write_text(json.dumps(experiments, indent=2) + "\n")
    print(f"Validated {len(experiments)} complete_metadata recipes in {CONFIG_PATH}")


if __name__ == "__main__":
    main()
