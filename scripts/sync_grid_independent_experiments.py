#!/usr/bin/env python3
"""Upsert FairDiverse experiments backed by the verified local GRID runs."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/reproduction/experiments_config_amazon-beauty_grid_tokenizers.json"
FIRST_EXPERIMENT_ID = 57
FAIR_RANKERS = ("APR", "FairDual", "IPS", "Minmax_SGD", "SDRO", "FOCF", "Reg")


def canonical_config(tokenizer: str, *, fair_ranker: str | None) -> dict:
    fair_rank = fair_ranker is not None
    config = {
        "model": "GRID",
        "data_type": "sequential",
        # Preprocessing is a shared prerequisite. Running it in every Slurm
        # array task creates races in recommendation/processed_dataset.
        "reprocess": False,
        "grid_data_source": "embedded_tfrecords",
        "grid_data_dir": "GRID/data/amazon_beauty_5core_tfrecords/temporal_loo",
        "grid_compact_ranking": True,
        "grid_tokenizer": tokenizer,
        "tokenizers": tokenizer,
        "embedding_model": "t5xl",
        "sid_path": (
            "GRID/outputs/beauty_5core/temporal_loo/"
            f"{tokenizer}/step2b_sid_inference/pickle/merged_predictions_tensor_suffix4.pt"
        ),
        "grid_item_mapping_path": (
            "GRID/data/amazon_beauty_5core_tfrecords/temporal_loo/item_id_mapping.json"
        ),
        "grid_checkpoint_path": "auto",
        "grid_eval_only": not fair_rank,
        # FairDiverse full-catalog metrics are primary for both the untouched
        # baseline and the fair-ranking variants, making their W&B results
        # directly comparable. Baselines additionally retain standalone-style
        # native beam metrics as a local diagnostic JSON.
        "grid_native_metrics": False,
        "grid_native_diagnostics": not fair_rank,
        # Fairness methods start from the verified native checkpoint and run
        # only a short, explicitly bounded in-processing phase.
        "grid_fair_finetune": fair_rank,
        "grid_fair_finetune_steps": 100 if fair_rank else 0,
        "num_hierarchies": 4,
        "codebook_size": 256,
        # Preserve all 40 history items after appending the fourth SID token.
        "history_length": 40,
        "sequence_length": 160,
        "padding_item_id": -1,
        "embedding_size": 128,
        "d_kv": 64,
        "inner_size": 1024,
        "n_heads": 6,
        "n_layers": 4,
        "hidden_dropout_prob": 0.15,
        # Generate enough native GRID beams to support item-level evaluation
        # beyond @10. All baseline and fair-ranking runs use the same width.
        "top_k_for_generation": 100,
        "use_beam_search_eval": True,
        "use_prefix_validation": False,
        "weight_tying": True,
        "mlp_layers": 2,
        "num_user_bins": None,
        "ranking_loss_weight": 0,
        "level_loss_weights": None,
        "use_causal_augmentation": True,
        "causal_pre_batch_size": 32,
        "causal_group_by_user": True,
        "batch_size": 512,
        "accumulate_grad_batches": 16,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "epoch": 3 if fair_rank else 100,
        "max_steps": 100 if fair_rank else 320000,
        "val_check_interval_batches": 1600,
        "eval_step": 10,
        "eval_type": "ranking",
        # Native recall@5 is HR@5 for temporal leave-one-out evaluation.
        "watch_metric": "hr@5",
        "early_stop_metrics": ["hr@5"],
        "early_stop_weights": [1.0],
        "early_stop_patience": 10,
        "early_stop_min_delta": 0.0,
        "topk": [5, 10, 20],
        "store_scores": fair_rank,
        "fairness_type": "Exposure",
        "mmf_eval_ratio": 0.5,
        "decimals": 4,
        "fair-rank": fair_rank,
        "use_llm": False,
        "use_wandb": True,
        "device": "cuda",
        "seed": 42,
    }
    if fair_ranker is not None:
        config["rank_model"] = fair_ranker
    return config


def experiment(exp_id: int, tokenizer: str, fair_ranker: str | None) -> dict:
    suffix = fair_ranker if fair_ranker is not None else "base"
    config = canonical_config(tokenizer, fair_ranker=fair_ranker)
    config["log_name"] = f"grid-independent-{tokenizer}-{suffix}"
    return {
        "task": "recommendation",
        "stage": "in-processing",
        "dataset": "amazon-beauty",
        "config": config,
        "exp_id": exp_id,
    }


def desired_experiments() -> list[dict]:
    result = []
    exp_id = FIRST_EXPERIMENT_ID
    for tokenizer in ("rkmeans", "rqvae"):
        result.append(experiment(exp_id, tokenizer, None))
        exp_id += 1
        for ranker in FAIR_RANKERS:
            result.append(experiment(exp_id, tokenizer, ranker))
            exp_id += 1
    # RVQ native parity is kept as a standalone checkpoint-validation run.
    # Fair-ranking variants can be added after the independent RVQ pipeline
    # has produced and validated its SID tensor and TIGER checkpoint.
    result.append(experiment(exp_id, "rvq", None))
    return result


def main() -> None:
    experiments = json.loads(CONFIG_PATH.read_text())
    desired = desired_experiments()
    desired_ids = {entry["exp_id"] for entry in desired}
    experiments = [entry for entry in experiments if entry.get("exp_id") not in desired_ids]
    experiments.extend(deepcopy(desired))
    CONFIG_PATH.write_text(json.dumps(experiments, indent=2) + "\n")
    print(f"Updated {CONFIG_PATH} with experiment IDs {min(desired_ids)}-{max(desired_ids)}")


if __name__ == "__main__":
    main()
