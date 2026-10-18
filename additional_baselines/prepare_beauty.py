"""Validate/load the original Beauty export; optionally serialize a package."""

import csv
import json
from pathlib import Path
import re

import numpy as np
from scipy.sparse import load_npz
import torch
import yaml

from .adapters import import_legacy
from .candidates import file_reference, save


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPORT = "artifacts/exports/amazon-beauty/rkmeans"
DEFAULT_DATA = "recommendation/processed_dataset/amazon-beauty"
DEFAULT_OUTPUT = "additional_baselines/artifacts/beauty_rkmeans_suffix4"


def load_beauty(*, root=ROOT, export_dir=DEFAULT_EXPORT,
               data_dir=DEFAULT_DATA, topk=(5, 10, 20), max_dense_bytes=4294967296,
               check_dense_memory=True, tokenizer="rkmeans", dataset="amazon-beauty"):
    """Return inputs, default configs and summary entirely in memory."""
    root = Path(root).resolve()

    def local(value):
        path = Path(value)
        return (root / path).resolve() if not path.is_absolute() else path.resolve()

    export_dir, data_dir = map(local, (export_dir, data_dir))
    if not topk or any(type(k) is not int or k <= 0 for k in topk) or len(set(topk)) != len(topk):
        raise ValueError("topk must contain distinct positive integers")
    if type(max_dense_bytes) is not int or max_dense_bytes <= 0:
        raise ValueError("max_dense_bytes must be a positive integer")
    config_path = export_dir / "config.yaml"
    config = yaml.safe_load(config_path.read_text())
    if tokenizer not in {"rkmeans", "rqvae", "rvq"}:
        raise ValueError("Unsupported Beauty GRID tokenizer")
    if dataset not in {"amazon-beauty", "amazon-toys", "amazon-sports", "amazon-pet-supplies"}:
        raise ValueError(f"Unsupported GRID dataset: {dataset}")
    expected = {"dataset": dataset, "model": "GRID", "grid_tokenizer": tokenizer,
                "num_hierarchies": 4, "grid_compact_ranking": True, "use_beam_search_eval": True,
                "grid_eval_only": True, "metadata_text_mode": "complete_metadata"}
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"Export must declare {key}={value!r}")
    checkpoint = local(config["grid_checkpoint_path"])
    sid_path = local(config["sid_path"])
    training_config_path = checkpoint.parent.parent / ".hydra/config.yaml"
    training = yaml.safe_load(training_config_path.read_text())
    if training.get("seed") != 42 or training.get("num_hierarchies") != 4:
        raise ValueError("Training configuration must confirm seed 42 and four SID tokens")
    if local(training["semantic_id_path"]) != sid_path:
        raise ValueError("Training and export SID mappings differ")
    # Independently verify inference seed; never fill it in from a default.
    log_path = root / "output" / f"{config['slurm_job_id']}_{config['slurm_array_task_id']}.out"
    log = log_path.read_text()
    seeds = {int(seed) for seed in re.findall(r"\[SEED\] seed=(\d+)", log)}
    if seeds != {42} or config["log_name"] not in log or config.get("seed", 42) != 42:
        raise ValueError("Export job log/config does not unambiguously confirm seed 42 for this run")
    manifest_path = data_dir / "grid_data_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if local(manifest["source_dir"]) != local(config["grid_data_dir"]) or manifest["split_strategy"] != "temporal_loo":
        raise ValueError("Processed Beauty tables come from a different GRID source/split")
    item_mapping_path = local(config["grid_item_mapping_path"])
    item_mapping = json.loads(item_mapping_path.read_text())
    iid_mapping_path = data_dir / "iid2item.json"
    iid_mapping = json.loads(iid_mapping_path.read_text())
    expected_iids = {str(int(iid)): str(item) for item, iid in item_mapping.items()}
    if iid_mapping != expected_iids or set(map(int, iid_mapping)) != set(range(config["item_num"])):
        raise ValueError("GRID and processed Beauty item IDs do not match")
    group_path = data_dir / "iid2pid.json"
    group_mapping = {int(item): str(group) for item, group in json.loads(group_path.read_text()).items()}
    if set(group_mapping) != set(range(config["item_num"])) or len(set(group_mapping.values())) != config["group_num"]:
        raise ValueError("Incomplete or incompatible brand mapping")
    process_config_path = data_dir / "process_config.yaml"
    process_config = yaml.safe_load(process_config_path.read_text())
    for key in ("group_id", "group_aggregation_threshold"):
        if process_config.get(key) != config.get(key):
            raise ValueError(f"Group construction setting differs from export: {key}")
    sid_table = torch.load(sid_path, map_location="cpu", weights_only=True)
    if not isinstance(sid_table, torch.Tensor) or tuple(sid_table.shape) not in {
        (4, config["item_num"]), (config["item_num"], 4)
    }:
        raise ValueError("SID tensor must have four tokens per catalog item")
    split_path = data_dir / f"{dataset}.test.ranking"
    with split_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    targets = load_npz(export_dir / "ground_truth.npz").tocsr()
    if targets.shape != (len(rows), config["item_num"]) or len(rows) != config["user_num"]:
        raise ValueError("Target matrix and Beauty request counts do not match")
    if not np.all(np.diff(targets.indptr) == 1) or not np.all(targets.data == 1):
        raise ValueError("Expected one positive held-out target per Beauty request")
    if targets.indices.tolist() != [int(row["target_item"]) for row in rows]:
        raise ValueError("Export targets do not match the current Beauty test row order")
    request_ids = [f"test:{row_index}:user:{row['user_id']}" for row_index, row in enumerate(rows)]
    sources = {"export_config": config_path, "training_config": training_config_path,
               "export_seed_log": log_path, "grid_manifest": manifest_path,
               "grid_item_mapping": item_mapping_path, "iid_mapping": iid_mapping_path,
               "group_mapping": group_path, "group_process_config": process_config_path}
    metadata = {
        "dataset": dataset, "checkpoint": file_reference(checkpoint),
        "sid_mapping": file_reference(sid_path), "split": {"name": "test", "source": file_reference(split_path)},
        "seed": 42, "training_seed": 42, "export_seed": 42,
        "generation": {key: config[key] for key in ("grid_tokenizer", "num_hierarchies", "top_k_for_generation",
                        "use_prefix_validation", "history_length", "sequence_length", "metadata_text_mode")},
        "score_convention": "probability", "candidate_membership": "compact_grid_stored_entries",
        "short_list_policy": "return_available", "request_order": "unaltered_test_table_row_order",
        "provenance": {name: file_reference(path) for name, path in sources.items()},
    }
    artifact = import_legacy(export_dir / "ranking_scores.npz", export_dir / "ground_truth.npz",
                             request_ids, {"brand": group_mapping}, metadata, export_format="compact_grid")
    counts = [len(r.candidate_ids) for r in artifact.requests]
    if min(counts) < max(topk):
        raise ValueError(f"Minimum candidate count {min(counts)} is below requested K={max(topk)}")
    estimate = 8 * len(group_mapping) * (len(rows) + len(set(group_mapping.values())) + 8)
    if check_dense_memory and estimate > max_dense_bytes:
        raise ValueError(f"Dense reranking estimate {estimate} exceeds max_dense_bytes={max_dense_bytes}")
    parameters = {"group": "brand", "horizon": None, "order_seed": None, "max_dense_bytes": max_dense_bytes}
    cpfair = {"method": "cpfair_fairdiverse", "topk": list(topk), "trace": False,
              "parameters": {**parameters, "fairness_lambda": 1.0}}
    pmmf = {"method": "pmmf_fairdiverse", "topk": list(topk), "trace": False,
            "parameters": {**parameters, "fairness_lambda": 30.0, "learning_rate": 0.5, "gamma": 0.2,
                           "weights": dict.fromkeys(sorted(set(group_mapping.values())), 1.0)}}
    summary = {"export_dir": str(export_dir), "request_count": len(rows), "item_count": len(group_mapping),
               "group_count": len(set(group_mapping.values())), "minimum_candidates": min(counts),
               "maximum_candidates": max(counts), "training_seed": 42, "export_seed": 42,
               "topk": list(topk), "estimated_dense_bytes": estimate,
               "pmmf_weights": "all ones, matching FairDiverse defaults; budgets generally nonbinding",
               "purpose": "fixed-default test evaluation, not validation tuning"}
    return artifact, {"CPFair": cpfair, "PMMF": pmmf}, summary


def prepare(output_dir=DEFAULT_OUTPUT, *, root=ROOT, export_dir=DEFAULT_EXPORT,
            data_dir=DEFAULT_DATA, topk=(5, 10, 20), max_dense_bytes=4294967296):
    """Optional persistent package; direct reranking uses load_beauty instead."""
    output_dir = Path(output_dir)
    if not output_dir.is_absolute():
        output_dir = Path(root).resolve() / output_dir
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}. Reuse inputs.json or choose a new --output-dir.")
    artifact, configs, summary = load_beauty(root=root, export_dir=export_dir, data_dir=data_dir,
                                          topk=topk, max_dense_bytes=max_dense_bytes)
    request_ids = [r.request_id for r in artifact.requests]
    summary["output_dir"] = str(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    save(artifact, output_dir / "inputs.json")
    for name, value in (("request_ids.json", request_ids), ("item_groups.json", artifact.item_groups),
                        ("metadata.json", artifact.metadata), ("cpfair.json", configs["CPFair"]), ("pmmf.json", configs["PMMF"]),
                        ("preparation_summary.json", summary)):
        with (output_dir / name).open("x") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
    return summary
