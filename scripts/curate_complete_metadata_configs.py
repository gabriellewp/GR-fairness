"""Refresh released GRID recipes from the completed comparison's saved run configs."""
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("amazon-beauty", "amazon-sports", "amazon-toys", "amazon-pet-supplies")
TOKENIZERS = ("rkmeans", "rqvae", "rvq")
RANKERS = ("APR", "FairDual", "IPS", "Minmax_SGD", "SDRO", "FOCF", "Reg")
ARTIFACT_KEYS = ("grid_data_dir", "grid_item_mapping_path", "sid_path", "grid_checkpoint_path")
EXTRA_SETTINGS = {"group_aggregation_threshold", "exposure_sample_num", "s_k", "fair_lambda",
                  "M", "beta", "eta", "alpha", "p", "lambd", "update_epoch", "use_8bit"}


def read_config(path, expected_sha256=None):
    raw = path.read_bytes()
    if expected_sha256 and hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError(f"Saved configuration changed since the completed comparison: {path.name}")
    config = yaml.safe_load(raw)
    if config.get("metadata_text_mode") != "complete_metadata":
        raise ValueError(f"Only complete_metadata run configurations are eligible: {path.name}")
    return config


def local_path(root, value):
    path = (root / value).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Recorded source must be inside --source-root")
    return path


def portable_config(template, recorded, source_root, dataset, tokenizer, ranker):
    expected = dict(dataset=dataset, model="GRID", grid_tokenizer=tokenizer)
    if any(recorded.get(key) != value for key, value in expected.items()):
        raise ValueError("Saved configuration differs from the declared dataset/tokenizer")
    if bool(recorded.get("fair-rank")) != bool(ranker):
        raise ValueError("Saved configuration differs from the declared training branch")
    if ranker and recorded.get("rank_model") != ranker:
        raise ValueError("Saved configuration differs from the declared fairness method")
    config = copy.deepcopy(template)
    for key in set(template) | EXTRA_SETTINGS:
        if key in recorded:
            config[key] = recorded[key]
    for key in ARTIFACT_KEYS:
        path = Path(recorded[key])
        if path.is_absolute():
            path = local_path(source_root, path).relative_to(source_root)
        if "complete_metadata" not in path.parts:
            raise ValueError(f"{key} must identify complete_metadata artifacts")
        config[key] = path.as_posix()
    config.update(metadata_text_mode="complete_metadata", embedding_model="google/flan-t5-xl",
                  use_wandb=False, log_name=f"paper-{dataset}-{tokenizer}-{ranker or 'base'}-complete-metadata")
    if not ranker:
        config.pop("rank_model", None)
    return config


def curate(source_root, release_root=ROOT):
    source_root, release_root = Path(source_root).resolve(), Path(release_root).resolve()
    if source_root == release_root:
        raise ValueError("--release-root must be separate from the source experiment workspace")
    comparison = source_root / "docs/rq2_comparison_draft_20261011"
    provenance = json.loads((comparison / "provenance.json").read_text())
    rows = list(csv.DictReader((comparison / "comparison_results.csv").open()))
    training = {(r["dataset"], r["tokenizer"], r["method"]): r for r in provenance["training_runs"]}
    originals = {(r["dataset"], r["tokenizer"]): r for r in rows
                 if r["method"] == "GRID: original (fresh evaluation)" and r["k"] == "10"}
    manifest = {"metadata_text_mode": "complete_metadata", "source": "Completed comparison and saved per-run configurations",
                "cases": [], "training_configurations": 0, "original_configurations": 0,
                "artifact_policy": "External checkpoints/data are required; only complete_metadata namespaces are eligible."}
    prepared = {}
    for dataset in DATASETS:
        target = release_root / "configs/reproduction" / f"experiments_config_{dataset}_grid_tokenizers.json"
        entries = json.loads(target.read_text())
        for tokenizer in TOKENIZERS:
            report = json.loads(local_path(source_root, originals[dataset, tokenizer]["source"]).read_text())
            config_reference = (report["case"]["config"] if "case" in report else
                                report["input_metadata"]["provenance"]["export_config"])
            config_path = local_path(source_root, config_reference["path"])
            base = read_config(config_path, config_reference["sha256"])
            case = {"dataset": dataset, "tokenizer": tokenizer, "experiment_ids": {},
                    "checkpoint": None, "sid_mapping": None}
            for ranker in (None, *RANKERS):
                matches = [entry for entry in entries if entry["config"].get("grid_tokenizer") == tokenizer
                           and (entry["config"].get("rank_model") if entry["config"].get("fair-rank") else None) == ranker]
                if len(matches) != 1:
                    raise ValueError("Expected exactly one recipe for each complete-metadata branch")
                entry = matches[0]
                if ranker:
                    label = "Minmax-SGD" if ranker == "Minmax_SGD" else ranker
                    record = training[dataset, tokenizer, f"GRID + {label} (training)"]
                    saved = read_config(local_path(source_root, record["config"]),
                                        provenance["sources"][record["config"]])
                else:
                    saved = base
                entry["config"] = portable_config(entry["config"], saved, source_root, dataset, tokenizer, ranker)
                case["experiment_ids"][ranker or "original"] = entry["exp_id"]
                manifest["training_configurations" if ranker else "original_configurations"] += 1
            for field, key in (("checkpoint", "grid_checkpoint_path"), ("sid_mapping", "sid_path")):
                path = Path(base[key])
                case[field] = (local_path(source_root, path).relative_to(source_root).as_posix()
                               if path.is_absolute() else path.as_posix())
            manifest["cases"].append(case)
        prepared[target] = entries
    # Validate the entire cohort before replacing any release file.
    for target, entries in prepared.items():
        target.write_text(json.dumps(entries, indent=2) + "\n")
    manifest_path = release_root / "configs/reproduction/complete_metadata_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True,
                        help="Local experiment workspace containing the completed comparison and saved configs")
    parser.add_argument("--release-root", type=Path, default=ROOT)
    args = parser.parse_args()
    manifest = curate(args.source_root, args.release_root)
    print(json.dumps({"metadata_text_mode": manifest["metadata_text_mode"], "cases": len(manifest["cases"]),
                      "original_configurations": manifest["original_configurations"],
                      "training_configurations": manifest["training_configurations"]}, indent=2))


if __name__ == "__main__":
    main()
