"""Generate validation/test candidates from a verified original GRID export."""

import argparse
import json
from pathlib import Path

from .candidates import file_reference, save
from .ifairlrs import popularity_groups
ROOT = Path(__file__).resolve().parents[1]

SUITES = {
    "amazon-beauty": None,
    "amazon-sports": "sports_rkmeans_continuation_retry",
    "amazon-toys": "toys_rkmeans_continuation",
    "amazon-pet-supplies": "pet_rkmeans_complete_metadata_continuation_20261004",
}

TOKENIZER_SUITES = {
    ("amazon-beauty", "rqvae"): "beauty_rqvae_complete_metadata_continuation_lr1e3_20261001_121336",
    ("amazon-beauty", "rvq"): "beauty_rvq_complete_metadata_continuation_lr1e3_20261001_121336",
    ("amazon-sports", "rqvae"): "sports_rqvae_continuation_retry",
    ("amazon-sports", "rvq"): "sports_rvq_continuation_retry",
    ("amazon-toys", "rqvae"): "toys_rqvae_continuation",
    ("amazon-toys", "rvq"): "toys_rvq_continuation_retry",
    ("amazon-pet-supplies", "rqvae"): "pet_rqvae_complete_metadata_continuation_20261004",
    ("amazon-pet-supplies", "rvq"): "pet_rvq_complete_metadata_continuation_20261004",
}


def source_export(dataset, tokenizer):
    if tokenizer == "rkmeans":
        suite_name = SUITES[dataset]
        if suite_name is None:
            from .prepare_beauty import DEFAULT_EXPORT
            return ROOT / DEFAULT_EXPORT
    else:
        suite_name = TOKENIZER_SUITES[dataset, tokenizer]
    suite = json.loads((ROOT / "additional_baselines/artifacts" / suite_name / "suite_config.json").read_text())
    return ROOT / suite["export_dir"]


def run(dataset, output_dir, device="cuda", limit_requests=None, tokenizer="rkmeans",
        generation_beam_width=None, require_min_candidates=None, export_dir=None, data_dir=None):
    for name, value in (("generation_beam_width", generation_beam_width),
                        ("require_min_candidates", require_min_candidates)):
        if value is not None and (type(value) is not int or value <= 0):
            raise ValueError(f"{name} must be a positive integer")
    import torch
    import yaml
    from .ifairlrs_grid import original_module, read_rows, regenerate, seed_everything, training_popularity, write_json
    from .prepare_beauty import load_beauty

    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(destination)
    export = Path(export_dir).resolve() if export_dir is not None else source_export(dataset, tokenizer)
    config = yaml.safe_load((export / "config.yaml").read_text())
    data = Path(data_dir).resolve() if data_dir is not None else ROOT / "recommendation/processed_dataset" / dataset
    verified, _, _ = load_beauty(export_dir=export, data_dir=data, dataset=dataset,
                                 tokenizer=tokenizer, topk=(1,), check_dense_memory=False)
    if generation_beam_width is not None:
        config["top_k_for_generation"] = generation_beam_width
    catalog = set(verified.item_groups["brand"])
    length = config["sequence_length"] // config["num_hierarchies"]
    train = data / f"{dataset}.train"
    training = read_rows(train, training=True, history_length=length, catalog=catalog)
    groups = {"brand": verified.item_groups["brand"],
              "popularity": popularity_groups(training_popularity(training), catalog, n_groups=min(5, len(catalog)))}
    checkpoint = verified.metadata["checkpoint"]["path"]
    provenance = {"dataset": dataset, "tokenizer": tokenizer, "training_data": file_reference(train),
                  "export_config": file_reference(export / "config.yaml"),
                  "scope": "smoke_prefix" if limit_requests else "full_splits",
                  "generation_beam_width": config["top_k_for_generation"],
                  "require_min_candidates": require_min_candidates}
    seed_everything(42)
    model = original_module("grid_model").GRIDBaseModel(config).to(device)
    destination.mkdir(parents=True)
    write_json(destination / "manifest.json", provenance)
    for split, suffix in (("validation", "valid"), ("test", "test")):
        path = data / f"{dataset}.{suffix}.ranking"
        rows = read_rows(path, training=False, history_length=length, catalog=catalog)
        if limit_requests:
            rows = rows[:limit_requests]
        seed_everything(42)
        artifact, histories = regenerate(model, rows, path, split, groups, checkpoint,
                                         verified.metadata["sid_mapping"]["path"], 42, device, provenance)
        artifact.metadata["dataset"] = dataset
        artifact.metadata["tokenizer"] = tokenizer
        save(artifact, destination / f"{split}_candidates.json")
        write_json(destination / f"{split}_histories.json", histories)
        counts = [len(request.candidate_ids) for request in artifact.requests]
        short = sum(count < require_min_candidates for count in counts) if require_min_candidates else 0
        write_json(destination / f"{split}_candidate_summary.json", {
            "request_count": len(counts), "min_candidate_count": min(counts),
            "max_candidate_count": max(counts), "beam_width": model.top_k_for_generation,
            "required_minimum": require_min_candidates, "below_required_minimum": short,
        })
        if short:
            raise ValueError(f"{split}: {short} requests have fewer than {require_min_candidates} "
                             f"eligible candidates at beam width {model.top_k_for_generation}; "
                             "candidate preparation is incomplete and downstream jobs must not run")
    write_json(destination / "complete.json", {"scope": provenance["scope"], "dataset": dataset, "tokenizer": tokenizer})
    print(f"Candidates: {destination}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(SUITES), required=True)
    parser.add_argument("--tokenizer", choices=("rkmeans", "rqvae", "rvq"), default="rkmeans")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--export-dir", help="Explicit FairDiverse score-export directory")
    parser.add_argument("--data-dir", help="Explicit matching processed dataset directory")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit-requests", type=int)
    parser.add_argument("--generation-beam-width", type=int)
    parser.add_argument("--require-min-candidates", type=int)
    args = parser.parse_args()
    if args.limit_requests is not None and args.limit_requests <= 0:
        parser.error("--limit-requests must be positive")
    run(**vars(args))


if __name__ == "__main__":
    main()
