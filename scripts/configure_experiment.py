"""Create a portable experiment config from the preserved reproduction protocol."""
import argparse
import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = {"amazon-beauty": "beauty", "amazon-sports": "sports",
            "amazon-toys": "toys", "amazon-pet-supplies": "pet_supplies"}


def build(args):
    dataset = args.dataset
    entries = json.loads((ROOT / "configs/reproduction" / f"experiments_config_{dataset}_grid_tokenizers.json").read_text())
    entry = next(e for e in entries if e["config"].get("grid_tokenizer") == args.tokenizer
                 and not e["config"].get("fair-rank", False))
    result = copy.deepcopy(entry)
    result["exp_id"] = 0
    config = result["config"]
    slug = DATASETS[dataset]
    source = args.grid_data_dir or f"GRID/data/amazon_{slug}_5core_tfrecords/temporal_loo/complete_metadata"
    if "description" in Path(source).parts:
        raise ValueError("Description-only data is excluded; use complete_metadata artifacts")
    config.update(metadata_text_mode="complete_metadata", grid_data_dir=source, grid_item_mapping_path=source+"/item_id_mapping.json",
                  use_wandb=False, reprocess=False, seed=args.seed, device=args.device,
                  log_name=f"{dataset}-{args.model.lower()}-{args.tokenizer}-complete-metadata-s{args.seed}",
                  store_scores=True)
    if args.model == "GRID":
        if not args.checkpoint:
            raise ValueError("GRID requires --checkpoint (the validation-selected native checkpoint)")
        for name, value in (("checkpoint", args.checkpoint), ("sid-path", args.sid_path)):
            if value and "description" in Path(value).parts:
                raise ValueError(f"--{name} must use complete_metadata artifacts")
            if value and not (ROOT / value).resolve().is_relative_to((ROOT / "GRID").resolve()):
                raise ValueError(f"--{name} must live under the embedded GRID directory")
        config.update(grid_checkpoint_path=args.checkpoint,
            sid_path=args.sid_path or f"GRID/outputs/{slug}_5core/temporal_loo/complete_metadata/{args.tokenizer}/step2b_sid_inference/pickle/merged_predictions_tensor_suffix4.pt",
            grid_eval_only=True, grid_fair_finetune=False, grid_fair_finetune_steps=0,
            grid_eval_generation_batch_size=args.eval_batch_size)
    else:
        config = {key: value for key, value in config.items() if not key.startswith("grid_")}
        config.update(model=args.model, data_type="sequential", expected_grid_source_dir=source,
                      padding_item_id=-1, epoch=args.epochs, batch_size=256, learning_rate=0.001,
                      eval_step=1, max_steps=0, embedding_size=64, hidden_size=64,
                      accumulate_grad_batches=1, n_heads=2, n_layers=2, inner_size=256)
        config.pop("sid_path", None)
        result["config"] = config
    config["fair-rank"] = False
    return [result]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--model", choices=["GRID", "SASRec", "GRU4Rec"], default="GRID")
    parser.add_argument("--tokenizer", choices=["rkmeans", "rqvae", "rvq"], default="rkmeans")
    parser.add_argument("--checkpoint")
    parser.add_argument("--sid-path")
    parser.add_argument("--grid-data-dir")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.epochs <= 0 or args.eval_batch_size <= 0:
        parser.error("--epochs and --eval-batch-size must be positive")
    try:
        result = build(args)
    except ValueError as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(args.output)


if __name__ == "__main__":
    main()
