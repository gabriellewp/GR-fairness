"""Freeze existing GRID evidence against Table 1 of arXiv:2507.22224v1.

Run with the project's scientific Python environment. No training or GPU needed.
The output inventory pins the inputs consumed by sbatch_rq1_grid_agreement.sh.
"""
import argparse
import csv
import json
from pathlib import Path
import sys

from scripts.paper.archive_paths import source_root, output_root
ROOT = source_root()
OUTPUT = output_root()
# Imports resolve from the released implementation, not the artifact workspace.
import yaml
from additional_baselines.candidates import file_reference

PAPER = {
    "amazon-beauty": {"rkmeans": [.0422, .0639, .0277, .0347],
                      "rvq": [.0422, .0638, .0282, .0351], "rqvae": [.0404, .0593, .0268, .0329]},
    "amazon-sports": {"rkmeans": [.0236, .0353, .0153, .0191],
                      "rvq": [.0234, .0352, .0151, .0189], "rqvae": [.0205, .0312, .0132, .0166]},
    "amazon-toys": {"rkmeans": [.0376, .0577, .0243, .0308],
                    "rvq": [.0327, .0493, .0209, .0262], "rqvae": [.0342, .0514, .0224, .0280]},
}
METRICS = ["hr@5", "hr@10", "ndcg@5", "ndcg@10"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()):
        raise FileExistsError("Audit destination must be empty")
    rows = list(csv.DictReader((ROOT / "docs/grid_update_october_2026_v1_METRICS.csv").open()))
    base = {(r["dataset"], r["tokenizer"]): r for r in rows if r["panel"] == "core" and r["method"] == "GRID base"}
    inventory, comparison, configurations = [], [], []
    for dataset in ["amazon-beauty", "amazon-sports", "amazon-toys", "amazon-pet-supplies"]:
        for tokenizer in ["rkmeans", "rvq", "rqvae"]:
            row = base[dataset, tokenizer]
            config_path = ROOT / (row["config"] or
                "fairdiverse/recommendation/log/2026-9-20_grid-independent-amazon-beauty-rkmeans-flan-t5-xl-complete-metadata-score-export/config.yaml")
            config = yaml.safe_load(config_path.read_text())
            checkpoint = ROOT / row["checkpoint"]
            native_path = checkpoint.parent.parent / ".hydra/config.yaml"
            native = yaml.safe_load(native_path.read_text())
            tokenizer_path = checkpoint.parents[2] / "step2_tokenizer_training/.hydra/config.yaml"
            tokenizer_config = yaml.safe_load(tokenizer_path.read_text())
            sid = ROOT / config["sid_path"]
            paper = PAPER.get(dataset, {}).get(tokenizer)
            item = dict(index=len(inventory), dataset=dataset, tokenizer=tokenizer,
                        source=file_reference(ROOT / row["source"]), config=file_reference(config_path),
                        checkpoint=file_reference(checkpoint), sid_mapping=file_reference(sid),
                        native_config=file_reference(native_path), seed=int(row["seed"]),
                        tokenizer_config=file_reference(tokenizer_path),
                        saved_metrics={k: float(row[k]) for k in METRICS},
                        saved_protocol="eligible generated candidates; log_probability" if dataset == "amazon-beauty" and tokenizer == "rkmeans"
                                       else "FairDiverse shifted full-catalog scores",
                        paper_table1=dict(zip(METRICS, paper)) if paper else None)
            for name in ("ranking_scores", "ground_truth"):
                path = config_path.parent / (name + ".npz")
                if path.exists():
                    item[name] = file_reference(path)
            inventory.append(item)
            out = dict(dataset=dataset, tokenizer=tokenizer, seed=item["seed"],
                       protocol=item["saved_protocol"], source=str(ROOT / row["source"]),
                       interpretation="adapted single-seed baseline; evaluator agreement pending")
            for i, metric in enumerate(METRICS):
                out["local_" + metric] = item["saved_metrics"][metric]
                out["paper_" + metric] = paper[i] if paper else ""
                out["local_minus_paper_" + metric] = item["saved_metrics"][metric] - paper[i] if paper else ""
            comparison.append(out)
            configurations.append(dict(dataset=dataset, tokenizer=tokenizer,
                native_config=str(native_path), optimizer=native["optim"]["optimizer"],
                checkpoint_monitor=native["callbacks"]["model_checkpoint"]["monitor"],
                val_check_interval=native["trainer"].get("val_check_interval"),
                max_steps=native["trainer"].get("max_steps"),
                accumulation=native["trainer"].get("accumulate_grad_batches"),
                train_loader=native["data_loading"]["train_dataloader_config"]["dataloader"],
                tokenizer_max_steps=tokenizer_config["trainer"]["max_steps"],
                tokenizer_devices=tokenizer_config["trainer"]["devices"],
                metadata="title/categories/brand/description; description truncated to 1000 characters; no price",
                beam_width=config["top_k_for_generation"], history_items=config["history_length"], seed=item["seed"]))
    for name, records in [("paper_vs_local_results.csv", comparison), ("local_configurations.csv", configurations)]:
        with (args.output_dir / name).open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    settings = [
        ("metadata", "title/categories/description/price", "title/categories/brand/description; no price", "different"),
        ("generator architecture", "4+4 layers; 6 heads; embedding 128; MLP 1024", "4+4 layers; 6 heads; embedding 128; MLP 1024", "agrees"),
        ("generator Adam learning rate", "0.0005", "0.001", "different"),
        ("generator weight decay", "0.000001", "0.0001", "different"),
        ("generator batch", "256", "32 users/microbatch; causal cap 512; accumulation 16", "different; windows are not users"),
        ("validation interval", "100 steps", "1600 microbatches (original three datasets)", "different; confirm step units"),
        ("checkpoint selection", "best validation Recall@10", "best validation Recall@5 (original three datasets)", "different"),
        ("early stopping", "10 validation intervals without NDCG@10 improvement", "inspect saved native callbacks per cell", "needs reconciliation"),
        ("seeds", "mean of 5", "seed 42", "missing 5-seed estimate"),
        ("tokenizer training", "RKMeans/RVQ 1000 steps/layer; RQVAE 15000 steps; 8 GPUs",
         "trainer caps: RKMeans 30, RVQ 3000, RQVAE 3000; 1 GPU", "reconcile effective per-layer updates; RQVAE budget differs"),
        ("dataset and split", "5-core Beauty/Sports/Toys; last=test, penultimate=validation",
         "5-core temporal leave-one-out; release/order fingerprints need comparison", "provenance agreement pending"),
    ]
    with (args.output_dir / "paper_vs_local_configuration.csv").open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["setting", "paper", "local_original_benchmarks", "status"])
        writer.writerows(settings)
    pet_root = ROOT / "additional_baselines/artifacts/pet_rvq_complete_metadata_continuation_20261004"
    pet_rows = []
    for mode in ["ordinary", "rw", "sprint"]:
        report_path = pet_root / mode / "continuation.json"
        report = json.loads(report_path.read_text())
        trace_path = pet_root / mode / "validation_trace.json"
        zero = json.loads(trace_path.read_text())[0]["metrics"]
        test = json.loads((pet_root / mode / "test_results.json").read_text())["metrics"]
        pet_rows.append(dict(mode=mode, selected_update=report["selected_update"],
            zero_validation_ndcg5=zero["at_k"]["5"]["ndcg"], zero_validation_recall5=zero["at_k"]["5"]["recall"],
            saved_test_ndcg5=test["at_k"]["5"]["ndcg"], saved_test_recall5=test["at_k"]["5"]["recall"],
            saved_test_status="invalid checkpoint attribution; reevaluate source" if mode == "sprint" else "selected positive-update checkpoint",
            continuation_report=str(report_path), validation_trace=str(trace_path)))
    with (args.output_dir / "pet_rvq_existing_results.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(pet_rows[0]))
        writer.writeheader()
        writer.writerows(pet_rows)
    issues = dict(
        reproduction_target="GRID Table 1, arXiv:2507.22224v1; Beauty/Sports/Toys x RKMeans/RVQ/RQVAE; five-seed means",
        paper_url="https://arxiv.org/html/2507.22224v1",
        paper_configuration=dict(metadata="title/categories/description/price", embedding="Flan-T5-XL mean pooling",
            generator="4 encoder + 4 decoder layers, 6 heads, 128 embeddings, 1024 MLP",
            optimizer="Adam lr=5e-4, weight_decay=1e-6; batch=256", seeds=5,
            validation="every 100 steps; stop after 10 NDCG@10 intervals; select best validation Recall@10",
            tokenizer_training="RKMeans/RVQ 1000 steps per layer; RQVAE 15000 steps; 8 GPUs, per-device batch 2048"),
        missing=["fresh pinned-checkpoint evaluator agreement on all nine original benchmark cells",
                 "fresh Pet RVQ source-checkpoint validation/test with restored selected weights",
                 "dataset/release and split fingerprints matched against paper/released benchmark",
                 "matched metadata/tokenizer/optimizer/batching/validation settings before claiming reproduction",
                 "five independent seeds and uncertainty; existing rows are seed 42"],
        invalidated_results=[dict(path="additional_baselines/artifacts/pet_rvq_complete_metadata_continuation_20261004/sprint/test_results.json",
            reason="selected_update=0 recorded, but runner tested final trained weights without restoring source; exclude until corrected",
            remedy="evaluate pinned source checkpoint; do not rerun SPRINT training to recover update-zero result")],
        already_verified=["output/pet_grid_rvq_serial_generation_review.json: serial batch=1, 512 requests, offset=300000, parity passed",
                          "output/pet_grid_beam_check_review_73.json: batched RVQ catalog ordering failed despite identical SID beams"],
        limits="Fresh evaluation will establish metrics for local checkpoints; it cannot retroactively match paper training or five-seed means.")
    (args.output_dir / "inventory.json").write_text(json.dumps(dict(cases=inventory, audit=issues), indent=2) + "\n")
    print(f"Wrote {len(inventory)} pinned baselines to {args.output_dir}")


if __name__ == "__main__":
    main()
