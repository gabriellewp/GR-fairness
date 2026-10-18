"""Beauty RKMeans: one frozen CPFair policy across four current checkpoints."""
import argparse
import csv
import gc
import json
from pathlib import Path

from .candidates import file_reference, load, save
from .rkmeans_selected_tests import prepare

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "additional_baselines/artifacts"
DATASET = "amazon-beauty"
BASE = ARTIFACTS / "beauty_ifairlrs_rr_popularity/original"
BRANCHES = ("beauty_native_control_lr1e3", "beauty_native_rw_lr1e3", "beauty_native_sprint_lr1e3")


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def audit(output):
    import yaml
    if output.exists():
        raise FileExistsError(output)
    candidate_dir, selection = prepare(DATASET, .05, sources={"cpfair": "beauty_cpfair_validation_tuning_351961"})
    reports = [json.loads((ARTIFACTS / branch / "continuation.json").read_text()) for branch in BRANCHES]
    identity = reports[0]["identity"]
    if any(r["identity"] != identity for r in reports):
        raise ValueError("Continuation branches have different training identities")
    base_metadata = load(BASE / "validation_candidates.json").metadata
    if base_metadata["checkpoint"] != identity["checkpoint"] or base_metadata["sid_mapping"] != identity["sid_mapping"]:
        raise ValueError("Validation-selected reranker and continuation source differ")
    export = ROOT / base_metadata["provenance"]["export_config"]["path"]
    config = yaml.safe_load(export.read_text())
    if config["grid_tokenizer"] != "rkmeans" or config["dataset"] != DATASET:
        raise ValueError("Require Beauty RKMeans")
    data = ROOT / "recommendation/processed_dataset" / DATASET
    inputs = {filename: file_reference(data / filename) for filename in (
        DATASET+".train", DATASET+".test.ranking", "iid2pid.json", "iid2item.json",
        "grid_data_manifest.json", "user_activity.csv", "user_activity_metadata.json")}
    cases = [dict(name="base", checkpoint=identity["checkpoint"], selected_update=0)]
    for name, branch, report in zip(("ordinary", "rw", "sprint"), BRANCHES, reports):
        # Historical reports predate selected_weights_restored. Fresh inference
        # loads the saved selected checkpoint directly, rather than reusing
        # their test predictions. A zero-update winner must be source weights.
        if report["selected_update"] == 0 and report["selected_checkpoint"] != identity["checkpoint"]:
            raise ValueError("Zero-update selection does not reference source weights")
        cases.append(dict(name=name, checkpoint=report["selected_checkpoint"], selected_update=report["selected_update"],
                          continuation=file_reference(ARTIFACTS / branch / "continuation.json")))
    for case in cases:
        if file_reference(case["checkpoint"]["path"]) != case["checkpoint"]:
            raise ValueError("Saved checkpoint changed")
    output.mkdir(parents=True)
    write(output / "inventory.json", dict(dataset=DATASET, tokenizer="rkmeans", cases=cases, config=config,
        export_config=file_reference(export), sid_mapping=identity["sid_mapping"], inputs=inputs,
        training_identity=identity, method="cpfair", selection=selection["cpfair"],
        protocol=dict(history_items=40, beam_width=identity["generation_beam_width"],
            candidate_pipeline="same GRID beam generation and teacher-forced native log likelihood for every pool",
            selection="base full validation only; frozen across all four pools",
            state="reset per pool and cutoff; same full test request order", cutoffs=[5,10,20])))
    print("Frozen CPFair lambda:", selection["cpfair"]["parameters"]["lambda"])


def checked_inventory(path):
    inventory = json.loads(path.read_text())
    for reference in [inventory["export_config"], inventory["sid_mapping"], *inventory["inputs"].values()]:
        if file_reference(reference["path"]) != reference:
            raise ValueError("Pinned input changed: "+reference["path"])
    return inventory


def load_selected_model(model_class, config, selected_checkpoint, source_checkpoint, device):
    """Initialize via the native source, then strictly restore saved continuation weights.

    GRIDBaseModel deliberately restricts its constructor artifacts to GRID/.
    Continuation checkpoints are saved outside that tree; restore them directly
    into the same native module after initialization, as continuation training does.
    """
    import torch
    model = model_class(dict(config, grid_checkpoint_path=str(source_checkpoint)))
    if Path(selected_checkpoint).resolve() != Path(source_checkpoint).resolve():
        checkpoint = torch.load(selected_checkpoint, map_location="cpu", weights_only=False)
        model.grid.load_state_dict(checkpoint.get("state_dict",checkpoint), strict=True)
        model.loaded_checkpoint_path = str(Path(selected_checkpoint).resolve())
    return model.to(device)


def generate(inventory_path, index, output, requests, device):
    import torch
    from .ifairlrs_grid import original_module, read_rows, regenerate, seed_everything, training_popularity
    from .ifairlrs import popularity_groups
    from .prepare_beauty import load_beauty
    inventory = checked_inventory(inventory_path)
    case = inventory["cases"][index]
    if output.exists():
        raise FileExistsError(output)
    if file_reference(case["checkpoint"]["path"]) != case["checkpoint"]:
        raise ValueError("Checkpoint changed")
    data = ROOT / "recommendation/processed_dataset" / DATASET
    export = Path(inventory["export_config"]["path"]).parent
    verified, _, _ = load_beauty(export_dir=export, data_dir=data, dataset=DATASET,
                                tokenizer="rkmeans", topk=(1,), check_dense_memory=False)
    catalog = set(verified.item_groups["brand"])
    training = read_rows(data / (DATASET+".train"), training=True, history_length=40, catalog=catalog)
    groups = dict(brand=verified.item_groups["brand"], popularity=popularity_groups(training_popularity(training), catalog))
    rows = read_rows(data / (DATASET+".test.ranking"), training=False, history_length=40, catalog=catalog)
    if requests:
        rows = rows[:requests]
    config = dict(inventory["config"], device=device, grid_checkpoint_path=case["checkpoint"]["path"],
                  top_k_for_generation=inventory["protocol"]["beam_width"])
    seed_everything(42)
    model = load_selected_model(original_module("grid_model").GRIDBaseModel, config,
        case["checkpoint"]["path"], inventory["training_identity"]["checkpoint"]["path"], device)
    artifact, _ = regenerate(model, rows, data / (DATASET+".test.ranking"), "test", groups,
        case["checkpoint"]["path"], inventory["sid_mapping"]["path"], 42, device,
        dict(controlled_inventory=file_reference(inventory_path), pool=case["name"]))
    if not requests and len(rows) != config["user_num"]:
        raise ValueError("Incomplete full test split")
    output.mkdir(parents=True)
    save(artifact, output / "test_candidates.json")
    write(output / "complete.json", dict(scope="smoke_prefix" if requests else "full_test",
        pool=case["name"], request_count=len(rows), candidates=file_reference(output / "test_candidates.json")))
    print("Prepared", case["name"], len(rows), "requests")


def validate_pool(inputs, reference):
    for key in ("split", "sid_mapping", "generation", "seed", "candidate_membership", "score_convention"):
        if inputs.metadata[key] != reference.metadata[key]:
            raise ValueError("Controlled pool differs in "+key)
    if inputs.item_groups != reference.item_groups:
        raise ValueError("Controlled pool group definitions differ")
    if [(r.request_id,r.targets) for r in inputs.requests] != [(r.request_id,r.targets) for r in reference.requests]:
        raise ValueError("Controlled pool request order or targets differ")


def evaluate_pools(inventory_path, pools_dir, output, bootstrap_samples):
    from .evaluation import evaluate
    from .rerank import FairDiverseReranker
    from .report_beauty_activity import validate_alignment, make_rows
    from fairdiverse.recommendation.user_fairness import load_activity, save_report
    inventory = checked_inventory(inventory_path)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    reference = load(pools_dir / "case_0/test_full/test_candidates.json")
    data = ROOT / "recommendation/processed_dataset" / DATASET
    activity_path = data / "user_activity.csv"
    activity = load_activity(activity_path)
    with (data / (DATASET+".test.ranking")).open() as stream:
        test_rows = list(csv.DictReader(stream, delimiter="\t"))
    comparison = []
    strength = inventory["selection"]["parameters"]["lambda"]
    for index, case in enumerate(inventory["cases"]):
        source = pools_dir / f"case_{index}/test_full/test_candidates.json"
        complete = json.loads((source.parent / "complete.json").read_text())
        if complete["scope"] != "full_test" or complete["candidates"] != file_reference(source):
            raise ValueError("Require completed, unchanged full test candidate pools")
        inputs = reference if index == 0 else load(source)
        validate_pool(inputs, reference)
        if inputs.metadata["checkpoint"] != case["checkpoint"]:
            raise ValueError("Pool is from the wrong checkpoint")
        users = validate_alignment(inputs, test_rows, activity, file_reference(data / (DATASET+".test.ranking")))
        control = evaluate(inputs, (5,10,20))
        rows = make_rows(inputs, users, None)
        rankings, metrics, configs = {}, {}, {}
        for k in (5,10,20):
            # Fresh backend state for every independent pool/cutoff.
            reranker = FairDiverseReranker(model="CPFair", group="brand", fairness_lambda=strength)
            ranked = reranker.rerank(inputs, k)
            rankings[str(k)], configs[str(k)] = ranked.rankings, ranked.config
            metrics[str(k)] = evaluate(inputs, (k,), ranked.rankings)["at_k"][str(k)]
            measured = make_rows(inputs, users, ranked.rankings)
            for row, new in zip(rows, measured):
                for metric in ("recall", "ndcg"):
                    row[f"{metric}@{k}"] = new[f"{metric}@{k}"]
            for variant, at_k in (("untreated",control["at_k"][str(k)]), ("cpfair",metrics[str(k)])):
                comparison.append(dict(pool=case["name"], variant=variant, k=k,
                    request_count=control["request_count"], coverage=control["candidate_target_recall"],
                    recall=at_k["recall"], ndcg=at_k["ndcg"], brand_gini=at_k["groups"]["brand"]["gini"],
                    delta_recall=at_k["recall"]-control["at_k"][str(k)]["recall"],
                    delta_brand_gini=at_k["groups"]["brand"]["gini"]-control["at_k"][str(k)]["groups"]["brand"]["gini"]))
        directory = output / case["name"]
        save_report(rows, activity_path, directory / "user_fairness", bootstrap_samples, 42,
            dict(model="CPFair", pool=case["name"], input_metadata=inputs.metadata))
        write(directory / "cpfair_test_results.json", dict(method="cpfair", fairness_lambda=strength,
            input_metadata=inputs.metadata, rankings_by_k=rankings, configs_by_k=configs,
            metrics=dict(at_k=metrics,candidate_target_recall=control["candidate_target_recall"]), control_metrics=control))
        print("Completed controlled CPFair:",case["name"],flush=True)
        del rankings, inputs, rows
        gc.collect()
    with (output / "controlled_comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)
    write(output / "protocol.json", dict(inventory=file_reference(inventory_path), protocol=inventory["protocol"],
        selection=inventory["selection"], interpretation="within-pool reranking changes and across-pool variation; one dataset/tokenizer/seed"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    audit_parser = sub.add_parser("audit")
    audit_parser.add_argument("--output-dir", type=Path, required=True)
    generator = sub.add_parser("generate")
    generator.add_argument("--inventory", type=Path, required=True)
    generator.add_argument("--case", type=int, choices=range(4), required=True)
    generator.add_argument("--output-dir", type=Path, required=True)
    generator.add_argument("--requests", type=int, default=0)
    generator.add_argument("--device", default="cuda")
    evaluator = sub.add_parser("evaluate")
    evaluator.add_argument("--inventory", type=Path, required=True)
    evaluator.add_argument("--pools-dir", type=Path, required=True)
    evaluator.add_argument("--output-dir", type=Path, required=True)
    evaluator.add_argument("--bootstrap-samples", type=int, default=1000)
    args = parser.parse_args()
    if args.command == "audit":
        audit(args.output_dir)
    elif args.command == "generate":
        if args.requests < 0:
            parser.error("--requests must be nonnegative")
        generate(args.inventory,args.case,args.output_dir,args.requests,args.device)
    else:
        evaluate_pools(args.inventory,args.pools_dir,args.output_dir,args.bootstrap_samples)


if __name__ == "__main__":
    main()
