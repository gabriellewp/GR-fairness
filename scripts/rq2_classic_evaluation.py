"""Audit and reevaluate pinned SASRec/GRU4Rec checkpoints for RQ2.

Common protocol: shared temporal-LOO files, last 40 items, -1 padding,
all real catalog IDs eligible (including history items), raw-logit ranking,
item-ID tie breaking, count exposure, bottom-50% MMF, entropy in bits.
No training, dense score export, or checkpoint selection using test results.
"""
import argparse
import ast
import csv
import itertools
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("USE_TF", "0")
import numpy as np
import yaml
from additional_baselines.candidates import file_reference

CUTOFFS = (5, 10, 20)


def raw_catalog_logits(model, feed, item_num, method="gemm", item_chunk_size=2048):
    """Rank raw logits; optionally retain the original single-user arithmetic.

    GEMM and batched Transformer kernels can change rounding on trained models.
    The serial reduction path keeps single-user encoding and full_predict's
    elementwise dot product, chunking catalog items to bound temporary memory.
    """
    import torch
    if method == "gemm":
        return model.get_user_embedding(feed) @ model.item_embedding.weight[:item_num].T
    if method != "serial-reduction" or item_chunk_size < 1:
        raise ValueError("Invalid scoring method or catalog chunk size")
    embeddings = torch.cat([
        model.get_user_embedding({key: value[i:i+1] for key, value in feed.items()})
        for i in range(feed["history_ids"].shape[0])
    ])
    return torch.cat([
        (embeddings[:, None, :] * model.item_embedding.weight[
            start:min(start + item_chunk_size, item_num)][None, :, :]).sum(-1)
        for start in range(0, item_num, item_chunk_size)
    ], dim=1)


def count_fairness(counts):
    values = np.sort(np.asarray(counts, dtype=np.float64))
    total, size = values.sum(), len(values)
    if not total:
        raise ValueError("No exposures")
    shares = values/total
    return dict(gini=float(((2*np.arange(1, size+1)-size-1)*values).sum()/(size*total)),
                mmf=float(values[:int(.5*size)].sum()/total),
                entropy_bits=float(-sum(p*math.log2(p) for p in shares if p)),
                min_max_ratio=float(values[0]/values[-1]), group_count=size)


def audit(destination):
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(destination)
    destination.mkdir(parents=True, exist_ok=True)
    snapshots = list(csv.DictReader((ROOT / "docs/grid_update_october_2026_v1_METRICS.csv").open()))
    baselines = {}
    for row in snapshots:
        if row["method"] in {"SASRec", "GRU4Rec"}:
            baselines[row["dataset"], row["method"]] = row
    experiments = json.loads((ROOT / "configs/reproduction/experiments_config_gru4rec_complete_metadata.json").read_text())
    template = yaml.safe_load((ROOT / baselines["amazon-beauty", "GRU4Rec"]["config"]).read_text())
    cases, table = [], []
    from fairdiverse.recommendation.grid_data_bridge import _source_fingerprint
    for ds in ("amazon-beauty", "amazon-sports", "amazon-toys", "amazon-pet-supplies"):
        data = ROOT / "recommendation/processed_dataset" / ds
        manifest = json.loads((data / "grid_data_manifest.json").read_text())
        if _source_fingerprint(ROOT / manifest["source_dir"]) != manifest["source_fingerprint"]:
            raise ValueError("GRID source changed since shared processed data were built")
        activity_metadata = json.loads((data / "user_activity_metadata.json").read_text())
        if activity_metadata["provenance"].get("source_fingerprint") != manifest["source_fingerprint"]:
            raise ValueError("Activity groups use a different training source")
        references = {name: file_reference(data / filename) for name, filename in
            [("train", ds+".train"), ("validation", ds+".valid.ranking"), ("test", ds+".test.ranking"),
             ("groups", "iid2pid.json"), ("manifest", "grid_data_manifest.json"),
             ("activity", "user_activity.csv"), ("activity_metadata", "user_activity_metadata.json"),
             ("item_mapping", "iid2item.json")]}
        for model in (("GRU4Rec", "SASRec") if ds == "amazon-pet-supplies" else ("SASRec", "GRU4Rec")):
            row = baselines.get((ds, model))
            if row is None and ds == "amazon-pet-supplies" and model == "SASRec":
                # Recognize the explicitly named matched run after it completes.
                completed = sorted(p for p in (ROOT / "fairdiverse/recommendation/log").glob(
                    "*_amazon-pet-supplies-SASRec-rq2-matched*/config.yaml")
                    if (p.parent / "best_model.pth").exists() and (p.parent / "test_result.json").exists())
                if completed:
                    # Prefer the explicit CUDA replacement over the slow CPU run.
                    p = max(completed, key=lambda p: (p.parent.name.endswith("-cuda"), p.parent.name))
                    row = dict(config=str(p.relative_to(ROOT)),
                               source=str((p.parent / "test_result.json").relative_to(ROOT)))
            if row:
                config_path = ROOT / row["config"]
                config = yaml.safe_load(config_path.read_text())
                checkpoint = config_path.parent / "best_model.pth"
                origin = "saved run config"
                config_reference = file_reference(config_path)
            elif ds == "amazon-pet-supplies" and model == "GRU4Rec":
                config = dict(template)
                experiment = next(e for e in experiments if e["dataset"] == ds)
                config.update(experiment["config"], dataset=ds)
                config.update(json.loads((data / "dataset_stats.json").read_text()))
                config["sample_num"] = manifest["item_num"]
                config["grid_data_dir"] = manifest["source_dir"]
                config["grid_item_mapping_path"] = manifest["source_dir"] + "/item_id_mapping.json"
                checkpoint = ROOT / "fairdiverse/recommendation/log/2026-10-8_amazon-pet-supplies-gru4rec-complete-metadata-base/best_model.pth"
                config_reference = file_reference(ROOT / "configs/reproduction/experiments_config_gru4rec_complete_metadata.json")
                origin = "reconstructed from GRU experiment specification + saved Beauty defaults + Pet dataset stats; strict checkpoint shape check required"
            else:
                table.append(dict(dataset=ds, model=model, status="missing checkpoint/result; matched training required",
                                  history_items="", padding="", saved_fairness="", saved_mmf="", checkpoint=""))
                continue
            if config["history_length"] != 40 or config.get("padding_item_id") != -1:
                raise ValueError(f"Unmatched history/padding: {ds}/{model}")
            if config["sample_num"] != manifest["item_num"] or config["item_num"] != manifest["item_num"]:
                raise ValueError("Saved catalog size differs from current GRID bridge")
            for key in ("user_num", "group_num"):
                if config[key] != manifest[key]:
                    raise ValueError(f"Saved {key} differs from current GRID bridge")
            if manifest["split_strategy"] != "temporal_loo":
                raise ValueError("Require temporal leave-one-out")
            declared = config.get("expected_grid_source_dir") or config.get("grid_data_dir")
            if declared and Path(declared).name == "complete_metadata" and (ROOT / declared).resolve() != (ROOT / manifest["source_dir"]).resolve():
                raise ValueError("Declared training source differs from processed bridge")
            case = dict(index=len(cases), dataset=ds, model=model, config=config,
                        config_reference=config_reference, config_origin=origin,
                        checkpoint=file_reference(checkpoint), inputs=references,
                        historical_training_provenance="Historical training split/mapping hashes were not saved; current files are pinned for reevaluation",
                        saved_result=file_reference(ROOT / row["source"]) if row else None)
            cases.append(case)
            table.append(dict(dataset=ds, model=model, status="reevaluate saved checkpoint",
                history_items=40, padding=-1, saved_fairness=config["fairness_type"],
                saved_mmf=config["mmf_eval_ratio"], checkpoint=str(checkpoint)))
    (destination / "inventory.json").write_text(json.dumps(dict(cases=cases, audit=table), indent=2)+"\n")
    with (destination / "baseline_audit.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    missing = [r for r in table if r["status"].startswith("missing")]
    print(f"Pinned {len(cases)} existing checkpoints; {len(missing)} missing baseline(s)")


def evaluate(args):
    import torch
    from additional_baselines.ifairlrs_grid import original_module, seed_everything
    from fairdiverse.recommendation.user_fairness import load_activity, ranking_metrics, save_report
    case = json.loads(args.inventory.read_text())["cases"][args.case]
    for ref in [case["checkpoint"], case["config_reference"], *case["inputs"].values()]:
        if file_reference(ref["path"]) != ref:
            raise ValueError(f"Pinned input changed: {ref['path']}")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    seed_everything(case["config"]["seed"])
    torch.set_num_threads(4)
    config = dict(case["config"], device=args.device)
    cls = getattr(original_module("sasrec" if case["model"] == "SASRec" else "gru4rec"), case["model"])
    model = cls(config).to(args.device)
    model.load_state_dict(torch.load(case["checkpoint"]["path"], map_location=args.device, weights_only=True), strict=True)
    model.eval()
    item_num = config["item_num"]
    mapping = {int(i): int(g) for i, g in json.loads(Path(case["inputs"]["groups"]["path"]).read_text()).items()}
    if set(mapping) != set(range(item_num)) or set(mapping.values()) != set(range(config["group_num"])):
        raise ValueError("Group map must cover the complete real catalog and all declared groups")
    activity = load_activity(case["inputs"]["activity"]["path"])
    counts = {k: np.zeros(config["group_num"], dtype=np.int64) for k in CUTOFFS}
    group_lookup = np.array([mapping[i] for i in range(item_num)])
    rows, diagnostic, seen = [], [], set()
    start_time = time.perf_counter()
    with Path(case["inputs"][args.split]["path"]).open() as stream, (args.output_dir / "top20.csv").open("x", newline="") as ranks:
        writer = csv.writer(ranks)
        writer.writerow(["request_id", "user_id", "target_item", "ranked_items"])
        reader = csv.DictReader(stream, delimiter="\t")
        if args.requests:
            reader = itertools.islice(reader, args.requests)
        with torch.no_grad():
            while True:
                batch = list(itertools.islice(reader, args.batch_size))
                if not batch:
                    break
                users = [int(r["user_id"]) for r in batch]
                targets = [int(r["target_item"]) for r in batch]
                histories = [ast.literal_eval(r["history_behaviors"])[-40:] for r in batch]
                for user, target, history in zip(users, targets, histories):
                    if user in seen or str(user) not in activity or not 0 <= target < item_num or any(not 0 <= i < item_num for i in history):
                        raise ValueError("User/target/history IDs do not align with catalog/activity mapping")
                    seen.add(user)
                feed = dict(user_ids=torch.tensor(users, device=args.device),
                    history_ids=torch.tensor([h+[-1]*(40-len(h)) for h in histories], device=args.device))
                logits = raw_catalog_logits(model, feed, item_num, args.score_method,
                                            args.item_chunk_size)
                if not torch.isfinite(logits).all():
                    raise ValueError("Nonfinite scores")
                # Stable sorting retains ascending item ID for exactly tied logits.
                ranked = torch.argsort(logits, dim=1, descending=True, stable=True)[:, :20].cpu().tolist()
                for i, (user, target, ranking) in enumerate(zip(users, targets, ranked)):
                    request_id = f"{args.split}:{len(rows)}:user:{user}"
                    metrics = ranking_metrics(ranking, [target], CUTOFFS)
                    rows.append(dict(user_id=str(user), request_id=request_id, **metrics))
                    writer.writerow([request_id, user, target, json.dumps(ranking)])
                    for k in CUTOFFS:
                        counts[k] += np.bincount(group_lookup[ranking[:k]], minlength=config["group_num"])
                    if len(diagnostic) < 16:
                        single = {name: value[i:i+1] for name, value in feed.items()}
                        catalog = np.concatenate(([target], np.delete(np.arange(item_num), target)))
                        legacy = model.full_predict(single, torch.tensor(catalog[None], device=args.device))[0].cpu().numpy()
                        fresh = torch.sigmoid(logits[i]).cpu().numpy()[catalog]
                        if not np.allclose(legacy, fresh, rtol=1e-5, atol=1e-6):
                            raise ValueError("Batched logits disagree with original single-user model scores")
                        legacy_ranking = catalog[np.argsort(legacy+np.float32(1))[::-1]][:20].tolist()
                        diagnostic.append(dict(request_id=request_id,
                            score_max_difference=float(np.max(np.abs(legacy-fresh))),
                            legacy_vs_raw_logit_top20_equal=legacy_ranking == ranking,
                            legacy_metrics=ranking_metrics(legacy_ranking, [target], CUTOFFS), common_metrics=metrics))
                print(f"{case['dataset']} {case['model']} {args.split}: {len(rows)} requests; {time.perf_counter()-start_time:.1f}s", flush=True)
    if not rows or (not args.requests and len(rows) != config["user_num"]):
        raise ValueError("Incomplete full split")
    metrics = {f"{name}@{k}": sum(r[f"{name}@{k}"] for r in rows)/len(rows)
               for name in ("ndcg", "recall") for k in CUTOFFS}
    for k in CUTOFFS:
        metrics.update({f"{key}@{k}": value for key, value in count_fairness(counts[k]).items()})
    protocol = dict(history_items=40, padding=-1, eligibility="full real catalog; retain history items; no padding ID",
        target_denominator="all requests; absent top-K target is a miss", ranking="raw logits descending; item ID ascending for ties",
        fairness="brand count exposure; include unexposed catalog groups", mmf_fraction=.5, entropy_unit="bits",
        candidate_target_coverage=1.0, grid_comparison="GRID eligibility uses the same catalog, but returned candidates depend on generated SID coverage")
    report = dict(case=case, split=args.split, full_split=args.requests == 0, request_count=len(rows),
        score_method=args.score_method, item_chunk_size=args.item_chunk_size,
        protocol=protocol, metrics=metrics, exposure_counts={str(k): counts[k].tolist() for k in CUTOFFS},
        serial_score_diagnostic=diagnostic, elapsed_seconds=time.perf_counter()-start_time,
        model_source=file_reference(ROOT / "fairdiverse/recommendation/base_model" / ("sasrec.py" if case["model"] == "SASRec" else "gru4rec.py")),
        evaluator_source=file_reference(Path(__file__)))
    (args.output_dir / "results.json").write_text(json.dumps(report, indent=2)+"\n")
    save_report(rows, case["inputs"]["activity"]["path"], args.output_dir / "user_fairness",
                args.bootstrap_samples, config["seed"], dict(model=case["model"], dataset=case["dataset"], split=args.split,
                                                             full_split=args.requests == 0, protocol=protocol))
    print(json.dumps(metrics, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    a = commands.add_parser("audit")
    a.add_argument("--output-dir", type=Path, required=True)
    e = commands.add_parser("evaluate")
    e.add_argument("--inventory", type=Path, default=ROOT / "docs/rq2_classic_audit_20261010/inventory.json")
    e.add_argument("--case", type=int, required=True)
    e.add_argument("--split", choices=["validation", "test"], default="test")
    e.add_argument("--output-dir", type=Path, required=True)
    e.add_argument("--device", default="cuda")
    e.add_argument("--requests", type=int, default=0)
    e.add_argument("--batch-size", type=int, default=32)
    e.add_argument("--score-method", choices=["gemm", "serial-reduction"], default="gemm")
    e.add_argument("--item-chunk-size", type=int, default=2048)
    e.add_argument("--bootstrap-samples", type=int, default=1000)
    args = parser.parse_args()
    if args.command == "audit":
        audit(args.output_dir)
    else:
        if args.requests < 0 or args.batch_size < 1 or args.bootstrap_samples < 0:
            parser.error("Invalid request, batch, or bootstrap count")
        evaluate(args)


if __name__ == "__main__":
    main()
