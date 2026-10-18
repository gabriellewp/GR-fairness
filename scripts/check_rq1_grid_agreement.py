"""Stream fresh serial GRID generation through native and eligible evaluators.

Consumes the fingerprinted audit inventory. Each output directory must be new.
Compares metric formulas on identical rankings separately from differences in
native SID/eligible-item/legacy full-catalog ranking protocols. No training.
"""
import argparse
import ast
import csv
import itertools
import importlib
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Evaluation uses the PyTorch implementation; avoid importing TensorFlow via
# Transformers' optional image utilities in this environment.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
import numpy as np
import torch
import yaml
from additional_baselines.adapters import GenerationRequest, generate_grid
from additional_baselines.candidates import file_reference
from additional_baselines.evaluation import evaluate
from additional_baselines.ifairlrs_grid import original_module, seed_everything

CUTOFFS = (5, 10, 20)


def merge_metrics(parts):
    """Combine request-weighted scalar metrics and global exposure counts."""
    n = sum(p["request_count"] for p in parts)
    if not n:
        raise ValueError("No evaluated requests")
    result = {"request_count": n, "at_k": {}}
    for key in parts[0]:
        if key not in {"request_count", "at_k"}:
            result[key] = sum(p[key] * p["request_count"] for p in parts) / n
    for k in map(str, CUTOFFS):
        rows = [p["at_k"][k] for p in parts]
        out = {key: sum(r[key] * p["request_count"] for r, p in zip(rows, parts)) / n
               for key in rows[0] if key != "groups"}
        out["groups"] = {}
        for group in rows[0]["groups"]:
            counts = {g: sum(r["groups"][group]["exposure_counts"][g] for r in rows)
                      for g in rows[0]["groups"][group]["exposure_counts"]}
            values = sorted(counts.values())
            total, size = sum(values), len(values)
            shares = {g: c / total if total else 0 for g, c in counts.items()}
            out["groups"][group] = dict(exposure_counts=counts, exposure_shares=shares,
                gini=sum((2*i-size-1)*v for i, v in enumerate(values, 1))/(size*total) if total else None,
                entropy=-sum(p*math.log(p) for p in shares.values() if p) if total else None,
                min_max_ratio=values[0]/values[-1] if total else None,
                exposed_group_fraction=sum(v > 0 for v in values)/size,
                candidate_group_coverage=sum(r["groups"][group]["candidate_group_coverage"] * p["request_count"]
                                             for r, p in zip(rows, parts))/n)
        result["at_k"][k] = out
    return result


def common_ranking_metrics(artifact, native_metrics):
    """Use strict synthetic scores to give both implementations identical ranks.

    Unmatched targets remain misses. Padding occupies only positions below the
    available list. This isolates metric formulas from tie/eligibility rules.
    """
    width = max(max(CUTOFFS), max(len(r.candidate_ids) for r in artifact.requests))
    preds = torch.arange(width, 0, -1, dtype=torch.float).repeat(len(artifact.requests), 1)
    target = torch.zeros_like(preds, dtype=torch.long)
    for i, request in enumerate(artifact.requests):
        for j, item in enumerate(artifact.ranking(request)):
            target[i, j] = item in request.targets
    return {str(k): {name: float(cls(top_k=k)._metric(preds, target).mean())
                     for name, cls in native_metrics.items()} for k in CUTOFFS}


def fairdiverse_common_metrics(artifact, metric_module):
    result = {}
    for k in CUTOFFS:
        ndcg, recall = [], []
        for request in artifact.requests:
            hits = [int(item in request.targets) for item in artifact.ranking(request)]
            ndcg.append(metric_module.NDCG(hits, [1], k))
            recall.append(metric_module.HR(np.array(hits), [1], k))
        result[str(k)] = dict(ndcg=float(np.mean(ndcg)), recall=float(np.mean(recall)))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=ROOT / "docs/rq1_grid_audit_20261010/inventory.json")
    parser.add_argument("--case", type=int, default=10, help="0..8 original cells; 10=Pet RVQ")
    parser.add_argument("--split", choices=["validation", "test"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--requests", type=int, default=0, help="0=full split")
    parser.add_argument("--start-row", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--legacy-diagnostic-requests", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if min(args.requests, args.start_row, args.legacy_diagnostic_requests) < 0 or args.chunk_size < 1:
        parser.error("Invalid request/offset/chunk limits")
    inventory = json.loads(args.inventory.read_text())
    case = inventory["cases"][args.case]
    for name in ("config", "checkpoint", "sid_mapping", "native_config", "source"):
        if file_reference(case[name]["path"]) != case[name]:
            raise RuntimeError(f"Pinned {name} changed since audit; regenerate inventory")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    torch.set_num_threads(4)
    seed_everything(case["seed"])
    config = yaml.safe_load(Path(case["config"]["path"]).read_text())
    config.update(device=args.device, grid_checkpoint_path=case["checkpoint"]["path"],
                  sid_path=case["sid_mapping"]["path"])
    model = original_module("grid_model").GRIDBaseModel(config).to(args.device).eval()
    fd_metrics = importlib.import_module("additional_baselines._grid_recommendation.metric")
    from src.components.eval_metrics import NDCG, Recall, SIDRetrievalEvaluator
    native_metrics = {"ndcg": NDCG, "recall": Recall}
    native = SIDRetrievalEvaluator(native_metrics, list(CUTOFFS))
    sids = {i: row for i, row in enumerate(model.tokenizer.sid_table.cpu().tolist())}
    reverse = {tuple(sid): i for i, sid in sids.items()}
    if len(reverse) != len(sids):
        raise ValueError("Nonunique catalog SID mapping")
    data_dir = ROOT / "recommendation/processed_dataset" / case["dataset"]
    split_path = data_dir / (case["dataset"] + (".valid.ranking" if args.split == "validation" else ".test.ranking"))
    brand = {int(i): str(g) for i, g in json.loads((data_dir / "iid2pid.json").read_text()).items()}
    if set(brand) != set(sids):
        raise ValueError("Group mapping and SID catalog disagree")
    groups = {"brand": brand}
    saved_scores, saved_targets = None, None
    source_export = dict(compared_requests=0, candidate_membership_differing_requests=0,
                         score_differing_requests=0, max_shifted_score_difference=0.0)
    if args.split == "test" and "ranking_scores" in case and "ground_truth" in case:
        from scipy.sparse import load_npz
        for name in ("ranking_scores", "ground_truth"):
            if file_reference(case[name]["path"]) != case[name]:
                raise RuntimeError(f"Pinned {name} changed since audit")
        saved_scores = load_npz(case["ranking_scores"]["path"]).tocsr()
        saved_targets = load_npz(case["ground_truth"]["path"]).tocsr()
    metadata = dict(checkpoint=case["checkpoint"], sid_mapping=case["sid_mapping"],
        split=dict(name=args.split, source=file_reference(split_path)), seed=case["seed"],
        generation=dict(beam_width=config["top_k_for_generation"], sequence_length=config["sequence_length"],
                        num_hierarchies=config["num_hierarchies"], generation_batch_size=1),
        score_convention="probability", short_list_policy="return_available")
    captured, parts, diagnostics, formula_errors = [], [], [], []
    generated = {"unknown_sid_count": 0, "ineligible_count": 0, "zero_probability_beams": 0,
                 "duplicate_sid_beams": 0, "requests": 0}
    original_generate = model.generate
    current = {}

    def audited_generate(histories, users=None):
        ids, probs = original_generate(histories, users)
        ids, probs_cpu = ids.detach().cpu(), probs.detach().cpu()
        captured.append((ids, probs_cpu, current["target"]))
        generated["zero_probability_beams"] += int((probs_cpu == 0).sum())
        generated["duplicate_sid_beams"] += len(ids[0])-len(set(map(tuple, ids[0].tolist())))
        if len(diagnostics) < args.legacy_diagnostic_requests:
            # Reproduce the legacy target-first catalog and float32 +1 order.
            target = current["target"]
            catalog = np.concatenate(([target], np.delete(np.arange(len(sids)), target)))
            raw = np.zeros(len(sids), dtype=np.float32)
            seen = set()
            for sid, probability in zip(ids[0].tolist(), probs_cpu[0].tolist()):
                item = reverse.get(tuple(sid))
                if item is not None and item not in seen:
                    raw[item] = probability
                    seen.add(item)
            legacy_rank = catalog[np.argsort(raw[catalog] + np.float32(1))[::-1]][:max(CUTOFFS)].tolist()
            diagnostics.append(dict(row=current["row"], target=target, legacy_top20=legacy_rank,
                shifted_to_one_count=int(((raw > 0) & (raw + np.float32(1) == 1)).sum())))
        generated["requests"] += 1
        return ids.to(probs.device), probs

    model.generate = audited_generate
    eligible = list(sids)
    with split_path.open() as stream:
        reader = itertools.islice(csv.DictReader(stream, delimiter="\t"), args.start_row,
                                  args.start_row + args.requests if args.requests else None)
        while True:
            batch = list(itertools.islice(reader, args.chunk_size))
            if not batch:
                break
            start = generated["requests"]
            def requests():
                for i, row in enumerate(batch):
                    target, user = int(row["target_item"]), int(row["user_id"])
                    current.update(target=target, row=args.start_row+start+i)
                    yield GenerationRequest(f"{args.split}:{args.start_row+start+i}:user:{user}",
                        ast.literal_eval(row["history_behaviors"])[-config["history_length"]:], [target], eligible, user)
            artifact = generate_grid(model, requests(), sids, groups, metadata, device=args.device)
            if saved_scores is not None:
                for i, (ids, probabilities, target) in enumerate(captured):
                    row = args.start_row + start + i
                    a, b = saved_targets.indptr[row:row+2]
                    if saved_targets.indices[a:b].tolist() != [target]:
                        raise RuntimeError(f"Saved source ground truth differs at row {row}")
                    first = {}
                    for sid, probability in zip(ids[0].tolist(), probabilities[0].tolist()):
                        item = reverse.get(tuple(sid))
                        if item is not None and item not in first:
                            first[item] = probability
                    positive = {item: float(np.float32(p) + np.float32(1))
                                for item, p in first.items() if p > 0}
                    a, b = saved_scores.indptr[row:row+2]
                    stored = dict(zip(saved_scores.indices[a:b].tolist(), saved_scores.data[a:b].tolist()))
                    source_export["compared_requests"] += 1
                    source_export["candidate_membership_differing_requests"] += set(positive) != set(stored)
                    delta = max((abs(positive[item]-stored[item]) for item in positive.keys() & stored.keys()), default=0)
                    source_export["max_shifted_score_difference"] = max(source_export["max_shifted_score_difference"], delta)
                    source_export["score_differing_requests"] += delta > 1e-6
            result = evaluate(artifact, CUTOFFS)
            parts.append(result)
            formula = common_ranking_metrics(artifact, native_metrics)
            fd_formula = fairdiverse_common_metrics(artifact, fd_metrics)
            for k in map(str, CUTOFFS):
                for key in native_metrics:
                    for evaluator, values in [("native_GRID", formula), ("FairDiverse", fd_formula)]:
                        difference = abs(values[k][key]-result["at_k"][k][key])
                        if difference > 1e-6:
                            formula_errors.append(dict(chunk_start=start, evaluator=evaluator,
                                                       k=k, metric=key, difference=difference))
            native(torch.cat([p for _, p, _ in captured]), torch.cat([s for s, _, _ in captured]),
                   torch.tensor([sids[t] for _, _, t in captured]))
            captured.clear()
            for key in ("unknown_sid_count", "ineligible_count"):
                generated[key] += artifact.metadata["generation_adapter"][key]
            for diagnostic in diagnostics:
                local_index = diagnostic["row"]-args.start_row-start
                if 0 <= local_index < len(artifact.requests):
                    rank = artifact.ranking(artifact.requests[local_index])[:max(CUTOFFS)]
                    diagnostic["eligible_top20"] = rank
                    diagnostic["legacy_vs_eligible_top20_equal"] = rank == diagnostic["legacy_top20"]
            print(f"{case['dataset']} {case['tokenizer']} {args.split}: {generated['requests']} requests", flush=True)
    eligible_metrics = merge_metrics(parts)
    if saved_scores is not None:
        if args.requests == 0 and args.start_row == 0 and generated["requests"] != saved_scores.shape[0]:
            raise RuntimeError("Full split and saved source request counts differ")
        source_export["passed"] = (source_export["candidate_membership_differing_requests"] == 0
                                    and source_export["score_differing_requests"] == 0)
    native_result = {key: float(metric.compute()) for key, metric in native.metrics.items()}
    differences = {f"{key}@{k}": native_result[f"{key}@{k}"]-eligible_metrics["at_k"][str(k)][key]
                   for k in CUTOFFS for key in native_metrics}
    report = dict(case=case, input_metadata=metadata, start_row=args.start_row, requested_limit=args.requests,
        full_split=args.requests == 0 and args.start_row == 0, metrics=eligible_metrics,
        native_sid_metrics=native_result, native_minus_eligible=differences,
        common_rank_metric_formula_agreement=not formula_errors, formula_errors=formula_errors,
        generated=generated, legacy_diagnostic=diagnostics,
        saved_source_export_agreement=source_export if saved_scores is not None else None,
        scope="Same generated beams; native SID and eligible raw-score metrics on full requested slice; legacy ordering diagnostic on prefix only",
        saved_source_comparison="Saved source uses its declared protocol; full legacy parity is not implied by this check",
        corrected_pet_sprint_source_result=case["dataset"] == "amazon-pet-supplies" and case["tokenizer"] == "rvq" and args.split == "test" and args.requests == 0 and args.start_row == 0)
    with (args.output_dir / "results.json").open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(dict(common_rank_agreement=not formula_errors, native_minus_eligible=differences), indent=2))
    if formula_errors:
        raise SystemExit("Metric formulas disagree on common rankings; inspect results.json before proceeding")
    if saved_scores is not None and not source_export["passed"]:
        raise SystemExit("Fresh generation differs from the saved source export; inspect results.json before proceeding")


if __name__ == "__main__":
    main()
