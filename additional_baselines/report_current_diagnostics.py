"""CPU diagnostics for current GRID continuations and selected rerankers.

Uses the deck snapshot for continuation identities and explicit selection files
for newer rerankers. Missing predictions are recorded, never synthesized.
"""
import argparse
import csv
import gc
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .candidates import file_reference, load
from .report_beauty_activity import validate_alignment, make_rows, check_aggregate
from fairdiverse.recommendation.user_fairness import load_activity, save_report

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "additional_baselines/artifacts"
CUTOFFS = (5, 10, 20)


def diagnostics(inputs, rows, activity):
    """Separate candidate misses from ranking quality, within fixed groups."""
    by_id = {r.request_id: r for r in inputs.requests}
    mapping = inputs.item_groups["brand"]
    group_count = len(set(mapping.values()))
    for row in rows:
        request = by_id[row["request_id"]]
        row["candidate_brand_fraction"] = len({mapping[i] for i in request.candidate_ids}) / group_count
    result = {}
    names = ["overall"] + sorted({activity[r["user_id"]]["activity_group"] for r in rows})
    for name in names:
        selected = rows if name == "overall" else [r for r in rows if activity[r["user_id"]]["activity_group"] == name]
        covered = [r for r in selected if r["candidate_coverage"] == 1]
        counts = np.asarray([r["candidate_count"] for r in selected])
        result[name] = dict(request_count=len(selected),
            target_coverage=float(np.mean([r["candidate_coverage"] for r in selected])),
            candidate_count_mean=float(counts.mean()),
            candidate_count_quantiles=dict(zip(("min", "p25", "p50", "p75", "p95", "max"),
                np.quantile(counts, (0, .25, .5, .75, .95, 1)).tolist())),
            empty_candidate_rate=float(np.mean(counts == 0)),
            candidate_brand_fraction=float(np.mean([r["candidate_brand_fraction"] for r in selected])),
            covered_request_count=len(covered),
            conditional_metrics={f"{m}@{k}": float(np.mean([r[f"{m}@{k}"] for r in covered])) if covered else None
                                 for m in ("recall", "ndcg") for k in CUTOFFS},
            at_k={str(k): dict(candidate_short_list_rate=float(np.mean(counts < k)),
                returned_short_list_rate=float(np.mean([r[f"returned_count@{k}"] < k for r in selected])),
                fill_rate=float(np.mean([r[f"returned_count@{k}"] / k for r in selected]))) for k in CUTOFFS})
    return result


def report(dataset, candidates, destination, result_path=None, method=None, bootstrap_samples=1000,
           expected_checkpoint=None, expected_checkpoint_reference=None, comparison_candidates=None):
    data = ROOT / "recommendation/processed_dataset" / dataset
    split = data / f"{dataset}.test.ranking"
    activity_path = data / "user_activity.csv"
    inputs = load(candidates)
    if expected_checkpoint and Path(inputs.metadata["checkpoint"]["path"]).resolve() != (ROOT / expected_checkpoint).resolve():
        raise ValueError("Candidate checkpoint differs from current deck checkpoint")
    if expected_checkpoint_reference and inputs.metadata["checkpoint"] != expected_checkpoint_reference:
        raise ValueError("Candidate checkpoint hash differs from selected continuation checkpoint")
    activity = load_activity(activity_path)
    activity_metadata = json.loads((data / "user_activity_metadata.json").read_text())
    manifest = json.loads((data / "grid_data_manifest.json").read_text())
    if activity_metadata["provenance"].get("source_fingerprint") != manifest["source_fingerprint"]:
        raise ValueError("Activity groups and current GRID data have different sources")
    with split.open() as stream:
        users = validate_alignment(inputs, list(csv.DictReader(stream, delimiter="\t")), activity, file_reference(split))
    sources = dict(candidates=file_reference(candidates), split=file_reference(split))
    rankings, expected = {}, {}
    if result_path:
        result = json.loads(Path(result_path).read_text())
        sources["results"] = file_reference(result_path)
        if "input_metadata" in result and result["input_metadata"] != inputs.metadata:
            raise ValueError("Results and candidate metadata differ")
        if method == "rr":
            # RR reports carry the frozen validation identity, not test metadata.
            frozen = result["frozen_correction"]["validation_metadata"]
            for key in ("checkpoint", "sid_mapping", "generation", "seed", "candidate_membership", "score_convention"):
                if frozen[key] != inputs.metadata[key]:
                    raise ValueError(f"Frozen RR correction differs in {key}")
            rankings = {k: result["rankings"] for k in CUTOFFS}
            expected = {k: result["metrics"] for k in CUTOFFS}
        elif method in ("cpfair", "pmmf"):
            if "runs" in result:
                rankings = {k: result["runs"][str(k)]["rankings"] for k in CUTOFFS}
                expected = {k: result["runs"][str(k)]["metrics"] for k in CUTOFFS}
            else:
                rankings = {k: result["rankings_by_k"][str(k)] for k in CUTOFFS}
                expected = {k: dict(result["metrics"], request_count=len(inputs.requests)) for k in CUTOFFS}
        else:
            expected = {k: result["metrics"] for k in CUTOFFS}
    rows = make_rows(inputs, users, rankings.get(5))
    for k in CUTOFFS:
        measured = rows if k == 5 else make_rows(inputs, users, rankings.get(k))
        if k in expected:
            check_aggregate(measured, expected[k], (k,))
        for row, correct in zip(rows, measured):
            for metric in ("recall", "ndcg"):
                row[f"{metric}@{k}"] = correct[f"{metric}@{k}"]
        for request, row in zip(inputs.requests, rows):
            ranking = inputs.ranking(request) if k not in rankings else rankings[k][request.request_id]
            row[f"returned_count@{k}"] = min(len(ranking), k)
    candidate_summary = diagnostics(inputs, rows, activity)
    summary = save_report(rows, activity_path, destination, bootstrap_samples, 42,
        dict(dataset=dataset, model=method or "GRID", input_metadata=inputs.metadata, sources=sources,
             cutoff_policy="selected CPFair/P-MMF allocations evaluated independently at each K"))
    (destination / "candidate_diagnostics.json").write_text(json.dumps(dict(
        by_activity=candidate_summary, sources=sources,
        conditional_scope="ranking metrics conditional on target present; unconditional metrics retain all requests"), indent=2)+"\n")
    item_groups = {}
    for name, mapping in inputs.item_groups.items():
        catalog_counts = Counter(mapping.values())
        reach, item_counts = Counter(), Counter()
        exposures = {k: Counter() for k in CUTOFFS}
        for request in inputs.requests:
            reach.update({mapping[i] for i in request.candidate_ids})
            item_counts.update(mapping[i] for i in request.candidate_ids)
            for k in CUTOFFS:
                ranking = inputs.ranking(request) if k not in rankings else rankings[k][request.request_id]
                exposures[k].update(mapping[i] for i in ranking[:k])
        item_groups[name] = {g: dict(catalog_items=catalog_counts[g],
            candidate_request_fraction=reach[g]/len(rows), candidate_item_occurrences=item_counts[g],
            exposure_counts={str(k): exposures[k][g] for k in CUTOFFS}) for g in sorted(catalog_counts)}
    (destination / "item_group_availability.json").write_text(json.dumps(item_groups, indent=2)+"\n")
    # Rerankers compare with their untreated candidate ordering. RW/SPRINT
    # compare with the matched ordinary continuation, supplied explicitly.
    if method or comparison_candidates:
        baseline = load(comparison_candidates) if comparison_candidates else inputs
        for key in ("sid_mapping", "generation", "seed", "candidate_membership", "score_convention"):
            if baseline.metadata[key] != inputs.metadata[key]:
                raise ValueError(f"Activity comparison baseline differs in {key}")
        with split.open() as stream:
            baseline_users = validate_alignment(baseline, list(csv.DictReader(stream, delimiter="\t")), activity, file_reference(split))
        baseline_rows = make_rows(baseline, baseline_users, None)
        changes = {}
        for name in ["overall"] + sorted({activity[r["user_id"]]["activity_group"] for r in rows}):
            pairs = [(r,b) for r,b in zip(rows, baseline_rows) if name == "overall" or activity[r["user_id"]]["activity_group"] == name]
            changes[name] = {f"{m}@{k}": dict(mean_signed_change=float(np.mean([r[f"{m}@{k}"]-b[f"{m}@{k}"] for r,b in pairs])),
                users_improved_fraction=float(np.mean([r[f"{m}@{k}"] > b[f"{m}@{k}"] for r,b in pairs])),
                users_worsened_fraction=float(np.mean([r[f"{m}@{k}"] < b[f"{m}@{k}"] for r,b in pairs])))
                for m in ("recall", "ndcg") for k in CUTOFFS}
        (destination / "activity_changes.json").write_text(json.dumps(dict(
            baseline=file_reference(comparison_candidates or candidates), by_activity=changes,
            uncertainty="signed point estimates; paired intervals are separate priority-6 work"), indent=2)+"\n")
    return dict(candidate_diagnostics=candidate_summary, activity=summary, sources=sources)


def discover(dataset):
    cases, gaps = [], []
    snapshot = ROOT / "docs/grid_update_october_2026_v1_METRICS.csv"
    with snapshot.open() as stream:
        rows = [r for r in csv.DictReader(stream) if r["dataset"] == dataset and r["panel"] == "core"]
    for row in rows:
        if row["method"] not in ("Ordinary continuation", "IFairLRS RW", "SPRINT"):
            continue
        if dataset == "amazon-pet-supplies" and row["tokenizer"] == "rvq" and row["method"] == "SPRINT":
            gaps.append(dict(method="Pet RVQ SPRINT", reason="update-zero checkpoint mismatch; requires corrected continuation test candidates"))
            continue
        result = ROOT / row["source"]
        candidates = result.parent / "test_candidates.json"
        continuation = json.loads((result.parent / "continuation.json").read_text())
        if (ROOT / continuation["identity"]["checkpoint"]["path"]).resolve() != (ROOT / row["checkpoint"]).resolve():
            raise ValueError("Continuation source differs from deck source checkpoint")
        cases.append(dict(name=row["tokenizer"]+"_"+row["method"], candidates=candidates,
                          result_path=result, expected_checkpoint_reference=continuation["selected_checkpoint"]))
        if row["method"] in ("IFairLRS RW", "SPRINT"):
            control = (ARTIFACTS / "beauty_native_control_lr1e3" if dataset == "amazon-beauty" and row["tokenizer"] == "rkmeans"
                       else result.parent.parent / "ordinary") / "test_candidates.json"
            cases[-1]["comparison_candidates"] = control
    # Existing full original candidate artifacts, plus new priority-7 preparations.
    candidates = [ARTIFACTS / ("beauty_ifairlrs_rr_popularity/original" if dataset == "amazon-beauty"
                               else dataset.removeprefix("amazon-")+"_rkmeans_rerank_candidates") / "test_candidates.json"]
    for tokenizer in ("rqvae", "rvq"):
        folders = sorted(ARTIFACTS.glob(dataset.removeprefix("amazon-")+f"_{tokenizer}_rerank_candidates_*/manifest.json"))
        complete = [p.parent / "test_candidates.json" for p in folders
                    if json.loads(p.read_text()).get("scope") == "full_splits"
                    and (p.parent / "complete.json").exists() and (p.parent / "test_candidates.json").exists()]
        if complete:
            candidates.append(complete[-1])
        else:
            gaps.append(dict(method=f"{tokenizer} original candidates", reason="needs priority-7 full candidate preparation"))
    cases.extend(dict(name="original_"+p.parent.name, candidates=p) for p in candidates)
    for path in sorted(ARTIFACTS.glob("*/selection.json")):
        selection = json.loads(path.read_text())
        if selection["dataset"] != dataset:
            continue
        candidate_ref = selection["test_candidates"]
        if file_reference(candidate_ref["path"]) != candidate_ref:
            raise ValueError("Selected test candidates changed")
        for method in selection["methods"]:
            result = path.parent / f"{method}_test_results.json"
            if result.exists():
                cases.append(dict(name=path.parent.name+"_"+method,
                    candidates=Path(candidate_ref["path"]), result_path=result, method=method))
            else:
                gaps.append(dict(method=str(result), reason="selected reranker test unfinished"))
    if dataset == "amazon-beauty":
        # These final native-score CPFair/P-MMF reports predate selection.json.
        for method in ("cpfair", "pmmf"):
            result = ARTIFACTS / "beauty_original_selected_rerank_352496" / f"{method}_test_results.json"
            cases.append(dict(name="beauty_selected_"+method, candidates=candidates[0], result_path=result, method=method))
    if not any(c.get("method") == "rr" for c in cases):
        gaps.append(dict(method="validation-selected RKMeans rerankers", reason="final tests not available; finish priority 2/7"))
    return cases, gaps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("amazon-beauty", "amazon-sports", "amazon-toys", "amazon-pet-supplies"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--inventory-only", action="store_true")
    args = parser.parse_args()
    cases, gaps = discover(args.dataset)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    manifest = dict(dataset=args.dataset, snapshot=file_reference(ROOT / "docs/grid_update_october_2026_v1_METRICS.csv"),
                    cases=[{k: str(v) if isinstance(v, Path) else v for k,v in c.items()} for c in cases], gaps=gaps)
    conventional = []
    for path in sorted((ROOT / "output/rq2_classic_evaluation").glob("**/test_full/results.json")):
        result = json.loads(path.read_text())
        if result["full_split"] and result["case"]["dataset"] == args.dataset:
            summary = path.parent / "user_fairness/user_fairness_summary.json"
            conventional.append(dict(model=result["case"]["model"], results=file_reference(path),
                                     activity_summary=file_reference(summary)))
    manifest["existing_conventional_activity_reports"] = conventional
    if not any(c["model"] == "SASRec" for c in conventional):
        gaps.append(dict(method="SASRec activity report", reason="waiting for priority-3 full evaluation"))
    (args.output_dir / "inventory.json").write_text(json.dumps(manifest, indent=2)+"\n")
    if args.inventory_only:
        print(json.dumps(manifest, indent=2))
        return
    comparison = []
    for case in cases:
        if not case["candidates"].exists():
            gaps.append(dict(method=case["name"], reason="missing test candidates"))
            continue
        print("Reporting", case["name"], flush=True)
        name = case["name"].replace(" ", "_")
        result = report(args.dataset, destination=args.output_dir / name,
               bootstrap_samples=args.bootstrap_samples, **{k:v for k,v in case.items() if k != "name"})
        for statistic, metrics in result["activity"]["statistics"].items():
            for metric, value in metrics.items():
                bounds = result["activity"].get("confidence_intervals", {}).get(statistic, {}).get(metric, [None,None])
                comparison.append(dict(method=name, statistic=statistic, metric=metric, value=value,
                                       ci95_low=bounds[0], ci95_high=bounds[1]))
        gc.collect()
    for saved in conventional:
        summary = json.loads(Path(saved["activity_summary"]["path"]).read_text())
        for statistic, metrics in summary["statistics"].items():
            for metric, value in metrics.items():
                bounds = summary.get("confidence_intervals", {}).get(statistic, {}).get(metric, [None,None])
                comparison.append(dict(method=saved["model"], statistic=statistic, metric=metric, value=value,
                                       ci95_low=bounds[0], ci95_high=bounds[1]))
    if comparison:
        with (args.output_dir / "all_methods_by_activity.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(comparison[0]))
            writer.writeheader()
            writer.writerows(comparison)
    (args.output_dir / "remaining_diagnostics.json").write_text(json.dumps(gaps, indent=2)+"\n")
    print("Reports:", args.output_dir, "Remaining gaps:", len(gaps))


if __name__ == "__main__":
    main()
