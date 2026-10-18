"""Backfill Beauty activity reports from saved, aligned test rankings on CPU.

Run ``python -m additional_baselines.report_beauty_activity --help``.
The W&B export is included only as aggregate context: it has no user rankings.
"""

import argparse
import csv
import json
from pathlib import Path

from .candidates import file_reference, load
from fairdiverse.recommendation.user_fairness import (
    load_activity, ranking_metrics, save_report,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = "recommendation/processed_dataset/amazon-beauty"
DEFAULT_EXPORT = "fairdiverse/recommendation/log/2026-9-20_grid-independent-amazon-beauty-rkmeans-flan-t5-xl-complete-metadata-score-export"
DEFAULT_OUTPUT = "additional_baselines/artifacts/beauty_rkmeans_suffix4"
CUTOFFS = (5, 10, 20)
RR = "additional_baselines/artifacts/beauty_ifairlrs_rr_popularity/original"
RW = "additional_baselines/artifacts/beauty_ifairlrs_rw_popularity"
SPRINT = "additional_baselines/artifacts/beauty_sprint_grid_encoder_351175"


def read_json(path):
    return json.loads(Path(path).read_text())


def read_rows(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def validate_alignment(inputs, test_rows, activity, split_source):
    if inputs.metadata["split"]["name"] != "test" or inputs.metadata["split"]["source"] != split_source:
        raise ValueError("Candidate artifact belongs to a different test split")
    if len(inputs.requests) != len(test_rows) or len(test_rows) != len(activity):
        raise ValueError("Candidate, test, and activity user counts disagree")
    request_users = {}
    for index, (request, row) in enumerate(zip(inputs.requests, test_rows)):
        user = row["user_id"]
        if request.request_id != f"test:{index}:user:{user}":
            raise ValueError(f"Request/user order mismatch at test row {index}")
        if request.targets != [int(row["target_item"])]:
            raise ValueError(f"Held-out target mismatch at test row {index}")
        if user not in activity:
            raise ValueError(f"No activity mapping for test user {user}")
        request_users[request.request_id] = user
    if set(request_users.values()) != set(activity):
        raise ValueError("Test users and activity users differ")
    return request_users


def make_rows(inputs, request_users, rankings):
    """Retain all requests, including generation misses, with eligible rankings."""
    if rankings is not None and set(rankings) != set(request_users):
        raise ValueError("Rankings must cover exactly the test requests")
    rows = []
    for request in inputs.requests:
        ranking = inputs.ranking(request) if rankings is None else rankings[request.request_id]
        if not isinstance(ranking, list) or len(ranking) != len(set(ranking)):
            raise ValueError(f"Duplicate or malformed ranking for {request.request_id}")
        if set(ranking) - set(request.candidate_ids):
            raise ValueError(f"Ineligible ranked item for {request.request_id}")
        rows.append({
            "user_id": request_users[request.request_id], "request_id": request.request_id,
            "candidate_coverage": len(set(request.targets) & set(request.candidate_ids)) / len(request.targets),
            "candidate_count": len(request.candidate_ids),
            **ranking_metrics(ranking, request.targets, CUTOFFS),
        })
    return rows


def check_aggregate(rows, expected, cutoffs=CUTOFFS):
    if expected["request_count"] != len(rows):
        raise ValueError("Saved aggregate request count disagrees with reconstructed rows")
    coverage = sum(row["candidate_coverage"] for row in rows) / len(rows)
    if abs(coverage - expected["candidate_target_recall"]) > 1e-10:
        raise ValueError("Saved candidate coverage disagrees with reconstructed rows")
    for k in cutoffs:
        for source_name, row_name in (("recall", f"recall@{k}"), ("ndcg", f"ndcg@{k}")):
            measured = sum(row[row_name] for row in rows) / len(rows)
            if abs(measured - expected["at_k"][str(k)][source_name]) > 1e-10:
                raise ValueError(f"Saved {source_name}@{k} disagrees with reconstructed rows")


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows for {path}")
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def wandb_context(path, local_control):
    """Select the same named configuration; never imply W&B has user data."""
    if path is None:
        return [], None
    path = Path(path)
    with path.open(newline="") as stream:
        rows = [row for row in csv.DictReader(stream)
                if row["dataset"] == "amazon-beauty"
                and row["model"] == "GRID"
                and row["grid_tokenizer"] == "rkmeans"
                and row["metadata_text_mode"] == "complete_metadata"
                and row["embedding_model"] == "google/flan-t5-xl"
                and row["seed"] == "42"]
    if not rows:
        raise ValueError("W&B snapshot has no matching Beauty complete-metadata rows")
    base = [row for row in rows if row["baseline_name"] == "base"]
    if len(base) != 1:
        raise ValueError("W&B snapshot must identify exactly one matching base row")
    checks = {}
    for k in CUTOFFS:
        for wandb_name, local_name in (("hr", "recall"), ("ndcg", "ndcg")):
            key = f"test/{wandb_name}@{k}"
            observed = float(base[0][key])
            expected = local_control["at_k"][str(k)][local_name]
            if abs(observed - expected) > 0.0001:
                raise ValueError(f"W&B base {key} differs from the saved candidate control")
            checks[key] = {"wandb_rounded": observed, "local": expected}
    return rows, {"source": file_reference(path), "base_run_id": base[0]["run_id"],
                  "base_rounded_metric_checks": checks,
                  "scope": "aggregate context only; no per-user predictions or checkpoint/SID fingerprint in this export"}


def run(root, output_dir, bootstrap_samples=1000, seed=42, wandb_csv=None):
    root, output_dir = Path(root).resolve(), Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    if type(bootstrap_samples) is not int or bootstrap_samples < 0:
        raise ValueError("bootstrap_samples must be nonnegative")
    data = root / DEFAULT_DATA
    activity_path = data / "user_activity.csv"
    activity = load_activity(activity_path)
    split_path = data / "amazon-beauty.test.ranking"
    split_source = file_reference(split_path)
    test_rows = read_rows(split_path)
    legacy_path = root / DEFAULT_OUTPUT / "inputs.json"
    legacy = load(legacy_path)
    legacy_users = validate_alignment(legacy, test_rows, activity, split_source)
    export = root / DEFAULT_EXPORT
    cp_path, pm_path = export / "cpfair_results.json", export / "pmmf_results.json"
    cp, pm = read_json(cp_path), read_json(pm_path)
    if cp["input_metadata"] != legacy.metadata or pm["input_metadata"] != legacy.metadata:
        raise ValueError("Legacy rerank reports do not match the saved candidate inputs")
    for result in (cp, pm):
        if result["evaluation_scope"]["evaluated_request_count"] != len(test_rows):
            raise ValueError("Legacy rerank report is not the full test split")

    sources = {}
    comparison = []
    def record(name, inputs, users, rankings, expected, source_files):
        rows = make_rows(inputs, users, rankings)
        check_aggregate(rows, expected)
        sources[name] = {label: file_reference(path) for label, path in source_files.items()}
        summary = save_report(rows, activity_path, output_dir / name, bootstrap_samples, seed,
                              {"model": name, "input_metadata": inputs.metadata, "sources": sources[name]})
        counts = summary["group_user_counts"]
        for statistic, metrics in summary["statistics"].items():
            selected = (rows if statistic == "overall" else
                        [row for row in rows if activity[row["user_id"]]["activity_group"] == statistic[6:]]
                        if statistic.startswith("group/") else [])
            coverage = sum(row["candidate_coverage"] for row in selected) / len(selected) if selected else ""
            for metric, value in metrics.items():
                bounds = summary.get("confidence_intervals", {}).get(statistic, {}).get(metric, ["", ""])
                comparison.append({"method": name, "statistic": statistic,
                                   "user_count": counts.get(statistic[6:], summary["user_count"] if statistic == "overall" else ""),
                                   "metric": metric, "value": value, "ci95_low": bounds[0], "ci95_high": bounds[1],
                                   "candidate_coverage": coverage})

    record("original_legacy", legacy, legacy_users, None, cp["control_metrics"],
           {"candidates": legacy_path, "results": cp_path})
    for name, result, result_path in (("cpfair", cp, cp_path), ("pmmf", pm, pm_path)):
        rows_by_id = {request.request_id: {"user_id": legacy_users[request.request_id],
                        "request_id": request.request_id,
                        "candidate_coverage": len(set(request.targets) & set(request.candidate_ids)) / len(request.targets),
                        "candidate_count": len(request.candidate_ids)} for request in legacy.requests}
        for k in CUTOFFS:
            variant = result["runs"][str(k)]
            rows = make_rows(legacy, legacy_users, variant["rankings"])
            check_aggregate(rows, variant["metrics"], (k,))
            for row in rows:
                for metric in (f"recall@{k}", f"ndcg@{k}"):
                    rows_by_id[row["request_id"]][metric] = row[metric]
        # Each cutoff is a separate allocation with its own budget and state.
        merged = list(rows_by_id.values())
        sources[name] = {"candidates": file_reference(legacy_path), "results": file_reference(result_path)}
        summary = save_report(merged, activity_path, output_dir / name, bootstrap_samples, seed,
                              {"model": name, "input_metadata": legacy.metadata, "sources": sources[name],
                               "cutoff_policy": "independent FairDiverse allocation per K"})
        for statistic, metrics in summary["statistics"].items():
            selected = (merged if statistic == "overall" else
                        [row for row in merged if activity[row["user_id"]]["activity_group"] == statistic[6:]]
                        if statistic.startswith("group/") else [])
            coverage = sum(row["candidate_coverage"] for row in selected) / len(selected) if selected else ""
            for metric, value in metrics.items():
                bounds = summary.get("confidence_intervals", {}).get(statistic, {}).get(metric, ["", ""])
                comparison.append({"method": name, "statistic": statistic,
                                   "user_count": summary["group_user_counts"].get(statistic[6:], summary["user_count"] if statistic == "overall" else ""),
                                   "metric": metric, "value": value, "ci95_low": bounds[0], "ci95_high": bounds[1],
                                   "candidate_coverage": coverage})

    native_specs = [
        ("original_fresh", RR, None), ("ifairlrs_rr", RR, "rankings"),
        ("ordinary_control", f"{RW}/ordinary_control", None),
        ("ifairlrs_rw", f"{RW}/ifairlrs_rw_grid_popularity_adaptation", None),
        ("sprint_grid_encoder", f"{SPRINT}/sprint_grid_encoder_adaptation", None),
    ]
    rr_result = read_json(root / RR / "test_results.json")
    for name, relative_dir, reranking in native_specs:
        directory = root / relative_dir
        candidate_path, result_path = directory / "test_candidates.json", directory / "test_results.json"
        inputs = load(candidate_path)
        users = validate_alignment(inputs, test_rows, activity, split_source)
        result = rr_result if relative_dir == RR else read_json(result_path)
        expected = result["control_metrics" if name == "original_fresh" else "metrics"] if relative_dir == RR else result["metrics"]
        ranks = result[reranking] if reranking else None
        record(name, inputs, users, ranks, expected,
               {"candidates": candidate_path, "results": result_path})

    # The SPRINT job recorded its own ordinary control. Verify that it is the
    # same checkpoint and evaluation before representing both with one row.
    sprint_control = root / SPRINT / "ordinary_control"
    sprint_control_result = read_json(sprint_control / "test_results.json")
    rw_control_result = read_json(root / RW / "ordinary_control/test_results.json")
    if sprint_control_result["metrics"] != rw_control_result["metrics"]:
        raise ValueError("SPRINT and RW ordinary controls have different metrics")
    if file_reference(sprint_control / "checkpoint.ckpt")["sha256"] != file_reference(root / RW / "ordinary_control/checkpoint.ckpt")["sha256"]:
        raise ValueError("SPRINT and RW ordinary controls have different checkpoints")
    rw_control_candidates = load(root / RW / "ordinary_control/test_candidates.json")
    sprint_control_candidates = load(sprint_control / "test_candidates.json")
    if rw_control_candidates.requests != sprint_control_candidates.requests:
        raise ValueError("SPRINT and RW ordinary controls have different test candidates")
    sources["ordinary_control"]["sprint_control_results"] = file_reference(sprint_control / "test_results.json")
    sources["ordinary_control"]["sprint_control_candidates"] = file_reference(sprint_control / "test_candidates.json")

    wandb_rows, wandb_meta = wandb_context(wandb_csv, cp["control_metrics"])
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "all_methods_by_activity.csv", comparison)
    if wandb_rows:
        write_csv(output_dir / "wandb_aggregate_context.csv", wandb_rows)
    manifest = {"dataset": "amazon-beauty", "setting": "GRID RKMeans complete_metadata suffix4",
                "test_request_count": len(test_rows), "activity": file_reference(activity_path),
                "test_split": split_source, "cutoffs": list(CUTOFFS),
                "bootstrap_samples": bootstrap_samples, "bootstrap_seed": seed,
                "methods": sources, "wandb": wandb_meta,
                "interpretation": "W&B rows are aggregate context; only saved candidate/ranking artifacts support activity-group analysis"}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {"output_dir": str(output_dir), "methods": list(sources),
            "test_request_count": len(test_rows), "wandb_context_rows": len(wandb_rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "additional_baselines/artifacts/beauty_complete_metadata_activity")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb-csv", type=Path, default=ROOT / "wandb_baselines_after_2026-08-27_grouped.csv")
    args = parser.parse_args()
    print(json.dumps(run(args.root, args.output_dir, args.bootstrap_samples, args.seed, args.wandb_csv), indent=2))


if __name__ == "__main__":
    main()
