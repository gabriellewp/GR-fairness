"""Sweep Beauty rerankers on saved validation predictions, never test targets.

Run ``python -m additional_baselines.tune_beauty_reranking --help``.
Each invocation handles one method and writes a new output directory. A request
limit is for integration checks only and is explicitly marked as a smoke test.
"""

import argparse
import csv
import json
import math
import resource
import time
from pathlib import Path

from .candidates import file_reference, load
from .evaluation import evaluate
from .ifairlrs import apply_rr, fairness, fit_rr


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANDIDATES = ROOT / "additional_baselines/artifacts/beauty_ifairlrs_rr_popularity/original/validation_candidates.json"
DEFAULT_HISTORIES = ROOT / "additional_baselines/artifacts/beauty_ifairlrs_rr_popularity/original/validation_histories.json"
DEFAULT_SPLIT = ROOT / "recommendation/processed_dataset/amazon-beauty/amazon-beauty.valid.ranking"
DEFAULT_ACTIVITY = ROOT / "recommendation/processed_dataset/amazon-beauty/user_activity.csv"
CUTOFFS = (5, 10, 20)
DEFAULT_STRENGTHS = {"cpfair": [0, 0.01, 0.1, 1], "pmmf": [0, 1, 10, 30],
                     "rr": [0, 0.01, 0.02, 0.05, 0.1]}
DEFAULT_FLOORS = [1e-6, 1e-3, 1e-2]


def task_plan(method, strengths=None, floors=None):
    if method not in DEFAULT_STRENGTHS:
        raise ValueError("Choose cpfair, pmmf, or rr")
    strengths = DEFAULT_STRENGTHS[method] if strengths is None else strengths
    floors = DEFAULT_FLOORS if floors is None else floors
    if not strengths or not floors or any(not math.isfinite(x) or x < 0 for x in strengths) or any(
            not math.isfinite(x) or not 0 < x <= 1 for x in floors):
        raise ValueError("Strengths must be nonnegative and floors in (0, 1]")
    if len(set(strengths)) != len(strengths) or len(set(floors)) != len(floors):
        raise ValueError("Grid values must be unique")
    if method == "rr":
        return [{"alpha": alpha, "floor": floor} for alpha in strengths
                for floor in ([floors[0]] if alpha == 0 else floors)]
    return [{"lambda": strength, "k": k} for strength in strengths for k in CUTOFFS]


def write_outputs(output_dir, rows, manifest):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    frontier = pareto(rows, manifest["fairness_axis"])
    for filename, data in (("validation_tradeoff.csv", rows), ("pareto_frontier.csv", frontier)):
        with (output_dir / filename).open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(data)
    (output_dir / "validation_rows.json").write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return {"output_dir": str(output_dir.resolve()), "pareto_rows": len(frontier), "scope": manifest["scope"]}


def read_rows(path, delimiter=","):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream, delimiter=delimiter))


def checked_inputs(candidates, split, activity, limit_requests=None, *, expected_request_ids=None):
    inputs = load(candidates)
    if inputs.metadata["split"] != {"name": "validation", "source": file_reference(split)}:
        raise ValueError("Candidates must match the complete, fingerprinted validation split")
    split_rows = read_rows(split, "\t")
    activity_rows = read_rows(activity)
    groups = {row["user_id"]: row["activity_group"] for row in activity_rows}
    if len(groups) != len(activity_rows) or len(inputs.requests) != len(split_rows):
        raise ValueError("Duplicate activity user or incomplete validation artifact")
    if limit_requests is not None and (limit_requests < 1 or limit_requests > len(split_rows)):
        raise ValueError("Request limit is outside the validation split")
    for index, (request, row) in enumerate(zip(inputs.requests, split_rows)):
        user = row["user_id"]
        if (request.request_id != f"validation:{index}:user:{user}"
                or request.targets != [int(row["target_item"])] or user not in groups):
            raise ValueError(f"Validation request/user/target alignment failed at row {index}")
    if limit_requests is None and set(groups) != {row["user_id"] for row in split_rows}:
        raise ValueError("Full validation split and activity mapping contain different users")
    if expected_request_ids is not None and expected_request_ids != {r.request_id for r in inputs.requests}:
        raise ValueError("Validation histories do not cover exactly the saved requests")
    if limit_requests is not None:
        inputs.requests = inputs.requests[:limit_requests]
    return inputs, groups


def activity_recall(inputs, groups, rankings, k):
    hits = {group: 0.0 for group in set(groups.values())}
    counts = dict.fromkeys(hits, 0)
    for request in inputs.requests:
        group = groups[request.request_id.rsplit(":user:", 1)[1]]
        ranked = inputs.ranking(request) if rankings is None else rankings[request.request_id]
        hits[group] += len(set(ranked[:k]) & set(request.targets)) / len(request.targets)
        counts[group] += 1
    means = {g: hits[g] / counts[g] for g in counts if counts[g]}
    return {"low": means.get("low"), "medium": means.get("medium"),
            "high": means.get("high"), "worst": min(means.values()),
            "gap": max(means.values()) - min(means.values())}


def record(method, parameters, inputs, groups, histories, rankings, metrics):
    rows = []
    cutoffs = tuple(map(int, metrics["at_k"]))
    rr_fairness = fairness(inputs, histories, "popularity", cutoffs, rankings) if method == "rr" else None
    for k in cutoffs:
        result = metrics["at_k"][str(k)]
        exposure = result["groups"]["popularity" if method == "rr" else "brand"]
        activity = activity_recall(inputs, groups, rankings, k)
        rows.append({"method": method, "parameters": json.dumps(parameters, sort_keys=True),
                     "k": k, "requests": len(inputs.requests),
                     "candidate_coverage": metrics["candidate_target_recall"],
                     "recall": result["recall"], "ndcg": result["ndcg"],
                     "exposure_gini": exposure["gini"],
                     "mgu": rr_fairness[str(k)]["MGU"] if rr_fairness else "",
                     "dgu": rr_fairness[str(k)]["DGU"] if rr_fairness else "",
                     "low_recall": activity["low"], "medium_recall": activity["medium"],
                     "high_recall": activity["high"], "worst_recall": activity["worst"],
                     "activity_gap": activity["gap"]})
    return rows


def pareto(rows, fairness_column):
    """Keep settings with no other setting at least as good on both axes."""
    frontier = []
    for row in rows:
        if row["parameters"] == "{}":
            continue
        peers = [other for other in rows if other["k"] == row["k"] and other is not row]
        dominated = any(other["recall"] >= row["recall"]
                        and other[fairness_column] <= row[fairness_column]
                        and (other["recall"] > row["recall"]
                             or other[fairness_column] < row[fairness_column]) for other in peers)
        if not dominated:
            frontier.append(row)
    return frontier


def run(method, output_dir, *, candidates=DEFAULT_CANDIDATES, histories_path=DEFAULT_HISTORIES,
        split=DEFAULT_SPLIT, activity=DEFAULT_ACTIVITY, limit_requests=None,
        strengths=None, floors=None, max_dense_bytes=4294967296, horizon=None, task_index=None):
    started = time.perf_counter()
    if method not in {"cpfair", "pmmf", "rr"}:
        raise ValueError("Choose cpfair, pmmf, or rr")
    if horizon is not None and (type(horizon) is not int or horizon <= 0):
        raise ValueError("Horizon must be a positive request count")
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Output directory exists: {output_dir}")
    strengths = list(DEFAULT_STRENGTHS[method] if strengths is None else strengths)
    floors = list(DEFAULT_FLOORS if floors is None else floors)
    plan = task_plan(method, strengths, floors)
    if task_index is not None and (type(task_index) is not int or not 0 <= task_index < len(plan)):
        raise ValueError("Task index outside the sweep grid")
    tasks = plan if task_index is None else [plan[task_index]]
    cutoffs = CUTOFFS if method == "rr" or task_index is None else (tasks[0]["k"],)
    histories = json.loads(Path(histories_path).read_text()) if method == "rr" else None
    inputs, groups = checked_inputs(candidates, split, activity, limit_requests,
                                   expected_request_ids=set(histories) if histories is not None else None)
    if histories is not None:
        histories = {r.request_id: histories[r.request_id] for r in inputs.requests}
    if min(len(r.candidate_ids) for r in inputs.requests) < max(CUTOFFS):
        raise ValueError("Every validation request must have at least 20 candidates")
    rows = record(method, {}, inputs, groups, histories, None, evaluate(inputs, cutoffs))
    if method == "rr":
        for task in tasks:
            alpha, epsilon = task["alpha"], task["floor"]
            frozen = fit_rr(inputs, histories, "popularity", (1, *CUTOFFS), alpha, epsilon)
            result = apply_rr(inputs, frozen, CUTOFFS)
            rows.extend(record(method, {"alpha": alpha, "floor": epsilon}, inputs,
                               groups, histories, result["rankings"], result["metrics"]))
            print(f"completed rr alpha={alpha:g} floor={epsilon:g}", flush=True)
    else:
        from .rerank import FairDiverseReranker
        brand_groups = set(inputs.item_groups["brand"].values())
        weights = {g: 1.0 for g in brand_groups}
        for task in tasks:
            strength, k = task["lambda"], task["k"]
            reranker = FairDiverseReranker(model="CPFair" if method == "cpfair" else "PMMF",
                group="brand", fairness_lambda=strength,
                weights=weights if method == "pmmf" else None,
                learning_rate=0.5, gamma=0.2, max_dense_bytes=max_dense_bytes, horizon=horizon)
            rankings = reranker.rerank(inputs, k).rankings
            metrics = evaluate(inputs, [k], rankings)
            rows.extend(record(method, {"lambda": strength}, inputs, groups, None,
                             rankings, {"at_k": {str(k): metrics["at_k"][str(k)]},
                                        "candidate_target_recall": metrics["candidate_target_recall"]}))
            print(f"completed {method} lambda={strength:g} k={k}", flush=True)
    fairness_column = "mgu" if method == "rr" else "exposure_gini"
    manifest = {"method": method, "scope": "full_validation" if limit_requests is None else "smoke_prefix",
                "request_count": len(inputs.requests), "cutoffs": CUTOFFS,
                "fairness_axis": fairness_column,
                "strengths": strengths, "floors": floors if method == "rr" else None,
                "horizon": horizon if method != "rr" else None,
                "max_dense_bytes": max_dense_bytes,
                "task_index": task_index, "task_count": len(plan), "task_plan": plan,
                "evaluated_cutoffs": cutoffs,
                "reranker_source": file_reference(Path(__file__).parent / "rerank/fairdiverse.py"),
                "tuner_source": file_reference(Path(__file__)),
                "implementation_sources": {
                    name: file_reference(Path(__file__).parent / name)
                    for name in ("candidates.py", "evaluation.py", "ifairlrs.py")},
                "selection": "Pareto frontier of validation recall versus lower-is-better fairness axis, separately at each K; no test selection",
                "sources": {"candidates": file_reference(candidates), "split": file_reference(split),
                            "activity": file_reference(activity),
                            **({"histories": file_reference(histories_path)} if method == "rr" else {})}}
    if method != "rr":
        backend = "CPFair" if method == "cpfair" else "PMMF"
        manifest["implementation_sources"]["backend"] = file_reference(
            ROOT / f"fairdiverse/recommendation/rerank_model/{backend}.py")
    manifest["resources"] = {"elapsed_seconds": time.perf_counter() - started,
                             "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2}
    return {**write_outputs(output_dir, rows, manifest), "trials": len(tasks)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=("cpfair", "pmmf", "rr"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--task-count", action="store_true", help="Print grid task count without loading candidates")
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--histories", dest="histories_path", type=Path, default=DEFAULT_HISTORIES)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--activity", type=Path, default=DEFAULT_ACTIVITY)
    parser.add_argument("--strengths", nargs="+", type=float)
    parser.add_argument("--floors", nargs="+", type=float)
    parser.add_argument("--limit-requests", type=int)
    parser.add_argument("--max-dense-bytes", type=int, default=4294967296)
    parser.add_argument("--horizon", type=int, help="Explicit allocation block size; resets CP-Fair/P-MMF state per block")
    args = parser.parse_args()
    if args.task_count:
        print(len(task_plan(args.method, args.strengths, args.floors)))
        return
    if args.output_dir is None:
        parser.error("--output-dir is required unless --task-count is used")
    del args.task_count
    print(json.dumps(run(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
