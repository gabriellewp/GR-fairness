"""Evaluate saved rerankings on the same popularity calibration reference as RR."""

import argparse
import csv
import json
import math
import os
from pathlib import Path

from .candidates import file_reference, load
from .ifairlrs import fairness, mapping_digest

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "additional_baselines/artifacts"
FOLDERS = {
    "Beauty": "beauty_rkmeans_validation_selected_test_356027",
    "Sports": "sports_rkmeans_validation_selected_test_356029",
    "Toys": "toys_rkmeans_validation_selected_test_356028",
}
EXPECTED_REQUESTS = {"Beauty": 22363, "Sports": 35598, "Toys": 19412}
CUTOFFS = (5, 10, 20)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(a, b):
    require(math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-12), f"Metric mismatch: {a} != {b}")


def check_stats(actual, saved):
    for k, values in actual.items():
        for name in ("MGU", "DGU"):
            close(values[name], saved[k][name])
        for name in ("GH", "GP", "GU"):
            require(values[name].keys() == saved[k][name].keys(), "Group IDs differ")
            for group, value in values[name].items():
                close(value, saved[k][name][group])


def check_rankings(inputs, rankings, k):
    require(set(rankings) == {r.request_id for r in inputs.requests}, "Ranking request IDs differ")
    for request in inputs.requests:
        items = rankings[request.request_id]
        require(len(items) >= k, f"Short ranking for {request.request_id} at K={k}")
        require(len(items) == len(set(items)), "Duplicate ranked items")
        require(set(items).issubset(request.candidate_ids), "Ranking contains a noncandidate item")


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def run(destination):
    destination.mkdir(parents=True, exist_ok=False)
    rows, references = [], []
    for dataset, folder in FOLDERS.items():
        selected = ARTIFACTS / folder
        selection_path = selected / "selection.json"
        selection = read(selection_path)
        candidate_path = Path(selection["test_candidates"]["path"])
        candidate_ref = file_reference(candidate_path)
        require(candidate_ref == selection["test_candidates"], "Candidate fingerprint changed")
        history_path = candidate_path.parent / "test_histories.json"
        histories = read(history_path)
        inputs = load(candidate_path)
        require(len(inputs.requests) == EXPECTED_REQUESTS[dataset], "Not the expected full test split")
        require(inputs.metadata["seed"] == 42 and inputs.metadata["split"]["name"] == "test", "Unexpected seed/split")
        require(file_reference(inputs.metadata["split"]["source"]["path"]) == inputs.metadata["split"]["source"], "Test split fingerprint changed")
        rr_path = selected / "rr_test_results.json"
        rr = read(rr_path)
        require(mapping_digest(inputs.item_groups["popularity"]) == rr["frozen_correction"]["group_mapping_sha256"], "Popularity mapping differs from RR")
        baseline = fairness(inputs, histories, "popularity", CUTOFFS)
        check_stats(baseline, rr["control_ifairlrs_fairness"])
        for k in CUTOFFS:
            check_rankings(inputs, rr["rankings"], k)
        check_stats(fairness(inputs, histories, "popularity", CUTOFFS, rr["rankings"]), rr["ifairlrs_fairness"])
        references.extend(file_reference(p) for p in (selection_path, history_path, rr_path))
        references.append(candidate_ref)
        del rr
        for method in ("cpfair", "pmmf"):
            source_folder = "beauty_original_selected_rerank_352496" if dataset == "Beauty" else folder
            report_path = ARTIFACTS / source_folder / f"{method}_test_results.json"
            report = read(report_path)
            require(report["input_metadata"] == inputs.metadata, "Saved reranking metadata differs from candidates")
            require(report["control_metrics"]["request_count"] == len(inputs.requests), "Control request count differs")
            if "input_artifact" in report:
                require(report["input_artifact"] == candidate_ref, "Saved reranking candidate fingerprint differs")
            results = {}
            for k in CUTOFFS:
                key = str(k)
                rankings = report["runs"][key]["rankings"] if "runs" in report else report["rankings_by_k"][key]
                check_rankings(inputs, rankings, k)
                values = fairness(inputs, histories, "popularity", [k], rankings)[key]
                # Check that these are the same top-K lists used for the saved relevance metrics.
                hits = sum(any(i in r.targets for i in rankings[r.request_id][:k]) for r in inputs.requests)
                saved_metrics = report["runs"][key]["metrics"]["at_k"][key] if "runs" in report else report["metrics"]["at_k"][key]
                close(hits/len(inputs.requests), saved_metrics["recall"])
                results[key] = values
                rows.append(dict(dataset=dataset, method="CPFair" if method == "cpfair" else "P-MMF", k=k,
                    seed=42, requests=len(inputs.requests), mgu=values["MGU"], dgu=values["DGU"],
                    control_mgu=baseline[key]["MGU"], control_dgu=baseline[key]["DGU"],
                    mgu_reduction_percent=100*(1-values["MGU"]/baseline[key]["MGU"]),
                    dgu_reduction_percent=100*(1-values["DGU"]/baseline[key]["DGU"])))
            write(destination / f"{dataset.lower()}_{method}_calibration.json", dict(
                dataset=dataset, method=method, request_count=len(inputs.requests), seed=42,
                group="popularity", group_mapping_sha256=mapping_digest(inputs.item_groups["popularity"]),
                baseline=baseline, reranked=results, candidates=candidate_ref,
                histories=file_reference(history_path), saved_rankings=file_reference(report_path)))
            references.append(file_reference(report_path))
            print(f"Completed {dataset} {method}: K=5/10/20", flush=True)
            del report, rankings
        del inputs, histories
    require(len(rows) == 18, "Missing calibration results")
    with (destination / "calibration_results.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write(destination / "manifest.json", dict(status="complete", slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        sources=references, code=[file_reference(Path(__file__)), file_reference(ROOT / "additional_baselines/ifairlrs.py")],
        definition="GH: pooled test request histories; GP: unweighted top-K popularity-group exposure; GU=GP-GH; MGU=mean(abs(GU)); DGU=max(GU)-min(GU)",
        verification=["candidate/test-split fingerprints", "matching saved rerank metadata", "full test request counts",
            "exact request IDs and eligible rankings", "existing RR baseline and reranked calibration reproduced",
            "saved reranked Recall@K reproduced"],
        note="Evaluation of saved rankings only; no training, reranking, or hyperparameter selection"))
    print(f"Saved 18 calibration rows to {destination}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    run(parser.parse_args().output_dir)
