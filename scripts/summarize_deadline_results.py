"""CPU-only readout of completed pilots; never launch or select another run."""

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]


def reference(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def read(path):
    return json.loads(Path(path).read_text())


def candidate_header(path):
    """Read mappings preceding requests, without loading millions of candidates."""
    prefix = ""
    with Path(path).open() as stream:
        while len(prefix) < 32 * 1024 * 1024:
            chunk = stream.read(65536)
            if not chunk:
                break
            prefix += chunk
            match = re.search(r'\n  "requests":\s*\[', prefix)
            if match:
                text = prefix[:match.start()].rstrip().rstrip(",")
                result = json.loads(text + "\n}")
                return result, dict(path=str(Path(path).resolve()),
                                    metadata_and_mapping_prefix_sha256=hashlib.sha256(text.encode()).hexdigest(),
                                    scope="metadata and group mappings only")
    raise ValueError(f"Candidate header unavailable: {path}")


def values(metrics, history):
    at5 = metrics["at_k"]["5"]
    counts = at5["groups"]["popularity"]["exposure_counts"]
    assert set(counts) == set(history)
    assert math.isclose(sum(history.values()), 1, abs_tol=1e-10)
    for group in at5["groups"].values():
        # Independent Lorenz-area calculation checks saved Gini values.
        ordered = sorted(group["exposure_counts"].values())
        total = sum(ordered)
        cumulative, area = 0, 0
        for count in ordered:
            area += 2 * cumulative + count
            cumulative += count
        independently_computed = 1 - area / (len(ordered) * total)
        assert math.isclose(independently_computed, group["gini"], abs_tol=1e-12)
        assert total == metrics["request_count"] * 5  # All these lists fill K=5.
    total = sum(counts.values())
    return dict(recall_at_5=at5["recall"],
                popularity_mgu_at_5=math.fsum(abs(counts[g] / total - history[g]) for g in counts) / len(counts),
                brand_gini_at_5=at5["groups"]["brand"]["gini"],
                most_popular_group_share_at_5=counts["4"] / total)


def summarize(destination):
    source = ROOT / "additional_baselines/artifacts/deadline_fairness_pilot_20261014"
    comparisons, outcomes, sources, cases = [], [], {}, {}
    for name in ("manifest_retry1.json", "manifest_retry3.json"):
        manifest_path = source / name
        sources[str(manifest_path)] = reference(manifest_path)
        for case in read(manifest_path)["cases"]:
            if not (Path(case["options"]["output_dir"]) / "complete.json").exists():
                continue
            key = case["dataset"], case["tokenizer"]
            if key in cases:
                raise ValueError(f"Multiple completed attempts for {key}")
            cases[key] = case
    completed_trials = 0
    ordinary_controls, history_cache = [], {}
    for case in cases.values():
        folder = Path(case["options"]["output_dir"])
        completed_path = folder / "complete.json"
        if not completed_path.exists():
            continue
        completed = read(completed_path)
        selected = read(completed["sample_selection"]["path"])
        assert reference(completed["sample_selection"]["path"]) == completed["sample_selection"]
        assert completed["no_test_based_retuning"] and not selected["selection_uses_test"]
        control = read(case["control"]["path"])
        assert reference(case["control"]["path"]) == case["control"]
        baseline_path = Path(case["control"]["path"]).with_name("test_results.json")
        if case["dataset"] == "amazon-pet-supplies" and case["tokenizer"] == "rqvae":
            assert control["selected_update"] == 0
            baseline_path = ROOT / "additional_baselines/artifacts/pet-supplies_rqvae_rerank_candidates_20261011_gpu4/source_test_results.json"
        baseline = read(baseline_path)
        assert baseline["input_metadata"]["checkpoint"] == control["selected_checkpoint"]
        groups = read(case["groups"]["path"])
        header, header_ref = candidate_header(baseline_path.with_name("test_candidates.json"))
        for key in ("checkpoint", "sid_mapping", "split"):
            assert header["metadata"][key] == baseline["input_metadata"][key]
        assert header["item_groups"] == groups
        test_source = baseline["input_metadata"]["split"]["source"]
        assert test_source == case["prepared_inputs"]["test"]
        generation = baseline["input_metadata"]["generation"]
        history_length = generation["sequence_length"] // generation["num_hierarchies"]
        cache_key = (case["dataset"], test_source["sha256"], history_length, case["groups"]["sha256"])
        if cache_key not in history_cache:
            assert reference(test_source["path"]) == test_source
            mapping = groups["popularity"]
            counts, requests = Counter(), 0
            with Path(test_source["path"]).open(newline="") as stream:
                for row in csv.DictReader(stream, delimiter="\t"):
                    history = json.loads(row["history_behaviors"])
                    assert history and all(type(i) is int and str(i) in mapping for i in history)
                    assert str(int(row["target_item"])) in mapping
                    counts.update(mapping[str(i)] for i in history[-history_length:])
                    requests += 1
            history_cache[cache_key] = ({g: counts[g] / sum(counts.values())
                                         for g in sorted(set(mapping.values()))}, requests)
        history, requests = history_cache[cache_key]
        assert baseline["metrics"]["request_count"] == requests
        ordinary_controls.append(dict(dataset=case["dataset"], tokenizer=case["tokenizer"],
            selected_update=control["selected_update"], source=reference(baseline_path),
            history_shares=history, test_request_count=requests,
            history_source=test_source, history_length=history_length,
            group_source=case["groups"], group_mapping_evidence=header_ref))
        sources[str(baseline_path)] = reference(baseline_path)
        completed_trials += len(list(folder.glob("completed_*.json")))
        sources[str(completed_path)] = reference(completed_path)
        for mode, result in completed["confirmations"].items():
            outcome = dict(dataset=case["dataset"], tokenizer=case["tokenizer"], method=mode,
                           status=result["status"], completed_case=reference(completed_path),
                           sampled_choice=selected["winners"][mode])
            if "objectives" in result:
                observed, ordinary = result["objectives"], result["full_validation_policy"]["control"]
                outcome["validation_relative_recall_change_percent"] = 100 * (observed["recall"] / ordinary["recall"] - 1)
                outcome["validation_brand_gini_change"] = observed["brand"] - ordinary["brand"]
                outcome["validation_popularity_mgu_change"] = observed["popularity"] - ordinary["popularity"]
            outcomes.append(outcome)
            if "test_result" not in result:
                continue
            test_path = Path(result["test_result"]["path"])
            assert reference(test_path) == result["test_result"]
            assert reference(result["checkpoint"]["path"]) == result["checkpoint"]
            test = read(test_path)
            assert baseline["input_metadata"]["checkpoint"] == control["selected_checkpoint"]
            assert test["input_metadata"]["checkpoint"] == result["checkpoint"]
            for key in ("sid_mapping", "split", "seed", "candidate_membership", "score_convention", "short_list_policy"):
                assert baseline["input_metadata"][key] == test["input_metadata"][key], key
            assert test["scope"] == "full_split"
            assert baseline["metrics"]["request_count"] == test["metrics"]["request_count"]
            for key in ("beam_width", "sequence_length", "num_hierarchies"):
                assert baseline["input_metadata"]["generation"][key] == test["input_metadata"]["generation"][key]
            assert baseline["input_metadata"]["generation"].get("use_prefix_validation", False) is False
            assert re.search(r"^use_prefix_validation:\s*false\s*$", Path(case["export_config"]["path"]).read_text(), re.M)
            assert test["history_shares"] == history
            ordinary = values(baseline["metrics"], test["history_shares"])
            candidate = values(test["metrics"], test["history_shares"])
            changes = {k: 100 * (candidate[k] / ordinary[k] - 1) for k in ordinary}
            fair = all(candidate[k] <= ordinary[k] + 1e-12 for k in ("popularity_mgu_at_5", "brand_gini_at_5"))
            recall = candidate["recall_at_5"] >= .98 * ordinary["recall_at_5"]
            comparisons.append(dict(dataset=case["dataset"], tokenizer=case["tokenizer"], method=mode,
                settings=selected["winners"][mode]["trial"], selected_update=selected["winners"][mode]["selected_update"],
                test_request_count=test["metrics"]["request_count"], control=ordinary, candidate=candidate,
                relative_changes_percent=changes, both_fairness_measures_no_worse_on_test=fair,
                test_recall_loss_within_2_percent=recall,
                test_checks_are_descriptive_only=True, validation_confirmed=True,
                baseline_source=reference(baseline_path), candidate_source=result["test_result"],
                baseline_group_mapping_evidence=header_ref,
                baseline_generation_batch_size=baseline["input_metadata"].get("generation_adapter", {}).get("evaluation_batch_size", 1),
                candidate_generation_batch_size=test["generation_batch_size"]))
            sources[str(baseline_path)] = reference(baseline_path)
            sources[str(test_path)] = result["test_result"]
    assert len(cases) == 12 and len(outcomes) == 24 and len(comparisons) == 10
    assert completed_trials == 48
    report = dict(readiness="share_with_caveats", completed_cases=len(cases), planned_cases=12,
        completed_screening_trials=completed_trials, planned_screening_trials=48,
        full_test_candidates=len(comparisons),
        descriptive_joint_test_passes=sum(r["both_fairness_measures_no_worse_on_test"] and
                                         r["test_recall_loss_within_2_percent"] for r in comparisons),
        selection_uses_test=False,
        caveats=["Single seed (42), two settings per method; no hyperparameter optimum or statistical significance established.",
                 "The 2% Recall guardrail applies to validation. Test checks here are descriptive; test results never reselect models.",
                 "Historical test controls used single-request generation; candidates used batches of 16. Top-20 parity was checked on 64 validation users per case, not the complete test splits. Very small differences need cautious interpretation.",
                 "All twelve planned cases completed screening and confirmation; only full-validation-confirmed candidates receive a full test. Ordinary is retained for rejected candidates.",
                 "This exploratory follow-up was conducted after earlier test results were available; its own saved selector does not use test metrics.",
                 "Pet Supplies RQ-VAE ordinary control selected update zero, so its matched test reference is the original checkpoint export."],
        ordinary_controls=ordinary_controls, method_outcomes=outcomes,
        test_comparisons=comparisons, sources=list(sources.values()))
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "completed_results.json").write_text(json.dumps(report, indent=2) + "\n")
    with (destination / "test_comparison.csv").open("w", newline="") as stream:
        columns = ["dataset", "tokenizer", "method", "setting", "selected_update", "test_request_count",
                   "control_recall_at_5", "candidate_recall_at_5", "recall_relative_change_percent",
                   "control_popularity_mgu_at_5", "candidate_popularity_mgu_at_5", "popularity_mgu_relative_change_percent",
                   "control_brand_gini_at_5", "candidate_brand_gini_at_5", "brand_gini_relative_change_percent",
                   "test_recall_loss_within_2_percent"]
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in comparisons:
            writer.writerow(dict(dataset=row["dataset"], tokenizer=row["tokenizer"], method=row["method"],
                setting=row["settings"]["name"], selected_update=row["selected_update"], test_request_count=row["test_request_count"],
                control_recall_at_5=row["control"]["recall_at_5"], candidate_recall_at_5=row["candidate"]["recall_at_5"],
                recall_relative_change_percent=row["relative_changes_percent"]["recall_at_5"],
                control_popularity_mgu_at_5=row["control"]["popularity_mgu_at_5"], candidate_popularity_mgu_at_5=row["candidate"]["popularity_mgu_at_5"],
                popularity_mgu_relative_change_percent=row["relative_changes_percent"]["popularity_mgu_at_5"],
                control_brand_gini_at_5=row["control"]["brand_gini_at_5"], candidate_brand_gini_at_5=row["candidate"]["brand_gini_at_5"],
                brand_gini_relative_change_percent=row["relative_changes_percent"]["brand_gini_at_5"],
                test_recall_loss_within_2_percent=row["test_recall_loss_within_2_percent"]))
    for row in comparisons:
        print(row["dataset"], row["tokenizer"], row["method"], json.dumps(row["relative_changes_percent"]))
    return destination / "completed_results.json"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=ROOT / "additional_baselines/artifacts/deadline_fairness_pilot_20261014/completed_readout_20261015")
    print(summarize(parser.parse_args().output_dir))
