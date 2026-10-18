"""Test explicit validation-selected CP-Fair/P-MMF settings and optional RR."""

import argparse
import json
import math
from pathlib import Path

from .candidates import file_reference, load
from .ifairlrs import apply_rr, fit_rr
from .rerank import FairDiverseReranker
from .evaluation import evaluate
from .tune_beauty_reranking import checked_inputs, ROOT


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def run(cpfair_lambda, pmmf_lambda, output_dir, dataset="amazon-beauty",
        artifact_dir="additional_baselines/artifacts/beauty_ifairlrs_rr_popularity/original",
        rr_alpha=None, rr_floor=1e-6, horizon=None, max_dense_bytes=4294967296,
        methods=None, selection_provenance=None):
    methods = set(methods if methods is not None else ["cpfair", "pmmf"] + (["rr"] if rr_alpha is not None else []))
    if not methods or methods - {"cpfair", "pmmf", "rr"} or ("rr" in methods and rr_alpha is None):
        raise ValueError("Select supported methods and supply rr_alpha when testing RR")
    strengths = (cpfair_lambda, pmmf_lambda) + (() if rr_alpha is None else (rr_alpha,))
    if any(not math.isfinite(v) or v < 0 for v in strengths):
        raise ValueError("Strengths must be finite and nonnegative")
    if not math.isfinite(rr_floor) or not 0 < rr_floor <= 1:
        raise ValueError("RR floor must be in (0, 1]")
    if horizon is not None and (type(horizon) is not int or horizon <= 0):
        raise ValueError("Horizon must be a positive request count")
    destination, source = Path(output_dir), Path(artifact_dir)
    if destination.exists():
        raise FileExistsError(destination)
    data = ROOT / "recommendation/processed_dataset" / dataset
    validation, _ = checked_inputs(source / "validation_candidates.json",
                                   data / f"{dataset}.valid.ranking", data / "user_activity.csv")
    test = load(source / "test_candidates.json")
    if test.metadata["split"] != {"name": "test", "source": file_reference(data / f"{dataset}.test.ranking")}:
        raise ValueError("Test candidates do not match the requested dataset's test split")
    for key in ("checkpoint", "sid_mapping", "generation", "seed", "candidate_membership", "score_convention"):
        if validation.metadata[key] != test.metadata[key]:
            raise ValueError(f"Validation/test {key} differs")
    if validation.item_groups != test.item_groups:
        raise ValueError("Validation/test group mappings differ")
    destination.mkdir(parents=True)
    write_json(destination / "selection.json", {
        "dataset": dataset, "cpfair_lambda": cpfair_lambda, "pmmf_lambda": pmmf_lambda,
        "rr_alpha": rr_alpha, "rr_floor": rr_floor,
        "horizon": horizon, "max_dense_bytes": max_dense_bytes,
        "selection": "Explicit settings supplied after validation tuning",
        "methods": sorted(methods), "validation_selection": selection_provenance,
        "validation_candidates": file_reference(source / "validation_candidates.json"),
        "test_candidates": file_reference(source / "test_candidates.json"),
    })
    weights = dict.fromkeys(set(test.item_groups["brand"].values()), 1.0)
    for method, model, strength in (("cpfair", "CPFair", cpfair_lambda), ("pmmf", "PMMF", pmmf_lambda)):
        if method not in methods:
            continue
        rankings, at_k = {}, {}
        for k in (5, 10, 20):
            reranker = FairDiverseReranker(model=model, group="brand", fairness_lambda=strength,
                                          weights=weights if method == "pmmf" else None,
                                          learning_rate=0.5, gamma=0.2, max_dense_bytes=max_dense_bytes, horizon=horizon)
            ranked = reranker.rerank(test, k).rankings
            metrics = evaluate(test, [k], ranked)
            rankings[str(k)] = ranked
            at_k[str(k)] = metrics["at_k"][str(k)]
            print(f"completed {method} k={k}", flush=True)
        write_json(destination / f"{method}_test_results.json", {
            "method": method, "fairness_lambda": strength, "rankings_by_k": rankings,
            "metrics": {"at_k": at_k, "candidate_target_recall": metrics["candidate_target_recall"]},
            "control_metrics": evaluate(test, (5, 10, 20)), "input_metadata": test.metadata,
        })
    if "rr" in methods:
        histories = json.loads((source / "validation_histories.json").read_text())
        frozen = fit_rr(validation, histories, "popularity", (1, 5, 10, 20), rr_alpha, rr_floor)
        write_json(destination / "rr_correction.json", frozen)
        test_histories = json.loads((source / "test_histories.json").read_text())
        write_json(destination / "rr_test_results.json", apply_rr(test, frozen, (5, 10, 20), test_histories))
    print(f"Results: {destination}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cpfair_lambda", type=float)
    parser.add_argument("pmmf_lambda", type=float)
    parser.add_argument("output_dir")
    parser.add_argument("--dataset", choices=("amazon-beauty", "amazon-sports", "amazon-toys", "amazon-pet-supplies"), default="amazon-beauty")
    parser.add_argument("--artifact-dir", default="additional_baselines/artifacts/beauty_ifairlrs_rr_popularity/original")
    parser.add_argument("--rr-alpha", type=float)
    parser.add_argument("--rr-floor", type=float, default=1e-6)
    parser.add_argument("--horizon", type=int)
    parser.add_argument("--max-dense-bytes", type=int, default=4294967296)
    parser.add_argument("--methods", nargs="+", choices=("cpfair", "pmmf", "rr"))
    run(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
