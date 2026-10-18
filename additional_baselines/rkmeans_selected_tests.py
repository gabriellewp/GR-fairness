"""Select RKMeans reranker settings from full validation, then test once.

Default policy: minimize the sweep's declared fairness axis at K=5 within a
5% relative Recall@5 loss budget. Record 0%, 1% and 5% validation sensitivity.
Beauty runs RR only; Sports, Toys and Pet run RR, CPFair and P-MMF.
"""
import argparse
import csv
import json
import math
from pathlib import Path

from .candidates import file_reference
from .selected_rerank_test import run
from .tune_beauty_reranking import ROOT

ARTIFACTS = ROOT / "additional_baselines/artifacts"
SOURCES = {
    "amazon-beauty": {"rr": "beauty_rr_validation_tuning_351960"},
    "amazon-sports": {"cpfair": "sports_cpfair_validation_tuning_array_355048/merged",
                      "pmmf": "sports_pmmf_validation_tuning_array_355050/merged",
                      "rr": "sports_rr_validation_tuning_array_355052/merged"},
    "amazon-toys": {"cpfair": "toys_cpfair_validation_tuning_array_355055/merged",
                    "pmmf": "toys_pmmf_validation_tuning_array_355057/merged",
                    "rr": "toys_rr_validation_tuning_array_355059/merged"},
    "amazon-pet-supplies": {"cpfair": "pet-supplies_cpfair_validation_tuning_array_355062/merged",
                            "pmmf": "pet-supplies_pmmf_validation_tuning_array_355064/merged",
                            "rr": "pet-supplies_rr_validation_tuning_array_355066/merged"},
}


def select(rows, axis, budget):
    rows = [r for r in rows if int(r["k"]) == 5]
    controls = [r for r in rows if r["parameters"] == "{}"]
    if len(controls) != 1:
        raise ValueError("Require exactly one full-validation control at K=5")
    control = controls[0]
    threshold = float(control["recall"]) * (1-budget)
    eligible = [r for r in rows if r["parameters"] != "{}" and
                float(r["recall"]) >= threshold-1e-12 and math.isfinite(float(r[axis]))]
    if not eligible:
        raise ValueError("No tested setting meets the relevance budget")
    best = min(eligible, key=lambda r: (float(r[axis]), -float(r["recall"]), r["parameters"]))
    return dict(parameters=json.loads(best["parameters"]), validation_row=best,
                control_row=control, max_relative_recall_loss=budget, fairness_axis=axis)


def prepare(dataset, budget, candidate_dir=None, sources=None):
    if not math.isfinite(budget) or not 0 <= budget < 1:
        raise ValueError("Relative loss budget must be in [0, 1)")
    candidate_dir = Path(candidate_dir) if candidate_dir is not None else ARTIFACTS / ("beauty_ifairlrs_rr_popularity/original" if dataset == "amazon-beauty"
                                else dataset.removeprefix("amazon-") + "_rkmeans_rerank_candidates")
    selected = {}
    for method, directory in (SOURCES[dataset] if sources is None else sources).items():
        directory = Path(directory)
        # CLI callers may pass repository-relative paths; built-in sources
        # remain relative to ARTIFACTS. Absolute paths work in either case.
        base = ROOT if directory.parts[:2] == ("additional_baselines", "artifacts") else ARTIFACTS
        source = base / directory
        manifest_path, rows_path = source / "manifest.json", source / "validation_tradeoff.csv"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Complete merged validation sweep required: {manifest_path}")
        manifest = json.loads(manifest_path.read_text())
        if manifest["scope"] != "full_validation" or manifest["method"] != method:
            raise ValueError("Selection requires the correct full-validation method")
        for key, path in [("candidates", candidate_dir / "validation_candidates.json"),
                          ("split", ROOT / "recommendation/processed_dataset" / dataset / (dataset + ".valid.ranking"))]:
            if manifest["sources"][key] != file_reference(path):
                raise ValueError(f"Validation {key} differs from the saved sweep")
        if method == "rr" and manifest["sources"]["histories"] != file_reference(candidate_dir / "validation_histories.json"):
            raise ValueError("RR validation histories differ from the saved sweep")
        with rows_path.open() as stream:
            rows = list(csv.DictReader(stream))
        selected[method] = select(rows, manifest["fairness_axis"], budget)
        selected[method].update(manifest=file_reference(manifest_path), rows=file_reference(rows_path),
            validation_sensitivity={str(b): select(rows, manifest["fairness_axis"], b) for b in (0, .01, .05)})
    return candidate_dir, selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=list(SOURCES), required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-relative-recall-loss", type=float, default=.05)
    parser.add_argument("--selection-only", action="store_true", help="Inspect validation settings without test evaluation")
    args = parser.parse_args()
    candidate_dir, selections = prepare(args.dataset, args.max_relative_recall_loss)
    print(json.dumps(selections, indent=2), flush=True)
    if args.selection_only:
        return
    rr = selections["rr"]["parameters"]
    run(selections.get("cpfair", {}).get("parameters", {}).get("lambda", 0),
        selections.get("pmmf", {}).get("parameters", {}).get("lambda", 0), args.output_dir,
        dataset=args.dataset, artifact_dir=candidate_dir, rr_alpha=rr["alpha"], rr_floor=rr["floor"],
        methods=list(selections), selection_provenance=dict(selection_cutoff=5,
            rule="minimum declared fairness axis within relative Recall@5 budget; ties prefer higher Recall@5 then parameter order",
            selections=selections))


if __name__ == "__main__":
    main()
