"""Summarize saved validation-selected RKMeans reranking tests without rerunning."""
import csv
import hashlib
import json
import math
from pathlib import Path

from scripts.paper.archive_paths import source_root, output_root
ROOT = source_root()
OUTPUT = output_root()
OUT = OUTPUT
ARTIFACTS = ROOT / "additional_baselines/artifacts"
FOLDERS = {
    "Beauty": "beauty_rkmeans_validation_selected_test_356027",
    "Sports": "sports_rkmeans_validation_selected_test_356029",
    "Toys": "toys_rkmeans_validation_selected_test_356028",
}
EXPECTED_REQUESTS = {"Beauty": 22363, "Sports": 35598, "Toys": 19412}
LABELS = {"control": "Untreated", "rr": "RR", "cpfair": "CPFair", "pmmf": "P-MMF"}
SOURCES = {}


def read(path):
    raw = path.read_bytes()
    SOURCES[str(path.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


def close(a, b):
    assert math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-12), (a, b)


def check_gini(metrics):
    values = sorted(metrics["groups"]["brand"]["exposure_counts"].values())
    size, total = len(values), sum(values)
    calculated = sum((2*i-size-1)*v for i, v in enumerate(values, 1))/(size*total)
    close(calculated, metrics["groups"]["brand"]["gini"])
    entropy = -sum((v/total)*math.log(v/total) for v in values if v)
    close(entropy, metrics["groups"]["brand"]["entropy"])


def check_calibration(fairness):
    gaps = [fairness["GP"][g]-fairness["GH"][g] for g in fairness["GH"]]
    close(sum(map(abs, gaps))/len(gaps), fairness["MGU"])
    close(max(gaps)-min(gaps), fairness["DGU"])


def write_csv(name, rows):
    with (OUT / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


rows, selections = [], []
for dataset, folder in FOLDERS.items():
    saved_selection = read(ARTIFACTS / folder / "selection.json")
    rr = read(ARTIFACTS / folder / "rr_test_results.json")
    control = rr["control_metrics"]
    assert control["request_count"] == EXPECTED_REQUESTS[dataset]
    for method in ("control", "rr", "cpfair", "pmmf"):
        if method == "control":
            report, source = rr, ARTIFACTS / folder / "rr_test_results.json"
        elif method == "rr":
            report, source = rr, ARTIFACTS / folder / "rr_test_results.json"
        else:
            source_folder = "beauty_original_selected_rerank_352496" if dataset == "Beauty" else folder
            source = ARTIFACTS / source_folder / f"{method}_test_results.json"
            report = read(source)
        assert report["control_metrics"]["request_count"] == EXPECTED_REQUESTS[dataset]
        close(report["control_metrics"]["candidate_target_recall"], control["candidate_target_recall"])
        calibration = None
        if method in ("cpfair", "pmmf"):
            calibration = read(ARTIFACTS / "rkmeans_calibration_357290" / f"{dataset.lower()}_{method}_calibration.json")
            assert calibration["saved_rankings"]["sha256"] == SOURCES[str(source.relative_to(ROOT))]
            assert calibration["request_count"] == EXPECTED_REQUESTS[dataset]
            for cutoff in ("5", "10", "20"):
                for metric in ("MGU", "DGU"):
                    close(calibration["baseline"][cutoff][metric], rr["control_ifairlrs_fairness"][cutoff][metric])
        for k in ("5", "10", "20"):
            matched = report["control_metrics"]["at_k"][k]
            baseline = control["at_k"][k]
            for key in ("recall", "ndcg", "fill_rate"):
                close(matched[key], baseline[key])
            assert matched["groups"]["brand"]["exposure_counts"] == baseline["groups"]["brand"]["exposure_counts"]
            metrics = (baseline if method == "control" else
                       report["runs"][k]["metrics"]["at_k"][k] if "runs" in report else
                       report["metrics"]["at_k"][k])
            check_gini(metrics)
            assert metrics["fill_rate"] == 1 and metrics["short_list_rate"] == 0
            if method in ("control", "rr"):
                fairness = rr["control_ifairlrs_fairness" if method == "control" else "ifairlrs_fairness"][k]
                check_calibration(fairness)
            else:
                fairness = calibration["reranked"][k]
                check_calibration(fairness)
            axis = "MGU" if method == "rr" else "brand_exposure_gini" if method != "control" else ""
            loss = fairness["MGU"] if method == "rr" else metrics["groups"]["brand"]["gini"]
            control_loss = rr["control_ifairlrs_fairness"][k]["MGU"] if method == "rr" else baseline["groups"]["brand"]["gini"]
            parameters = ({"alpha": rr["frozen_correction"]["alpha"], "floor": rr["frozen_correction"]["epsilon"]}
                          if method == "rr" else {"lambda": 1 if method == "cpfair" else 30} if method != "control" else {})
            if method in ("cpfair", "pmmf"):
                actual_lambda = (report["runs"][k]["config"]["backend_config"]["lambda"]
                                 if "runs" in report else report["fairness_lambda"])
                assert actual_lambda == parameters["lambda"]
            rows.append(dict(dataset=dataset, method=LABELS[method], k=int(k), seed=42,
                requests=control["request_count"], parameters=json.dumps(parameters, sort_keys=True),
                candidate_target_coverage=control["candidate_target_recall"],
                recall=metrics["recall"], ndcg=metrics["ndcg"],
                control_recall=baseline["recall"], control_ndcg=baseline["ndcg"],
                delta_recall=metrics["recall"]-baseline["recall"],
                relative_delta_recall_percent=100*(metrics["recall"]/baseline["recall"]-1),
                relative_delta_ndcg_percent=100*(metrics["ndcg"]/baseline["ndcg"]-1),
                brand_exposure_gini=metrics["groups"]["brand"]["gini"],
                control_brand_exposure_gini=baseline["groups"]["brand"]["gini"],
                brand_exposure_entropy=metrics["groups"]["brand"]["entropy"],
                control_brand_exposure_entropy=baseline["groups"]["brand"]["entropy"],
                relative_delta_brand_entropy_percent=100*(metrics["groups"]["brand"]["entropy"]/baseline["groups"]["brand"]["entropy"]-1),
                mgu=fairness.get("MGU", ""), dgu=fairness.get("DGU", ""),
                control_mgu=rr["control_ifairlrs_fairness"][k]["MGU"],
                control_dgu=rr["control_ifairlrs_fairness"][k]["DGU"],
                mgu_reduction_percent=100*(1-fairness["MGU"]/rr["control_ifairlrs_fairness"][k]["MGU"]),
                dgu_reduction_percent=100*(1-fairness["DGU"]/rr["control_ifairlrs_fairness"][k]["DGU"]),
                selection_fairness_axis=axis,
                fairness_loss_reduction_percent=100*(1-loss/control_loss) if method != "control" else 0,
                source=str(source.relative_to(ROOT))))
        if method == "control":
            continue
        prior = (saved_selection.get("validation_selection") or {}).get("selections", {}).get(method)
        if prior:
            selected = prior["parameters"]
            val, val_control = float(prior["validation_row"]["recall"]), float(prior["control_row"]["recall"])
            axis = prior["fairness_axis"]
            record = "saved automatic validation selection; 5% relative Recall@5 loss budget"
        else:
            assert dataset == "Beauty" and method in ("cpfair", "pmmf")
            suffix = "351961" if method == "cpfair" else "351975"
            sweep = ARTIFACTS / f"beauty_{method}_validation_tuning_{suffix}"
            manifest = read(sweep / "manifest.json")
            assert manifest["scope"] == "full_validation"
            with (sweep / "validation_tradeoff.csv").open() as stream:
                vals = [r for r in csv.DictReader(stream) if int(r["k"]) == 5]
            base = next(r for r in vals if r["parameters"] == "{}")
            eligible = [r for r in vals if r["parameters"] != "{}" and float(r["recall"]) >= .95*float(base["recall"])]
            best = min(eligible, key=lambda r: (float(r["exposure_gini"]), -float(r["recall"]), r["parameters"]))
            selected, val, val_control, axis = json.loads(best["parameters"]), float(best["recall"]), float(base["recall"]), "exposure_gini"
            record = "earlier explicit validation-selected setting; independently confirmed to match the 5% rule, not a new test selection"
        assert selected == parameters
        selections.append(dict(dataset=dataset, method=LABELS[method], parameters=json.dumps(selected, sort_keys=True),
            selection_k=5, selection_fairness_axis=axis, validation_recall=val, validation_control_recall=val_control,
            validation_relative_delta_recall_percent=100*(val/val_control-1), selection_record=record))

assert len(rows) == 36 and len(selections) == 9
# The canonical comparison includes the completed Pet Supplies tests. Use its
# matched saved-candidate control, rather than the fresh-generation baseline.
comparison_path = ROOT / "docs/rq2_comparison_draft_20261011/comparison_results.csv"
comparison_bytes = comparison_path.read_bytes()
SOURCES[str(comparison_path.relative_to(ROOT))] = hashlib.sha256(comparison_bytes).hexdigest()
pet_methods = {"GRID: saved reranking reference": "Untreated", "GRID + RR": "RR",
               "GRID + CPFair": "CPFair", "GRID + P-MMF": "P-MMF"}
pet_rows = []
for row in csv.DictReader(comparison_bytes.decode().splitlines()):
    if (row["dataset"] == "amazon-pet-supplies" and row["tokenizer"] == "rkmeans"
            and row["method"] in pet_methods):
        assert row["status"] == "verified"
        pet_rows.append(dict(dataset="Pet Supplies", method=pet_methods[row["method"]],
                             k=int(row["k"]), recall=float(row["recall"]),
                             ndcg=float(row["ndcg"]), source=row["source"]))
assert len(pet_rows) == 12
for method in LABELS.values():
    assert sorted(r["k"] for r in pet_rows if r["method"] == method) == [5, 10, 20]
pet_selection_path = ARTIFACTS / "pet-supplies_rkmeans_validation_selected_test_357020/selection.json"
pet_selection = read(pet_selection_path)
assert pet_selection["dataset"] == "amazon-pet-supplies"
assert set(pet_selection["validation_selection"]["selections"]) == {"rr", "cpfair", "pmmf"}
relevance_rows = rows + pet_rows
write_csv("relevance_results.csv", [
    {key: r[key] for key in ("dataset", "method", "k", "recall", "ndcg", "source")}
    for r in relevance_rows])
write_csv("test_results.csv", rows)
write_csv("validation_selection.csv", selections)
(OUT / "provenance.json").write_text(json.dumps(dict(
    scope="Four-dataset relevance table; detailed three-dataset tradeoff audit; original RKMeans checkpoints, seed 42, matched saved candidate exports",
    pet_relevance_source=str(comparison_path.relative_to(ROOT)),
    sources=SOURCES, validation_selection_note="Beauty CPFair/P-MMF tests predate the automatic selector; selected values independently agree with the same 5% validation rule",
    verification=["all matched controls agree", "full request counts", "all K=5/10/20 lists full", "Gini and natural-log entropy recomputed from exposure counts", "RR MGU/DGU recomputed from shares", "selected parameters agree with saved test settings"],
    limitations=["one seed; no training-seed uncertainty", "RR and brand rerankers optimize different fairness axes", "5% validation loss allowance is not a test guarantee", "Beauty historical export has three-request candidate discrepancy relative to fresh RQ1 generation", "CPFair and P-MMF retain full-artifact allocation state and saved request order", "P-MMF is the FairDiverse variant with all-one unnormalized provider weights"]), indent=2)+"\n")

tex = [r"% Requires \usepackage{booktabs}. Relevance values come from relevance_results.csv.",
       r"\begin{table*}[t]", r"\centering\small", r"\setlength{\tabcolsep}{4pt}",
       r"\caption{Validation-selected RKMeans reranking. Each Recall/NDCG cell lists K=5, 10, 20. Untreated controls use the same saved candidates. Seed 42.}",
       r"\label{tab:rkmeans-rerank-relevance}", r"\begin{tabular}{llcc}", r"\toprule",
       r"Dataset & Method & Recall (5 / 10 / 20) $\uparrow$ & NDCG (5 / 10 / 20) $\uparrow$ \\", r"\midrule"]
for dataset in [*FOLDERS, "Pet Supplies"]:
    for method in LABELS.values():
        group = sorted((r for r in relevance_rows if r["dataset"] == dataset and r["method"] == method),
                       key=lambda r: r["k"])
        label = "w/o reranking" if method == "Untreated" else method
        tex.append(f"{dataset} & {label} & "+" / ".join(f"{r['recall']:.4f}" for r in group)+" & "+" / ".join(f"{r['ndcg']:.4f}" for r in group)+r" \\")
    tex.append(r"\midrule")
tex[-1] = r"\bottomrule"
tex.extend([r"\end{tabular}", r"\end{table*}", "", r"% Requires booktabs and graphicx.",
    r"\begin{table*}[t]", r"\centering\small", r"\setlength{\tabcolsep}{4pt}",
    r"\caption{Relative changes against GRID RKMeans recommendations without reranking. Each cell lists $K=5/10/20$. MGU and DGU measure popularity calibration; Gini and entropy measure brand exposure. Positive loss reductions and entropy increases indicate improvements in their respective measures; negative values indicate deterioration. All percentages are relative changes computed from unrounded values. Seed 42.}",
    r"\label{tab:rkmeans-rerank-tradeoff}", r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{llccccc}", r"\toprule",
    r"Dataset & Method & Recall change (\%) & MGU reduction (\%) & DGU reduction (\%) & Brand Gini reduction (\%) & Brand entropy increase (\%) \\", r"\midrule"])
for dataset in FOLDERS:
    for method in ("RR", "CPFair", "P-MMF"):
        group = [r for r in rows if r["dataset"] == dataset and r["method"] == method]
        recall = " / ".join(f"{r['relative_delta_recall_percent']:+.2f}" for r in group)
        mgu = " / ".join(f"{r['mgu_reduction_percent']:.2f}" for r in group)
        dgu = " / ".join(f"{r['dgu_reduction_percent']:.2f}" for r in group)
        gini = " / ".join(f"{100*(1-r['brand_exposure_gini']/r['control_brand_exposure_gini']):.2f}" for r in group)
        entropy = " / ".join(f"{r['relative_delta_brand_entropy_percent']:+.2f}" for r in group)
        tex.append(" & ".join((dataset, method, recall, mgu, dgu, gini, entropy))+r" \\")
    tex.append(r"\midrule")
tex[-1] = r"\bottomrule"
tex.extend([r"\end{tabular}%", "}", r"\end{table*}"])
(OUT / "tables.tex").write_text("\n".join(tex)+"\n")
for dataset in FOLDERS:
    for method in LABELS.values():
        group = [r for r in rows if r["dataset"] == dataset and r["method"] == method]
        print(dataset, method, "R", "/".join(f"{r['recall']:.4f}" for r in group), "N", "/".join(f"{r['ndcg']:.4f}" for r in group))
print("Saved 48 relevance rows, 36 detailed tradeoff rows, nine detailed selection records, source hashes, and LaTeX tables.")
