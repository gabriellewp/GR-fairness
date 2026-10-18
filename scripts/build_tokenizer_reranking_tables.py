"""Build separate validation-selected RVQ/RQ-VAE test tables and exact deltas.

Metrics come from the verified comparison CSV. Selection choices are checked
against their saved full-validation sweeps; recommendation models are not run.
"""
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/tokenizer_reranking_analysis_20261014"
COMPARISON = ROOT / "docs/rq2_comparison_draft_20261011/comparison_results.csv"
DATASETS = {"amazon-beauty": "Beauty", "amazon-sports": "Sports",
            "amazon-toys": "Toys", "amazon-pet-supplies": "Pet Supplies"}
TOKENIZERS = {"rvq": "RVQ", "rqvae": "RQ-VAE"}
METHODS = {"GRID: saved reranking reference": "w/o reranking", "GRID + RR": "RR",
           "GRID + CPFair": "CPFair", "GRID + P-MMF": "P-MMF"}


def reference(path):
    path = path.resolve()
    return dict(path=str(path.relative_to(ROOT)), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def write_csv(name, rows):
    with (OUT / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def candidate_metadata(path):
    # Metadata precedes the large item-group maps and request arrays. Reading
    # this prefix does not hash or load the multi-GB candidate file again.
    with path.open() as stream:
        prefix = stream.read(65536).split('"item_groups"', 1)[0]
    return json.loads(prefix.rstrip().rstrip(",") + "}")["metadata"]


def render(tokenizer, rows, beam_widths):
    label = TOKENIZERS[tokenizer]
    caption = (f"Validation-selected {label} reranking. Test Recall/NDCG cells list "
               r"$K=5,10,20$. Untreated controls use the same saved candidates from the original source checkpoints. Seed 42. "
               r"Settings minimize popularity MGU (RR) or brand exposure Gini (CPFair/P-MMF), allowing up to 5\% relative validation Recall@5 loss. ")
    if tokenizer == "rvq":
        caption += "Sports uses beam width 300; the other datasets use 100. "
    tex = [r"% Requires booktabs. Generated from relevance_results.csv; values are test metrics.",
           r"\begin{table*}[t]", r"\centering\small", r"\setlength{\tabcolsep}{4pt}",
           r"\caption{" + caption.rstrip() + "}",
           r"\label{tab:" + tokenizer + r"-rerank-relevance}", r"\begin{tabular}{llcc}",
           r"\toprule", r"Dataset & Method & Recall (5 / 10 / 20) $\uparrow$ & NDCG (5 / 10 / 20) $\uparrow$ \\",
           r"\midrule"]
    for dataset, name in DATASETS.items():
        assert beam_widths[dataset, tokenizer] == (300 if dataset == "amazon-sports" and tokenizer == "rvq" else 100)
        for method in METHODS.values():
            group = sorted((r for r in rows if r["tokenizer"] == tokenizer and
                            r["dataset"] == name and r["method"] == method), key=lambda r: r["k"])
            assert [r["k"] for r in group] == [5, 10, 20]
            cells = [" / ".join(f"{r[metric]:.4f}" for r in group) for metric in ("recall", "ndcg")]
            tex.append(" & ".join([name, method] + cells) + r" \\")
        tex.append(r"\midrule")
    tex[-1] = r"\bottomrule"
    return "\n".join(tex + [r"\end{tabular}", r"\end{table*}", ""])


def tradeoffs():
    """Use independently recomputed calibration, once all eight cases exist."""
    paths = [OUT / "calibration" / f"{ds}_{tok}.json" for tok in TOKENIZERS for ds in DATASETS]
    if not all(path.exists() for path in paths):
        return [], []
    rows = []
    comparison_sha = reference(COMPARISON)["sha256"]
    for path in paths:
        report = json.loads(path.read_text())
        assert report["comparison"]["sha256"] == comparison_sha
        dataset, tok = report["dataset"], report["tokenizer"]
        assert report["seed"] == 42
        for method, label in (("rr", "RR"), ("cpfair", "CPFair"), ("pmmf", "P-MMF")):
            for k in (5, 10, 20):
                base = report["results"]["control"][str(k)]
                new = report["results"][method][str(k)]
                row = dict(dataset=DATASETS[dataset], tokenizer=tok, method=label, k=k,
                           seed=42, requests=report["request_count"], calibration_source=str(path.relative_to(ROOT)))
                for metric in ("recall", "ndcg", "MGU", "DGU", "brand_gini", "entropy_bits"):
                    assert base[metric] > 0
                    row["control_" + metric] = base[metric]
                    row[metric] = new[metric]
                    # Positive fairness-loss reductions mean lower MGU/DGU/Gini.
                    sign = -1 if metric in ("MGU", "DGU", "brand_gini") else 1
                    row[metric + "_change_percent"] = sign * 100 * (new[metric] - base[metric]) / base[metric]
                rows.append(row)
    assert len(rows) == 72
    write_csv("tradeoff_results.csv", rows)
    rendered = []
    for tok, label in TOKENIZERS.items():
        caption = (f"Relative changes against GRID {label} recommendations without reranking. "
                   r"Each cell lists $K=5/10/20$. MGU and DGU measure popularity calibration; Gini and entropy measure brand exposure. "
                   r"Positive loss reductions and entropy increases indicate improvements in their respective measures; negative values indicate deterioration. "
                   r"All percentages are relative changes computed from unrounded values. Seed 42.")
        if tok == "rvq":
            caption += " Sports uses beam width 300; other datasets use 100."
        tex = [r"% Requires booktabs and graphicx. Values come from tradeoff_results.csv.",
               r"\begin{table*}[t]", r"\centering\small", r"\setlength{\tabcolsep}{4pt}",
               r"\caption{" + caption + "}", r"\label{tab:" + tok + r"-rerank-tradeoff}",
               r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{llccccc}", r"\toprule",
               r"Dataset & Method & Recall change (\%) & MGU reduction (\%) & DGU reduction (\%) & Brand Gini reduction (\%) & Brand entropy increase (\%) \\",
               r"\midrule"]
        for ds in DATASETS.values():
            for method in ("RR", "CPFair", "P-MMF"):
                group = sorted((r for r in rows if r["dataset"] == ds and r["tokenizer"] == tok
                                and r["method"] == method), key=lambda r: r["k"])
                assert [r["k"] for r in group] == [5, 10, 20]
                cells = []
                for metric in ("recall", "MGU", "DGU", "brand_gini", "entropy_bits"):
                    spec = "+.2f" if metric in ("recall", "entropy_bits") else ".2f"
                    cells.append(" / ".join(format(0. if round(r[metric + "_change_percent"], 2) == 0
                                                       else r[metric + "_change_percent"], spec) for r in group))
                tex.append(" & ".join([ds, method] + cells) + r" \\")
            tex.append(r"\midrule")
        tex[-1] = r"\bottomrule"
        tex += [r"\end{tabular}%", "}", r"\end{table*}", ""]
        content = "\n".join(tex)
        (OUT / (tok + "_tradeoff.tex")).write_text(content)
        rendered.append(content)
    return rendered, paths


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    comparison = list(csv.DictReader(COMPARISON.open()))
    lookup = {}
    for row in comparison:
        if row["tokenizer"] in TOKENIZERS and row["method"] in METHODS:
            key = row["dataset"], row["tokenizer"], row["method"], int(row["k"])
            assert key not in lookup and row["status"] == "verified", key
            lookup[key] = row
    assert len(lookup) == 96
    rows, deltas, selections, sources, inputs = [], [], [], [reference(COMPARISON)], []
    beam_widths = {}
    for tokenizer in TOKENIZERS:
        for dataset, name in DATASETS.items():
            cp = lookup[dataset, tokenizer, "GRID + CPFair", 5]
            selection_path = ROOT / Path(cp["source"]).parent / "selection.json"
            selection = json.loads(selection_path.read_text())
            sources.append(reference(selection_path))
            assert selection["dataset"] == dataset
            meta = candidate_metadata(Path(selection["test_candidates"]["path"]))
            assert meta["seed"] == 42 and meta["split"]["name"] == "test"
            assert meta["dataset"] == dataset and meta["tokenizer"] == tokenizer
            beam_widths[dataset, tokenizer] = meta["generation"]["beam_width"]
            inputs.append(dict(dataset=dataset, tokenizer=tokenizer,
                               checkpoint=meta["checkpoint"], seed=42,
                               test_candidates=selection["test_candidates"],
                               split=meta["split"], generation=meta["generation"]))
            for method in ("rr", "cpfair", "pmmf"):
                chosen = selection["validation_selection"]["selections"][method]
                sweep = Path(chosen["rows"]["path"])
                ref = reference(sweep)
                assert ref["sha256"] == chosen["rows"]["sha256"]
                sources.append(ref)
                sweep_rows = [r for r in csv.DictReader(sweep.open()) if int(r["k"]) == 5]
                controls = [r for r in sweep_rows if json.loads(r["parameters"]) == {}]
                assert len(controls) == 1 and chosen["max_relative_recall_loss"] == .05
                control = controls[0]
                axis = chosen["fairness_axis"]
                assert axis == ("mgu" if method == "rr" else "exposure_gini")
                threshold = .95 * float(control["recall"])
                eligible = [r for r in sweep_rows if json.loads(r["parameters"]) and
                            float(r["recall"]) >= threshold - 1e-12 and math.isfinite(float(r[axis]))]
                best = min(eligible, key=lambda r: (float(r[axis]), -float(r["recall"]), r["parameters"]))
                assert json.loads(best["parameters"]) == chosen["parameters"]
                params = ({"alpha": selection["rr_alpha"], "floor": selection["rr_floor"]}
                          if method == "rr" else {"lambda": selection[method + "_lambda"]})
                assert params == chosen["parameters"]
                selections.append(dict(dataset=name, tokenizer=tokenizer, method=method,
                    selected_parameters=json.dumps(params, sort_keys=True), fairness_axis=axis,
                    selection_k=5, allowed_validation_recall_loss_percent=5,
                    validation_control_recall=float(control["recall"]),
                    validation_selected_recall=float(best["recall"]),
                    validation_control_fairness=float(control[axis]),
                    validation_selected_fairness=float(best[axis]), selection_source=str(selection_path.relative_to(ROOT))))
            for raw_method, method in METHODS.items():
                for k in (5, 10, 20):
                    entry = lookup[dataset, tokenizer, raw_method, k]
                    base = lookup[dataset, tokenizer, "GRID: saved reranking reference", k]
                    row = dict(dataset=name, tokenizer=tokenizer, method=method, k=k, seed=42,
                               recall=float(entry["recall"]), ndcg=float(entry["ndcg"]),
                               brand_gini=float(entry["brand_gini"]), entropy_bits=float(entry["brand_entropy_bits"]),
                               source=entry["source"])
                    assert 0 <= row["ndcg"] <= row["recall"] <= 1
                    rows.append(row)
                    if raw_method == "GRID: saved reranking reference":
                        continue
                    delta = dict(row)
                    for metric, source_key in (("recall", "recall"), ("ndcg", "ndcg"),
                                               ("brand_gini", "brand_gini"), ("entropy_bits", "brand_entropy_bits")):
                        baseline = float(base[source_key])
                        change = row[metric] - baseline
                        delta["control_" + metric] = baseline
                        delta["delta_" + metric] = change
                        delta["relative_delta_percent_" + metric] = 100 * change / baseline
                        if metric in ("recall", "ndcg"):
                            delta[metric + "_gain_hidden_by_rounding"] = change > 1e-12 and round(row[metric], 4) == round(baseline, 4)
                    deltas.append(delta)
    write_csv("relevance_results.csv", rows)
    write_csv("test_deltas.csv", deltas)
    write_csv("validation_selection.csv", selections)
    rendered = []
    for tokenizer in TOKENIZERS:
        tex = render(tokenizer, rows, beam_widths)
        (OUT / (tokenizer + "_relevance.tex")).write_text(tex)
        rendered.append(tex)
    tradeoff_tex, calibration_paths = tradeoffs()
    rendered.extend(tradeoff_tex)
    sources.extend(reference(path) for path in calibration_paths)
    (OUT / "tables.tex").write_text("\n".join(rendered))
    preview_inputs = "\n".join(r"\input{" + tok + "_relevance.tex}" for tok in TOKENIZERS)
    if tradeoff_tex:
        preview_inputs += "\n" + r"\clearpage" + "\n" + "\n".join(
            r"\input{" + tok + "_tradeoff.tex}" for tok in TOKENIZERS)
    (OUT / "preview.tex").write_text(r"\documentclass[10pt]{article}" + "\n" +
        r"\usepackage[a4paper,margin=15mm]{geometry}" + "\n" + r"\usepackage{booktabs,graphicx}" + "\n" +
        r"\begin{document}" + "\n" + preview_inputs + "\n" + r"\end{document}" + "\n")
    (OUT / "provenance.json").write_text(json.dumps(dict(
        sources=sources, inputs=inputs, rows=len(rows), selected_settings=len(selections),
        code=[reference(Path(__file__)), reference(ROOT / "scripts/compute_tokenizer_rerank_tradeoffs.py")],
        metrics_scope="Saved full-test relevance metrics from canonical comparison CSV; tradeoff calibration and all displayed tradeoff inputs independently reproduced from saved test rankings when calibration records are present",
        tradeoff_tables=len(tradeoff_tex), calibration_records=len(calibration_paths),
        calibration_definition="GH: pooled test histories; GP: unweighted top-K popularity shares; GU=GP-GH; MGU=mean(abs(GU)); DGU=max(GU)-min(GU)",
        selection_verification="All 24 selections independently checked against their saved full-validation sweep and recorded sweep hash",
        baseline="Same saved-candidate untreated control; source checkpoint before the later standard and fairness-aware fine-tuning",
        delta_formula="100 * (reranked - untreated) / untreated; computed before rounding",
        fairness_selection="Minimum MGU for RR or exposure Gini for CPFair/P-MMF subject to <=5% validation Recall@5 loss; untreated control excluded from active settings",
        limitations=["One seed; no statistical-significance claim", "The validation loss allowance is not a test-set guarantee",
                     "Relevance-only builds use canonical CSV; completed tradeoff builds include independent candidate/ranking fingerprint, eligibility and metric checks recorded per calibration case",
                     "Sports RVQ uses beam width 300; other reranking candidates here use 100",
                     "Values within 1e-12 are treated as equal when describing gains"]), indent=2) + "\n")
    for tok in TOKENIZERS:
        subset = [r for r in deltas if r["tokenizer"] == tok]
        print(tok, {metric + "_wins": sum(r["delta_" + metric] > 1e-12 for r in subset)
                    for metric in ("recall", "ndcg")}, "of", len(subset), "method/dataset/cutoff comparisons")
    print(f"Saved {len(rendered)} tables, 96 metric rows, 72 delta rows and 24 validated setting selections to", OUT)


if __name__ == "__main__":
    main()
