"""Add the completed joint-fairness pilot to the existing RQ2 comparison.

Historical Recall-selected rows stay in their original cohort. This supplement
records every confirmation outcome and the full tests of frozen winners.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from scripts.summarize_deadline_results import summarize


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/rq2_comparison_draft_20261011"
PILOT = ROOT / "additional_baselines/artifacts/deadline_fairness_pilot_20261014"
BEGIN = "% BEGIN JOINT FAIRNESS PILOT COMPARISON"
END = "% END JOINT FAIRNESS PILOT COMPARISON"
DATASETS = {"amazon-beauty": "Beauty", "amazon-sports": "Sports",
            "amazon-toys": "Toys", "amazon-pet-supplies": "Pet Supplies"}
TOKENS = {"rkmeans": "RKMeans", "rvq": "RVQ", "rqvae": "RQ-VAE"}


def read(path):
    return json.loads(Path(path).read_text())


def metrics_at(report, history, cutoff):
    """Recompute exposure statistics directly from all saved catalog groups."""
    metrics = report["metrics"]["at_k"][str(cutoff)]
    counts = metrics["groups"]["brand"]["exposure_counts"]
    ordered = sorted(counts.values())
    total, n = sum(ordered), len(ordered)
    gini = sum((2 * i - n - 1) * count for i, count in enumerate(ordered, 1)) / (n * total)
    entropy = -math.fsum(count / total * math.log2(count / total) for count in ordered if count)
    assert math.isclose(gini, metrics["groups"]["brand"]["gini"], abs_tol=1e-12)
    assert math.isclose(entropy * math.log(2), metrics["groups"]["brand"]["entropy"], abs_tol=1e-12)
    popularity = metrics["groups"]["popularity"]["exposure_counts"]
    assert set(popularity) == set(history)
    exposure = sum(popularity.values())
    mgu = math.fsum(abs(popularity[g] / exposure - history[g]) for g in history) / len(history)
    return dict(recall=metrics["recall"], ndcg=metrics["ndcg"], brand_gini=gini,
                brand_entropy_bits=entropy, popularity_mgu=mgu)


def write_joint_fairness_comparison(destination=OUT):
    destination = Path(destination)
    report_path = summarize(PILOT / "final_readout_20261015")
    report = read(report_path)
    comparisons = {(r["dataset"], r["tokenizer"], r["method"]): r
                   for r in report["test_comparisons"]}
    outcomes = {(r["dataset"], r["tokenizer"], r["method"]): r
                for r in report["method_outcomes"]}
    controls = {(r["dataset"], r["tokenizer"]): r for r in report["ordinary_controls"]}
    assert len(controls) == 12
    tex = [BEGIN, r"\begin{table*}[p]", r"\centering\small",
           r"\caption{Completed joint-fairness screening study: all 12 cases and 48 screening trials. "
           r"Each case screens history/uniform RW at strength 0.25 and two smaller SPRINT coefficient triples. "
           r"Screening uses 10,000 fixed stratified validation users; sampled choices are frozen before full validation. "
           r"Recall@5 must retain at least 98\% of ordinary, neither brand Gini nor popularity MGU may worsen, "
           r"and at least one fairness axis must improve. Tested denotes full-validation confirmation plus a full test, "
           r"not guaranteed satisfaction of those criteria on test. An ordinary fallback is a completed selection outcome.}",
           r"\label{tab:joint-fairness-outcomes}", r"\resizebox{\textwidth}{!}{%",
           r"\begin{tabular}{llll}", r"\toprule",
           r"Dataset & Tokenizer & RW outcome & SPRINT outcome \\", r"\midrule"]
    for dataset, label in DATASETS.items():
        for tokenizer, token_label in TOKENS.items():
            cells = []
            for mode in ("rw", "sprint"):
                outcome = outcomes[dataset, tokenizer, mode]
                selected = outcome["sampled_choice"]
                if outcome["status"] == "full_validation_confirmed":
                    setting = "uniform, $s=0.25$" if mode == "rw" else (
                        "$a=0.001$" if selected["trial"]["sprint_settings"]["attention_weight"] == .001
                        else "$a=0.01$")
                    cells.append(f"Tested: {setting}, update {selected['selected_update']}")
                elif outcome["status"] == "failed_full_validation_keep_ordinary":
                    cells.append("Ordinary: full validation rejected")
                else:
                    assert outcome["status"] == "no_sampled_improvement_keep_ordinary"
                    cells.append("Ordinary: no sampled improvement")
            tex.append(" & ".join([label, token_label] + cells) + r" \\")
        tex.append(r"\midrule")
    tex[-1] = r"\bottomrule"
    tex += [r"\end{tabular}%", "}", r"\par\smallskip\footnotesize "
            r"SPRINT coefficients are $(a,0.001a,0.1a)$ for attention, feed-forward, and embeddings; "
            r"temperature 4, one power iteration. RW uniform target is a group-balanced adaptation, "
            r"distinct from the historical IFairLRS history-target weights.", r"\end{table*}", ""]
    rows = []
    for part, (dataset, label) in enumerate(DATASETS.items(), 1):
        tex += [r"\begin{table*}[p]", r"\centering\small", r"\setlength{\tabcolsep}{3pt}"]
        if part == 1:
            tex += [r"\caption{Joint-fairness study: full-test metrics and qualification outcomes for all "
                r"four datasets and three tokenizers (part 1/4: Beauty). Every case includes its ordinary "
                r"test control and the RW/SPRINT outcome. A not-qualified entry denotes completed screening "
                r"with no accepted candidate; no candidate test was run and ordinary is retained. "
                r"It does not denote an unfinished job or a zero score. Rows report $K=5,10,20$ separately; "
                r"entropy is in bits. Ten candidates passed full validation; seven meet the same joint "
                r"criteria descriptively on test at $K=5$. $\ddagger$ marks test-side exceptions: Beauty "
                r"RQ-VAE RW loses 2.12\% Recall@5, Sports RQ-VAE RW loses 7.45\%, and Toys RVQ SPRINT "
                r"increases MGU@5 by 0.26\%. Test outcomes do not reselect models. These are distinct from "
                r"the historical Recall-selected rows.}", r"\label{tab:joint-fairness-tests}"]
        else:
            tex += [r"\addtocounter{table}{-1}",
                    r"\caption[]{Joint-fairness study: full-test metrics and qualification outcomes "
                    f"(continued, part {part}/4: {label}). "
                    r"Not qualified means screening completed, no candidate full test, and ordinary retained. "
                    r"$\ddagger$ marks failure of the joint criteria on test at $K=5$; entropy is in bits.}"]
        tex += [r"\label{tab:joint-fairness-tests-" + dataset.removeprefix("amazon-") + "}",
                r"\begin{tabular}{llcccccc}", r"\toprule",
                r"Dataset / tokenizer & Variant & $K$ & Recall $\uparrow$ & NDCG $\uparrow$ & "
                r"Brand Gini $\downarrow$ & Entropy $\uparrow$ & MGU $\downarrow$ \\", r"\midrule"]
        for tokenizer, token_label in TOKENS.items():
            control = controls[dataset, tokenizer]
            history, baseline = control["history_shares"], control["source"]
            for mode in ("ordinary", "rw", "sprint"):
                candidate = comparisons.get((dataset, tokenizer, mode))
                outcome = outcomes.get((dataset, tokenizer, mode))
                if mode != "ordinary" and candidate is None:
                    variant = "RW" if mode == "rw" else "SPRINT"
                    assert outcome["status"] in {"failed_full_validation_keep_ordinary", "no_sampled_improvement_keep_ordinary"}
                    reason = ("Full validation rejected; ordinary retained" if outcome["status"] == "failed_full_validation_keep_ordinary"
                              else "Screening rejected; ordinary retained")
                    tex.append(f"{label} / {token_label} & {variant} (not qualified) & -- & "
                               + r"\multicolumn{5}{l}{" + reason + r"} \\")
                    for cutoff in (5, 10, 20):
                        rows.append(dict(dataset=dataset, tokenizer=tokenizer, method=mode, k=cutoff,
                            **{name: "" for name in ("recall", "ndcg", "brand_gini", "brand_entropy_bits", "popularity_mgu")},
                            request_count="", selected_update="", setting="no accepted candidate; ordinary retained",
                            selection_cohort="joint_fairness_pilot_20261014", test_joint_criteria_pass_at_5="",
                            source="", source_sha256="", status=outcome["status"], test_evaluated=False,
                            qualification_evidence=str(Path(outcome["completed_case"]["path"]).relative_to(ROOT))))
                    continue
                source = baseline if mode == "ordinary" else candidate["candidate_source"]
                if candidate:
                    assert candidate["baseline_source"] == baseline
                variant = "Ordinary" if mode == "ordinary" else "Uniform RW" if mode == "rw" else "SPRINT"
                full = read(source["path"])
                values = [metrics_at(full, history, k) for k in (5, 10, 20)]
                passed = (candidate["both_fairness_measures_no_worse_on_test"] and
                          candidate["test_recall_loss_within_2_percent"]) if candidate else None
                display = variant + (r"$^{\ddagger}$" if passed is False else "")
                for cutoff, value in zip((5, 10, 20), values):
                    cells = [f"{value[name]:.4f}" for name in
                             ("recall", "ndcg", "brand_gini", "brand_entropy_bits", "popularity_mgu")]
                    tex.append(" & ".join([f"{label} / {token_label}" if cutoff == 5 else "",
                                           display if cutoff == 5 else "", str(cutoff)] + cells) + r" \\")
                    rows.append(dict(dataset=dataset, tokenizer=tokenizer, method=mode, k=cutoff, **value,
                        request_count=full["metrics"]["request_count"],
                        selected_update=candidate["selected_update"] if candidate else control["selected_update"],
                        setting=json.dumps(candidate["settings"], sort_keys=True) if candidate else "matched ordinary control",
                        selection_cohort="joint_fairness_pilot_20261014",
                        test_joint_criteria_pass_at_5=passed if candidate else "",
                        source=str(Path(source["path"]).relative_to(ROOT)), source_sha256=source["sha256"],
                        status="full_validation_confirmed" if candidate else "ordinary_control",
                        test_evaluated=True,
                        qualification_evidence=str(Path(outcome["completed_case"]["path"]).relative_to(ROOT)) if outcome else ""))
            tex.append(r"\midrule")
        tex[-1] = r"\bottomrule"
        tex += [r"\end{tabular}", r"\par\smallskip\footnotesize "
                r"The historical name ``pilot'' denotes the limited search: two settings per method, "
                r"one seed (42), and 32 additional updates. Screening used 10,000 validation users; "
                r"confirmation used full validation and numerical test results use all test users. "
                r"Controls used single-request generation; candidates used batches of 16. Reported-rank "
                r"parity was checked on 64 validation users per case, not the whole test split. "
                r"Differences are descriptive; tiny changes require caution.", r"\end{table*}", ""]
    tex += [END, ""]
    assert len(rows) == 108  # All 12 cases, three variants, three cutoffs.
    assert sum(r["test_evaluated"] for r in rows) == 66
    with (destination / "joint_fairness_results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    provenance = dict(report=report, report_source=str(report_path.relative_to(ROOT)),
                      report_sha256=hashlib.sha256(report_path.read_bytes()).hexdigest(),
                      historical_results_unchanged=True, supplemental_cutoff_rows=len(rows),
                      supplemental_logical_rows=36, populated_cutoff_rows=66,
                      not_qualified_cutoff_rows=42, ordinary_full_test_controls=12,
                      new_full_test_candidates=10)
    (destination / "joint_fairness_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return "\n".join(tex)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    section = write_joint_fairness_comparison(args.output_dir)
    path = args.output_dir / "comparison_tables.tex"
    existing = path.read_text()
    if BEGIN in existing:
        before, remainder = existing.split(BEGIN, 1)
        _, after = remainder.split(END, 1)
        existing = before.rstrip() + "\n" + after.lstrip("\n")
    path.write_text(existing.rstrip() + "\n\n" + section)


if __name__ == "__main__":
    main()
