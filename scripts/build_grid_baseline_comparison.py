"""Render the local/published GRID table from verified original-baseline rows."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/rq2_comparison_draft_20261011"
PAPER_CSV = ROOT / "docs/rq1_grid_analysis_20261011/paper_vs_fresh_local_results.csv"
BEGIN = "% BEGIN GRID LOCAL/PUBLISHED COMPARISON"
END = "% END GRID LOCAL/PUBLISHED COMPARISON"


def write_baseline_comparison(rows, *, dataset_statistics_path=None):
    paper_rows = list(csv.DictReader(PAPER_CSV.open()))
    published = {(r["dataset"], r["tokenizer"]): r for r in paper_rows}
    pet_path = OUT / "pet_published_reference.json"
    pet_reference = json.loads(pet_path.read_text())
    stats_path = (Path(dataset_statistics_path) if dataset_statistics_path else
                  ROOT / "recommendation/processed_dataset/amazon-pet-supplies/dataset_stats.json")
    stats = json.loads(stats_path.read_text())
    stats = stats.get("amazon-pet-supplies", stats)
    assert stats["user_num"] == pet_reference["users"] and stats["item_num"] == pet_reference["items"]
    assert pet_reference["table"] == 11 and pet_reference["published_model"] == "TIGER"
    for metric, value in pet_reference["metrics"].items():
        assert abs(value - pet_reference["table_values_percent"][metric] / 100) < 1e-12
    originals = {(r["dataset"], r["tokenizer"], int(r["k"])): r for r in rows
                 if r["method"] == "GRID: original (fresh evaluation)"}
    tex = [BEGIN, r"% Requires booktabs and graphicx; no multirow package needed.",
           r"\begin{table*}[t]", r"    \centering\small",
           r"    \setlength{\tabcolsep}{3pt}", r"    \renewcommand{\arraystretch}{1.35}",
           r"    \caption{Local GRID results compared with the published results.",
           r"        Each metric cell shows \textbf{local / published}.",
           r"        $\Delta R@10$ and $\Delta N@10$ give the relative differences in",
           r"        Recall@10 and NDCG@10, respectively: $100(M_{\mathrm{local}}/M_{\mathrm{published}}-1)$.",
           r"        Deltas use unrounded local scores. Local results use seed 42; published Beauty/Sports/Toys results average five seeds.",
           r"        Positive values indicate higher observed local scores; these descriptive differences do not indicate statistical significance.",
           r"        $\ddagger$ Pet Supplies RQ-VAE uses TIGER's test results from Lepage et al. (2025), arXiv:2508.14910, Table~11:",
           r"        a contextual comparison across models and quantizer settings, rather than a matched GRID reproduction.",
           r"        No matching Pet Supplies RKMeans/RVQ test reference was verified (\textemdash); validation-only results are excluded.}",
           r"    \label{tab:grid-baseline-comparison}",
           r"    \begin{tabular}{@{}clccccrr@{}}", r"        \toprule",
           r"        & Tokenizer & Recall@5 & Recall@10 & NDCG@5 & NDCG@10 & $\Delta R@10$ & $\Delta N@10$ \\",
           r"        \midrule"]
    audit = []
    for ds, label in (("amazon-beauty", "Beauty"), ("amazon-sports", "Sports"),
                      ("amazon-toys", "Toys"), ("amazon-pet-supplies", "Pet Supplies")):
        for tok, token_label in (("rkmeans", "RKMeans"), ("rvq", "RVQ"), ("rqvae", "RQ-VAE")):
            r5, r10 = originals[ds, tok, 5], originals[ds, tok, 10]
            assert r5["status"].startswith("verified") and r10["status"].startswith("verified")
            local = [float(r5["recall"]), float(r10["recall"]), float(r5["ndcg"]), float(r10["ndcg"])]
            paper = published.get((ds, tok))
            metrics = ("recall@5", "recall@10", "ndcg@5", "ndcg@10")
            values = [float(paper["paper_" + m]) for m in metrics] if paper else None
            contextual = ds == "amazon-pet-supplies" and tok == "rqvae"
            if contextual:
                assert pet_reference["dataset"] == ds and pet_reference["tokenizer"] == tok
                values = [pet_reference["metrics"][m] for m in metrics]
                token_label += r"$^{\ddagger}$"
            cells = [f"{v:.4f} / {values[i]:.4f}" if values else f"{v:.4f} / \\textemdash"
                     for i, v in enumerate(local)]
            deltas = [100 * (local[i] / values[i] - 1) for i in (1, 3)] if values else None
            cells += [f"${v:+.1f}\\%$" for v in deltas] if deltas else [r"\textemdash"] * 2
            # Put the label in the middle row, centered across the three rows.
            # A zero-height rotated box avoids increasing that row's height.
            rotated = (r"\raisebox{\dimexpr0.5ex-0.5\height\relax}[0pt][0pt]{\rotatebox{90}{\footnotesize "
                       + label + "}}") if tok == "rvq" else ""
            tex.append("        " + " & ".join([rotated, token_label] + cells) + r" \\")
            audit.append(dict(dataset=ds, tokenizer=tok, source=r5["source"],
                              local=dict(zip(metrics, local)), published=dict(zip(metrics, values)) if values else None,
                              published_source=pet_reference["pdf_url"] + "#page=13" if contextual else
                                               "https://arxiv.org/html/2507.22224v1#S3.T1" if values else None,
                              published_model="TIGER" if contextual else "GRID" if values else None,
                              comparison_kind=pet_reference["comparison_kind"] if contextual else
                                              "local GRID versus published GRID" if values else "no verified published reference",
                              relative_delta_percent=dict(zip(("recall@10", "ndcg@10"), deltas)) if deltas else None))
        tex.append(r"        \midrule" if ds != "amazon-pet-supplies" else r"        \bottomrule")
    tex += [r"    \end{tabular}", r"\end{table*}", END]
    rendered = "\n".join(tex) + "\n"
    (OUT / "baseline_comparison.tex").write_text(rendered)
    (OUT / "baseline_comparison_provenance.json").write_text(json.dumps(dict(
        published_source="https://arxiv.org/html/2507.22224v1#S3.T1",
        published_reference_csv=str(PAPER_CSV),
        additional_published_references=[dict(path=str(pet_path),
            sha256=hashlib.sha256(pet_path.read_bytes()).hexdigest(),
            source=pet_reference["pdf_url"] + "#page=13", table=11)],
        published_reference_sha256=hashlib.sha256(PAPER_CSV.read_bytes()).hexdigest(), rows=audit), indent=2) + "\n")
    return rendered


if __name__ == "__main__":
    rendered = write_baseline_comparison(list(csv.DictReader((OUT / "comparison_results.csv").open())))
    path = OUT / "comparison_tables.tex"
    existing = path.read_text()
    if BEGIN in existing:
        start, tail = existing.split(BEGIN, 1)
        _, rest = tail.split(END, 1)
        existing = start + rendered + rest.lstrip("\n")
    else:
        header, rest = existing.split("\n", 1)
        existing = header + "\n" + rendered + "\n" + rest
    path.write_text(existing)
