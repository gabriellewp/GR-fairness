#!/usr/bin/env python3
"""Build a fine-tuning/reranking table and figure from comparison_tables.tex.

Run:
    python -m scripts.paper.make_training_interventions.py

Changes are paired within dataset/tokenizer. Fine-tuning uses original GRID;
reranking uses the untreated ranking of its matched saved candidate pool.
All calculations use the four-decimal values actually printed in the input TeX.
"""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import tempfile

os.environ.setdefault("MPLCONFIGDIR", "/tmp/training-interventions-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, MultipleLocator
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "paper/inputs/comparison_tables.tex"
DATASETS = {"beauty": "Beauty", "sports": "Sports", "toys": "Toys",
            "pet-supplies": "Pet Supplies"}
TOKENIZERS = ("RKMeans", "RVQ", "RQ-VAE")
ORIGINAL = "GRID: original (fresh evaluation)"
RERANKING_REFERENCE = "GRID: saved reranking reference"
METHODS = {
    "Standard FT": "GRID: standard fine-tuning",
    "IFairLRS RW": "GRID + IFairLRS RW (fairness-aware fine-tuning)",
    "SPRINT": "GRID + SPRINT (fairness-aware fine-tuning)",
    "IPS": "GRID + IPS (training)",
    "FOCF": "GRID + FOCF (training)",
    "Reg": "GRID + Reg (training)",
    "FairDual": "GRID + FairDual (training)",
    "Minmax-SGD": "GRID + Minmax-SGD (training)",
    "APR": "GRID + APR (training)",
    "SDRO": "GRID + SDRO (training)",
}
CONTINUATION = {"Standard FT", "IFairLRS RW", "SPRINT"}
FAIRDIVERSE = set(METHODS) - CONTINUATION
RERANKERS = {"CPFair": "GRID + CPFair", "P-MMF": "GRID + P-MMF", "RR": "GRID + RR"}
METHODS.update(RERANKERS)
BLUE, ORANGE, TEAL = "#2a78d6", "#eb6834", "#159b88"


def parse_tables(source):
    """Select the four named dataset tables and all required methods/references."""
    tables = re.findall(r"\\begin\{table\*\}.*?\\end\{table\*\}",
                        source.read_text(), re.S)
    records = {}
    selected_methods = {ORIGINAL, RERANKING_REFERENCE, *METHODS.values()}
    for dataset in DATASETS:
        label = rf"\label{{tab:comparison-{dataset}}}"
        matches = [table for table in tables if label in table]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one source table for {dataset}")
        for line in matches[0].splitlines():
            cells = [cell.strip() for cell in line.split("&")]
            if len(cells) != 6:
                continue
            variant = re.sub(r"\$\^\{[^}]*\}\$", "", cells[0]).strip()
            if variant not in selected_methods:
                continue
            tokenizer = cells[1]
            if tokenizer not in TOKENIZERS:
                raise ValueError(f"Unknown tokenizer: {tokenizer}")
            key = dataset, tokenizer, variant
            if key in records:
                raise ValueError(f"Duplicate source row: {key}")
            metrics = {}
            for name, cell in zip(("recall", "ndcg", "gini"), cells[2:5]):
                values = [float(value.strip()) for value in cell.split("/")]
                if len(values) != 3 or not all(math.isfinite(v) and 0 <= v <= 1
                                               for v in values):
                    raise ValueError(f"Invalid {name} values: {key}")
                metrics.update({f"{name}@{k}": v for k, v in zip((5, 10, 20), values)})
            if any(metrics[f"ndcg@{k}"] > metrics[f"recall@{k}"] for k in (5, 10, 20)):
                raise ValueError(f"NDCG exceeds single-target Recall: {key}")
            records[key] = metrics
    expected = {(ds, tok, variant) for ds in DATASETS for tok in TOKENIZERS
                for variant in selected_methods}
    if set(records) != expected:
        raise ValueError(f"Incomplete source rows; missing {expected - set(records)}")
    return records


def calculate_changes(records):
    changes = []
    for method, variant in METHODS.items():
        for dataset in DATASETS:
            for tokenizer in TOKENIZERS:
                original = records[dataset, tokenizer, ORIGINAL]
                reference_variant = RERANKING_REFERENCE if method in RERANKERS else ORIGINAL
                reference = records[dataset, tokenizer, reference_variant]
                result = records[dataset, tokenizer, variant]
                if reference["recall@10"] <= 0:
                    raise ValueError(f"Nonpositive reference recall: {dataset}, {tokenizer}")
                changes.append(dict(
                    dataset=dataset, tokenizer=tokenizer, method=method, variant=variant,
                    evaluation_status=("saved-candidate reranking evaluation" if method in RERANKERS
                                       else "continuation evaluation" if method in CONTINUATION
                                       else "recorded FairDiverse evaluation"),
                    intervention_group=("reranking" if method in RERANKERS
                                        else "resumed-Adam fine-tuning" if method in CONTINUATION
                                        else "fresh-Adam FairDiverse training"),
                    optimizer_state=("not applicable; no model updates" if method in RERANKERS
                                     else "resumed source Adam" if method in CONTINUATION
                                     else "fresh Adam"),
                    reference_variant=reference_variant,
                    reference_recall_at_10=reference["recall@10"],
                    reference_brand_gini_at_10=reference["gini@10"],
                    original_recall_at_10=original["recall@10"],
                    method_recall_at_10=result["recall@10"],
                    original_brand_gini_at_10=original["gini@10"],
                    method_brand_gini_at_10=result["gini@10"],
                    recall_at_10_relative_change_percent=100 * (
                        result["recall@10"] / reference["recall@10"] - 1),
                    brand_gini_at_10_absolute_change=result["gini@10"] - reference["gini@10"],
                ))
    expected_changes = len(METHODS) * len(DATASETS) * len(TOKENIZERS)
    if len(changes) != expected_changes:
        raise ValueError(f"Expected {expected_changes} paired changes; found {len(changes)}")
    summaries = []
    for method in METHODS:
        cases = [r for r in changes if r["method"] == method]
        if len(cases) != 12:
            raise ValueError(f"Expected twelve cases for {method}")
        summary = dict(method=method, cases=len(cases))
        for field, label in (("recall_at_10_relative_change_percent", "recall_change_percent"),
                             ("brand_gini_at_10_absolute_change", "gini_absolute_change")):
            values = [r[field] for r in cases]
            summary.update({f"{label}_median": statistics.median(values),
                            f"{label}_min": min(values), f"{label}_max": max(values)})
        summaries.append(summary)
    return changes, summaries


def signed(value, precision):
    value = round(value, precision)
    return f"{0.0 if value == 0 else value:+.{precision}f}"


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_latex(summaries, output):
    lines = [
        r"% Generated by docs/make_training_interventions.py; requires booktabs.",
        r"\begin{table}[!tb]", r"\centering\small",
        r"\caption{Fine-tuning and reranking interventions. Fine-tuning is compared",
        r"with original GRID; reranking is compared with the untreated ranking of",
        r"the same saved candidates, always within the same dataset and tokenizer.",
        r"Values are the unweighted median",
        r"[minimum, maximum] across 12 cases (four datasets $\times$ three tokenizers),",
        r"computed from the rounded source tables; seed 42.",
        r"$\Delta R@10=100(R_{m}/R_{0}-1)$ is a relative percentage change;",
        r"$\Delta G@10=G_{m}-G_{0}$ is an absolute brand Gini change",
        r"(negative means fairer). The range is descriptive, not a confidence interval.",
        r"Standard FT, RW and SPRINT fine-tune with the source Adam state resumed",
        r"and a 32-update cap; update zero is eligible.",
        r"$\dagger$ marks recorded FairDiverse trainer evaluations, with different",
        r"training schedules and a fresh Adam optimizer, that have not been freshly reevaluated under the",
        r"common protocol. These differences do not isolate causal method effects.}",
        r"\label{tab:training-interventions}", r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{@{}lcc@{}}", r"\toprule",
        r"Method & $\Delta R@10$ (\%) & $\Delta G@10$ \\", r"\midrule",
    ]
    for index, row in enumerate(summaries):
        if index in (3, 10):
            lines.append(r"\midrule")
        method = row["method"] + (r"$^{\dagger}$" if row["method"] in FAIRDIVERSE else "")
        cells = []
        for field, precision in (("recall_change_percent", 1), ("gini_absolute_change", 4)):
            cells.append("{} [{}, {}]".format(
                signed(row[f"{field}_median"], precision),
                signed(row[f"{field}_min"], precision),
                signed(row[f"{field}_max"], precision)))
        lines.append(f"{method} & {cells[0]} & {cells[1]} " + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    output.write_text("\n".join(lines))


def extreme_cases(cases, field):
    """Return every tied minimum and maximum, allowing float subtraction noise."""
    values = [row[field] for row in cases]
    return [(side, value, [row for row in cases
                          if math.isclose(row[field], value, rel_tol=0, abs_tol=1e-12)])
            for side, value in (("minimum", min(values)), ("maximum", max(values)))]


def case_label(cases):
    grouped = {}
    for row in cases:
        grouped.setdefault(row["dataset"], []).append(row["tokenizer"])
    return "; ".join(
        f"{'Pet' if ds == 'pet-supplies' else DATASETS[ds]} · {'/'.join(tokens)}"
        for ds, tokens in grouped.items())


def draw_case_columns(ax, case_axes, changes, field, offsets):
    """Identify endpoint cases in aligned columns outside the numerical plot."""
    labels = []
    for label_ax, title in zip(case_axes, ("Min case", "Max case")):
        label_ax.set_xlim(0, 1)
        label_ax.set_ylim(len(METHODS) - 0.5, -0.5)
        label_ax.axis("off")
        label_ax.set_title(title, loc="left", fontsize=7, pad=7)
        for boundary in (2.5, 9.5):
            label_ax.axhline(boundary, color="#d5d4d1", linewidth=0.5)
    for i, method in enumerate(METHODS):
        cases = [row for row in changes if row["method"] == method]
        for label_ax, (_, value, tied) in zip(case_axes, extreme_cases(cases, field)):
            indices = [cases.index(row) for row in tied]
            ax.scatter([value] * len(indices), i + offsets[indices], marker="D", s=23,
                       facecolors="none", edgecolors="#262626", linewidths=0.55, zorder=5)
            text = label_ax.text(0.01, i, case_label(tied), va="center", ha="left",
                                 fontsize=6.2, color="#353534")
            labels.append((label_ax, text))
    return labels


def write_extrema(changes, output):
    rows = []
    for method in METHODS:
        cases = [row for row in changes if row["method"] == method]
        for field in ("recall_at_10_relative_change_percent", "brand_gini_at_10_absolute_change"):
            for side, value, tied in extreme_cases(cases, field):
                for row in tied:
                    rows.append(dict(method=method, metric=field, extremum=side,
                                     value=value, dataset=row["dataset"],
                                     tokenizer=row["tokenizer"], tied_cases=len(tied),
                                     plotted_label=case_label(tied)))
    write_csv(output, rows)


def draw_figure(changes, output):
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 7.5, "axes.labelsize": 7.5,
        "xtick.labelsize": 6.5, "ytick.labelsize": 7, "legend.fontsize": 6.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "axes.edgecolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e",
        "pdf.fonttype": 42,
    })
    fig = plt.figure(figsize=(4.8, 5.6))
    layout = fig.add_gridspec(2, 3, width_ratios=(1.9, 1, 1),
                              left=0.17, right=0.99, bottom=0.095, top=0.84,
                              wspace=0.08, hspace=0.38)
    # The same case has the same vertical offset in both panels and every row.
    offsets = np.random.default_rng(42).uniform(-0.12, 0.12, 12)
    panels = (("recall_at_10_relative_change_percent", "Recall@10 change (%)",
               "Positive = higher recall"),
              ("brand_gini_at_10_absolute_change", "Brand Gini@10 change",
               "Negative = fairer"))
    labels = []
    for panel_index, (field, title, xlabel) in enumerate(panels):
        ax = fig.add_subplot(layout[panel_index, 0])
        case_axes = [fig.add_subplot(layout[panel_index, col]) for col in (1, 2)]
        for i, method in enumerate(METHODS):
            values = np.array([r[field] for r in changes if r["method"] == method])
            color = (TEAL if method in RERANKERS else BLUE if method in CONTINUATION else ORANGE)
            ax.scatter(values, i + offsets, s=12, facecolor=color, alpha=0.85,
                       edgecolor="white", linewidth=0.3, zorder=3)
            ax.plot([np.median(values)] * 2, [i - 0.32, i + 0.32],
                    color="#0b0b0b", linewidth=1.2, zorder=4)
        ax.axvline(0, color="#52514e", linewidth=0.6)
        ax.axhline(2.5, color="#d5d4d1", linewidth=0.5)
        ax.axhline(9.5, color="#d5d4d1", linewidth=0.5)
        ax.set_title(title, loc="left", fontsize=8, fontweight="bold", pad=7)
        ax.set_xlabel(xlabel)
        ax.xaxis.set_major_locator(MultipleLocator(20 if field.startswith("recall") else 0.05))
        if field.startswith("brand_gini"):
            ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.grid(axis="x", color="#ecebe8", linewidth=0.5)
        ax.set_axisbelow(True)
        ax.set_ylim(len(METHODS) - 0.5, -0.5)
        ax.margins(x=0.08)
        ax.set_yticks(range(len(METHODS)))
        ax.set_yticklabels([m + (r"$^{\dagger}$" if m in FAIRDIVERSE else "")
                            for m in METHODS])
        labels.extend(draw_case_columns(ax, case_axes, changes, field, offsets))
    handles = [
        Line2D([], [], marker="o", linestyle="", color=BLUE, markersize=3.5,
               label="Fine-tuning (32-update cap)\nResumed Adam state"),
        Line2D([], [], marker="o", linestyle="", color=ORANGE, markersize=3.5,
               label="FairDiverse trainer$^{\\dagger}$\nFresh Adam optimizer"),
        Line2D([], [], marker="o", linestyle="", color=TEAL, markersize=3.5,
               label="Reranking"),
        Line2D([], [], marker="|", linestyle="", color="#0b0b0b", markersize=7,
               markeredgewidth=1.2, label="Median of 12 cases"),
        Line2D([], [], marker="D", linestyle="", color="#262626", markersize=4,
               markerfacecolor="none", markeredgewidth=0.55, label="Extreme cases"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.0), columnspacing=1, handletextpad=0.3)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for label_ax, text in labels:
        box = text.get_window_extent(renderer)
        column = label_ax.get_window_extent(renderer)
        if box.x0 < column.x0 - 1 or box.x1 > column.x1 + 1:
            raise ValueError(f"An extreme-case label exceeds its column: {text.get_text()}")
    boxes = [text.get_window_extent(renderer) for _, text in labels]
    if any(a.overlaps(b) for i, a in enumerate(boxes) for b in boxes[i + 1:]):
        raise ValueError("Extreme-case labels overlap")
    fig.savefig(output, metadata={"Title": "GRID fine-tuning and reranking interventions"})
    fig.savefig(output.with_suffix(".png"), dpi=240)
    plt.close(fig)


def write_figure_caption(output):
    output.write_text("\n".join([
        r"\caption{Effects of fine-tuning and reranking on GRID.",
        r"Each row contains 12 dataset--tokenizer cases (four datasets $\times$ three tokenizers).",
        r"Each case uses its own untreated reference: original GRID for fine-tuning,",
        r"and the ordering of the same saved candidates before reranking.",
        r"Upper panel: relative Recall@10 change, $100(R_m/R_0-1)$; positive means higher recall.",
        r"Lower panel: absolute brand Gini@10 change, $G_m-G_0$; negative means fairer exposure.",
        r"Thin vertical zero lines indicate no change; short black bars show the median across 12 cases.",
        r"Outlined diamonds mark the minimum and maximum changes in each row.",
        r"The adjacent \emph{Min case} and \emph{Max case} columns identify their",
        r"dataset--tokenizer combinations separately for each metric, listing all ties.",
        r"Pet denotes Pet Supplies. Vertical jitter only separates overlapping dots.",
        r"Blue: Standard FT, RW and SPRINT resume GRID's Adam state with a 32-update cap",
        r"(update zero eligible). Orange ($\dagger$): FairDiverse methods use fresh Adam",
        r"optimizers and different training schedules; their recorded evaluations have",
        r"not been freshly recomputed under the common protocol.",
        r"Teal: CPFair, P-MMF and RR rerank without model updates.",
        r"Sports--RVQ reranking and its control use beam width 300; fresh GRID uses 100.",
        r"Comparisons are descriptive; seed 42; calculations use rounded source-table values.}",
        r"\label{fig:training-interventions}", "",
    ]))


def render_table(table, output):
    """Compile a preview at the 122 mm LNCS text width and reject overflow."""
    with tempfile.TemporaryDirectory(prefix="training-table-") as directory:
        stage = Path(directory)
        shutil.copyfile(table, stage / "table.tex")
        (stage / "preview.tex").write_text(
            "\\documentclass[10pt]{article}\n"
            "\\usepackage[paperwidth=142mm,paperheight=170mm,margin=10mm]{geometry}\n"
            "\\usepackage{booktabs}\n\\pagestyle{empty}\n"
            "\\begin{document}\n\\input{table.tex}\n\\end{document}\n")
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "preview.tex"],
            cwd=stage, capture_output=True, text=True)
        if result.returncode or "Overfull" in result.stdout:
            raise RuntimeError(result.stdout)
        shutil.copyfile(stage / "preview.pdf", output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--figure-dir", type=Path, default=ROOT / "artifacts/paper/figures")
    parser.add_argument("--table-dir", type=Path, default=ROOT / "artifacts/paper/tables")
    parser.add_argument("--latex-preview", action="store_true", help="Also compile a table preview with pdflatex")
    args = parser.parse_args()
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    args.table_dir.mkdir(parents=True, exist_ok=True)
    records = parse_tables(args.source)
    changes, summaries = calculate_changes(records)
    write_csv(args.table_dir / "training_interventions_changes.csv", changes)
    write_csv(args.table_dir / "training_interventions_summary.csv", summaries)
    write_extrema(changes, args.table_dir / "training_interventions_extrema.csv")
    table = args.table_dir / "training_interventions_table.tex"
    write_latex(summaries, table)
    figure = args.figure_dir / "fig_training_interventions.pdf"
    draw_figure(changes, figure)
    write_figure_caption(args.figure_dir / "fig_training_interventions_caption.tex")
    if args.latex_preview:
        render_table(table, args.figure_dir / "training_interventions_table_preview.pdf")
    provenance = dict(
        source=str(args.source.resolve()),
        source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_precision="Four decimal places as printed in comparison_tables.tex",
        references={"fine_tuning": ORIGINAL, "reranking": RERANKING_REFERENCE},
        cases_per_method=12, paired_changes=len(changes), seed=42,
        recall_formula="100 * (method_recall_at_10 / reference_recall_at_10 - 1)",
        gini_formula="method_brand_gini_at_10 - reference_brand_gini_at_10",
        aggregation="Unweighted median and observed minimum/maximum over the twelve paired cases",
        caveats=["Ranges are across dataset/tokenizer cases, not confidence intervals.",
                 "The seven FairDiverse methods use recorded evaluations and different training schedules.",
                 "Fine-tuning branches resume source Adam state and have a 32-update cap; selected checkpoints may use fewer updates.",
                 "FairDiverse trainer methods load source model weights but initialize fresh Adam state.",
                 "These comparisons do not isolate causal effects of the fairness penalties.",
                 "CPFair, P-MMF and RR use matched saved-candidate references and no model updates.",
                 "Sports RVQ reranking/control uses beam 300; fresh original GRID uses beam 100.",
                 "Classical models and later pilot tables are excluded."],
    )
    (args.table_dir / "training_interventions_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n")
    print(f"Parsed {len(records)} source rows; computed {len(changes)} paired changes.")
    print(f"Table: {table}\nFigure: {figure}")
    for row in summaries:
        print(f"{row['method']:12s}  median Recall change "
              f"{row['recall_change_percent_median']:+.3f}%  "
              f"median Gini change {row['gini_absolute_change_median']:+.5f}")


if __name__ == "__main__":
    main()
