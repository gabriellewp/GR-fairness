"""Apply the shared full-validation selection policy to an explicit tokenizer case."""
import argparse
from pathlib import Path
from . import selected_rerank_test as selected
from .rerank.fast_pmmf import FastPMMFReranker
from .candidates import file_reference
from .rkmeans_selected_tests import prepare
from .selected_rerank_test import run
from .candidates import load


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--tokenizer", choices=("rkmeans", "rqvae", "rvq"), required=True)
    parser.add_argument("--artifact-dir", required=True)
    for method in ("rr", "cpfair", "pmmf"):
        parser.add_argument(f"--{method}-validation", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-relative-recall-loss", type=float, default=.05)
    args = parser.parse_args()
    sources = {m: getattr(args, m+"_validation") for m in ("rr", "cpfair", "pmmf")}
    candidate_dir, selections = prepare(args.dataset, args.max_relative_recall_loss,
                                         candidate_dir=args.artifact_dir, sources=sources)
    validation = load(candidate_dir / "validation_candidates.json")
    if validation.metadata.get("tokenizer") != args.tokenizer:
        raise ValueError("Candidate tokenizer differs from requested tokenizer")
    rr = selections["rr"]["parameters"]
    original = selected.FairDiverseReranker
    selected.FairDiverseReranker = FastPMMFReranker
    try:
        run(selections["cpfair"]["parameters"]["lambda"],
            selections["pmmf"]["parameters"]["lambda"], args.output_dir,
            dataset=args.dataset, artifact_dir=candidate_dir, rr_alpha=rr["alpha"], rr_floor=rr["floor"],
            methods=list(selections), selection_provenance=dict(tokenizer=args.tokenizer,
                selection_cutoff=5, max_relative_recall_loss=args.max_relative_recall_loss,
                rule="minimum declared fairness axis within relative Recall@5 budget",
                selections=selections, projection="exact_sorted_l1",
                projection_source=file_reference(Path(__file__).parent / "rerank/fast_pmmf.py")))
    finally:
        selected.FairDiverseReranker = original


if __name__ == "__main__":
    main()
