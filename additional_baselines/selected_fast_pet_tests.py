"""Pet validation-selected final tests using the verified exact P-MMF projection."""
import argparse
import json
from pathlib import Path

from . import selected_rerank_test as selected
from .rkmeans_selected_tests import prepare
from .rerank.fast_pmmf import FastPMMFReranker
from .candidates import file_reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",required=True,type=Path)
    parser.add_argument("--max-relative-recall-loss",type=float,default=.05)
    parser.add_argument("--selection-only",action="store_true")
    args = parser.parse_args()
    candidates,selections = prepare("amazon-pet-supplies",args.max_relative_recall_loss)
    if args.selection_only:
        print(json.dumps(selections,indent=2))
        return
    rr = selections["rr"]["parameters"]
    original = selected.FairDiverseReranker
    selected.FairDiverseReranker = FastPMMFReranker
    try:
        selected.run(selections["cpfair"]["parameters"]["lambda"], selections["pmmf"]["parameters"]["lambda"],
            args.output_dir,dataset="amazon-pet-supplies",artifact_dir=candidates,
            rr_alpha=rr["alpha"],rr_floor=rr["floor"],methods=list(selections),
            selection_provenance=dict(selection_cutoff=5,selections=selections,
                max_relative_recall_loss=args.max_relative_recall_loss,
                projection="exact_sorted_l1",projection_source=file_reference(Path(__file__).parent/"rerank/fast_pmmf.py")))
    finally:
        selected.FairDiverseReranker = original


if __name__ == "__main__":
    main()
