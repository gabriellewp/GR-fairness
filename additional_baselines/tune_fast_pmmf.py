"""Opt-in fast projection with the unchanged validation sweep/evaluation pipeline."""
import argparse
import json
from pathlib import Path

from . import rerank
from . import tune_beauty_reranking as tuner
from .rerank.fast_pmmf import FastPMMFReranker
from .candidates import file_reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--activity", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--limit-requests", type=int)
    args = parser.parse_args()
    # Process-local injection preserves the original tuner and pending jobs.
    # Only its P-MMF factory changes; parsing, controls, metrics and grid remain.
    original = rerank.FairDiverseReranker
    rerank.FairDiverseReranker = FastPMMFReranker
    try:
        result = tuner.run("pmmf",args.output_dir,candidates=args.candidates,split=args.split,
            activity=args.activity,task_index=args.task_index,limit_requests=args.limit_requests)
    finally:
        rerank.FairDiverseReranker = original
    path = args.output_dir / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["projection_backend"] = "exact_sorted_l1"
    manifest["implementation_sources"]["fast_projection"] = file_reference(Path(__file__).parent / "rerank/fast_pmmf.py")
    manifest["implementation_sources"]["fast_tuner"] = file_reference(Path(__file__))
    path.write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
