"""Prepare and execute the twelve matched dataset/tokenizer fairness sweeps."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from additional_baselines.candidates import file_reference
from additional_baselines.ifairlrs_grid import write_json
from additional_baselines.prepare_beauty import load_beauty
from additional_baselines.prepare_rerank_candidates import SUITES, TOKENIZER_SUITES, source_export
from additional_baselines.tune_continuation import run_sweep

DATASETS = ("amazon-beauty", "amazon-sports", "amazon-toys", "amazon-pet-supplies")
TOKENIZERS = ("rkmeans", "rqvae", "rvq")
CODE = ("additional_baselines/continuation_grid.py", "additional_baselines/continuation_selection.py",
        "additional_baselines/ifairlrs.py", "additional_baselines/ifairlrs_grid.py",
        "additional_baselines/sprint_grid.py", "additional_baselines/tune_continuation.py",
        "scripts/run_joint_fairness_matrix.py")


def prepare_case(index, dataset, tokenizer, output_root):
    artifacts = ROOT / "additional_baselines/artifacts"
    if dataset == "amazon-beauty" and tokenizer == "rkmeans":
        control = artifacts / "beauty_native_control_lr1e3/continuation.json"
    else:
        suite = SUITES[dataset] if tokenizer == "rkmeans" else TOKENIZER_SUITES[dataset, tokenizer]
        control = artifacts / suite / "ordinary/continuation.json"
    export = source_export(dataset, tokenizer)
    data = ROOT / "recommendation/processed_dataset" / dataset
    options = dict(control_report=str(control), export_dir=str(export),
        output_dir=str(output_root / f"{dataset}_{tokenizer}"), dataset=dataset, tokenizer=tokenizer,
        data_dir=str(data), objective="both", max_relative_recall_loss=0.02,
        device="cuda", evaluate_winners=True)
    plan = run_sweep(**options, dry_run=True)
    verified, _, summary = load_beauty(export_dir=export, data_dir=data, tokenizer=tokenizer,
                                     dataset=dataset, topk=(1,), check_dense_memory=False)
    identity = json.loads(control.read_text())["identity"]
    for key in ("checkpoint", "sid_mapping"):
        if verified.metadata[key] != identity[key]:
            raise ValueError(f"Verified source differs from ordinary control: {dataset}/{tokenizer}/{key}")
    inputs = {name: file_reference(data / f"{dataset}.{suffix}") for name, suffix in
              (("train", "train"), ("validation", "valid.ranking"), ("test", "test.ranking"))}
    return dict(index=index, dataset=dataset, tokenizer=tokenizer, sweep_options=options,
                plan=plan, inputs=inputs, input_verification=summary,
                control_trace=file_reference(control.with_name("validation_trace.json")))


def prepare(manifest_dir, output_root):
    folder, output_root = Path(manifest_dir).resolve(), Path(output_root).resolve()
    if folder.exists() or output_root.exists():
        raise FileExistsError("Use fresh manifest and experiment output directories")
    combinations = [(i, ds, tok) for i, (ds, tok) in enumerate(
        (ds, tok) for ds in DATASETS for tok in TOKENIZERS)]
    cases = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        tasks = {pool.submit(prepare_case, i, ds, tok, output_root): (ds, tok)
                 for i, ds, tok in combinations}
        for task in as_completed(tasks):
            case = task.result()
            cases.append(case)
            print(f"Verified {case['dataset']}/{case['tokenizer']}: "
                  f"{case['input_verification']['request_count']} test requests; "
                  f"{len(case['plan']['trials'])} trials", flush=True)
    cases.sort(key=lambda case: case["index"])
    manifest = dict(datasets=list(DATASETS), tokenizers=list(TOKENIZERS), cases=cases,
                    case_count=len(cases), trial_count=sum(len(c['plan']['trials']) for c in cases),
                    max_parallel_sweeps=4, output_root=str(output_root),
                    code={path: file_reference(ROOT / path) for path in CODE},
                    note="Full validation; seed 42; both fairness axes; 2% relative Recall@5 budget; test only frozen winners")
    folder.mkdir(parents=True)
    write_json(folder / "manifest.json", manifest)
    return folder / "manifest.json"


def run_case(manifest_path, index):
    manifest = json.loads(Path(manifest_path).read_text())
    if not 0 <= index < len(manifest["cases"]):
        raise ValueError("Array index is outside the prepared matrix")
    case = manifest["cases"][index]
    assert case["index"] == index
    references = [*manifest["code"].values(), *case["inputs"].values(),
                  case["control_trace"], case["plan"]["control"], case["plan"]["export_config"]]
    for ref in references:
        if file_reference(ref["path"]) != ref:
            raise ValueError(f"Prepared input or implementation changed: {ref['path']}")
    print(f"Starting case {index}: {case['dataset']}/{case['tokenizer']}", flush=True)
    print(json.dumps(run_sweep(**case["sweep_options"]), indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--manifest-dir", required=True)
    prep.add_argument("--output-root", required=True)
    run = sub.add_parser("run-case")
    run.add_argument("--manifest", required=True)
    run.add_argument("--index", required=True, type=int)
    args = parser.parse_args()
    if args.command == "prepare":
        print(prepare(args.manifest_dir, args.output_root), flush=True)
    else:
        run_case(args.manifest, args.index)


if __name__ == "__main__":
    main()
