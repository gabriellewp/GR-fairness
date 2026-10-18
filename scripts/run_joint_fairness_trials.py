"""Run independent trials and finalize the verified joint-fairness matrix."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from additional_baselines.candidates import file_reference
from additional_baselines.continuation_grid import run_continuation, evaluate_selected_continuation
from additional_baselines.continuation_selection import select_checkpoint
from additional_baselines.ifairlrs_grid import write_json
from additional_baselines.tune_continuation import choose_winners


def verify(references):
    for ref in references:
        if file_reference(ref["path"]) != ref:
            raise ValueError(f"Prepared input or implementation changed: {ref['path']}")


def initialize(matrix_path):
    matrix_path = Path(matrix_path).resolve()
    matrix = json.loads(matrix_path.read_text())
    verify(matrix["code"].values())
    entries = []
    for case in matrix["cases"]:
        output = Path(case["sweep_options"]["output_dir"])
        output.mkdir(parents=True)
        write_json(output / "plan.json", case["plan"])
        for trial_index, settings in enumerate(case["plan"]["trials"]):
            entries.append(dict(index=len(entries), case_index=case["index"],
                                trial_index=trial_index, settings=settings))
    # Interleave datasets/tokenizers so every case starts before later trials
    # of the first case consume the queue.
    entries.sort(key=lambda entry: (entry["trial_index"], entry["case_index"]))
    for index, entry in enumerate(entries):
        entry["index"] = index
    manifest = dict(matrix=file_reference(matrix_path), trials=entries,
                    runner=file_reference(__file__), max_parallel_gpu_tasks=4,
                    case_count=len(matrix["cases"]), trial_count=len(entries))
    path = matrix_path.with_name("trial_manifest.json")
    write_json(path, manifest)
    return path


def load_manifest(path):
    manifest = json.loads(Path(path).read_text())
    verify([manifest["matrix"], manifest["runner"]])
    matrix = json.loads(Path(manifest["matrix"]["path"]).read_text())
    verify(matrix["code"].values())
    return manifest, matrix


def verify_case(case):
    verify([*case["inputs"].values(), case["control_trace"],
            case["plan"]["control"], case["plan"]["export_config"]])


def run_trial(path, index):
    manifest, matrix = load_manifest(path)
    if not 0 <= index < len(manifest["trials"]):
        raise ValueError("Trial array index is outside the prepared matrix")
    entry = manifest["trials"][index]
    case = matrix["cases"][entry["case_index"]]
    verify_case(case)
    settings = entry["settings"]
    assert settings == case["plan"]["trials"][entry["trial_index"]]
    output = Path(case["sweep_options"]["output_dir"])
    run_dir = output / settings["name"]
    print(f"Starting trial {index}: {case['dataset']}/{case['tokenizer']}/{settings['name']}", flush=True)
    options = {k: v for k, v in settings.items() if k != "name"}
    report = run_continuation(output_dir=run_dir, **case["plan"]["common"], **options)
    write_json(output / f"completed_{settings['name']}.json",
               dict(settings=settings, run_dir=str(run_dir), report=report))
    print(f"Completed trial {index}: {report['status']}", flush=True)


def finalize_case(path, index):
    manifest, matrix = load_manifest(path)
    if not 0 <= index < len(matrix["cases"]):
        raise ValueError("Finalization index is outside the prepared matrix")
    case = matrix["cases"][index]
    verify_case(case)
    output = Path(case["sweep_options"]["output_dir"])
    required = [output / f"completed_{s['name']}.json" for s in case["plan"]["trials"]]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        write_json(output / "incomplete.json", dict(missing=missing, selection_performed=False,
                                                    test_evaluated=False))
        raise RuntimeError(f"Cannot select a winner: {len(missing)} of {len(required)} trials are incomplete")
    completed = [json.loads(p.read_text()) for p in required]
    control_path = case["plan"]["common"]["control_report"]
    control = json.loads(Path(control_path).read_text())
    for settings, trial in zip(case["plan"]["trials"], completed):
        report = trial["report"]
        if trial["settings"] != settings or report["identity"] != control["identity"]:
            raise ValueError("Completed trial differs from frozen settings or matched control")
        saved = json.loads((Path(trial["run_dir"]) / "continuation.json").read_text())
        if saved != report:
            raise ValueError("Completed trial receipt differs from saved continuation")
        trace = json.loads((Path(trial["run_dir"]) / "validation_trace.json").read_text())
        best = select_checkpoint(trace, report["selection"])
        if (best["update"] if best else None) != report["selected_update"]:
            raise ValueError("Trial selection differs from saved validation trace")
    winners = choose_winners(completed, control_path)
    selection = dict(dataset=case["dataset"], tokenizer=case["tokenizer"],
                     control=case["plan"]["control"], objective="both",
                     max_relative_recall_loss=0.02, winners=winners,
                     trial_count=len(completed), selection_uses_test=False)
    write_json(output / "progress.json", completed)
    selection_path = output / "selection.json"
    write_json(selection_path, selection)
    tests = {}
    for mode, winner in winners.items():
        if winner["status"] != "validation_fairness_improved":
            continue
        folder = output / "winner_tests" / mode
        evaluate_selected_continuation(winner["run_dir"], folder,
                                       device=case["plan"]["common"]["device"])
        tests[mode] = file_reference(folder / "test_results.json")
    write_json(output / "test_evaluation.json",
               dict(frozen_selection=file_reference(selection_path), tests=tests))
    print(json.dumps(selection, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("initialize", "trial", "finalize"))
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    if args.phase == "initialize":
        print(initialize(args.manifest))
    elif args.index is None:
        parser.error("--index is required for trial/finalize")
    elif args.phase == "trial":
        run_trial(args.manifest, args.index)
    else:
        finalize_case(args.manifest, args.index)


if __name__ == "__main__":
    main()
