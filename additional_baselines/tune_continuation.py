"""Tune RW/SPRINT on validation, then optionally evaluate frozen winners once."""

import argparse
import json
from pathlib import Path

from .candidates import file_reference
from .continuation_grid import evaluate_selected_continuation, run_continuation
from .continuation_selection import checkpoint_key, select_checkpoint, selection_policy
from .ifairlrs_grid import write_json
from .prepare_beauty import ROOT


def default_trials():
    trials = []
    for target, group in (("history", "popularity"), ("uniform", "popularity"), ("uniform", "brand")):
        for strength in (0.25, 0.5, 1.0):
            trials.append(dict(name=f"rw_{target}_{group}_{str(strength).replace('.', 'p')}",
                               mode="rw", group=group, rw_target=target, rw_strength=strength))
    base = dict(attention_weight=1.0, ffn_weight=0.001, embedding_weight=0.1,
                temperature=4.0, iterations=1)
    for scale in (0.001, 0.01, 0.1, 1.0):
        settings = {k: v * scale if k.endswith("_weight") else v for k, v in base.items()}
        trials.append(dict(name=f"sprint_scale_{str(scale).replace('.', 'p')}", mode="sprint",
                           sprint_settings=settings))
    for term in ("attention", "ffn", "embedding"):
        settings = {k: (v * 0.1 if k == term + "_weight" else 0.0) if k.endswith("_weight") else v
                    for k, v in base.items()}
        trials.append(dict(name=f"sprint_{term}_only", mode="sprint", sprint_settings=settings))
    return trials


def validate_trials(trials):
    from .sprint_grid import resolve_settings
    if not trials:
        raise ValueError("At least one tuning trial is required")
    seen = set()
    allowed = {"name", "mode", "group", "rw_target", "rw_strength", "sprint_settings"}
    for trial in trials:
        name = trial.get("name", "")
        if not name or not all(c.isalnum() or c in "_-" for c in name) or name in seen:
            raise ValueError("Trial names must be unique safe directory names")
        seen.add(name)
        if set(trial) - allowed or trial.get("mode") not in {"rw", "sprint"}:
            raise ValueError("Trials may only configure RW/SPRINT method-specific settings")
        if trial.get("group", "popularity") not in {"popularity", "brand"}:
            raise ValueError("Invalid RW group")
        if trial["mode"] == "sprint":
            if any(k in trial for k in ("rw_target", "rw_strength")):
                raise ValueError("RW settings cannot configure SPRINT")
            resolve_settings(trial.get("sprint_settings", {}))
        else:
            import math
            if "sprint_settings" in trial or trial.get("rw_target", "history") not in {"history", "uniform"}:
                raise ValueError("Invalid RW trial")
            strength = trial.get("rw_strength", 1.0)
            if not math.isfinite(strength) or not 0 <= strength <= 1:
                raise ValueError("RW strength must be in [0, 1]")


def choose_winners(completed, control_report):
    winners = {}
    for mode in ("rw", "sprint"):
        eligible = []
        for index, trial in enumerate(completed):
            if trial["settings"]["mode"] != mode:
                continue
            report = trial["report"]
            if report["selected_update"] is None:
                continue
            row = dict(update=report["selected_update"], metrics=report["selected_validation_metrics"])
            key = checkpoint_key(row, report["selection"])
            if key is not None:
                eligible.append((key, -index, trial))
        if eligible:
            chosen = max(eligible, key=lambda item: item[:2])[2]
            winners[mode] = dict(status="validation_fairness_improved", trial=chosen["settings"],
                                 variant=chosen["report"].get("variant", mode),
                                 run_dir=chosen["run_dir"], selected_update=chosen["report"]["selected_update"],
                                 selected_checkpoint=chosen["report"]["selected_checkpoint"],
                                 selection=chosen["report"]["selection"])
        else:
            winners[mode] = dict(status="no_feasible_improvement_keep_ordinary",
                                 ordinary_control=str(control_report))
    return winners


def run_sweep(*, control_report, export_dir, output_dir, dataset, tokenizer, data_dir,
              objective="both", max_relative_recall_loss=0.02, device="cuda",
              trials=None, dry_run=False, evaluate_winners=False):
    if objective not in {"both", "popularity", "brand"}:
        raise ValueError("Sweep must declare a fairness objective")
    selection_policy("recall", max_relative_recall_loss=max_relative_recall_loss)
    control_report = Path(control_report).resolve()
    control = json.loads(control_report.read_text())
    if control.get("mode") != "ordinary":
        raise ValueError("Sweep requires an ordinary control")
    # A fairness-selected ordinary branch would not be a matched accuracy control.
    trace = json.loads(control_report.with_name("validation_trace.json").read_text())
    best = select_checkpoint(trace, selection_policy())
    if best["update"] != control["selected_update"] or best["metrics"] != control["selected_validation_metrics"]:
        raise ValueError("Control must select maximum validation Recall@5 with earlier ties")
    trials = default_trials() if trials is None else trials
    validate_trials(trials)
    identity = control["identity"]
    common = dict(control_report=str(control_report), export_dir=str(Path(export_dir).resolve()),
                  data_dir=str(Path(data_dir).resolve()), dataset=dataset, tokenizer=tokenizer,
                  seed=identity["seed"], max_updates=identity["max_updates"],
                  validation_interval=identity["validation_interval"],
                  validation_requests=identity["validation_requests"], train_users=identity["train_users"],
                  generation_beam_width=identity["generation_beam_width"],
                  learning_rate=identity["learning_rate"], optimizer_state=identity["optimizer_state"],
                  selection_objective=objective, max_relative_recall_loss=max_relative_recall_loss,
                  require_control_improvement=False, evaluate_test=False, device=device)
    if identity["validation_requests"] is not None or identity["train_users"] is not None:
        raise ValueError("Tuning requires full validation and training inputs, not smoke-test controls")
    # Check the named export against the control before starting any trial.
    import yaml
    export_config_path = Path(common["export_dir"]) / "config.yaml"
    export_config = yaml.safe_load(export_config_path.read_text())
    if (export_config.get("dataset"), export_config.get("grid_tokenizer")) != (dataset, tokenizer):
        raise ValueError("Source export dataset/tokenizer differs from sweep")
    for key, ref_key in (("grid_checkpoint_path", "checkpoint"), ("sid_path", "sid_mapping")):
        path = Path(export_config[key])
        ref = file_reference(path if path.is_absolute() else ROOT / path)
        if ref != identity[ref_key]:
            raise ValueError("Source export differs from matched ordinary checkpoint/SID mapping")
    plan = dict(control=file_reference(control_report), common=common, trials=trials,
                export_config=file_reference(export_config_path),
                zero_strength_reference="Matched ordinary control; zero-strength settings need no duplicate training",
                test_policy="No per-trial test evaluation; freeze winners before any optional test evaluation")
    if dry_run:
        return plan
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    write_json(destination / "plan.json", plan)
    completed = []
    for settings in trials:
        folder = destination / settings["name"]
        kwargs = {k: v for k, v in settings.items() if k != "name"}
        report = run_continuation(output_dir=folder, **common, **kwargs)
        completed.append(dict(settings=settings, run_dir=str(folder), report=report))
        write_json(destination / f"completed_{settings['name']}.json", completed[-1])
    write_json(destination / "progress.json", completed)
    winners = choose_winners(completed, control_report)
    selection = dict(control=plan["control"], objective=objective,
                     max_relative_recall_loss=max_relative_recall_loss, winners=winners,
                     trial_count=len(completed), selection_uses_test=False)
    selection_path = destination / "selection.json"
    write_json(selection_path, selection)
    if evaluate_winners:
        tests = {}
        for mode, winner in winners.items():
            if winner["status"] != "validation_fairness_improved":
                continue
            folder = destination / "winner_tests" / mode
            evaluate_selected_continuation(winner["run_dir"], folder, device=device)
            tests[mode] = file_reference(folder / "test_results.json")
        write_json(destination / "test_evaluation.json",
                   dict(frozen_selection=file_reference(selection_path), tests=tests))
    return selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-report", required=True)
    parser.add_argument("--export-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset", default="amazon-beauty")
    parser.add_argument("--tokenizer", choices=("rkmeans", "rqvae", "rvq"), default="rkmeans")
    parser.add_argument("--data-dir")
    parser.add_argument("--objective", choices=("both", "popularity", "brand"), default="both")
    parser.add_argument("--max-relative-recall-loss", type=float, default=0.02)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--trials", help="Optional JSON list replacing the default validation search")
    parser.add_argument("--dry-run", action="store_true", help="Print the complete trial plan without training or writing outputs")
    parser.add_argument("--evaluate-winners", action="store_true", help="Evaluate only the frozen per-method winners on test")
    args = vars(parser.parse_args())
    args["data_dir"] = args["data_dir"] or str(ROOT / "recommendation/processed_dataset" / args["dataset"])
    if args["trials"] is not None:
        args["trials"] = json.loads(Path(args["trials"]).read_text())
    print(json.dumps(run_sweep(**args), indent=2))


if __name__ == "__main__":
    main()
