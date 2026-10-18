"""Run a tokenizer's matched ordinary, RW and SPRINT GRID continuations."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

from .continuation_grid import run_continuation
from .prepare_beauty import ROOT, DEFAULT_EXPORT


DEFAULT_EXPORT_PYTHON = sys.executable
EXPORT_CONFIGS = {
    "amazon-beauty": "configs/reproduction/experiments_config_beauty_continuation_exports.json",
    "amazon-sports": "configs/reproduction/experiments_config_sports_continuation_exports.json",
    "amazon-toys": "configs/reproduction/experiments_config_amazon-toys_grid_tokenizers.json",
    "amazon-pet-supplies": "configs/reproduction/experiments_config_pet_continuation_exports.json",
}


def run_suite(tokenizer, output_dir, export_dir=None, dataset="amazon-beauty",
              generation_beam_width=None, require_control_improvement=True):
    if generation_beam_width is not None and (type(generation_beam_width) is not int or generation_beam_width <= 0):
        raise ValueError("generation_beam_width must be a positive integer")
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    if dataset not in EXPORT_CONFIGS:
        raise ValueError(f"Unsupported GRID dataset: {dataset}")
    entries = json.loads((ROOT / EXPORT_CONFIGS[dataset]).read_text())
    experiment = next((e for e in entries if e["config"]["grid_tokenizer"] == tokenizer
                       and e["config"].get("grid_eval_only")), None)
    if experiment is None and dataset == "amazon-beauty" and tokenizer == "rkmeans":
        export_dir = export_dir or DEFAULT_EXPORT
    if experiment is None and dataset == "amazon-pet-supplies" and tokenizer == "rkmeans":
        # The existing Pet Supplies configuration trains a fairness branch.
        # Derive an evaluation-only source export from its original checkpoint.
        experiment = next(e for e in entries if e["config"]["grid_tokenizer"] == tokenizer)
        experiment["config"].update(grid_eval_only=True, grid_fair_finetune=False,
                                    grid_fair_finetune_steps=0, **{"fair-rank": False})
    if experiment is None and export_dir is None:
        raise ValueError(f"No evaluation-only {tokenizer} source for {dataset}")
    # The source export must contain the compact scores and targets verified by load_beauty.
    if experiment is not None:
        experiment["config"]["store_scores"] = True
    data_dir = f"recommendation/processed_dataset/{dataset}"
    required = ("grid_data_manifest.json", "iid2item.json", "iid2pid.json",
                f"{dataset}.train", f"{dataset}.valid.ranking", f"{dataset}.test.ranking")
    missing = [name for name in required if not (ROOT / data_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Processed {dataset} continuation inputs missing: {', '.join(missing)}")
    # Check the complete-metadata starting artifacts before creating outputs.
    if experiment is not None:
        for key in ("grid_data_dir", "grid_item_mapping_path", "sid_path", "grid_checkpoint_path"):
            if not (ROOT / experiment["config"][key]).exists():
                raise FileNotFoundError(experiment["config"][key])
    elif not (ROOT / export_dir / "config.yaml").exists():
        raise FileNotFoundError(f"Source export config is missing: {export_dir}/config.yaml")
    export_python = os.environ.get("GRID_EXPORT_PYTHON", DEFAULT_EXPORT_PYTHON)
    if export_dir is None:
        # FairDiverse eagerly imports sklearn and its other training dependencies.
        # Check its separate runtime before creating an output directory.
        subprocess.run([export_python, "-c",
                        "from fairdiverse.recommendation.trainer import RecTrainer; "
                        "import tensorflow, rootutils"], cwd=ROOT, check=True)
    destination.mkdir(parents=True)
    if export_dir is None:
        # The source export is a prerequisite, not a trained continuation branch.
        # Unique names prevent replacing historical FairDiverse exports.
        run_id = os.environ.get("SLURM_JOB_ID", "manual") + "-" + uuid.uuid4().hex[:12]
        label = dataset.removeprefix("amazon-")
        name = f"{label}-{tokenizer}-complete-metadata-continuation-export-{run_id}"
        experiment["config"]["log_name"] = name
        config_path = destination / "export_experiment.json"
        config_path.write_text(json.dumps([experiment], indent=2) + "\n")
        log_path = ROOT / "output" / f"{run_id}_export.out"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        command = [export_python, "-u", str(ROOT / "run_experiment.py"),
                   "--config_file", str(config_path), "--exp_id", str(experiment["exp_id"]),
                   "--slurm_job_id", run_id, "--slurm_array_task_id", "export"]
        print(f"Preparing {tokenizer} starting scores; log: {log_path}", flush=True)
        with log_path.open("x") as stream:
            try:
                subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
            except subprocess.CalledProcessError as error:
                raise RuntimeError(f"Starting-score export failed; see {log_path}") from error
        exports = list((ROOT / "fairdiverse/recommendation/log").glob(f"*_{name}"))
        if len(exports) != 1:
            raise RuntimeError(f"Expected exactly one source export; found {exports}")
        export_dir = exports[0]
        missing_exports = [filename for filename in ("ranking_scores.npz", "ground_truth.npz")
                           if not (export_dir / filename).is_file()]
        if missing_exports:
            raise RuntimeError(
                f"Source export {export_dir} is missing {', '.join(missing_exports)}; "
                f"check store_scores in {config_path} and {log_path}"
            )
    common = dict(tokenizer=tokenizer, dataset=dataset, export_dir=str(Path(export_dir).resolve()),
                  data_dir=data_dir, device="cuda", seed=42, max_updates=32,
                  validation_interval=8, learning_rate=0.001, optimizer_state="require",
                  require_control_improvement=require_control_improvement)
    if generation_beam_width is not None:
        common["generation_beam_width"] = generation_beam_width
    (destination / "suite_config.json").write_text(json.dumps(common, indent=2) + "\n")
    control_dir = destination / "ordinary"
    control = run_continuation(mode="ordinary", output_dir=control_dir, **common)
    if require_control_improvement and control["selected_update"] == 0:
        print("Ordinary control did not improve validation Recall@5; RW and SPRINT remain gated.", flush=True)
        return
    for mode in ("rw", "sprint"):
        run_continuation(mode=mode, output_dir=destination / mode,
                         control_report=control_dir / "continuation.json", **common)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", choices=("rkmeans", "rqvae", "rvq"), required=True)
    parser.add_argument("--dataset", choices=tuple(EXPORT_CONFIGS), default="amazon-beauty")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--export-dir", help="Optional existing verified source export; otherwise create one")
    parser.add_argument("--generation-beam-width", type=int)
    parser.add_argument("--allow-unimproved-control", dest="require_control_improvement",
                        action="store_false", help="Run matched RW/SPRINT even when the control selects update zero")
    args = parser.parse_args()
    os.chdir(ROOT)
    run_suite(args.tokenizer, args.output_dir, args.export_dir, args.dataset,
              args.generation_beam_width, args.require_control_improvement)


if __name__ == "__main__":
    main()
