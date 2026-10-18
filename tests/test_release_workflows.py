"""Exercise portable configuration and Slurm handoffs without running jobs."""
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("release_config", ROOT / "scripts/configure_experiment.py")
config_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config_module)


@pytest.mark.parametrize("dataset", config_module.DATASETS)
@pytest.mark.parametrize("model", ["GRID", "SASRec", "GRU4Rec"])
def test_fresh_configs_keep_matched_data_and_valid_architecture(dataset, model):
    options = SimpleNamespace(dataset=dataset, tokenizer="rkmeans", model=model,
        grid_data_dir=None, checkpoint="GRID/outputs/fixture.ckpt", sid_path=None,
        seed=42, device="cpu", epochs=3, eval_batch_size=2)
    result = config_module.build(options)
    config = result[0]["config"]
    assert config["metadata_text_mode"] == "complete_metadata"
    assert config["use_wandb"] is False
    assert config["padding_item_id"] == -1
    if model == "GRID":
        assert config["grid_eval_only"] and not config["grid_fair_finetune"]
        assert config["grid_eval_generation_batch_size"] == 2
        assert config["sequence_length"] == config["history_length"] * config["num_hierarchies"]
    else:
        assert config["model"] == model
        assert "complete_metadata" in config["expected_grid_source_dir"]
        assert config["embedding_size"] % config["n_heads"] == 0
        assert not any(key.startswith("grid_") for key in config)


def test_array_submission_preserves_noncontiguous_ids_and_resource_arguments(tmp_path):
    config = tmp_path / "config with spaces.json"
    config.write_text(json.dumps([{"exp_id": 57}, {"exp_id": 80}]))
    result = subprocess.run(["bash", str(ROOT / "scripts/slurm/submit_experiments.sh"),
        "--config_file", str(config), "--exp_ids", "80 57", "--max_concurrent", "3",
        "--mem", "48G", "--dependency", "afterok:123", "--dry-run", "--", "--seed", "7"],
        cwd=tmp_path, env={**os.environ, "FAIRDIVERSE_ROOT": str(ROOT), "FAIRDIVERSE_PYTHON": sys.executable},
        capture_output=True, text=True, check=True)
    assert "--array=80\\,57%3" in result.stdout
    assert "--mem=48G" in result.stdout
    assert "--dependency=afterok:123" in result.stdout
    assert "--seed 7" in result.stdout
    invalid = subprocess.run(["bash", str(ROOT / "scripts/slurm/submit_experiments.sh"),
        "--config_file", str(config), "--exp_ids", "99", "--dry-run"],
        env={**os.environ, "FAIRDIVERSE_ROOT": str(ROOT), "FAIRDIVERSE_PYTHON": sys.executable},
        capture_output=True, text=True)
    assert invalid.returncode != 0
    assert "exist in the config" in invalid.stderr


def test_experiment_wrapper_uses_submission_root_when_running_a_spool_copy(tmp_path):
    spool = tmp_path / "job.sh"
    spool.write_text((ROOT / "scripts/slurm/experiment.sh").read_text())
    interpreter = tmp_path / "python"
    interpreter.write_text('#!/bin/bash\nprintf "%s\\n" "$PWD" "$@" > "$ARG_LOG"\n')
    interpreter.chmod(0o755)
    log = tmp_path / "args"
    environment = {**os.environ, "SLURM_SUBMIT_DIR": str(ROOT), "SLURM_ARRAY_TASK_ID": "80",
        "SLURM_JOB_ID": "123", "FAIRDIVERSE_PYTHON": str(interpreter), "ARG_LOG": str(log)}
    environment.pop("FAIRDIVERSE_ROOT", None)
    subprocess.run(["bash", str(spool), "config.json"], cwd=tmp_path, env=environment, check=True)
    arguments = log.read_text().splitlines()
    assert arguments[0] == str(ROOT)
    assert arguments[arguments.index("--exp_id")+1] == "80"
    assert arguments[arguments.index("--slurm_job_id")+1] == "123"
