"""Reject metadata-mode mixing before data conversion or experiment execution."""
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from scripts import configure_experiment, curate_complete_metadata_configs, sync_grid_independent_experiments
from fairdiverse.recommendation import grid_data_bridge as bridge

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("arguments", [
    ["--metadata-text-mode", "description"],
    ["--tiger-ckpt", "outputs/description/old.ckpt"],
    ["--data-dir", "GRID/data/description"],
])
def test_pipeline_rejects_description_only_before_creating_outputs(tmp_path, arguments):
    result = subprocess.run(["bash", str(ROOT / "GRID/run_independent_pipeline.sh"),
        *arguments, "--preprocess-only"], cwd=tmp_path,
        env={**os.environ, "FAIRDIVERSE_GRID_ROOT": str(ROOT / "GRID")}, capture_output=True, text=True)
    assert result.returncode == 2
    assert "requires complete_metadata" in result.stderr
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("field,value", [
    ("grid_data_dir", "GRID/data/description"),
    ("checkpoint", "GRID/outputs/description/old.ckpt"),
    ("sid_path", "GRID/outputs/description/sids.pt"),
])
def test_fresh_configuration_rejects_description_artifacts(field, value):
    args = SimpleNamespace(dataset="amazon-beauty", model="GRID", tokenizer="rkmeans",
        grid_data_dir=None, checkpoint="GRID/outputs/complete_metadata/fixture.ckpt", sid_path=None,
        seed=42, device="cpu", epochs=3, eval_batch_size=2)
    setattr(args, field, value)
    with pytest.raises(ValueError, match="complete_metadata"):
        configure_experiment.build(args)


def test_bridge_rejects_a_description_run_configuration():
    with pytest.raises(ValueError, match="requires complete_metadata"):
        bridge._merged_process_config({"dataset": "amazon-beauty", "metadata_text_mode": "description"})


@pytest.mark.parametrize("recorded_mode", ["description", None])
def test_bridge_checks_actual_split_metadata_before_reusing_data(tmp_path, monkeypatch, recorded_mode):
    source = tmp_path / "GRID/data/complete_metadata"
    source.mkdir(parents=True)
    (source / "split_config.json").write_text(json.dumps({
        "strategy": "temporal_loo", "metadata_text_mode": recorded_mode}))
    monkeypatch.setattr(bridge, "__file__", str(tmp_path / "fairdiverse/recommendation/grid_data_bridge.py"))
    monkeypatch.setattr(bridge, "_merged_process_config", lambda config: config)
    monkeypatch.setattr(bridge, "_ensure_grid_tfrecords", lambda *args: "reused")
    with pytest.raises(ValueError, match="split_config.json must confirm complete_metadata"):
        bridge.ensure_grid_fairdiverse_data({"dataset": "amazon-beauty",
            "grid_data_dir": "GRID/data/complete_metadata", "metadata_text_mode": "complete_metadata"})
    assert not (tmp_path / "recommendation").exists()


def test_curator_rejects_a_description_run_even_if_its_name_looks_complete(tmp_path):
    path = tmp_path / "complete_metadata.yaml"
    path.write_text("metadata_text_mode: description\n")
    with pytest.raises(ValueError, match="Only complete_metadata"):
        curate_complete_metadata_configs.read_config(path)


def test_curator_rejects_configs_changed_since_table_assembly(tmp_path):
    path = tmp_path / "run.yaml"
    path.write_text("metadata_text_mode: complete_metadata\nseed: 99\n")
    with pytest.raises(ValueError, match="changed since the completed comparison"):
        curate_complete_metadata_configs.read_config(path, "0" * 64)


def test_curated_release_covers_the_complete_cohort():
    directory = ROOT / "configs/reproduction"
    manifest = json.loads((directory / "complete_metadata_manifest.json").read_text())
    assert len(manifest["cases"]) == 12
    assert manifest["original_configurations"] == 12
    assert manifest["training_configurations"] == 84
    for dataset in curate_complete_metadata_configs.DATASETS:
        entries = json.loads((directory / f"experiments_config_{dataset}_grid_tokenizers.json").read_text())
        assert len(entries) == 24
        for tokenizer in curate_complete_metadata_configs.TOKENIZERS:
            case = [entry["config"] for entry in entries if entry["config"]["grid_tokenizer"] == tokenizer]
            assert len(case) == 8
            for key in curate_complete_metadata_configs.ARTIFACT_KEYS:
                assert len({config[key] for config in case}) == 1
        for entry in entries:
            config = entry["config"]
            assert config["metadata_text_mode"] == "complete_metadata"
            for key in curate_complete_metadata_configs.ARTIFACT_KEYS:
                assert "complete_metadata" in Path(config[key]).parts
                assert not Path(config[key]).is_absolute()
            if config.get("rank_model") == "FairDual":
                assert config["exposure_sample_num"] == 200
                assert config["max_steps"] == 50
                assert config["accumulate_grad_batches"] == 4


def test_sync_preserves_all_three_tokenizers_and_curated_settings(tmp_path, monkeypatch):
    copied = tmp_path / "recipes.json"
    copied.write_bytes(sync_grid_independent_experiments.CONFIG_PATH.read_bytes())
    before = json.loads(copied.read_text())
    monkeypatch.setattr(sync_grid_independent_experiments, "CONFIG_PATH", copied)
    sync_grid_independent_experiments.main()
    assert json.loads(copied.read_text()) == before
