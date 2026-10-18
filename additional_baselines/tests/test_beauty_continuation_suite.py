import json
from pathlib import Path

import pytest

from additional_baselines import beauty_continuation_suite as suite


def processed_inputs(root, dataset):
    (root / "configs/reproduction").mkdir(parents=True, exist_ok=True)
    data = root / "recommendation" / "processed_dataset" / dataset
    data.mkdir(parents=True)
    for name in ("grid_data_manifest.json", "iid2item.json", "iid2pid.json",
                 f"{dataset}.train", f"{dataset}.valid.ranking", f"{dataset}.test.ranking"):
        (data / name).write_text("source")


@pytest.mark.parametrize("improved", [False, True])
@pytest.mark.parametrize("require_improvement", [False, True])
def test_suite_gates_fairness_and_keeps_matched_controls(tmp_path, monkeypatch, improved, require_improvement):
    processed_inputs(tmp_path, "amazon-beauty")
    artifact = tmp_path / "source"
    artifact.write_text("source")
    config = {key: "source" for key in
              ("grid_data_dir", "grid_item_mapping_path", "sid_path", "grid_checkpoint_path")}
    config["grid_tokenizer"] = "rvq"
    config["grid_eval_only"] = True
    (tmp_path / "configs/reproduction/experiments_config_beauty_continuation_exports.json").write_text(
        json.dumps([{"config": config}]))
    monkeypatch.setattr(suite, "ROOT", tmp_path)
    calls = []

    def continuation(**kwargs):
        calls.append(kwargs)
        return {"selected_update": 8 if improved else 0}

    monkeypatch.setattr(suite, "run_continuation", continuation)
    suite.run_suite("rvq", tmp_path / "results", export_dir=tmp_path / "export",
                    require_control_improvement=require_improvement)
    run_branches = improved or not require_improvement
    assert [c["mode"] for c in calls] == (["ordinary", "rw", "sprint"] if run_branches else ["ordinary"])
    assert all(c["require_control_improvement"] == require_improvement for c in calls)
    if run_branches:
        for branch in calls[1:]:
            assert branch["control_report"] == tmp_path / "results/ordinary/continuation.json"
            for key in ("tokenizer", "export_dir", "data_dir", "seed", "max_updates",
                        "validation_interval", "learning_rate", "optimizer_state"):
                assert branch[key] == calls[0][key]
    with pytest.raises(FileExistsError):
        suite.run_suite("rvq", tmp_path / "results", export_dir=tmp_path / "export")


def test_sports_suite_selects_rvq_source_and_dataset(tmp_path, monkeypatch):
    processed_inputs(tmp_path, "amazon-sports")
    for name in ("grid_data_dir", "grid_item_mapping_path", "sid_path", "grid_checkpoint_path"):
        (tmp_path / name).write_text("source")
    config = {name: name for name in
              ("grid_data_dir", "grid_item_mapping_path", "sid_path", "grid_checkpoint_path")}
    config["grid_tokenizer"] = "rvq"
    config["grid_eval_only"] = True
    (tmp_path / "configs/reproduction/experiments_config_sports_continuation_exports.json").write_text(
        json.dumps([{"config": config}]))
    monkeypatch.setattr(suite, "ROOT", tmp_path)
    calls = []
    monkeypatch.setattr(suite, "run_continuation", lambda **kwargs: calls.append(kwargs) or {"selected_update": 8})
    suite.run_suite("rvq", tmp_path / "results", export_dir=tmp_path / "export", dataset="amazon-sports",
                    generation_beam_width=300)
    assert [call["mode"] for call in calls] == ["ordinary", "rw", "sprint"]
    assert all(call["dataset"] == "amazon-sports" and
               call["data_dir"] == "recommendation/processed_dataset/amazon-sports" for call in calls)
    assert all(call["generation_beam_width"] == 300 for call in calls)
    saved = json.loads((tmp_path / "results/suite_config.json").read_text())
    assert saved["generation_beam_width"] == 300


@pytest.mark.parametrize("tokenizer", ["rqvae", "rvq"])
def test_toys_suite_uses_evaluation_export(tmp_path, monkeypatch, tokenizer):
    processed_inputs(tmp_path, "amazon-toys")
    for name in ("grid_data_dir", "grid_item_mapping_path", "sid_path", "grid_checkpoint_path"):
        (tmp_path / name).write_text("source")
    config = {name: name for name in
              ("grid_data_dir", "grid_item_mapping_path", "sid_path", "grid_checkpoint_path")}
    config.update(grid_tokenizer=tokenizer, grid_eval_only=True)
    (tmp_path / "configs/reproduction/experiments_config_amazon-toys_grid_tokenizers.json").write_text(
        json.dumps([{"config": config}]))
    monkeypatch.setattr(suite, "ROOT", tmp_path)
    calls = []
    monkeypatch.setattr(suite, "run_continuation", lambda **kwargs: calls.append(kwargs) or {"selected_update": 0})
    suite.run_suite(tokenizer, tmp_path / "results", export_dir=tmp_path / "export", dataset="amazon-toys")
    assert len(calls) == 1
    assert calls[0]["data_dir"] == "recommendation/processed_dataset/amazon-toys"


def test_toys_rvq_has_complete_metadata_evaluation_source():
    entries = json.loads((suite.ROOT / suite.EXPORT_CONFIGS["amazon-toys"]).read_text())
    source = next(e for e in entries if e["config"]["grid_tokenizer"] == "rvq"
                  and e["config"].get("grid_eval_only"))
    config = source["config"]
    assert source["dataset"] == "amazon-toys"
    assert config["metadata_text_mode"] == "complete_metadata"
    assert config["grid_fair_finetune"] is False
    assert config["fair-rank"] is False
    assert config["store_scores"] is True
    assert config["num_hierarchies"] == 4
    for key in ("grid_data_dir", "grid_item_mapping_path", "sid_path", "grid_checkpoint_path"):
        assert "complete_metadata" in Path(config[key]).parts


def test_pet_suite_reports_missing_processed_inputs(tmp_path, monkeypatch):
    (tmp_path / "configs/reproduction").mkdir(parents=True)
    config = {"grid_tokenizer": "rkmeans", "grid_eval_only": False}
    (tmp_path / "configs/reproduction/experiments_config_pet_continuation_exports.json").write_text(
        json.dumps([{"config": config}]))
    monkeypatch.setattr(suite, "ROOT", tmp_path)
    with pytest.raises(FileNotFoundError, match="Processed amazon-pet-supplies"):
        suite.run_suite("rkmeans", tmp_path / "results", export_dir=tmp_path / "export",
                        dataset="amazon-pet-supplies")
