import json
import subprocess
import sys

import pytest
from scipy.sparse import csr_matrix, save_npz
import torch
import yaml

from additional_baselines import load
from additional_baselines.prepare_beauty import prepare, load_beauty
from additional_baselines.rerank_beauty import rerank_beauty
from additional_baselines.run_rerank import run


@pytest.fixture
def source(tmp_path):
    export = tmp_path / "export"
    data = tmp_path / "data"
    for directory in (export, data, tmp_path / "grid", tmp_path / "output",
                      tmp_path / "training/checkpoints", tmp_path / "training/.hydra"):
        directory.mkdir(parents=True, exist_ok=True)
    (tmp_path / "training/checkpoints/model.ckpt").write_bytes(b"fixture checkpoint")
    torch.save(torch.tensor([[0, 1, 2, 3]] * 4), tmp_path / "grid/sids.pt")
    config = {"dataset": "amazon-beauty", "model": "GRID", "grid_tokenizer": "rkmeans",
              "num_hierarchies": 4, "grid_compact_ranking": True, "use_beam_search_eval": True,
              "grid_eval_only": True, "metadata_text_mode": "complete_metadata",
              "grid_checkpoint_path": "training/checkpoints/model.ckpt", "sid_path": "grid/sids.pt",
              "slurm_job_id": "123", "slurm_array_task_id": "0", "log_name": "fixture-export",
              "grid_data_dir": "grid", "grid_item_mapping_path": "grid/item_id_mapping.json",
              "item_num": 4, "user_num": 2, "group_num": 2, "group_id": "brand:token",
              "group_aggregation_threshold": 10, "top_k_for_generation": 4,
              "use_prefix_validation": False, "history_length": 40, "sequence_length": 160}
    (export / "config.yaml").write_text(yaml.safe_dump(config))
    (tmp_path / "training/.hydra/config.yaml").write_text(yaml.safe_dump(
        {"seed": 42, "num_hierarchies": 4, "semantic_id_path": "grid/sids.pt"}))
    (tmp_path / "output/123_0.out").write_text("fixture-export\n[SEED] seed=42, deterministic=False\n")
    (data / "process_config.yaml").write_text(yaml.safe_dump(config))
    (data / "grid_data_manifest.json").write_text(json.dumps({"source_dir": "grid", "split_strategy": "temporal_loo"}))
    (tmp_path / "grid/item_id_mapping.json").write_text(json.dumps({f"item{i}": i for i in range(4)}))
    (data / "iid2item.json").write_text(json.dumps({str(i): f"item{i}" for i in range(4)}))
    (data / "iid2pid.json").write_text(json.dumps({"0": 0, "1": 0, "2": 1, "3": 1}))
    (data / "amazon-beauty.test.ranking").write_text("user_id\thistory_behaviors\ttarget_item\n8\t[0]\t3\n9\t[1]\t0\n")
    save_npz(export / "ranking_scores.npz", csr_matrix([[1.9, 1.8, 1.7, 0], [1.5, 0, 1.7, 1.9]]))
    save_npz(export / "ground_truth.npz", csr_matrix([[0, 0, 0, 1], [1, 0, 0, 0]]))
    return tmp_path


def package(root, **kwargs):
    return prepare("prepared", root=root, export_dir="export", data_dir="data", topk=(1, 2), **kwargs)


def test_preparation_preserves_targets_provenance_and_real_groups(source):
    summary = package(source)
    artifact = load(source / "prepared/inputs.json")
    assert artifact.requests[0].targets == [3]
    assert 3 not in artifact.requests[0].candidate_ids
    assert artifact.requests[0].request_id == "test:0:user:8"
    assert artifact.metadata["training_seed"] == artifact.metadata["export_seed"] == 42
    assert artifact.metadata["provenance"]["export_seed_log"]["sha256"]
    assert summary["minimum_candidates"] == 3
    config = json.loads((source / "prepared/pmmf.json").read_text())
    assert config["parameters"]["weights"] == {"0": 1, "1": 1}
    assert config["parameters"]["horizon"] is None
    with pytest.raises(FileExistsError):
        package(source)


@pytest.mark.parametrize("tokenizer", ["rqvae", "rvq"])
def test_continuation_export_requires_matching_tokenizer(source, tokenizer):
    path = source / "export/config.yaml"
    config = yaml.safe_load(path.read_text())
    config["grid_tokenizer"] = tokenizer
    path.write_text(yaml.safe_dump(config))
    artifact, _, _ = load_beauty(root=source, export_dir="export", data_dir="data",
                                topk=(1,), tokenizer=tokenizer)
    assert artifact.metadata["generation"]["grid_tokenizer"] == tokenizer
    with pytest.raises(ValueError, match="grid_tokenizer"):
        load_beauty(root=source, export_dir="export", data_dir="data", topk=(1,))


def test_sports_continuation_uses_sports_split_and_identity(source):
    path = source / "export/config.yaml"
    config = yaml.safe_load(path.read_text())
    config["dataset"] = "amazon-sports"
    path.write_text(yaml.safe_dump(config))
    (source / "data/amazon-beauty.test.ranking").rename(source / "data/amazon-sports.test.ranking")
    artifact, _, _ = load_beauty(root=source, export_dir="export", data_dir="data",
                                topk=(1,), dataset="amazon-sports")
    assert artifact.metadata["dataset"] == "amazon-sports"
    assert artifact.metadata["split"]["source"]["path"].endswith("amazon-sports.test.ranking")
    with pytest.raises(ValueError, match="dataset"):
        load_beauty(root=source, export_dir="export", data_dir="data", topk=(1,))


@pytest.mark.parametrize("fault", ["seed", "targets", "mapping", "sid", "candidates", "memory"])
def test_preparation_rejects_mismatched_sources_before_writing(source, fault):
    kwargs = {}
    if fault == "seed":
        (source / "output/123_0.out").write_text("fixture-export\n[SEED] seed=7\n")
    elif fault == "targets":
        save_npz(source / "export/ground_truth.npz", csr_matrix([[1, 0, 0, 0], [0, 0, 0, 1]]))
    elif fault == "mapping":
        (source / "data/iid2item.json").write_text("{}")
    elif fault == "sid":
        torch.save(torch.zeros(3, 4), source / "grid/sids.pt")
    elif fault == "candidates":
        save_npz(source / "export/ranking_scores.npz", csr_matrix([[1.5, 0, 0, 0], [1.5, 0, 1.7, 1.9]]))
    elif fault == "memory":
        kwargs["max_dense_bytes"] = 1
    with pytest.raises(ValueError):
        package(source, **kwargs)
    assert not (source / "prepared").exists()


def test_smoke_cli_marks_scope_and_retains_source_fingerprint(source):
    package(source)
    directory = source / "prepared"
    output = directory / "smoke.json"
    subprocess.run([sys.executable, "-m", "additional_baselines", "rerank", str(directory / "inputs.json"),
                    "--config", str(directory / "cpfair.json"), "--limit-requests", "1",
                    "--output", str(output)], check=True)
    report = json.loads(output.read_text())
    assert report["evaluation_scope"] == {"kind": "smoke_test_prefix", "original_request_count": 2, "evaluated_request_count": 1}
    assert report["input_artifact"]["sha256"]
    assert report["control_metrics"]["request_count"] == 1


@pytest.mark.parametrize("model", ["CPFair", "PMMF"])
def test_direct_matches_packaged_and_writes_only_results(source, model):
    package(source)
    before = {path.relative_to(source): path.read_bytes() for path in source.rglob("*") if path.is_file()}
    config_name = "cpfair.json" if model == "CPFair" else "pmmf.json"
    config = json.loads((source / "prepared" / config_name).read_text())
    expected = run(load(source / "prepared/inputs.json"), config)
    output = source / "export/direct_results.json"
    actual = rerank_beauty(model, output, root=source, export_dir="export", data_dir="data", topk=(1, 2))
    assert actual["runs"] == expected["runs"]
    assert actual["control_metrics"] == expected["control_metrics"]
    assert actual["evaluation_scope"]["kind"] == "full_export"
    assert actual["input_metadata"]["legacy_import"]["scores"]["sha256"]
    after = {path.relative_to(source): path.read_bytes() for path in source.rglob("*") if path.is_file()}
    assert set(after) - set(before) == {output.relative_to(source)}
    assert all(after[path] == value for path, value in before.items())


def test_direct_needs_no_package_and_checks_lookup_workspace_memory(source):
    output = source / "export/smoke.json"
    report = rerank_beauty("CPFair", output, root=source, export_dir="export", data_dir="data",
                           topk=(1, 2), limit_requests=1, horizon=1, max_dense_bytes=1024)
    assert report["evaluation_scope"] == {"kind": "smoke_test_prefix", "original_request_count": 2, "evaluated_request_count": 1}
    assert not (source / "prepared").exists()
    assert not (source / "additional_baselines").exists()


@pytest.mark.parametrize("failure", ["targets", "seed", "limit", "memory"])
def test_direct_failure_writes_no_output(source, failure):
    options = {}
    if failure == "targets":
        save_npz(source / "export/ground_truth.npz", csr_matrix([[1, 0, 0, 0], [0, 0, 0, 1]]))
    elif failure == "seed":
        (source / "output/123_0.out").write_text("fixture-export\n[SEED] seed=7\n")
    elif failure == "limit":
        options["limit_requests"] = 3
    elif failure == "memory":
        options["max_dense_bytes"] = 1
    output = source / "export/failed.json"
    with pytest.raises(ValueError):
        rerank_beauty("CPFair", output, root=source, export_dir="export", data_dir="data", topk=(1, 2), **options)
    assert not output.exists()


def test_direct_does_not_overwrite_existing_result(source):
    output = source / "export/existing.json"
    output.write_text("existing result")
    with pytest.raises(FileExistsError):
        rerank_beauty("CPFair", output, root=source, export_dir="export", data_dir="data", topk=(1, 2))
    assert output.read_text() == "existing result"
