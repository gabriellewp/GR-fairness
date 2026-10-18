import ast
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from scipy.sparse import coo_matrix, csr_matrix, save_npz
import torch

from additional_baselines import ExperimentInputs, Request, load, save
from additional_baselines.adapters import GenerationRequest, generate_grid, import_legacy
from additional_baselines.candidates import file_reference
from additional_baselines.evaluation import evaluate


@pytest.fixture
def metadata(tmp_path):
    source = tmp_path / "source"
    source.write_text("fixture")
    ref = file_reference(source)
    return {
        "checkpoint": ref, "sid_mapping": ref,
        "split": {"name": "test", "source": ref}, "seed": 42,
        "generation": {"beam_width": 4}, "score_convention": "probability",
        "candidate_membership": "explicit_fixture", "short_list_policy": "return_available",
    }


@pytest.fixture
def inputs(metadata):
    return ExperimentInputs(metadata, {"brand": {0: "a", 1: "a", 2: "b", 3: "c"}}, [
        Request("r0", [0, 2], [0.7, 0.2], [0, 1]),
        Request("r1", [], [], [3]),
    ])


def test_roundtrip_and_no_overwrite(inputs, tmp_path):
    path = tmp_path / "inputs.json"
    save(inputs, path)
    assert load(path) == inputs
    with pytest.raises(FileExistsError):
        save(inputs, path)


@pytest.mark.parametrize("change", [
    lambda x: x.requests[0].candidate_ids.append(0),
    lambda x: x.requests[0].scores.__setitem__(0, float("nan")),
    lambda x: x.requests[0].scores.__setitem__(0, 1.1),
    lambda x: x.requests[0].targets.clear(),
    lambda x: x.item_groups["brand"].pop(1),
    lambda x: x.metadata.pop("seed"),
    lambda x: x.metadata.__setitem__("score_convention", "unknown"),
    lambda x: setattr(x.requests[1], "request_id", "r0"),
])
def test_invalid_contract(inputs, change):
    change(inputs)
    with pytest.raises(ValueError):
        inputs.validate()


def test_missed_targets_short_lists_and_zero_exposure_groups(inputs):
    result = evaluate(inputs, [3])
    metrics = result["at_k"]["3"]
    assert result["candidate_target_recall"] == 0.25
    assert result["candidate_hit_rate"] == 0.5
    assert result["empty_candidate_rate"] == 0.5
    assert metrics["hit_rate"] == 0.5
    assert metrics["recall"] == 0.25
    assert metrics["ndcg"] == pytest.approx(0.5 / (1 + 1 / math.log2(3)))
    assert metrics["mrr"] == 0.5
    assert metrics["fill_rate"] == pytest.approx(1 / 3)
    assert metrics["short_list_rate"] == 1
    assert metrics["groups"]["brand"]["exposure_counts"] == {"a": 1, "b": 1, "c": 0}
    assert metrics["groups"]["brand"]["gini"] == pytest.approx(1 / 3)
    assert metrics["groups"]["brand"]["entropy"] == pytest.approx(math.log(2))
    assert metrics["groups"]["brand"]["entropy_bits"] == pytest.approx(1.0)
    assert metrics["groups"]["brand"]["candidate_group_coverage"] == pytest.approx(1 / 3)


def test_empty_exposure_is_undefined_not_perfect_fairness(inputs):
    result = evaluate(inputs, [1], {"r0": [], "r1": []})
    assert result["at_k"]["1"]["groups"]["brand"]["gini"] is None
    assert result["at_k"]["1"]["groups"]["brand"]["entropy_bits"] is None


@pytest.mark.parametrize("ranking", [[0, 0], [1]])
def test_reject_invalid_reranked_lists(inputs, ranking):
    with pytest.raises(ValueError):
        evaluate(inputs, [2], {"r0": ranking, "r1": []})


def test_score_direction_and_stable_ties(inputs):
    inputs.metadata["score_convention"] = "negative_log_probability"
    assert inputs.ranking(inputs.requests[0]) == [2, 0]
    inputs.requests[0].scores = [1, 1]
    assert inputs.ranking(inputs.requests[0]) == [0, 2]


def legacy_files(tmp_path, values=(1.7, 1.2)):
    scores = tmp_path / "scores.npz"
    targets = tmp_path / "targets.npz"
    save_npz(scores, csr_matrix((values, ([0, 0], [0, 2])), shape=(1, 4)))
    # Legacy evaluator writes COO ground truth.
    save_npz(targets, coo_matrix(([1], ([0], [3])), shape=(1, 4)))
    return scores, targets


def test_legacy_conversion_keeps_missed_target(metadata, inputs, tmp_path):
    scores, targets = legacy_files(tmp_path)
    artifact = import_legacy(scores, targets, ["r"], inputs.item_groups, metadata, export_format="compact_grid")
    assert artifact.requests[0].candidate_ids == [0, 2]
    assert artifact.requests[0].scores == pytest.approx([0.7, 0.2])
    assert artifact.requests[0].targets == [3]
    assert evaluate(artifact, [2])["at_k"]["2"]["ndcg"] == 0
    assert metadata["candidate_membership"] == "explicit_fixture"


def test_dense_requires_mask_and_mask_can_retain_zero(metadata, inputs, tmp_path):
    scores, targets = legacy_files(tmp_path, (1.0, 1.2))
    with pytest.raises(ValueError, match="explicit candidate mask"):
        import_legacy(scores, targets, ["r"], inputs.item_groups, metadata, export_format="explicit_mask")
    with pytest.raises(ValueError, match="regenerate"):
        import_legacy(scores, targets, ["r"], inputs.item_groups, metadata, export_format="compact_grid")
    mask = tmp_path / "mask.npz"
    save_npz(mask, csr_matrix(([1], ([0], [0])), shape=(1, 4)))
    artifact = import_legacy(scores, targets, ["r"], inputs.item_groups, metadata,
                             export_format="explicit_mask", candidate_mask_path=mask)
    assert artifact.requests[0].candidate_ids == [0]
    assert artifact.requests[0].scores == [0]


def test_duplicate_sparse_scores_rejected(metadata, inputs, tmp_path):
    scores, targets = legacy_files(tmp_path)
    save_npz(scores, coo_matrix(([1.1, 1.2], ([0, 0], [0, 0])), shape=(1, 4)))
    with pytest.raises(ValueError, match="Duplicate"):
        import_legacy(scores, targets, ["r"], inputs.item_groups, metadata, export_format="compact_grid")


class FakeGRID(torch.nn.Module):
    def generate(self, history, users):
        assert not self.training
        assert not torch.is_grad_enabled()
        assert history.tolist() == [[2, 0]]
        return torch.tensor([[[0, 0], [0, 0], [0, 2], [9, 9], [0, 1]]]), torch.tensor([[0.2, 0.4, 0.0, 0.1, 0.3]])


def test_fresh_generation_membership_not_probability(metadata, inputs):
    model = FakeGRID()
    request = GenerationRequest("r", [2, 0], [3], [0, 2, 3])
    result = generate_grid(model, [request], {i: [0, i] for i in range(4)}, inputs.item_groups, metadata)
    assert model.training
    assert result.requests[0].candidate_ids == [0, 2]
    assert result.requests[0].scores == pytest.approx([0.4, 0])
    assert result.requests[0].targets == [3]
    assert result.metadata["generation_adapter"]["unknown_sid_count"] == 1
    assert result.metadata["generation_adapter"]["ineligible_count"] == 1


def test_ambiguous_sids_rejected(metadata, inputs):
    with pytest.raises(ValueError, match="Ambiguous"):
        generate_grid(FakeGRID(), [], {0: [0], 1: [0]}, inputs.item_groups, metadata)


def test_cli_evaluation(inputs, tmp_path):
    artifact, report = tmp_path / "inputs.json", tmp_path / "report.json"
    save(inputs, artifact)
    subprocess.run([sys.executable, "-m", "additional_baselines", "evaluate", str(artifact),
                    "--topk", "3", "--output", str(report)], check=True)
    assert json.loads(report.read_text())["at_k"]["3"]["recall"] == 0.25


def test_single_target_metrics_match_existing_functions(inputs):
    # Execute the actual, unmodified ranking function definitions without the
    # module's unrelated sklearn AUC dependency or model/registry side effects.
    path = Path(__file__).resolve().parents[2] / "fairdiverse/recommendation/metric.py"
    tree = ast.parse(path.read_text())
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in {"dcg", "NDCG", "HR", "MRR"}]
    legacy = {"np": np}
    exec(compile(tree, str(path), "exec"), legacy)
    inputs.requests = [Request("r", [0, 2], [0.7, 0.2], [2])]
    metrics = evaluate(inputs, [2])["at_k"]["2"]
    assert metrics["ndcg"] == pytest.approx(legacy["NDCG"]([0, 1], [1, 0], 2))
    assert metrics["hit_rate"] == pytest.approx(legacy["HR"]([0, 1], [1, 0], 2))
    assert metrics["mrr"] == pytest.approx(legacy["MRR"]([0, 1], 2))


def test_generation_restores_model_mode_on_failure(metadata, inputs):
    class BrokenGRID(FakeGRID):
        def generate(self, *args):
            raise RuntimeError("fixture failure")
    model = BrokenGRID()
    with pytest.raises(RuntimeError, match="fixture failure"):
        generate_grid(model, [GenerationRequest("r", [0], [3], [0])],
                      {i: [0, i] for i in range(4)}, inputs.item_groups, metadata)
    assert model.training


def test_generation_rejects_incorrect_settings(metadata, inputs):
    model = FakeGRID()
    model.top_k_for_generation = 10
    with pytest.raises(ValueError, match="metadata disagrees"):
        generate_grid(model, [], {i: [0, i] for i in range(4)}, inputs.item_groups, metadata)


def test_cli_import_then_evaluate(metadata, inputs, tmp_path):
    scores, targets = legacy_files(tmp_path)
    for name, data in (("ids", ["r"]), ("groups", inputs.item_groups), ("metadata", metadata)):
        (tmp_path / f"{name}.json").write_text(json.dumps(data))
    artifact = tmp_path / "imported.json"
    subprocess.run([
        sys.executable, "-m", "additional_baselines", "import-legacy",
        "--scores", str(scores), "--targets", str(targets),
        "--request-ids", str(tmp_path / "ids.json"), "--groups", str(tmp_path / "groups.json"),
        "--metadata", str(tmp_path / "metadata.json"), "--format", "compact_grid",
        "--output", str(artifact),
    ], check=True)
    loaded = load(artifact)
    assert loaded.metadata["request_order_source"] == file_reference(tmp_path / "ids.json")
    assert evaluate(loaded, [5])["candidate_hit_rate"] == 0
