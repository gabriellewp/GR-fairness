from copy import deepcopy

import pytest
import torch

from additional_baselines.candidates import ExperimentInputs, Request, file_reference
from additional_baselines.ifairlrs import apply_rr, fairness, fit_rr, popularity_groups, rw_weights, weighted_loss
from additional_baselines.ifairlrs_grid import plan_digest, replay_batches, train_branch, training_popularity


@pytest.fixture
def validation(tmp_path):
    source = tmp_path / "source"
    source.write_text("fixture")
    ref = file_reference(source)
    metadata = {"checkpoint": ref, "sid_mapping": ref, "split": {"name": "validation", "source": ref},
                "seed": 42, "generation": {"beam_width": 3}, "score_convention": "negative_log_probability",
                "candidate_membership": "explicit", "short_list_policy": "return_available"}
    return ExperimentInputs(metadata, {"popularity": {0: "a", 1: "b", 2: "b"}},
                            [Request("r", [0, 1], [1.0, 1.1], [2])])


def test_rr_paper_aggregation_freeze_and_eligibility(validation):
    histories = {"r": [0, 1]}
    frozen = fit_rr(validation, histories, cutoffs=(1, 2), alpha=1)
    assert frozen["raw_punishment"] == pytest.approx({"a": 1 / 6, "b": -1 / 6})
    assert frozen["punishment"] == pytest.approx({"a": 1, "b": -1})
    test = deepcopy(validation)
    test.metadata["split"]["name"] = "test"
    result = apply_rr(test, frozen, (1, 5), histories)
    assert result["rankings"]["r"] == [1, 0]
    assert result["metrics"]["candidate_target_recall"] == 0
    assert frozen == fit_rr(validation, histories, cutoffs=(1, 2), alpha=1)
    assert result["ifairlrs_fairness"]["5"]["MGU"] == 0
    with pytest.raises(ValueError, match="validation only"):
        fit_rr(test, histories)


@pytest.mark.parametrize("convention,scores", [("probability", [0.8, 0.5]), ("log_probability", [-0.2, -0.7]),
                                             ("negative_log_probability", [0.2, 0.7])])
def test_rr_alpha_zero_and_uniform_shares(validation, convention, scores):
    validation.metadata["score_convention"] = convention
    validation.requests[0].scores = scores
    correction = fit_rr(validation, {"r": [0, 1]}, cutoffs=(2,), alpha=0)
    assert correction["punishment"] == {"a": 0, "b": 0}
    assert apply_rr(validation, correction)["rankings"]["r"] == validation.ranking(validation.requests[0])


def test_rr_rejects_drift_and_unrecoverable_scores(validation):
    correction = fit_rr(validation, {"r": [0, 1]})
    modified = deepcopy(validation)
    modified.metadata["checkpoint"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="checkpoint differs"):
        apply_rr(modified, correction)
    modified = deepcopy(validation)
    modified.item_groups["popularity"][0] = "b"
    with pytest.raises(ValueError, match="mapping differs"):
        apply_rr(modified, correction)
    validation.metadata["score_convention"] = "probability"
    validation.requests[0].scores = [0, 0.5]
    with pytest.raises(ValueError, match="Zero probability"):
        apply_rr(validation, correction)


def test_popularity_events_not_repeated_prefixes():
    rows = [(0, [0], 1), (0, [0, 1], 2), (1, [0], 2)]
    assert training_popularity(rows) == {0: 2, 1: 1, 2: 2}
    groups = popularity_groups({0: 5, 1: 2}, range(5), 5)
    assert [groups[i] for i in [2, 3, 4, 1, 0]] == ["0", "1", "2", "3", "4"]


def test_rw_counts_actual_capped_batches_and_replay_rng():
    rows = [(0, [0, 1, 2], 0), (1, [1, 2], 2), (2, [0], 1)]
    options = dict(epochs=2, seed=42, history_length=3, pre_batch_size=2, batch_size=2)
    torch.manual_seed(8)
    before = torch.get_rng_state().clone()
    batches = list(replay_batches(rows, **options))
    assert torch.equal(before, torch.get_rng_state())
    digest = plan_digest(batches)
    torch.rand(100)
    assert plan_digest(replay_batches(rows, **options)) == digest
    assert all(len(b[2]) <= 2 for b in batches)
    mapping = {0: "a", 1: "b", 2: "b", 3: "unused"}
    stats = rw_weights(batches, mapping)
    assert sum(stats["target_counts"].values()) == sum(len(b[2]) for b in batches)
    assert sum(stats["history_counts"].values()) == sum(int((b[1] != -1).sum()) for b in batches)
    assert stats["weights"]["unused"] == 0
    for g in ("a", "b"):
        assert stats["weights"][g] == pytest.approx(stats["history_shares"][g] / stats["target_shares"][g])


def test_replay_never_seeds_cuda(monkeypatch):
    def unexpected_seed(seed):
        raise AssertionError("CPU augmentation must not seed CUDA")

    monkeypatch.setattr(torch.cuda, "manual_seed_all", unexpected_seed)
    rows = [(0, [0, 1, 2], 1), (1, [2, 1], 0)]
    before = torch.get_rng_state().clone()
    batches = list(replay_batches(rows, epochs=2, seed=42, history_length=3,
                                  pre_batch_size=1, batch_size=1))
    assert len(batches) == 4
    assert torch.equal(before, torch.get_rng_state())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA RNG preservation requires a GPU")
def test_replay_preserves_cuda_rng_across_epochs():
    rows = [(0, [0, 1, 2], 1), (1, [2, 1], 0)]
    batches = iter(replay_batches(rows, epochs=2, seed=42, history_length=3,
                                 pre_batch_size=1, batch_size=1))
    for _ in range(4):
        torch.rand(3, device="cuda")
        before = torch.cuda.get_rng_state_all()
        next(batches)
        assert all(torch.equal(a, b) for a, b in zip(before, torch.cuda.get_rng_state_all()))


def test_regenerate_prepares_catalog_once_and_preserves_outputs(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock
    from additional_baselines import ifairlrs_grid as workflow

    class CountedTable:
        conversions = 0

        def detach(self):
            return self

        def cpu(self):
            return self

        def tolist(self):
            self.conversions += 1
            return [[0, 0], [1, 1], [2, 2]]

    class FakeGrid(torch.nn.Module):
        top_k_for_generation, sequence_length, num_hierarchies = 4, 4, 2
        config = {}

        def __init__(self):
            super().__init__()
            self.tokenizer = SimpleNamespace(sid_table=CountedTable())
            self.seen = []

        def generate(self, history, users):
            self.seen.append((history.tolist(), users.tolist()))
            if len(self.seen) == 1:
                return torch.tensor([[[0, 0], [0, 0], [9, 9], [2, 2]]]), torch.tensor([[0.4, 0.8, 0.1, 0.0]])
            return torch.tensor([[[9, 9]] * 4]), torch.full((1, 4), 0.1)

        def compute_loss(self, interaction):
            return interaction["item_ids"].float() + 1

    source = tmp_path / "source"
    source.write_text("fixture")
    spy = Mock(wraps=workflow.generate_grid)
    monkeypatch.setattr(workflow, "generate_grid", spy)
    model = FakeGrid()
    artifact, histories = workflow.regenerate(model, [(7, [0], 1), (8, [1, 0], 2)],
                                               source, "test", {"brand": {0: "a", 1: "b", 2: "b"}},
                                               source, source, 42, "cpu", {})
    assert spy.call_count == 1
    assert model.tokenizer.sid_table.conversions == 2  # Construct mapping once; validate it once.
    first, second = artifact.requests
    assert first.request_id == "test:0:user:7"
    assert first.candidate_ids == [0, 2] and first.scores == [-1.0, -3.0]
    assert first.targets == [1]  # Held-out target remains despite missing generation.
    assert second.request_id == "test:1:user:8"
    assert second.candidate_ids == [] and second.targets == [2]
    assert histories == {first.request_id: [0], second.request_id: [1, 0]}
    assert model.seen == [([[0]], [7]), ([[1, 0]], [8])]
    assert artifact.metadata["generation_adapter"]["unknown_sid_count"] == 5
    assert artifact.metadata["generation_adapter"]["ineligible_count"] == 0
    assert artifact.metadata["score_convention"] == "log_probability"


def test_rw_gradient_is_weighted_before_reduction():
    losses = torch.tensor([2.0, 4.0], requires_grad=True)
    weighted_loss(losses, torch.tensor([0.5, 2.0])).backward()
    assert losses.grad.tolist() == [0.25, 1.0]
    assert weighted_loss(losses, torch.ones(2)) == losses.mean()
    with pytest.raises(ValueError, match="per-example"):
        weighted_loss(losses.mean(), torch.ones(2))


def test_paired_control_identical_when_weights_one():
    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.value = torch.nn.Parameter(torch.tensor([1.0, 2.0]))

        def compute_loss(self, interaction):
            return self.value[interaction["item_ids"]].square()

    batch = (torch.tensor([0, 1]), torch.tensor([[0], [1]]), torch.tensor([0, 1]), torch.zeros(2, dtype=torch.long))
    ordinary, rw = TinyModel(), TinyModel()
    kwargs = dict(device="cpu", learning_rate=0.01, weight_decay=0, clip_norm=0)
    for model in (ordinary, rw):
        stats = train_branch(model, [batch], {0: "a", 1: "b"}, {"a": 1, "b": 1}, **kwargs)
        assert stats["training_plan_sha256"] == plan_digest([batch])
    assert torch.equal(ordinary.value, rw.value)


def test_shortlists_and_missing_histories(validation):
    with pytest.raises(ValueError, match="match request IDs"):
        fit_rr(validation, {"other": [0]})
    validation.requests.append(Request("empty", [], [], [2]))
    histories = {"r": [0], "empty": [1]}
    correction = fit_rr(validation, histories)
    result = apply_rr(validation, correction, histories=histories)
    assert result["rankings"]["empty"] == []
    assert fairness(validation, histories, "popularity", (5,))["5"]["GP"] == {"a": 0.5, "b": 0.5}
