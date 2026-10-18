from copy import deepcopy
import json

import pytest
import torch

from additional_baselines.adapters import GenerationRequest, generate_grid
from additional_baselines.batched_generation import generate_grid_batched
from additional_baselines.candidates import file_reference
from additional_baselines.deadline_pilot import stratified_sample


def test_stratified_sample_is_paired_reproducible_and_proportional():
    rows = [(i, [0, 1], i % 2) for i in range(100)]
    activity = {i: "low" if i < 70 else "high" for i in range(100)}
    result = stratified_sample(rows, activity, {0: "rare", 1: "popular"}, 20)
    assert result == stratified_sample(rows, activity, {0: "rare", 1: "popular"}, 20)
    assert len(set(result["user_ids"])) == 20
    assert sum(v for s, v in result["sample_strata"].items() if s.startswith("low")) == 14
    assert sum(result["sample_strata"].values()) == 20
    full = stratified_sample(rows, activity, {0: "rare", 1: "popular"}, 1000)
    assert full["user_ids"] == list(range(100))


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_batched_generation_preserves_order_targets_eligibility_and_mode(tmp_path, batch_size):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def generate(self, history, users):
            self.calls += 1
            # Depend on history and user so cross-request mixing is detected.
            item = (history.clamp_min(0).sum(1) + users) % 5
            ids = torch.stack([item, (item + 1) % 5, item], dim=1).unsqueeze(-1)
            return ids, torch.tensor([[.8, .4, .2]]).expand(len(history), -1)

    source = tmp_path / "source"
    source.write_text("fixture")
    ref = file_reference(source)
    metadata = dict(checkpoint=ref, sid_mapping=ref, split=dict(name="validation", source=ref),
                    seed=42, generation=dict(beam_width=3), score_convention="probability",
                    candidate_membership="explicit", short_list_policy="return_available")
    requests = [GenerationRequest(str(i), [i % 5] * (i % 3 + 1), [4], [0, 1, 2, 3, 4], i)
                for i in range(7)]
    requests[2].eligible_ids = [0, 1]
    mapping = {i: [i] for i in range(5)}
    groups = {"brand": {i: str(i % 2) for i in range(5)}}
    ordinary, batched = Model(), Model()
    expected = generate_grid(ordinary, deepcopy(requests), mapping, groups, metadata)
    actual = generate_grid_batched(batched, deepcopy(requests), mapping, groups, metadata, batch_size=batch_size)
    assert actual.requests == expected.requests
    assert batched.calls == (len(requests) + batch_size - 1) // batch_size
    assert batched.training is True
    assert actual.metadata["generation_adapter"]["ineligible_count"] == expected.metadata["generation_adapter"]["ineligible_count"]


def test_sampling_rejects_duplicate_validation_users():
    with pytest.raises(ValueError, match="one validation request per user"):
        stratified_sample([(0, [0], 0), (0, [0], 0)], {0: "low"}, {0: "rare"}, 2)


@pytest.mark.parametrize("confirmed", [True, False])
@pytest.mark.parametrize("module", ["deadline_pilot", "deadline_pilot_v2"])
def test_sampled_selection_is_frozen_and_full_validation_gates_test(tmp_path, monkeypatch, confirmed, module):
    import importlib
    workflow = importlib.import_module(f"additional_baselines.{module}")
    from additional_baselines.continuation_selection import selection_policy

    def metrics(brand=.6, counts=(80, 20)):
        return {"at_k": {"5": {"recall": .5, "groups": {"brand": {"gini": brand},
                "popularity": {"exposure_counts": dict(zip(("a", "b"), counts))}}}}}

    gh = {"a": .5, "b": .5}
    refs = {}
    for name in ("sample", "groups", "source", "export"):
        p = tmp_path / name
        p.write_text(json.dumps({"user_ids": [0, 1]}) if name == "sample" else "{}")
        refs[name] = file_reference(p)
    control = dict(identity=dict(max_updates=32, validation_interval=8, learning_rate=.001,
        optimizer_state="require", generation_beam_width=100), selected_checkpoint=refs["source"],
        selected_validation_metrics=metrics())
    control_path = tmp_path / "control.json"
    control_path.write_text(json.dumps(control))
    case_output = (tmp_path / "missing_parent" / "results" if module == "deadline_pilot_v2"
                   else tmp_path / "results")
    case = dict(index=0, dataset="amazon-beauty", tokenizer="rkmeans", options=dict(output_dir=str(case_output),
        export_dir=str(tmp_path), data_dir=str(tmp_path)), prepared_inputs={}, control=file_reference(control_path),
        export_config=refs["export"], sample=refs["sample"], groups=refs["groups"],
        trials=[dict(name=f"{mode}_{i}", mode=mode) for mode in ("rw", "sprint") for i in range(2)])
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(dict(cases=[case], code={})))
    calls = []

    def evaluate(case, checkpoint, split, *, sample=None, output=None):
        calls.append((split, sample is not None, checkpoint == refs["source"]))
        if sample is None:
            assert (case_output / "sampled_selection.json").exists()
        value = metrics() if checkpoint == refs["source"] else metrics(
            brand=.5 if confirmed else .7, counts=(60, 40))
        report = dict(metrics=value, history_shares=gh)
        if output:
            output.write_text(json.dumps(report))
        return report

    def train(**kwargs):
        return dict(selected_update=8, selected_validation_metrics=metrics(brand=.5, counts=(60, 40)),
            selected_checkpoint=dict(path="trained", sha256="fixture"),
            selection=selection_policy("both", control_metrics=metrics(), history_shares=gh))

    monkeypatch.setattr(workflow, "evaluate_checkpoint", evaluate)
    monkeypatch.setattr(workflow, "run_deadline_continuation", train)
    workflow.run_case(manifest_path, 0)
    assert sum(split == "test" for split, _, _ in calls) == (2 if confirmed else 0)
    complete = json.loads((case_output / "complete.json").read_text())
    assert complete["no_test_based_retuning"] is True
    expected = "full_validation_confirmed" if confirmed else "failed_full_validation_keep_ordinary"
    assert all(c["status"] == expected for c in complete["confirmations"].values())
