import pytest
import torch

from additional_baselines.continuation_grid import native_batches, native_schedule, reconstruct_users, resume_optimizer, selected_step
from additional_baselines.prepare_beauty import ROOT


def test_reconstructs_full_users_and_rejects_misaligned_prefix():
    rows = [(1, [3], 4), (1, [3, 4], 5), (2, [8], 9)]
    assert reconstruct_users(rows) == [(1, [3, 4, 5]), (2, [8, 9])]
    with pytest.raises(ValueError, match="Nonchronological"):
        reconstruct_users(rows + [(1, [3, 6, 5], 7)])


def test_native_windows_and_capped_replay_are_deterministic():
    users = [(1, [3, 4, 5])]
    settings = dict(updates=2, seed=42, users_per_batch=1, cap=10, accumulation=2, history_length=4)
    batches = list(native_batches(users, **settings))
    assert len(batches) == 4
    assert batches[0][2].tolist() == [4, 5, 5]
    assert batches[0][1].tolist() == [[3, -1, -1, -1], [3, 4, -1, -1], [4, -1, -1, -1]]
    settings["cap"] = 2
    first = list(native_batches(users, **settings))
    second = list(native_batches(users, **settings))
    assert all(batch[2].numel() == 2 for batch in first)
    assert all(all((a == b).all() for a, b in zip(x, y)) for x, y in zip(first, second))


def test_native_training_schedule_has_room_for_target(tmp_path):
    import yaml
    config = yaml.safe_load((ROOT / "GRID/configs/experiment/tiger_train_flat.yaml").read_text())
    # The pipeline supplies this interpolation parameter when launching TIGER.
    config["num_hierarchies"] = 4
    path = tmp_path / "native_training.yaml"
    path.write_text(yaml.safe_dump(config))
    schedule = native_schedule(path)
    assert (schedule["users_per_batch"], schedule["cap"], schedule["accumulation"]) == (32, 512, 16)
    assert schedule["history_length"] == 39 and schedule["model_history_length"] == 40


def test_zero_step_eligible_and_earlier_tie_wins():
    def row(update, value):
        return {"update": update, "metrics": {"at_k": {"5": {"recall": value}}}}
    assert selected_step([row(0, .2), row(4, .1)]) == 0
    assert selected_step([row(0, .2), row(4, .3), row(8, .3)]) == 4


def test_resume_adam_moments_and_override_learning_rate(tmp_path):
    class Model:
        def __init__(self):
            self.grid = torch.nn.Linear(2, 1)

    source = Model()
    optimizer = torch.optim.Adam(source.grid.parameters(), lr=.001, weight_decay=.0001)
    source.grid(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    checkpoint = tmp_path / "native.ckpt"
    torch.save({"optimizer_states": [optimizer.state_dict()], "global_step": 1}, checkpoint)
    target = Model()
    resumed, audit = resume_optimizer(target, checkpoint, learning_rate=.0001, weight_decay=.0001, policy="require")
    assert audit["resumed"] and audit["source_adam_step"] == [1]
    assert resumed.param_groups[0]["lr"] == .0001
    assert all(torch.equal(a["exp_avg"], b["exp_avg"]) for a, b in zip(
        optimizer.state.values(), resumed.state.values()))
    missing = tmp_path / "missing.ckpt"
    torch.save({"optimizer_states": []}, missing)
    with pytest.raises(RuntimeError, match="Cannot resume"):
        resume_optimizer(target, missing, learning_rate=.0001, weight_decay=.0001, policy="require")


def test_unimproved_control_requires_explicit_opt_out_and_keeps_identity_checks():
    from additional_baselines.continuation_grid import validate_control
    identity = {"max_updates": 32, "seed": 42, "plan_sha256": "matched"}
    control = {"mode": "ordinary", "identity": identity, "selected_update": 0}
    with pytest.raises(ValueError, match="must improve"):
        validate_control(control, identity)
    validate_control(control, identity, require_control_improvement=False)
    validate_control({**control, "selected_update": 8}, identity)
    for invalid in ({**control, "mode": "rw"},
                    {**control, "identity": {**identity, "seed": 43}}):
        with pytest.raises(ValueError, match="identical run identity"):
            validate_control(invalid, identity, require_control_improvement=False)
    for update in (None, -1, 33, True):
        with pytest.raises(ValueError, match="invalid selected update"):
            validate_control({**control, "selected_update": update}, identity,
                             require_control_improvement=False)


@pytest.mark.parametrize("feasible", [True, False])
def test_joint_selection_restores_best_weights_and_never_evaluates_trial_test(tmp_path, monkeypatch, feasible):
    import json
    from types import SimpleNamespace
    from additional_baselines import continuation_grid as workflow
    from additional_baselines.candidates import file_reference

    checkpoint = tmp_path / "source.ckpt"
    torch.save({"state_dict": {"weight": torch.zeros(1, 1)}}, checkpoint)
    sid = tmp_path / "sids.pt"
    sid.write_text("sids")
    export = tmp_path / "export"
    export.mkdir()
    config = dict(top_k_for_generation=100, sequence_length=20, num_hierarchies=4)
    (export / "config.yaml").write_text(json.dumps(config))
    data = tmp_path / "data"
    data.mkdir()
    for suffix in ("train", "valid.ranking", "test.ranking"):
        (data / ("amazon-beauty." + suffix)).write_text("pinned input")
    mapping = {i: str(i) for i in range(5)}
    metadata = dict(checkpoint=file_reference(checkpoint), sid_mapping=file_reference(sid))
    monkeypatch.setattr(workflow, "ROOT", tmp_path)
    monkeypatch.setattr(workflow, "load_beauty", lambda **kwargs: (
        SimpleNamespace(metadata=metadata, item_groups={"brand": mapping}), None, None))
    schedule = dict(users_per_batch=1, cap=5, accumulation=1, history_length=4,
                    model_history_length=5, weight_decay=0)
    monkeypatch.setattr(workflow, "native_schedule", lambda path: schedule)
    monkeypatch.setattr(workflow, "read_rows", lambda path, **kwargs: [(0, list(range(5)), 4)])
    batch = (torch.tensor([0]), torch.tensor([[0, 1, 2, 3]]), torch.tensor([4]), torch.tensor([0]))
    monkeypatch.setattr(workflow, "native_batches", lambda users, **kwargs: iter([batch] * kwargs["updates"]))

    class TinyModel(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.grid = torch.nn.Linear(1, 1, bias=False)
            self.grid.weight.data.zero_()
            self.tokenizer = SimpleNamespace(sid_table=torch.arange(5).reshape(5, 1))
            if config.get("grid_checkpoint_path"):
                self.grid.load_state_dict(torch.load(config["grid_checkpoint_path"], weights_only=True)["state_dict"])

        def compute_loss(self, interaction):
            return (self.grid.weight.reshape(-1) - 1).square()

    monkeypatch.setattr(workflow, "original_module", lambda name: SimpleNamespace(GRIDBaseModel=TinyModel))
    audit = dict(resumed=True)
    monkeypatch.setattr(workflow, "resume_optimizer", lambda model, *args, **kwargs: (
        torch.optim.SGD(model.grid.parameters(), lr=.1), audit))

    def metrics(recall, gini, counts):
        return dict(candidate_target_recall=.8, at_k={"5": dict(recall=recall, groups={
            "brand": dict(gini=gini), "popularity": dict(exposure_counts=dict(zip(map(str, range(5)), counts)))})})

    baseline = metrics(.5, .6, [0, 0, 0, 0, 100])
    best_metrics = metrics(.495, .5 if feasible else .7, [20] * 5)
    validation = iter([baseline, best_metrics, metrics(.5, .7, [20] * 5)])
    generated = []
    generated_weights = []

    def generate(model, requests, sids, mapping, metadata, **kwargs):
        generated.append(metadata["split"]["name"])
        generated_weights.append(model.grid.weight.item())
        return SimpleNamespace(metadata=metadata)

    monkeypatch.setattr(workflow, "generate_grid", generate)
    monkeypatch.setattr(workflow, "evaluate", lambda *args: next(validation))
    plan = dict(updates=2, seed=42, **{k: schedule[k] for k in
                ("users_per_batch", "cap", "accumulation", "history_length")})
    identity = dict(**metadata, seed=42, plan_sha256=workflow.batch_digest(workflow.native_batches(None, **plan)),
                    max_updates=2, validation_interval=1, validation_requests=None, learning_rate=.1,
                    optimizer_state="require", generation_beam_width=100, train_users=None)
    control = tmp_path / "control.json"
    control.write_text(json.dumps(dict(mode="ordinary", identity=identity, selected_update=1,
        selected_validation_metrics=baseline, optimizer_audit=audit)))
    output = tmp_path / "trial"
    result = workflow.run_continuation(mode="rw", output_dir=output, control_report=control,
        device="cpu", max_updates=2, validation_interval=1, learning_rate=.1, export_dir=export,
        data_dir=data, selection_objective="both", evaluate_test=False, rw_strength=0)
    assert generated == ["validation"] * 3
    assert not (output / "test_results.json").exists()
    if feasible:
        assert result["selected_update"] == 1
        selected = torch.load(output / "selected_checkpoint.ckpt", weights_only=True)
        assert selected["state_dict"]["weight"].item() == pytest.approx(.2)
        assert result["status"] == "fairness_improved"
        monkeypatch.setattr(workflow, "evaluate", lambda *args: baseline)
        monkeypatch.setattr(workflow, "save", lambda artifact, path: path.write_text("saved test candidates"))
        tested = workflow.evaluate_selected_continuation(output, tmp_path / "winner_test", device="cpu")
        assert generated[-1] == "test" and generated_weights[-1] == pytest.approx(.2)
        assert tested["input_metadata"]["checkpoint"] == result["selected_checkpoint"]
        assert (tmp_path / "winner_test/test_results.json").exists()
        sid.write_text("changed source")
        with pytest.raises(ValueError, match="input changed"):
            workflow.evaluate_selected_continuation(output, tmp_path / "drifted_test", device="cpu")
    else:
        assert result["selected_update"] is None and result["selected_checkpoint"] is None
        assert result["status"] == "no_feasible_fairness_checkpoint"
        assert not (output / "selected_checkpoint.ckpt").exists()
