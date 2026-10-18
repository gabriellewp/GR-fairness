import json
from pathlib import Path

import pytest

from additional_baselines import tune_continuation as tuning
from additional_baselines.candidates import file_reference
from additional_baselines.continuation_selection import selection_policy


def metrics(recall=.5, brand=.6, counts=(80, 20)):
    return {"at_k": {"5": {"recall": recall, "groups": {
        "brand": {"gini": brand}, "popularity": {"exposure_counts": dict(zip(("a", "b"), counts))}}}}}


@pytest.fixture
def sweep(tmp_path):
    control_dir = tmp_path / "ordinary"
    control_dir.mkdir()
    identity = dict(seed=42, max_updates=32, validation_interval=8, validation_requests=None,
                    train_users=None, generation_beam_width=100, learning_rate=.001, optimizer_state="require")
    for name in ("checkpoint", "sid_mapping"):
        path = tmp_path / name
        path.write_text(name)
        identity[name] = file_reference(path)
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    (export_dir / "config.yaml").write_text(json.dumps(dict(dataset="amazon-beauty", grid_tokenizer="rkmeans",
        grid_checkpoint_path=identity["checkpoint"]["path"], sid_path=identity["sid_mapping"]["path"])))
    control = dict(mode="ordinary", identity=identity, selected_update=8, selected_validation_metrics=metrics())
    (control_dir / "continuation.json").write_text(json.dumps(control))
    (control_dir / "validation_trace.json").write_text(json.dumps([
        dict(update=0, metrics=metrics(recall=.45)), dict(update=8, metrics=metrics())]))
    return dict(control_report=control_dir / "continuation.json", export_dir=export_dir,
                data_dir=tmp_path / "data", output_dir=tmp_path / "sweep",
                dataset="amazon-beauty", tokenizer="rkmeans")


def test_dry_run_contains_full_search_and_writes_nothing(sweep):
    plan = tuning.run_sweep(**sweep, dry_run=True)
    assert len(plan["trials"]) == 16
    assert plan["common"]["evaluate_test"] is False
    assert not sweep["output_dir"].exists()
    assert any(t.get("rw_target") == "uniform" and t.get("group") == "brand" for t in plan["trials"])


@pytest.mark.parametrize("feasible", [True, False])
def test_sweep_freezes_selection_before_testing_and_keeps_ordinary_when_infeasible(sweep, monkeypatch, feasible):
    calls, evaluations = [], []
    p = selection_policy("both", control_metrics=metrics(), history_shares={"a": .5, "b": .5})

    def run(**kwargs):
        calls.append(kwargs)
        assert kwargs["evaluate_test"] is False
        folder = Path(kwargs["output_dir"])
        folder.mkdir()
        return dict(selected_update=8 if feasible else None,
                    selected_validation_metrics=metrics(brand=.5, counts=(60, 40)) if feasible else None,
                    selected_checkpoint={"path": "frozen", "sha256": "fixture"} if feasible else None,
                    selection=p)

    def evaluate(run_dir, output_dir, **kwargs):
        selection = json.loads((sweep["output_dir"] / "selection.json").read_text())
        assert selection["selection_uses_test"] is False
        evaluations.append(run_dir)
        Path(output_dir).mkdir(parents=True)
        (Path(output_dir) / "test_results.json").write_text("{}")

    monkeypatch.setattr(tuning, "run_continuation", run)
    monkeypatch.setattr(tuning, "evaluate_selected_continuation", evaluate)
    trials = [dict(name="rw", mode="rw", rw_target="uniform"), dict(name="sprint", mode="sprint")]
    result = tuning.run_sweep(**sweep, trials=trials, evaluate_winners=True)
    assert len(calls) == 2 and len(evaluations) == (2 if feasible else 0)
    expected = "validation_fairness_improved" if feasible else "no_feasible_improvement_keep_ordinary"
    assert all(w["status"] == expected for w in result["winners"].values())


def test_trials_cannot_change_matched_identity_or_escape_output():
    for trial in (dict(name="bad", mode="rw", learning_rate=1), dict(name="../escape", mode="rw"),
                  dict(name="bad", mode="sprint", rw_strength=.5)):
        with pytest.raises(ValueError):
            tuning.validate_trials([trial])
