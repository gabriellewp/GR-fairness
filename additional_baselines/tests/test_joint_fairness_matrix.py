import json
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "joint_fairness_trials", Path(__file__).resolve().parents[2] / "scripts/run_joint_fairness_trials.py")
matrix = importlib.util.module_from_spec(spec)
spec.loader.exec_module(matrix)


def test_trial_matrix_covers_each_case_and_setting_exactly_once(tmp_path):
    cases = [dict(index=i, sweep_options={"output_dir": str(tmp_path / f"case_{i}")},
                  plan={"trials": [dict(name=f"trial_{j}", mode="rw") for j in range(16)]})
             for i in range(12)]
    source = tmp_path / "matrix.json"
    source.write_text(json.dumps(dict(code={}, cases=cases)))
    manifest_path = matrix.initialize(source)
    prepared, original = matrix.load_manifest(manifest_path)
    assert len(prepared["trials"]) == 192
    assert {(t["case_index"], t["trial_index"]) for t in prepared["trials"]} == {
        (c, t) for c in range(12) for t in range(16)}
    assert [t["index"] for t in prepared["trials"]] == list(range(192))
    assert [t["case_index"] for t in prepared["trials"][:12]] == list(range(12))
    assert original["cases"] == cases
    source.write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        matrix.load_manifest(manifest_path)


def test_incomplete_case_cannot_select_or_evaluate_test(tmp_path, monkeypatch):
    case = dict(sweep_options={"output_dir": str(tmp_path)},
                plan={"trials": [dict(name="missing", mode="rw")]})
    monkeypatch.setattr(matrix, "load_manifest", lambda path: ({}, {"cases": [case]}))
    monkeypatch.setattr(matrix, "verify_case", lambda case: None)
    monkeypatch.setattr(matrix, "choose_winners", lambda *args: pytest.fail("Incomplete search must not select"))
    monkeypatch.setattr(matrix, "evaluate_selected_continuation", lambda *args, **kwargs:
                        pytest.fail("Incomplete search must not evaluate test"))
    with pytest.raises(RuntimeError, match="incomplete"):
        matrix.finalize_case("fixture", 0)
    result = json.loads((tmp_path / "incomplete.json").read_text())
    assert result["selection_performed"] is result["test_evaluated"] is False


def test_array_trial_uses_frozen_case_and_method_settings(tmp_path, monkeypatch):
    settings = dict(name="rw_uniform", mode="rw", rw_target="uniform", rw_strength=.25)
    case = dict(dataset="amazon-toys", tokenizer="rvq",
                sweep_options={"output_dir": str(tmp_path)},
                plan={"trials": [settings], "common": dict(evaluate_test=False, seed=42)})
    prepared = dict(trials=[dict(index=0, case_index=0, trial_index=0, settings=settings)])
    monkeypatch.setattr(matrix, "load_manifest", lambda path: (prepared, {"cases": [case]}))
    monkeypatch.setattr(matrix, "verify_case", lambda case: None)
    calls = []

    def continuation(**kwargs):
        calls.append(kwargs)
        return dict(status="no_feasible_fairness_checkpoint")

    monkeypatch.setattr(matrix, "run_continuation", continuation)
    matrix.run_trial("fixture", 0)
    assert calls == [dict(output_dir=tmp_path / "rw_uniform", evaluate_test=False, seed=42,
                          mode="rw", rw_target="uniform", rw_strength=.25)]
    assert (tmp_path / "completed_rw_uniform.json").exists()
