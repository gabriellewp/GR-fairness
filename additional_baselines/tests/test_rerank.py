"""Integration checks for delegation to the unchanged FairDiverse classes."""

import inspect
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from additional_baselines import ExperimentInputs, Request, save
from additional_baselines.candidates import file_reference
from additional_baselines.rerank import FairDiverseReranker
from additional_baselines.rerank.legacy import load_class, ROOT
from additional_baselines.run_rerank import run


@pytest.fixture
def inputs(tmp_path):
    source = tmp_path / "source"
    source.write_text("fixture")
    ref = file_reference(source)
    return ExperimentInputs({
        "checkpoint": ref, "sid_mapping": ref,
        "split": {"name": "validation", "source": ref}, "seed": 42,
        "generation": {"beam_width": 4}, "score_convention": "utility",
        "candidate_membership": "explicit_fixture", "short_list_policy": "return_available",
    }, {"brand": {0: "a", 1: "a", 2: "b", 3: "b"}}, [
        Request("r0", [0, 1, 2, 3], [0.9, 0.8, 0.7, 0.6], [0]),
        Request("r1", [0, 2], [0.9, 0.8], [3]),
    ])


def adapter(model, **options):
    config = {"group": "brand", "fairness_lambda": 1.0}
    if model == "PMMF":
        config["weights"] = {"a": 1.0, "b": 1.0}
    config.update(options)
    return FairDiverseReranker(model=model, **config)


@pytest.mark.parametrize("model", ["CPFair", "PMMF"])
def test_adapter_matches_direct_original_constructor_and_method(inputs, model, monkeypatch):
    cls = load_class(model)
    assert Path(inspect.getfile(cls)).resolve() == (ROOT / "rerank_model" / f"{model}.py").resolve()
    # Let the real legacy constructor read the fixture matrix instead of disk.
    # Only this direct-reference test patches the data loader; production doesn't.
    base = sys.modules[cls.__bases__[0].__module__]
    matrix = np.array([[1., 0.], [1., 0.], [0., 1.], [0., 1.]])
    monkeypatch.setattr(base, "Build_Adjecent_Matrix", lambda _: (matrix.copy(), {0: 0, 1: 0, 2: 1, 3: 1}))
    config = {"item_num": 4, "group_num": 2, "lambda": 1.0, "learning_rate": 0.5, "gamma": 0.2}
    direct = cls(config)
    scores = np.array([[0.9, 0.8, 0.7, 0.6], [0.9, -np.inf, 0.8, -np.inf]])
    expected = direct.rerank(scores, 2)
    result = adapter(model).rerank(inputs, 2)
    assert list(result.rankings.values()) == [row.tolist() for row in expected]
    assert result.config["backend_source"] == file_reference(ROOT / "rerank_model" / f"{model}.py")
    assert not any(name.endswith(".TaxRank") for name in sys.modules if name.startswith("additional_baselines._legacy"))


@pytest.mark.parametrize("model", ["CPFair", "PMMF"])
def test_delegates_once_per_horizon_with_full_catalog(inputs, model, monkeypatch):
    cls = load_class(model)
    original = cls.rerank
    calls = []
    def spy(self, scores, k):
        calls.append((scores.copy(), self.item_num, self.config.copy()))
        return original(self, scores, k)
    monkeypatch.setattr(cls, "rerank", spy)
    inputs.requests = [Request("r0", [2], [-20000], [0]), Request("r1", [0], [0], [3])]
    result = adapter(model, matrix_mode="dense").rerank(inputs, 1)
    assert result.rankings == {"r0": [2], "r1": [0]}
    assert len(calls) == 1
    assert calls[0][0].shape == (2, 4)
    assert calls[0][1] == 4
    assert np.isneginf(calls[0][0][0, 0])
    assert calls[0][0][1, 0] == 0  # Zero utility is an eligible score.


@pytest.mark.parametrize("model", ["CPFair", "PMMF"])
def test_insufficient_candidates_fail_before_backend_call(inputs, model, monkeypatch):
    def unexpected(*args):
        pytest.fail("Backend must not run with insufficient candidates")
    monkeypatch.setattr(load_class(model), "rerank", unexpected)
    inputs.requests[1] = Request("r1", [], [], [3])
    with pytest.raises(ValueError, match="fewer than k"):
        adapter(model).rerank(inputs, 1)


def test_native_budget_overshoot_is_reported_then_exhausted_provider_rejected(inputs):
    inputs.requests = [Request("r0", [0, 1], [0.9, 0.8], [0]), Request("r1", [2, 3], [0.9, 0.8], [3])]
    method = adapter("PMMF", weights={"a": 0.3, "b": 1.0})
    result = method.rerank(inputs, 2, trace=True)
    assert result.diagnostics["horizons"][0]["overshoot"]["a"] == pytest.approx(0.8)
    assert result.trace[0]["budget_after"]["a"] == pytest.approx(-0.8)
    inputs.requests[1] = Request("r1", [0, 1], [0.9, 0.8], [3])
    with pytest.raises(ValueError, match="exhausted provider"):
        method.rerank(inputs, 2)


@pytest.mark.parametrize("model", ["CPFair", "PMMF"])
def test_state_resets_each_call_cutoff_and_horizon(inputs, model):
    method = adapter(model, horizon=1, order_seed=42)
    result = method.rerank(inputs, 2, trace=True)
    assert result == method.rerank(inputs, 2, trace=True)
    assert [h["length"] for h in result.diagnostics["horizons"]] == [1, 1]
    variant = "cpfair_fairdiverse" if model == "CPFair" else "pmmf_fairdiverse"
    config = {"group": "brand", "fairness_lambda": 1.0, "horizon": 1, "order_seed": 42}
    if model == "PMMF":
        config["weights"] = {"a": 1, "b": 1}
    report = run(inputs, {"method": variant, "parameters": config, "topk": [1, 2]})
    for k in (1, 2):
        assert report["runs"][str(k)]["rankings"] == method.rerank(inputs, k).rankings


def test_noncontiguous_ids_and_negative_log_scores(inputs):
    inputs.item_groups = {"brand": {10: "a", 20: "b", 30: "b"}}
    inputs.requests = [Request("r0", [20, 10], [2., 1.], [30])]
    inputs.metadata["score_convention"] = "negative_log_probability"
    result = adapter("CPFair", fairness_lambda=0).rerank(inputs, 1)
    assert result.rankings == {"r0": [10]}


@pytest.mark.parametrize("options", [
    {"weights": None}, {"weights": {"a": 1}}, {"weights": {"a": 0, "b": 1}},
    {"gamma": 1.5}, {"learning_rate": -1}, {"horizon": 0},
    {"fairness_lambda": float("nan")}, {"max_dense_bytes": 1},
])
def test_bad_configuration_fails(inputs, options):
    with pytest.raises(ValueError):
        adapter("PMMF", **options).rerank(inputs, 1)


def test_rejects_removed_authors_variant(inputs):
    with pytest.raises(ValueError, match="method"):
        run(inputs, {"method": "pmmf_authors", "parameters": {}, "topk": [1]})


@pytest.mark.parametrize("model", ["CPFair", "PMMF"])
def test_cli_runs_native_backend(inputs, tmp_path, model):
    artifact, config, output = [tmp_path / name for name in ("inputs.json", "config.json", "output.json")]
    save(inputs, artifact)
    parameters = {"group": "brand", "fairness_lambda": 1}
    if model == "PMMF":
        parameters["weights"] = {"a": 1, "b": 1}
    variant = "cpfair_fairdiverse" if model == "CPFair" else "pmmf_fairdiverse"
    config.write_text(json.dumps({"method": variant, "parameters": parameters, "topk": [1, 2], "trace": True}))
    command = [sys.executable, "-m", "additional_baselines", "rerank", str(artifact),
               "--config", str(config), "--output", str(output)]
    subprocess.run(command, check=True)
    report = json.loads(output.read_text())
    assert report["runs"]["1"]["variant"] == variant
    assert report["input_artifact"] == file_reference(artifact)
    assert report["control_metrics"]["candidate_hit_rate"] == 0.5
    before = output.read_bytes()
    assert subprocess.run(command, capture_output=True).returncode != 0
    assert output.read_bytes() == before
