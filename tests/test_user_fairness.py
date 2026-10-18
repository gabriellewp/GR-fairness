import csv
import json
import math

import pytest

from fairdiverse.recommendation.user_fairness import (
    load_activity, prepare_activity, ranking_metrics, save_report, summarize, write_activity,
)


def test_ties_and_cold_start(tmp_path):
    write_activity({"zero": 0, "a": 1, "b": 1, "c": 5, "d": 9, "e": 9}, tmp_path, {})
    mapping = load_activity(tmp_path / "user_activity.csv")
    assert mapping["zero"]["activity_group"] == "cold_start"
    assert mapping["a"]["activity_group"] == mapping["b"]["activity_group"] == "low"
    assert mapping["c"]["activity_group"] == "medium"
    assert mapping["d"]["activity_group"] == mapping["e"]["activity_group"] == "high"
    write_activity({"a": 5, "b": 5}, tmp_path, {})
    assert {r["activity_group"] for r in load_activity(tmp_path / "user_activity.csv").values()} == {"warm"}
    path = tmp_path / "user_activity.csv"
    path.write_text(path.read_text().replace(",5,", ",6,"))
    with pytest.raises(ValueError, match="fingerprint"):
        load_activity(path)


def test_rank_metrics_include_misses_and_multiple_targets():
    metrics = ranking_metrics([3, 2], [2, 7], [1, 3])
    assert metrics["recall@1"] == 0
    assert metrics["recall@3"] == 0.5
    assert metrics["ndcg@3"] == pytest.approx((1 / math.log2(3)) / (1 + 1 / math.log2(3)))
    assert ranking_metrics([], [1], [10])["ndcg@10"] == 0
    with pytest.raises(ValueError, match="duplicate"):
        ranking_metrics([1, 1], [1], [2])


def test_user_macro_average_and_known_gap(tmp_path):
    write_activity({"a": 1, "b": 1, "c": 10}, tmp_path, {})
    rows = [{"user_id": user, "request_id": str(i), "ndcg@10": value, "recall@10": value}
            for i, (user, value) in enumerate([("a", 1), ("a", 1), ("b", 0), ("c", 1)])]
    users, report = summarize(rows, load_activity(tmp_path / "user_activity.csv"), 30)
    assert report["statistics"]["overall"]["ndcg@10"] == pytest.approx(2 / 3)
    assert report["statistics"]["group/low"]["ndcg@10"] == 0.5
    assert report["statistics"]["gap"]["ndcg@10"] == 0.5
    assert report["statistics"]["high_minus_low"]["ndcg@10"] == 0.5
    assert users[0]["request_count"] == 2
    assert summarize(rows, load_activity(tmp_path / "user_activity.csv"), 30)[1] == report
    saved = save_report(rows, tmp_path / "user_activity.csv", tmp_path / "report", 30)
    assert json.loads((tmp_path / "report/user_fairness_summary.json").read_text()) == saved
    assert len(list(csv.DictReader((tmp_path / "report/per_user_metrics.csv").open()))) == 3
    with pytest.raises(ValueError, match="Missing activity"):
        summarize(rows, {})
    with pytest.raises(ValueError, match="Duplicate evaluation"):
        summarize(rows + rows[:1], load_activity(tmp_path / "user_activity.csv"))


def test_cached_standard_counts_ignore_heldout_and_negative_rows(tmp_path):
    (tmp_path / "process_config.yaml").write_text("user_id: uid\nlabel_id: label\n")
    (tmp_path / "sample.train").write_text("uid\tlabel\nhistory_user\t1\nhistory_user\t1\nhistory_user\t0\n")
    for split in ("valid", "test"):
        (tmp_path / f"sample.{split}.ranking").write_text("user_id\nnew_user\nhistory_user\n")
    prepare_activity(tmp_path, "sample", tmp_path)
    mapping = load_activity(tmp_path / "user_activity.csv")
    assert mapping["history_user"]["train_interaction_count"] == "2"
    assert mapping["new_user"]["train_interaction_count"] == "0"


def test_cached_grid_uses_original_sequence_length(tmp_path, monkeypatch):
    from fairdiverse.recommendation import grid_data_bridge as bridge
    monkeypatch.setattr(bridge, "_read_tfrecords", lambda *a: [(10, [1, 2, 3]), (20, [4])])
    monkeypatch.setattr(bridge, "_source_fingerprint", lambda *a: "same")
    (tmp_path / "grid_data_manifest.json").write_text(json.dumps({"source_dir": "source", "source_fingerprint": "same"}))
    prepare_activity(tmp_path, "sample", tmp_path)
    mapping = load_activity(tmp_path / "user_activity.csv")
    assert mapping["0"]["train_interaction_count"] == "3"
    assert mapping["1"]["train_interaction_count"] == "1"
    monkeypatch.setattr(bridge, "_source_fingerprint", lambda *a: "changed")
    with pytest.raises(ValueError, match="source changed"):
        prepare_activity(tmp_path, "sample", tmp_path)


def test_ranking_evaluator_preserves_aggregate_and_exports_users(tmp_path):
    import torch
    from fairdiverse.recommendation.evaluator import Ranking_Evaluator

    class Model:
        IR_type = ["retrieval"]

        def eval(self):
            pass

        def full_predict(self, users, items):
            return torch.tensor([[0.1, 0.8, 0.2]])

    evaluator = Ranking_Evaluator.__new__(Ranking_Evaluator)
    evaluator.config = {"model": "SASRec", "topk": [1, 2], "group_num": 2,
                        "fairness_type": "Exposure", "mmf_eval_ratio": 0.5,
                        "decimals": 4, "device": "cpu", "item_num": 3}
    evaluator.iid2pid = {0: 0, 1: 0, 2: 1}
    batch = (torch.tensor([7, 8]), torch.tensor([[0], [0]]),
             torch.tensor([[0, 1, 2], [1, 0, 2]]), torch.tensor([1, 2]))
    before = evaluator.eval([batch], Model())
    write_activity({7: 1, 8: 10}, tmp_path, {})
    evaluator.config.update(user_fairness=True, user_activity_path=str(tmp_path / "user_activity.csv"),
                            user_fairness_bootstrap_samples=10)
    after, scores = evaluator.eval([batch], Model(), store_scores=True)
    assert before == after
    assert scores.shape == (2, 3)
    assert evaluator.last_user_metrics[0]["recall@2"] == 0
    assert evaluator.last_user_metrics[1]["recall@1"] == 0.5
    summary = evaluator.save_user_fairness(tmp_path / "report")
    assert summary["user_count"] == 2
    evaluator.eval([batch], Model())
    assert len(evaluator.last_user_metrics) == 2  # reset between validation/test calls


def test_native_grid_hits_misses_and_single_target_guard(tmp_path):
    import torch
    from fairdiverse.recommendation.evaluator import Ranking_Evaluator

    class Model:
        def eval(self):
            pass

        def generate(self, history, user):
            return torch.tensor([[[1, 1], [2, 2]]]), None

        def _target_sids(self, items):
            return items[:, None].repeat(1, 2)

    write_activity({0: 1, 1: 9}, tmp_path, {})
    evaluator = Ranking_Evaluator.__new__(Ranking_Evaluator)
    evaluator.config = {"model": "GRID", "grid_native_metrics": True, "topk": [1, 2],
                        "device": "cpu", "decimals": 4, "user_fairness": True,
                        "user_activity_path": str(tmp_path / "user_activity.csv")}
    batch = (torch.tensor([0, 1]), torch.tensor([[1], [1]]), torch.tensor([[2], [3]]), torch.tensor([1, 1]))
    result = evaluator.eval([batch], Model())
    assert result["hr@2"] == 0.5
    assert evaluator.last_user_metrics[0]["ndcg@2"] == pytest.approx(1 / math.log2(3))
    assert evaluator.last_user_metrics[1]["recall@2"] == 0
    with pytest.raises(ValueError, match="exactly one target"):
        evaluator.eval([(*batch[:3], torch.tensor([2, 1]))], Model())


def test_rerank_uses_explicit_request_user_mapping(tmp_path):
    from additional_baselines.candidates import ExperimentInputs, Request, file_reference
    from additional_baselines.evaluation import evaluate

    source = tmp_path / "source"
    source.write_text("fixture")
    ref = file_reference(source)
    inputs = ExperimentInputs({"checkpoint": ref, "sid_mapping": ref,
        "split": {"name": "test", "source": ref}, "seed": 42,
        "generation": {"beam_width": 2}, "score_convention": "utility",
        "candidate_membership": "explicit", "short_list_policy": "return_available"},
        {"brand": {0: "a", 1: "b"}}, [Request("request0", [0, 1], [0.9, 0.1], [0])])
    write_activity({"user7": 3}, tmp_path, {})
    options = {"activity_path": tmp_path / "user_activity.csv", "output_dir": tmp_path / "report",
               "request_users": {"request0": "user7"}, "bootstrap_samples": 0}
    result = evaluate(inputs, [1], {"request0": [1]}, user_fairness=options)
    assert result["user_fairness"]["statistics"]["overall"]["recall@1"] == 0
    with pytest.raises(ValueError, match="request_users"):
        evaluate(inputs, [1], user_fairness={**options, "request_users": {}})
    from additional_baselines.run_rerank import run
    run(inputs, {"method": "cpfair_fairdiverse", "parameters": {"group": "brand", "fairness_lambda": 0},
                 "topk": [1, 2], "user_fairness": options})
    for name in ("control", "reranked_1", "reranked_2"):
        assert (tmp_path / "report" / name / "user_fairness_summary.json").exists()
