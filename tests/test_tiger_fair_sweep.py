import importlib
import json

import numpy as np
from scipy.sparse import csr_matrix


reranker_module = importlib.import_module("fairdiverse.recommendation.reranker")
RecReRanker = reranker_module.RecReRanker


class FakeTIGERFair:
    def __init__(self, config):
        self.config = config
        self.iid2pid = {0: 0, 1: 1}
        self.last_candidate_group_coverage = 1.0

    def rerank(self, ranking_scores, k):
        return np.argsort(ranking_scores, axis=-1)[:, ::-1][:, :k]


def test_sweep_reuses_scores_and_writes_every_variant(monkeypatch, tmp_path):
    config = {
        "dataset": "synthetic",
        "model": "TIGERFair",
        "rank_model": "TIGERFair",
        "fair-rank": True,
        "log_name": "tigerfair-pilot",
        "ranking_store_path": "unused",
        "topk": [1],
        "fairness_metrics": [],
        "fairness_type": "Exposure",
        "fairness_modes": ["positive_deficit", "normalized_debt"],
        "lambda_values": [0.1, 0.5],
        "group_num": 2,
        "item_num": 2,
        "decimals": 4,
    }
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(reranker_module, "TIGERFair", FakeTIGERFair)
    monkeypatch.setattr(
        RecReRanker,
        "load_configs",
        lambda self, directory: dict(self.train_config),
    )

    results = RecReRanker(config).rerank(
        ranking_scores=np.array([[0.8, 0.9]], dtype=np.float64)
    )

    assert set(results) == {
        "positive_deficit/lambda=0.1",
        "positive_deficit/lambda=0.5",
        "normalized_debt/lambda=0.1",
        "normalized_debt/lambda=0.5",
    }
    summary_paths = list(
        tmp_path.glob(
            "fairdiverse/recommendation/log/*_tigerfair-pilot-sweep/"
            "sweep_results.json"
        )
    )
    assert len(summary_paths) == 1
    with summary_paths[0].open() as stream:
        assert set(json.load(stream)) == set(results)


def test_reranker_reports_separate_retention_and_ground_truth_metrics(
    monkeypatch, tmp_path
):
    config = {
        "dataset": "synthetic",
        "model": "TIGERFair",
        "rank_model": "TIGERFair",
        "fair-rank": True,
        "log_name": "tigerfair-ground-truth",
        "topk": [1, 2],
        "fairness_metrics": [],
        "fairness_type": "Exposure",
        "group_num": 2,
        "item_num": 2,
        "decimals": 4,
    }
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(reranker_module, "TIGERFair", FakeTIGERFair)
    monkeypatch.setattr(
        RecReRanker,
        "load_configs",
        lambda self, directory: dict(self.train_config),
    )

    scores = np.array([[0.8, 0.9]], dtype=np.float64)
    # Item 0 is relevant, while the score-preserving order ranks item 1 first.
    ground_truth = csr_matrix(np.array([[1, 0]], dtype=np.int8))
    result = RecReRanker(config).rerank(scores, ground_truth=ground_truth)

    assert result["ranking_retention_ndcg@1"] == 1.0
    assert result["ground_truth_ndcg@1"] == 0.0
    assert result["ground_truth_hr@1"] == 0.0
    assert result["ground_truth_mrr@1"] == 0.0
    assert result["ground_truth_ndcg@2"] == 0.6309
    assert result["ground_truth_hr@2"] == 1.0
    assert result["ground_truth_mrr@2"] == 0.5
    assert result["ndcg@2"] == result["ground_truth_ndcg@2"]
    assert result["hr@2"] == result["ground_truth_hr@2"]
    assert result["mrr@2"] == result["ground_truth_mrr@2"]
    for metric in ("mmf", "gini", "entropy", "maxminratio"):
        assert f"{metric}@1" in result
        assert f"{metric}@2" in result


def test_sweep_summary_reports_relative_ndcg_loss_and_budget():
    scores = np.array([[0.9, 0.8], [0.9, 0.8]], dtype=np.float64)
    ground_truth = csr_matrix(np.array([[1, 0], [0, 1]], dtype=np.int8))
    results = {
        "acceptable": {"ground_truth_ndcg@1": 0.5},
        "too-lossy": {"ground_truth_ndcg@1": 0.4},
    }

    summary = RecReRanker._summarize_sweep(
        scores, ground_truth, [1], results, max_loss=0.05
    )

    assert summary["baseline"]["ground_truth_ndcg@1"] == 0.5
    assert summary["variants"]["acceptable"]["relative_ndcg_loss@1"] == 0.0
    assert summary["variants"]["acceptable"]["meets_relevance_budget"] is True
    assert summary["variants"]["too-lossy"]["relative_ndcg_loss@1"] == 0.2
    assert summary["variants"]["too-lossy"]["meets_relevance_budget"] is False
    assert summary["eligible_variants"] == ["acceptable"]
