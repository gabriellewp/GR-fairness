import numpy as np
import pytest

from fairdiverse.recommendation.rerank_model.TIGERFair import TIGERFair


def make_reranker(config=None, groups=None):
    """Build a small reranker without depending on a processed dataset."""
    groups = groups or {0: 0, 1: 1, 2: 0, 3: 1}
    reranker = TIGERFair.__new__(TIGERFair)
    reranker.config = {
        "lambda": 0.5,
        "invalid_score_threshold": -999.0,
        "relevance_normalization": "minmax",
        **(config or {}),
    }
    reranker.item_num = len(groups)
    reranker.group_num = max(groups.values()) + 1
    reranker.iid2pid = groups
    reranker.M = np.zeros((reranker.item_num, reranker.group_num))
    for item, group in groups.items():
        reranker.M[item, group] = 1
    reranker.last_exposure = np.zeros(reranker.group_num)
    return reranker


def test_zero_fairness_weight_preserves_generated_ranking():
    reranker = make_reranker({"lambda": 0.0})
    scores = np.array([[0.8, 0.9, -1000.0, 0.7]])

    result = reranker.rerank(scores, 2)

    assert result[0].tolist() == [1, 0]
    assert 2 not in result[0]


def test_exposure_deficit_balances_groups_across_users():
    reranker = make_reranker({"lambda": 3.0})
    scores = np.tile(np.array([1.0, 0.9, -1000.0, -1000.0]), (4, 1))

    result = reranker.rerank(scores, 1)
    selected_groups = [reranker.iid2pid[int(items[0])] for items in result]

    assert selected_groups.count(0) == 2
    assert selected_groups.count(1) == 2
    assert reranker.last_candidate_group_coverage == 1.0


def test_historical_exposure_favours_underexposed_group():
    reranker = make_reranker(
        {"lambda": 2.0, "initial_exposure": [10.0, 0.0]}
    )
    scores = np.array([[1.0, 0.9, -1000.0, -1000.0]])

    result = reranker.rerank(scores, 1)

    assert result[0].tolist() == [1]


def test_requires_enough_generated_candidates():
    reranker = make_reranker()
    scores = np.array([[1.0, -1000.0, -1000.0, -1000.0]])

    with pytest.raises(ValueError, match="fewer than k=2"):
        reranker.rerank(scores, 2)


def test_rejects_wrong_target_shape():
    reranker = make_reranker({"target_exposure": [1.0]})
    scores = np.array([[1.0, 0.9, -1000.0, -1000.0]])

    with pytest.raises(ValueError, match="one value per group"):
        reranker.rerank(scores, 1)


def test_rejects_negative_fairness_weight():
    reranker = make_reranker({"lambda": -1.0})
    scores = np.array([[1.0, 0.9, -1000.0, -1000.0]])

    with pytest.raises(ValueError, match="lambda"):
        reranker.rerank(scores, 1)


def test_fairness_modes_implement_distinct_deficit_equations():
    target = np.array([0.5, 0.5])
    exposure = np.array([8.0, 2.0])

    proposed = make_reranker({"fairness_mode": "positive_deficit"})
    np.testing.assert_allclose(
        proposed._fairness_bonus(target, exposure, 1.0), [0.0, 0.3]
    )

    original = make_reranker({"fairness_mode": "cumulative_debt"})
    np.testing.assert_allclose(
        original._fairness_bonus(target, exposure, 1.0), [-5.0, 7.0]
    )

    primary = make_reranker({"fairness_mode": "normalized_debt"})
    np.testing.assert_allclose(
        primary._fairness_bonus(target, exposure, 1.0), [0.0, 3.5 / 5.5]
    )


def test_rejects_unknown_fairness_mode():
    reranker = make_reranker({"fairness_mode": "unknown"})
    scores = np.array([[1.0, 0.9, -1000.0, -1000.0]])

    with pytest.raises(ValueError, match="fairness_mode"):
        reranker.rerank(scores, 1)
