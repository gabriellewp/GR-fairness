from copy import deepcopy
from types import SimpleNamespace

import pytest

from additional_baselines.candidates import Request
from additional_baselines.generation_parity import generation_parity


def artifact():
    request = Request("u", list(range(30)), [1 - i / 100 for i in range(30)], [0])
    # Sorting is the same stable score ordering used by ExperimentInputs.
    return SimpleNamespace(requests=[request], ranking=lambda r: [i for _, i in sorted(
        zip(r.scores, r.candidate_ids), reverse=True)])


def test_bottom_rank_roundoff_is_recorded_without_changing_reported_ranks():
    a = artifact()
    a.requests[0].scores[-2:] = [.3, .3 - 1e-8]
    b = deepcopy(a)
    b.requests[0].scores[-2:] = [.3 - 1e-8, .3]
    result = generation_parity(a, b)
    assert result["passed"] and not result["exact_rankings"]
    assert result["lower_rank_differences"] == [dict(request_id="u", first_different_rank=29)]
    assert all(result["top_k_rankings_equal"].values())


@pytest.mark.parametrize("change", ["top20", "membership", "score", "target", "count"])
def test_parity_rejects_reported_rank_membership_and_score_changes(change):
    a, b = artifact(), artifact()
    if change == "top20":
        a.requests[0].scores[18:20] = [.8, .8 - 1e-8]
        b.requests[0].scores[18:20] = [.8 - 1e-8, .8]
    elif change == "membership":
        b.requests[0].candidate_ids[-1] = 999
    elif change == "score":
        b.requests[0].scores[-1] -= .1
    elif change == "target":
        b.requests[0].targets = [1]
    else:
        b.requests = []
    assert not generation_parity(a, b)["passed"]


def test_scores_are_compared_by_item_even_when_candidate_storage_order_changes():
    a, b = artifact(), artifact()
    b.requests[0].candidate_ids.reverse()
    b.requests[0].scores.reverse()
    assert generation_parity(a, b)["passed"]
