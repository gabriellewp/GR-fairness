import pytest

from additional_baselines.candidates import ExperimentInputs, Request, file_reference
from additional_baselines.report_beauty_activity import (
    check_aggregate, make_rows, validate_alignment,
)


def test_backfill_rejects_row_and_target_drift_and_keeps_generation_miss(tmp_path):
    split = tmp_path / "test.ranking"
    split.write_text("user_id\ttarget_item\n7\t4\n8\t5\n")
    ref = file_reference(split)
    inputs = ExperimentInputs(
        {"checkpoint": ref, "sid_mapping": ref, "split": {"name": "test", "source": ref},
         "seed": 42, "generation": {"beam": 2}, "score_convention": "utility",
         "candidate_membership": "fixture", "short_list_policy": "return_available"},
        {"group": {1: "a", 4: "b", 5: "b"}},
        [Request("test:0:user:7", [1], [1.0], [4]),
         Request("test:1:user:8", [5], [1.0], [5])],
    ).validate()
    test_rows = [{"user_id": "7", "target_item": "4"},
                 {"user_id": "8", "target_item": "5"}]
    activity = {"7": {"activity_group": "low"}, "8": {"activity_group": "high"}}
    users = validate_alignment(inputs, test_rows, activity, ref)
    rows = make_rows(inputs, users, None)
    assert rows[0]["candidate_coverage"] == rows[0]["recall@10"] == 0
    assert rows[1]["candidate_coverage"] == rows[1]["recall@10"] == 1
    expected = {"request_count": 2, "candidate_target_recall": 0.5,
                "at_k": {str(k): {"recall": 0.5, "ndcg": 0.5} for k in (5, 10, 20)}}
    check_aggregate(rows, expected)
    with pytest.raises(ValueError, match="disagrees"):
        check_aggregate(rows, {**expected, "candidate_target_recall": 1})
    inputs.requests[0].request_id = "test:0:user:8"
    with pytest.raises(ValueError, match="Request/user order"):
        validate_alignment(inputs, test_rows, activity, ref)
    inputs.requests[0].request_id = "test:0:user:7"
    inputs.requests[0].targets = [5]
    with pytest.raises(ValueError, match="Held-out target"):
        validate_alignment(inputs, test_rows, activity, ref)


def test_backfill_rejects_extra_and_ineligible_rankings(tmp_path):
    split = tmp_path / "split"
    split.write_text("fixture")
    ref = file_reference(split)
    inputs = ExperimentInputs(
        {"checkpoint": ref, "sid_mapping": ref, "split": {"name": "test", "source": ref},
         "seed": 42, "generation": {"beam": 1}, "score_convention": "utility",
         "candidate_membership": "fixture", "short_list_policy": "return_available"},
        {"group": {1: "a", 2: "a"}}, [Request("test:0:user:7", [1], [1.0], [2])],
    ).validate()
    with pytest.raises(ValueError, match="exactly"):
        make_rows(inputs, {"test:0:user:7": "7"}, {})
    with pytest.raises(ValueError, match="Ineligible"):
        make_rows(inputs, {"test:0:user:7": "7"}, {"test:0:user:7": [2]})
