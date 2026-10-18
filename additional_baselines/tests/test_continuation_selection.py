import pytest
import torch

from additional_baselines.continuation_selection import checkpoint_key, select_checkpoint, selection_policy
from additional_baselines.ifairlrs import rw_weights


def metrics(recall=.5, brand=.6, counts=(80, 20)):
    return {"at_k": {"5": {"recall": recall, "groups": {
        "brand": {"gini": brand}, "popularity": {"exposure_counts": dict(zip(("a", "b"), counts))}}}}}


def policy(**kwargs):
    return selection_policy("both", control_metrics=metrics(), history_shares={"a": .5, "b": .5}, **kwargs)


def row(update, **kwargs):
    return dict(update=update, metrics=metrics(**kwargs))


def test_both_rejects_fairness_tradeoffs_and_recall_loss():
    p = policy(max_relative_recall_loss=.02)
    assert checkpoint_key(row(1, brand=.7, counts=(60, 40)), p) is None
    assert checkpoint_key(row(1, brand=.5, counts=(90, 10)), p) is None
    assert checkpoint_key(row(1, recall=.489, brand=.5, counts=(60, 40)), p) is None
    assert checkpoint_key(row(1, recall=.49, brand=.5, counts=(60, 40)), p) is not None
    assert select_checkpoint([row(0), row(1, brand=.7, counts=(60, 40))], p) is None


def test_balanced_relative_improvement_and_ties():
    trace = [row(0), row(1, brand=.59, counts=(50, 50)),
             row(2, brand=.48, counts=(70, 30)), row(3, brand=.48, counts=(70, 30))]
    # The near-zero MGU at update 1 cannot outweigh negligible brand improvement.
    assert select_checkpoint(trace, policy())["update"] == 2
    trace.append(row(4, recall=.51, brand=.48, counts=(70, 30)))
    assert select_checkpoint(trace, policy())["update"] == 4


def test_zero_fairness_baseline_and_invalid_values():
    p = selection_policy("both", control_metrics=metrics(counts=(50, 50)),
                         history_shares={"a": .5, "b": .5})
    assert checkpoint_key(row(1, brand=.5, counts=(50, 50)), p) is not None
    assert checkpoint_key(row(1, brand=.5, counts=(51, 49)), p) is None
    for budget in (-.1, 1, float("nan")):
        with pytest.raises(ValueError, match="budget"):
            policy(max_relative_recall_loss=budget)
    with pytest.raises(ValueError, match="finite"):
        checkpoint_key(row(1, brand=float("nan")), policy())


def test_uniform_rw_balances_observed_targets_and_zero_strength_recovers_control():
    targets = torch.tensor([0] * 9 + [1])
    batch = (torch.zeros(10, dtype=torch.long), targets[:, None], targets, torch.zeros(10))
    mapping = {0: "popular", 1: "rare", 2: "unused"}
    legacy = rw_weights([batch], mapping)
    assert legacy["weights"] == {"popular": 1, "rare": 1, "unused": 0}
    uniform = rw_weights([batch], mapping, target_distribution="uniform")
    assert uniform["weights"]["rare"] > 1 > uniform["weights"]["popular"]
    weighted_mass = {g: uniform["weights"][g] * uniform["target_shares"][g] for g in mapping.values()}
    assert weighted_mass == pytest.approx({"popular": .5, "rare": .5, "unused": 0})
    zero = rw_weights([batch], mapping, target_distribution="uniform", strength=0)
    assert [zero["weights"][mapping[int(t)]] for t in targets] == [1] * len(targets)
    assert uniform["variant"] != legacy["variant"]


@pytest.mark.parametrize("kwargs", [{"target_distribution": "inverse"}, {"strength": -1}, {"strength": float("nan")}])
def test_rw_rejects_invalid_weighting_before_counting(kwargs):
    with pytest.raises(ValueError):
        rw_weights([], {}, **kwargs)
