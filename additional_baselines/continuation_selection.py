"""Validation-only fairness selection against a matched ordinary continuation."""

import math


OBJECTIVES = ("recall", "popularity", "brand", "both")


def objective_values(metrics, history_shares):
    at5 = metrics["at_k"]["5"]
    counts = at5["groups"]["popularity"]["exposure_counts"]
    if set(counts) != set(history_shares) or sum(counts.values()) <= 0:
        raise ValueError("Popularity groups must match nonempty validation histories")
    total = sum(counts.values())
    values = dict(recall=at5["recall"], brand=at5["groups"]["brand"]["gini"],
                  popularity=sum(abs(counts[g] / total - history_shares[g])
                                 for g in counts) / len(counts))
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in values.values()):
        raise ValueError("Validation objectives must be finite values in [0, 1]")
    return values


def selection_policy(objective="recall", *, max_relative_recall_loss=0.02,
                     control_metrics=None, history_shares=None):
    if objective not in OBJECTIVES:
        raise ValueError("Unknown validation selection objective")
    if not math.isfinite(max_relative_recall_loss) or not 0 <= max_relative_recall_loss < 1:
        raise ValueError("Recall loss budget must be in [0, 1)")
    policy = dict(objective=objective, cutoff=5,
                  max_relative_recall_loss=max_relative_recall_loss)
    if objective != "recall":
        if control_metrics is None or history_shares is None:
            raise ValueError("Fairness selection requires matched ordinary validation metrics and histories")
        policy["control"] = objective_values(control_metrics, history_shares)
        policy["history_shares"] = history_shares
        policy["recall_floor"] = (1 - max_relative_recall_loss) * policy["control"]["recall"]
        policy["rule"] = ("No selected fairness axis may worsen; at least one must improve. "
                          "Minimize worst relative fairness ratio, then sum of ratios, "
                          "then maximize Recall@5, then prefer earlier update.")
    else:
        policy["rule"] = "Maximum validation Recall@5; earlier update wins ties; step zero eligible"
    return policy


def checkpoint_key(row, policy):
    """Larger is better; None means infeasible. Test metrics never enter here."""
    if type(row["update"]) is not int or row["update"] < 0:
        raise ValueError("Invalid checkpoint update")
    if policy["objective"] == "recall":
        recall = row["metrics"]["at_k"]["5"]["recall"]
        if not math.isfinite(recall) or not 0 <= recall <= 1:
            raise ValueError("Validation Recall@5 must be finite and in [0, 1]")
        return recall, -row["update"]
    values = objective_values(row["metrics"], policy["history_shares"])
    control = policy["control"]
    axes = ("popularity", "brand") if policy["objective"] == "both" else (policy["objective"],)
    if values["recall"] < policy["recall_floor"] - 1e-12:
        return None
    if any(values[a] > control[a] + 1e-12 for a in axes):
        return None
    if not any(values[a] < control[a] - 1e-12 for a in axes):
        return None
    # A zero baseline has no room to improve. Neutral zero/zero contributes 1.
    ratios = [values[a] / control[a] if control[a] > 1e-12 else 1.0 for a in axes]
    return -max(ratios), -sum(ratios), values["recall"], -row["update"]


def select_checkpoint(trace, policy):
    if not trace or trace[0]["update"] != 0:
        raise ValueError("Selection requires zero-step validation")
    eligible = [(checkpoint_key(row, policy), row) for row in trace]
    eligible = [(key, row) for key, row in eligible if key is not None]
    return max(eligible, key=lambda pair: pair[0])[1] if eligible else None
