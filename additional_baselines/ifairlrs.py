"""IFairLRS equations 8–14, with explicit GRID distance/group adaptations."""

from collections import Counter
import hashlib
import json
import math

from .evaluation import evaluate


def mapping_digest(mapping):
    return hashlib.sha256(json.dumps(sorted(mapping.items()), separators=(",", ":")).encode()).hexdigest()


def popularity_groups(counts, item_ids, n_groups=5):
    """Equal catalog-size bins, low to high popularity; item ID breaks ties."""
    items = sorted(item_ids, key=lambda i: (counts.get(i, 0), i))
    if type(n_groups) is not int or not 1 <= n_groups <= len(items):
        raise ValueError("Popularity group count must be between 1 and catalog size")
    return {item: str(min(index * n_groups // len(items), n_groups - 1))
            for index, item in enumerate(items)}


def shares(items, mapping):
    counts = Counter(mapping[item] for item in items)
    total = sum(counts.values())
    if not total:
        raise ValueError("Group shares require at least one interaction/recommendation")
    return {g: counts[g] / total for g in sorted(set(mapping.values()))}


def history_shares(histories, request_ids, mapping):
    if set(histories) != set(request_ids):
        raise ValueError("Histories must match request IDs exactly")
    return shares((i for rid in request_ids for i in histories[rid]), mapping)


def fairness(inputs, histories, group, cutoffs, rankings=None):
    mapping = inputs.item_groups[group]
    gh = history_shares(histories, [r.request_id for r in inputs.requests], mapping)
    rankings = rankings or {r.request_id: inputs.ranking(r) for r in inputs.requests}
    result = {}
    for k in cutoffs:
        items = [i for r in inputs.requests for i in rankings[r.request_id][:k]]
        if not items:
            result[str(k)] = {"GH": gh, "GP": None, "GU": None, "MGU": None, "DGU": None}
            continue
        gp = shares(items, mapping)
        gu = {g: gp[g] - gh[g] for g in gh}
        result[str(k)] = {"GH": gh, "GP": gp, "GU": gu,
                          "MGU": sum(map(abs, gu.values())) / len(gu),
                          "DGU": max(gu.values()) - min(gu.values())}
    return result


def fit_rr(validation, histories, group="popularity", cutoffs=(1, 5, 10, 20), alpha=0.05, epsilon=1e-6):
    validation.validate()
    if validation.metadata["split"]["name"] not in {"valid", "validation"}:
        raise ValueError("RR corrections must be fitted on validation only")
    cutoffs = list(cutoffs)
    if not cutoffs or len(set(cutoffs)) != len(cutoffs) or any(type(k) is not int or k <= 0 for k in cutoffs):
        raise ValueError("Cutoffs must be unique positive integers")
    if not math.isfinite(alpha) or alpha < 0 or not math.isfinite(epsilon) or not 0 < epsilon <= 1:
        raise ValueError("Require alpha >= 0 and 0 < epsilon <= 1")
    stats = fairness(validation, histories, group, cutoffs)
    if any(s["GU"] is None for s in stats.values()):
        raise ValueError("Cannot fit RR without validation recommendations")
    mapping = validation.item_groups[group]
    punishment = {g: sum(k / sum(cutoffs) * stats[str(k)]["GU"][g] for k in cutoffs)
                  for g in set(mapping.values())}
    scale = max(map(abs, punishment.values()))
    normalized = {g: u / scale if scale else 0.0 for g, u in punishment.items()}
    return {"schema_version": 1, "variant": f"ifairlrs_rr_grid_{group}_adaptation", "group": group,
            "alpha": alpha, "epsilon": epsilon, "cutoffs": cutoffs,
            "punishment": normalized, "raw_punishment": punishment,
            "validation_fairness": stats, "validation_metadata": validation.metadata,
            "group_mapping_sha256": mapping_digest(mapping),
            "distance": "negative_log_sid_probability", "short_list_policy": "return_available"}


def apply_rr(inputs, frozen, cutoffs=(5, 10, 20), histories=None):
    inputs.validate()
    group = frozen["group"]
    if frozen.get("schema_version") != 1 or frozen.get("distance") != "negative_log_sid_probability":
        raise ValueError("Unsupported RR correction artifact")
    alpha, epsilon = frozen["alpha"], frozen["epsilon"]
    if not math.isfinite(alpha) or alpha < 0 or not math.isfinite(epsilon) or not 0 < epsilon <= 1:
        raise ValueError("Invalid frozen alpha/epsilon")
    mapping = inputs.item_groups[group]
    if mapping_digest(mapping) != frozen["group_mapping_sha256"]:
        raise ValueError("Frozen RR group mapping differs from evaluation mapping")
    for key in ("checkpoint", "sid_mapping", "generation", "seed", "candidate_membership"):
        if inputs.metadata[key] != frozen["validation_metadata"][key]:
            raise ValueError(f"Validation/test {key} differs; fit a separate correction")
    correction = frozen["punishment"]
    if set(correction) != set(mapping.values()) or any(not math.isfinite(u) or not -1 <= u <= 1 for u in correction.values()):
        raise ValueError("Invalid frozen group punishment")
    convention = inputs.metadata["score_convention"]
    if convention == "utility":
        raise ValueError("RR needs probability or log-probability scores")
    rankings = {}
    for request in inputs.requests:
        if alpha == 0:
            rankings[request.request_id] = inputs.ranking(request)
            continue
        distances = []
        for item, score in zip(request.candidate_ids, request.scores):
            if convention == "probability":
                if score == 0:
                    raise ValueError("Zero probability cannot recover RR distance; regenerate native log scores")
                distance = -math.log(score)
            else:
                distance = -score if convention == "log_probability" else score
            # Rank in log distance space to avoid exponent overflow. D=0 stays 0.
            distances.append(-math.inf if distance == 0 else math.log(distance) - alpha * math.log(max(epsilon, 1 - correction[mapping[item]])))
        rankings[request.request_id] = [request.candidate_ids[j] for j in sorted(range(len(distances)), key=distances.__getitem__)]
    report = {"variant": frozen["variant"], "frozen_correction": frozen, "rankings": rankings,
              "control_metrics": evaluate(inputs, cutoffs), "metrics": evaluate(inputs, cutoffs, rankings=rankings)}
    if histories is not None:
        report["control_ifairlrs_fairness"] = fairness(inputs, histories, group, cutoffs)
        report["ifairlrs_fairness"] = fairness(inputs, histories, group, cutoffs, rankings)
    return report


def rw_weights(batches, mapping, padding=-1, *, target_distribution="history", strength=1.0):
    """Count the actual replayed augmented batches, including sampled caps."""
    if target_distribution not in {"history", "uniform"}:
        raise ValueError("RW target distribution must be history or uniform")
    if not math.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError("RW strength must be in [0, 1]")
    history, target = Counter(), Counter()
    for users, histories, targets, groups in batches:
        history.update(mapping[i] for row in histories.tolist() for i in row if i != padding)
        target.update(mapping[i] for i in targets.tolist())
    if not history or not target:
        raise ValueError("RW requires nonempty history and target interactions")
    hs = {g: history[g] / sum(history.values()) for g in sorted(set(mapping.values()))}
    ts = {g: target[g] / sum(target.values()) for g in hs}
    # No target => no sample to weight: recorded zero, no division/smoothing.
    active = sum(value > 0 for value in ts.values())
    desired = hs if target_distribution == "history" else {g: 1 / active if ts[g] else 0.0 for g in hs}
    weights = {g: (1 - strength) + strength * desired[g] / ts[g] if ts[g] else 0.0 for g in hs}
    return {"history_counts": dict(history), "target_counts": dict(target),
            "history_shares": hs, "target_shares": ts, "weights": weights,
            "zero_target_policy": "zero_weight_unused_group", "reduction": "mean(weight * per_example_loss)",
            "target_distribution": target_distribution, "strength": strength, "desired_shares": desired,
            "variant": ("ifairlrs_rw_grid_adaptation" if target_distribution == "history" else
                        "uniform_group_balanced_rw_grid_adaptation")}


def weighted_loss(losses, weights):
    import torch
    if losses.ndim != 1 or weights.shape != losses.shape or not losses.numel():
        raise ValueError("RW requires unreduced per-example losses and matching weights")
    if not torch.isfinite(losses).all() or not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Nonfinite loss or invalid sample weight")
    return (losses * weights).mean()
