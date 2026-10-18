"""Binary relevance and count-based group exposure on eligible candidates."""

import math


def evaluate(inputs, cutoffs=(5, 10), rankings=None, *, user_fairness=None):
    """Evaluate default score ordering or explicit per-request reranked lists.

    Missing positions are misses. IDCG includes targets outside the candidate
    set. All mapped catalog groups, including unexposed groups, enter fairness.
    """
    inputs.validate()
    cutoffs = list(cutoffs)
    if not cutoffs or any(type(k) is not int or k <= 0 for k in cutoffs):
        raise ValueError("cutoffs must be positive integers")
    if rankings is not None and set(rankings) != {r.request_id for r in inputs.requests}:
        raise ValueError("Rankings must contain exactly the artifact request IDs")
    ordered = {}
    for request in inputs.requests:
        ranking = inputs.ranking(request) if rankings is None else list(rankings[request.request_id])
        if len(ranking) != len(set(ranking)) or set(ranking) - set(request.candidate_ids):
            raise ValueError("Rankings contain duplicate or ineligible items")
        ordered[request.request_id] = ranking
    n = len(inputs.requests)
    result = {
        "request_count": n,
        "candidate_target_recall": sum(len(set(r.targets) & set(r.candidate_ids)) / len(r.targets) for r in inputs.requests) / n,
        "candidate_hit_rate": sum(bool(set(r.targets) & set(r.candidate_ids)) for r in inputs.requests) / n,
        "mean_candidate_count": sum(len(r.candidate_ids) for r in inputs.requests) / n,
        "empty_candidate_rate": sum(not r.candidate_ids for r in inputs.requests) / n,
        "at_k": {},
    }
    for k in cutoffs:
        metrics = dict.fromkeys(("hit_rate", "recall", "ndcg", "mrr", "fill_rate", "short_list_rate"), 0.0)
        exposures = {name: dict.fromkeys(sorted(set(mapping.values())), 0) for name, mapping in inputs.item_groups.items()}
        for request in inputs.requests:
            selected = ordered[request.request_id][:k]
            targets = set(request.targets)
            hits = [item in targets for item in selected]
            metrics["hit_rate"] += bool(sum(hits))
            metrics["recall"] += sum(hits) / len(targets)
            ideal = sum(1 / math.log2(i + 2) for i in range(min(k, len(targets))))
            metrics["ndcg"] += sum(hit / math.log2(i + 2) for i, hit in enumerate(hits)) / ideal
            metrics["mrr"] += next((1 / (i + 1) for i, hit in enumerate(hits) if hit), 0)
            metrics["fill_rate"] += len(selected) / k
            metrics["short_list_rate"] += len(selected) < k
            for name, mapping in inputs.item_groups.items():
                for item in selected:
                    exposures[name][mapping[item]] += 1
        metrics = {name: value / n for name, value in metrics.items()}
        metrics["groups"] = {}
        for name, counts in exposures.items():
            values = sorted(counts.values())
            total = sum(values)
            count = len(values)
            shares = {group: value / total if total else 0.0 for group, value in counts.items()}
            metrics["groups"][name] = {
                "exposure_counts": counts, "exposure_shares": shares,
                "gini": sum((2 * i - count - 1) * v for i, v in enumerate(values, 1)) / (count * total) if total else None,
                "entropy": -sum(p * math.log(p) for p in shares.values() if p) if total else None,
                "min_max_ratio": values[0] / values[-1] if total else None,
                "exposed_group_fraction": sum(v > 0 for v in values) / count,
                "candidate_group_coverage": sum(len({inputs.item_groups[name][i] for i in r.candidate_ids}) / count for r in inputs.requests) / n,
            }
        result["at_k"][str(k)] = metrics
    if user_fairness is not None:
        from fairdiverse.recommendation.user_fairness import ranking_metrics, save_report
        # Request IDs are not necessarily user IDs. Require an explicit mapping.
        users = user_fairness["request_users"]
        if set(users) != {r.request_id for r in inputs.requests}:
            raise ValueError("request_users must map exactly the evaluated request IDs")
        rows = [
            {"user_id": str(users[r.request_id]), "request_id": r.request_id,
             **ranking_metrics(ordered[r.request_id], r.targets, cutoffs)}
            for r in inputs.requests
        ]
        result["user_fairness"] = save_report(
            rows, user_fairness["activity_path"], user_fairness["output_dir"],
            user_fairness.get("bootstrap_samples", 1000), user_fairness.get("seed", 42),
            inputs.metadata,
        )
    return result
