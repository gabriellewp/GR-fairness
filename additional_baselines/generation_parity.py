"""Check batching against the rankings used by reported metrics."""


def generation_parity(single, batched, ks=(5, 10, 20)):
    import torch

    aligned = len(single.requests) == len(batched.requests)
    membership = aligned
    top_k_equal = {str(k): aligned for k in ks}
    exact = aligned
    scores_close = aligned
    differences = []
    for a, b in zip(single.requests, batched.requests):
        same_request = a.request_id == b.request_id and a.targets == b.targets
        same_members = same_request and set(a.candidate_ids) == set(b.candidate_ids)
        membership = membership and same_members
        ra, rb = single.ranking(a), batched.ranking(b)
        exact = exact and same_request and ra == rb
        for k in ks:
            top_k_equal[str(k)] = top_k_equal[str(k)] and same_request and ra[:k] == rb[:k]
        if ra != rb:
            positions = [i + 1 for i, (x, y) in enumerate(zip(ra, rb)) if x != y]
            differences.append(dict(request_id=a.request_id,
                                    first_different_rank=min(positions) if positions else min(len(ra), len(rb)) + 1))
        if same_members:
            # Align by item ID: an allowed lower-rank reorder must not mix scores.
            other = dict(zip(b.candidate_ids, b.scores))
            x = torch.tensor(a.scores, dtype=torch.float64)
            y = torch.tensor([other[i] for i in a.candidate_ids], dtype=torch.float64)
            scores_close = scores_close and bool(torch.allclose(x, y, rtol=1e-4, atol=1e-7))
        else:
            scores_close = False
    return dict(requests=len(single.requests), request_counts_equal=aligned,
                candidate_membership_equal=membership, exact_rankings=exact,
                top_k_rankings_equal=top_k_equal, scores_close=scores_close,
                lower_rank_differences=differences,
                passed=membership and scores_close and all(top_k_equal.values()),
                scope="64-user diagnostic; full controls and candidates use the same batched protocol")
