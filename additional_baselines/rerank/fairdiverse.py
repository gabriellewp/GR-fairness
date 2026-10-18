"""FairDiverse CPFair/PMMF equations with matrix-free group lookups."""

from dataclasses import dataclass
import math
from pathlib import Path
import sys

import numpy as np

from ..candidates import file_reference
from .common import RerankResult, number, prepare, utilities
from .legacy import ROOT, load_class


@dataclass(frozen=True)
class FairDiverseReranker:
    model: str
    group: str
    fairness_lambda: float
    learning_rate: float = 0.5
    gamma: float = 0.2
    weights: dict[str, float] | None = None
    horizon: int | None = None
    order_seed: int | None = None
    max_dense_bytes: int = 536870912
    matrix_mode: str = "lookup"

    def _lookup_rankings(self, inputs, batch, k, catalog, item_index, providers, weights, module):
        """Keep native state and sorting; replace one-hot products with indexing."""
        exposure = np.zeros(len(weights))
        budget = len(batch) * k * weights
        dual = np.zeros(len(weights))
        previous_gradient = np.zeros(len(weights))
        eta = self.learning_rate / math.sqrt(len(catalog))
        # A single full-catalog row retains NumPy's original quicksort tie order.
        scores = np.full(len(catalog), -np.inf)
        for row in module.trange(len(batch)):
            request = batch[row]
            ids = np.asarray([item_index[item] for item in request.candidate_ids])
            values = np.asarray(utilities(inputs, request))
            candidate_groups = providers[ids]
            scores.fill(-np.inf)
            if self.model == "CPFair":
                adjustment = self.fairness_lambda * (1 - exposure / (np.sum(exposure) + 1e-5))
                scores[ids] = values + adjustment[candidate_groups]
                chosen = np.argsort(scores)[::-1][:k]
                exposure += np.bincount(providers[chosen], minlength=len(weights))
            else:
                mask = (1.0 - (budget[candidate_groups] > 0).astype(np.float32)) * -10000.0
                scores[ids] = values - dual[candidate_groups] + mask
                chosen = np.argsort(scores)[::-1][:k]
                # Native PMMF reorders the selected set by original relevance.
                original = dict(zip(ids.tolist(), values.tolist()))
                chosen = chosen[np.argsort([original[int(i)] for i in chosen])[::-1]]
                counts = np.bincount(providers[chosen], minlength=len(weights))
                budget -= counts
                gradient = -counts / k + weights
                gradient = self.gamma * gradient + (1 - self.gamma) * previous_gradient
                previous_gradient = gradient
                dual = module.compute_next_dual(eta, weights, dual, gradient, self.fairness_lambda)
            yield chosen

    def rerank(self, inputs, k, *, trace=False):
        if self.model not in {"CPFair", "PMMF"}:
            raise ValueError("model must be CPFair or PMMF")
        if self.matrix_mode not in {"lookup", "dense"}:
            raise ValueError("matrix_mode must be lookup or dense")
        requests, mapping = prepare(inputs, self.group, k, self.order_seed)
        number(self.fairness_lambda, "fairness_lambda")
        if self.horizon is not None and (type(self.horizon) is not int or self.horizon <= 0):
            raise ValueError("horizon must be a positive request count or null for the whole artifact")
        if type(self.max_dense_bytes) is not int or self.max_dense_bytes <= 0:
            raise ValueError("max_dense_bytes must be a positive integer")
        groups = sorted(set(mapping.values()))
        group_index = {g: i for i, g in enumerate(groups)}
        catalog = sorted(mapping)
        item_index = {item: i for i, item in enumerate(catalog)}
        providers = np.array([group_index[mapping[item]] for item in catalog])
        if self.model == "CPFair" and self.weights is not None:
            raise ValueError("CPFair does not use provider weights")
        if self.model == "PMMF":
            number(self.learning_rate, "learning_rate")
            number(self.gamma, "gamma")
            if self.gamma > 1:
                raise ValueError("gamma must be in [0, 1] (weight on new gradient)")
            if self.weights is None or set(self.weights) != set(groups):
                raise ValueError("PMMF requires explicit weights for exactly every catalog group")
            for weight in self.weights.values():
                number(weight, "weights", positive=True)
        weights = np.array([self.weights[g] for g in groups]) if self.weights is not None else np.ones(len(groups))
        for request in requests:
            if len(request.candidate_ids) < k:
                raise ValueError(f"Request {request.request_id} has fewer than k={k} eligible candidates; unchanged FairDiverse requires full lists. Regenerate candidates or explicitly choose a smaller K.")
        horizon = self.horizon or len(requests)
        # Guard array storage, not total Python/candidate/solver process memory.
        # Lookup mode needs only catalog-sized work rows and group vectors.
        estimate = (8 * len(catalog) * (min(horizon, len(requests)) + len(groups) + 8)
                    if self.matrix_mode == "dense" else 8 * (8 * len(catalog) + 12 * len(groups)))
        if estimate > self.max_dense_bytes:
            raise ValueError(f"Reranking array estimate {estimate} exceeds max_dense_bytes={self.max_dense_bytes}; configure a larger memory allowance or use lookup mode")
        backend_config = {"item_num": len(catalog), "group_num": len(groups), "lambda": self.fairness_lambda}
        if self.model == "PMMF":
            backend_config.update(learning_rate=self.learning_rate, gamma=self.gamma)
        cls = load_class(self.model)
        result = RerankResult(
            "cpfair_fairdiverse" if self.model == "CPFair" else "pmmf_fairdiverse",
            {"backend": f"fairdiverse/recommendation/rerank_model/{self.model}.py",
             "backend_source": file_reference(ROOT / "rerank_model" / f"{self.model}.py"),
             "label": f"{self.model} — FairDiverse equations ({self.matrix_mode})",
             "matrix_mode": self.matrix_mode,
             "adapter_source": file_reference(Path(__file__)),
             "estimated_array_bytes": estimate,
             "group": self.group, "k": k, "backend_config": backend_config,
             "weights": dict(zip(groups, weights.tolist())), "group_order": groups,
             "horizon": self.horizon, "effective_horizon": horizon, "tail_policy": "actual_length",
             "order_seed": self.order_seed, "state_reset": "each_call_and_horizon",
             "score_convention": inputs.metadata["score_convention"],
             "score_transform": "negate_nll_otherwise_identity", "candidate_mask": "negative_infinity",
             "infeasibility": "error", "budget_policy": "native_request_gate_audited_after_run",
             "tie_policy": "legacy_numpy_order_over_sorted_catalog",
             "projection": "unchanged CVXPY default solver" if self.model == "PMMF" else None,
             "max_dense_bytes": self.max_dense_bytes},
            [r.request_id for r in requests],
        )
        totals = np.zeros(len(groups), dtype=int)
        result.diagnostics = {"short_list_count": 0, "horizons": []}
        for start in range(0, len(requests), horizon):
            batch = requests[start:start + horizon]
            if self.matrix_mode == "dense":
                scores = np.full((len(batch), len(catalog)), -np.inf)
                for row, request in enumerate(batch):
                    scores[row, [item_index[item] for item in request.candidate_ids]] = utilities(inputs, request)
                backend = cls.__new__(cls)
                backend.config = dict(backend_config)
                backend.item_num, backend.group_num = len(catalog), len(groups)
                backend.weights = weights.copy()
                backend.iid2pid = dict(enumerate(providers.tolist()))
                backend.M = np.zeros((len(catalog), len(groups)))
                backend.M[np.arange(len(catalog)), providers] = 1
                with np.errstate(over="raise", invalid="raise", divide="raise"):
                    rankings = backend.rerank(scores, k)
                if len(rankings) != len(batch):
                    raise RuntimeError("FairDiverse returned the wrong request count")
            else:
                rankings = self._lookup_rankings(inputs, batch, k, catalog, item_index,
                                                providers, weights, sys.modules[cls.__module__])
            capacity = len(batch) * k * weights
            remaining = capacity.copy()
            if not np.all(np.isfinite(capacity)):
                raise ValueError("Nonfinite capacities")
            def checked_rankings():
                with np.errstate(over="raise", invalid="raise", divide="raise"):
                    yield from rankings
            for ranked, request in zip(checked_rankings(), batch):
                indices = np.asarray(ranked)
                if indices.ndim != 1 or len(indices) != k or not np.issubdtype(indices.dtype, np.integer) or np.any(indices < 0) or np.any(indices >= len(catalog)):
                    raise RuntimeError("Invalid FairDiverse ranking shape or item index")
                selected = [catalog[index] for index in indices]
                if len(set(selected)) != k or set(selected) - set(request.candidate_ids):
                    raise RuntimeError(f"FairDiverse returned duplicate/ineligible items for {request.request_id}; no report will be saved")
                selected_providers = providers[indices]
                if self.model == "PMMF" and np.any(remaining[selected_providers] <= 0):
                    raise ValueError(f"PMMF selected an exhausted provider for request {request.request_id}; its finite penalty cannot guarantee feasibility. Review weights/horizon/K. No report will be saved.")
                counts = np.bincount(selected_providers, minlength=len(groups))
                totals += counts
                remaining -= counts
                result.rankings[request.request_id] = selected
                if trace:
                    step = {"request_id": request.request_id, "horizon_start": start,
                            "allocation_counts": dict(zip(groups, counts.tolist()))}
                    if self.model == "PMMF":
                        step["budget_after"] = dict(zip(groups, remaining.tolist()))
                    result.trace.append(step)
            summary = {"start": start, "length": len(batch)}
            if self.model == "PMMF":
                summary.update(initial_capacity=dict(zip(groups, capacity.tolist())),
                               remaining_capacity=dict(zip(groups, remaining.tolist())),
                               overshoot=dict(zip(groups, np.maximum(-remaining, 0).tolist())))
            result.diagnostics["horizons"].append(summary)
        result.diagnostics["exposure_counts"] = dict(zip(groups, totals.tolist()))
        return result
