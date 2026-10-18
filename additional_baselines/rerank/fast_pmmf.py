"""Exact P-MMF projection, avoiding a fresh CVXPY problem per request.

For sorted z=rho*tilde_dual, the prefix constraints project its negative
part onto an L1 ball of radius lambda. Positive entries stay unchanged.
The clipping threshold satisfies sum(max(-z-threshold, 0))=lambda.
This is the same convex objective and feasible optimum as the legacy QP.
"""
import math
from pathlib import Path

import numpy as np

from .fairdiverse import FairDiverseReranker
from .common import utilities
from ..candidates import file_reference


def project_sorted(ordered_tilde_dual, rho, strength):
    rho = np.asarray(rho, dtype=np.float64)
    z = rho * np.asarray(ordered_tilde_dual, dtype=np.float64)
    if rho.shape != z.shape or np.any(rho <= 0) or not np.isfinite(z).all() or strength < 0 or not math.isfinite(strength):
        raise ValueError("Invalid P-MMF projection input")
    if np.any(np.diff(z) < 0):
        raise ValueError("Projection requires sorted rho*dual")
    negative = np.maximum(-z, 0)
    if negative.sum() <= strength:
        return z/rho
    if strength == 0:
        return np.maximum(z, 0)/rho
    threshold = (np.cumsum(negative)-strength)/np.arange(1,len(z)+1)
    active = np.flatnonzero(negative > threshold)
    threshold = threshold[active[-1]]
    return (np.maximum(z,0)-np.maximum(negative-threshold,0))/rho


def next_dual(eta, rho, dual, gradient, strength):
    tilde = dual-eta*gradient/rho/rho
    order = np.argsort(tilde*rho)
    projected = project_sorted(tilde[order],rho[order],strength)
    result = np.empty_like(projected)
    result[order] = projected
    return result


class FastPMMFReranker(FairDiverseReranker):
    def _lookup_rankings(self, inputs, batch, k, catalog, item_index, providers, weights, module):
        if self.model != "PMMF":
            yield from super()._lookup_rankings(inputs,batch,k,catalog,item_index,providers,weights,module)
            return
        budget = len(batch)*k*weights
        dual = np.zeros(len(weights))
        previous_gradient = np.zeros(len(weights))
        eta = self.learning_rate/math.sqrt(len(catalog))
        scores = np.full(len(catalog),-np.inf)
        for row in module.trange(len(batch)):
            request = batch[row]
            ids = np.asarray([item_index[item] for item in request.candidate_ids])
            values = np.asarray(utilities(inputs,request))
            candidate_groups = providers[ids]
            mask = (1.0-(budget[candidate_groups]>0).astype(np.float32))*-10000.0
            scores.fill(-np.inf)
            scores[ids] = values-dual[candidate_groups]+mask
            chosen = np.argsort(scores)[::-1][:k]
            original = dict(zip(ids.tolist(),values.tolist()))
            chosen = chosen[np.argsort([original[int(i)] for i in chosen])[::-1]]
            counts = np.bincount(providers[chosen],minlength=len(weights))
            budget -= counts
            gradient = -counts/k+weights
            gradient = self.gamma*gradient+(1-self.gamma)*previous_gradient
            previous_gradient = gradient
            dual = next_dual(eta,weights,dual,gradient,self.fairness_lambda)
            yield chosen

    def rerank(self, inputs, k, *, trace=False):
        if self.model == "PMMF" and self.matrix_mode != "lookup":
            raise ValueError("Exact P-MMF requires lookup mode")
        result = super().rerank(inputs,k,trace=trace)
        if self.model == "PMMF":
            result.config.update(projection="exact sorted negative-part L1 projection",
                                 projection_source=file_reference(Path(__file__)))
        return result
