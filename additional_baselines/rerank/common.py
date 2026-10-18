"""Shared ordering and reporting; no implicit score normalization."""

from dataclasses import dataclass, field
import math
import random


def number(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if value < 0 or (positive and value == 0):
        raise ValueError(f"{name} must be {'positive' if positive else 'nonnegative'}")


def prepare(inputs, group, k, order_seed):
    inputs.validate()
    if group not in inputs.item_groups:
        raise ValueError(f"Unknown group mapping: {group}")
    if type(k) is not int or k <= 0:
        raise ValueError("k must be a positive integer")
    if order_seed is not None and type(order_seed) is not int:
        raise ValueError("order_seed must be an integer or None")
    requests = list(inputs.requests)
    if order_seed is not None:
        random.Random(order_seed).shuffle(requests)
    return requests, inputs.item_groups[group]


def utilities(inputs, request):
    scores = request.scores
    if inputs.metadata["score_convention"] == "negative_log_probability":
        scores = [-score for score in scores]
    return list(scores)


@dataclass
class RerankResult:
    variant: str
    config: dict
    request_order: list[str]
    rankings: dict[str, list[int]] = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)
    trace: list[dict] = field(default_factory=list)
