"""Run independent allocations for each cutoff and evaluate against the control."""

from dataclasses import asdict
from pathlib import Path

from .evaluation import evaluate
from .rerank import FairDiverseReranker


def run(inputs, config):
    allowed = {"method", "parameters", "topk", "trace", "user_fairness"}
    if set(config) - allowed:
        raise ValueError(f"Unknown experiment settings: {sorted(set(config) - allowed)}")
    methods = {"cpfair_fairdiverse": "CPFair", "pmmf_fairdiverse": "PMMF"}
    if config.get("method") not in methods:
        raise ValueError(f"method must be one of {sorted(methods)}")
    if not isinstance(config.get("parameters"), dict):
        raise ValueError("parameters must be an explicit configuration object")
    topk = config.get("topk")
    if not isinstance(topk, list) or not topk or any(type(k) is not int or k <= 0 for k in topk) or len(set(topk)) != len(topk):
        raise ValueError("topk must be a nonempty list of distinct positive integers")
    trace = config.get("trace", False)
    if type(trace) is not bool:
        raise ValueError("trace must be boolean")
    reranker = FairDiverseReranker(model=methods[config["method"]], **config["parameters"])
    def fairness_options(name):
        options = config.get("user_fairness")
        if options is None:
            return None
        return {**options, "output_dir": str(Path(options["output_dir"]) / name)}

    report = {"schema_version": 1, "input_metadata": inputs.metadata,
              "control_metrics": evaluate(inputs, topk, user_fairness=fairness_options("control")), "runs": {}}
    for k in topk:
        # Do not truncate one large-K allocation: budgets and updates depend on K.
        allocation = reranker.rerank(inputs, k, trace=trace)
        report["runs"][str(k)] = {
            **asdict(allocation), "metrics": evaluate(inputs, [k], allocation.rankings,
                                                     user_fairness=fairness_options(f"reranked_{k}")),
        }
    return report
