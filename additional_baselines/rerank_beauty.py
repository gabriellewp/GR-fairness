"""Read Beauty's original export and write only the requested result report."""

import json
from pathlib import Path

from .candidates import file_reference
from .prepare_beauty import DEFAULT_DATA, DEFAULT_EXPORT, ROOT, load_beauty
from .run_rerank import run


def rerank_beauty(model, output, *, root=ROOT, export_dir=DEFAULT_EXPORT,
                  data_dir=DEFAULT_DATA, topk=(5, 10, 20), fairness_lambda=None,
                  learning_rate=0.5, gamma=0.2, weights_path=None, horizon=None,
                  order_seed=None, max_dense_bytes=4294967296, limit_requests=None,
                  trace=False):
    if model not in {"CPFair", "PMMF"}:
        raise ValueError("model must be CPFair or PMMF")
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    if not output.parent.is_dir():
        raise FileNotFoundError(f"Output parent directory does not exist: {output.parent}")
    if limit_requests is not None and (type(limit_requests) is not int or limit_requests <= 0):
        raise ValueError("limit-requests must be a positive integer")
    if weights_path is not None and model != "PMMF":
        raise ValueError("Provider weights apply only to PMMF")
    inputs, configs, summary = load_beauty(root=root, export_dir=export_dir, data_dir=data_dir,
                                         topk=topk, max_dense_bytes=max_dense_bytes, check_dense_memory=False)
    config = configs[model]
    parameters = config["parameters"]
    if fairness_lambda is not None:
        parameters["fairness_lambda"] = fairness_lambda
    parameters.update(horizon=horizon, order_seed=order_seed)
    config["trace"] = trace
    if model == "PMMF":
        parameters.update(learning_rate=learning_rate, gamma=gamma)
        if weights_path is not None:
            parameters["weights"] = json.loads(Path(weights_path).read_text())
    original_count = len(inputs.requests)
    if limit_requests is not None:
        if limit_requests > original_count:
            raise ValueError("limit-requests exceeds the export request count")
        inputs.requests = inputs.requests[:limit_requests]
    report = run(inputs, config)
    report["input_export"] = summary["export_dir"]
    report["input_loading"] = "original_files_validated_in_memory"
    report["resolved_experiment_config"] = config
    report["source_summary"] = summary
    report["evaluation_scope"] = {
        "kind": "smoke_test_prefix" if limit_requests is not None else "full_export",
        "original_request_count": original_count, "evaluated_request_count": len(inputs.requests),
    }
    if weights_path is not None:
        report["weights_source"] = file_reference(weights_path)
    with output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return report
