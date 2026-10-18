"""Shared inputs and direct or packaged FairDiverse reranking."""

import argparse
import json
from pathlib import Path

from .adapters import import_legacy
from .candidates import load, save, file_reference
from .evaluation import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    continuation = commands.add_parser("continue-beauty", help="Native-schedule GRID continuation with validation checkpoint selection")
    continuation.add_argument("--mode", choices=("ordinary", "rw", "sprint"), required=True)
    continuation.add_argument("--output-dir", required=True)
    continuation.add_argument("--control-report", help="Ordinary continuation.json; required for RW/SPRINT")
    continuation.add_argument("--allow-unimproved-control", dest="require_control_improvement",
                              action="store_false", help="Allow a matched ordinary control that selected update zero")
    continuation.add_argument("--group", choices=("popularity", "brand"), default="popularity")
    continuation.add_argument("--rw-target", choices=("history", "uniform"), default="history",
                              help="Uniform is a separately named group-balanced RW variant")
    continuation.add_argument("--rw-strength", type=float, default=1.0)
    continuation.add_argument("--selection-objective", choices=("recall", "popularity", "brand", "both"), default="recall")
    continuation.add_argument("--max-relative-recall-loss", type=float, default=0.02,
                              help="Fairness selection budget relative to matched ordinary validation Recall@5")
    continuation.add_argument("--validation-only", dest="evaluate_test", action="store_false",
                              help="Save selected checkpoint without evaluating test")
    continuation.add_argument("--device", default="cuda")
    continuation.add_argument("--seed", type=int, default=42)
    continuation.add_argument("--max-updates", type=int, default=32)
    continuation.add_argument("--validation-interval", type=int, default=8)
    continuation.add_argument("--learning-rate", type=float, default=0.001)
    continuation.add_argument("--optimizer-state", choices=("require", "auto", "fresh"), default="require")
    continuation.add_argument("--export-dir")
    continuation.add_argument("--tokenizer", choices=("rkmeans", "rqvae", "rvq"), default="rkmeans")
    continuation.add_argument("--dataset", choices=("amazon-beauty", "amazon-toys", "amazon-sports", "amazon-pet-supplies"), default="amazon-beauty")
    continuation.add_argument("--data-dir")
    continuation.add_argument("--validation-requests", type=int, help="Smoke-test validation prefix; never use for reported runs")
    continuation.add_argument("--train-users", type=int, help="Smoke-test training-user prefix")
    continuation.add_argument("--beam-width", dest="generation_beam_width", type=int)
    continuation.add_argument("--attention-weight", type=float, default=1.0)
    continuation.add_argument("--ffn-weight", type=float, default=0.001)
    continuation.add_argument("--embedding-weight", type=float, default=0.1)
    continuation.add_argument("--temperature", type=float, default=4.0)
    continuation.add_argument("--iterations", type=int, default=1)
    selected = commands.add_parser("evaluate-continuation", help="Evaluate a frozen continuation checkpoint without retraining")
    selected.add_argument("--run-dir", required=True)
    selected.add_argument("--output-dir", required=True)
    selected.add_argument("--device", default="cuda")
    sprint = commands.add_parser("sprint-beauty", help="Paired ordinary GRID and encoder SPRINT adaptation training")
    sprint.add_argument("--output-dir", required=True)
    sprint.add_argument("--device", default="cuda")
    sprint.add_argument("--seed", type=int, default=42)
    sprint.add_argument("--epochs", type=int, default=1)
    sprint.add_argument("--attention-weight", type=float, default=1.0)
    sprint.add_argument("--ffn-weight", type=float, default=0.001)
    sprint.add_argument("--embedding-weight", type=float, default=0.1)
    sprint.add_argument("--temperature", type=float, default=4.0)
    sprint.add_argument("--iterations", type=int, default=1)
    sprint.add_argument("--beam-width", dest="generation_beam_width", type=int, help="Default: original beam width 100; recorded for both branches")
    sprint.add_argument("--export-dir")
    sprint.add_argument("--data-dir")
    sprint.add_argument("--limit-requests", type=int)
    sprint.add_argument("--limit-train-rows", type=int)
    ifair = commands.add_parser("ifairlrs-beauty", help="IFairLRS GRID adaptation: validation-fitted RR or paired ordinary/RW training")
    ifair.add_argument("--mode", choices=("rr", "rw"), required=True)
    ifair.add_argument("--group", choices=("popularity", "brand"), default="popularity")
    ifair.add_argument("--output-dir", required=True)
    ifair.add_argument("--device", default="cuda")
    ifair.add_argument("--seed", type=int, default=42)
    ifair.add_argument("--epochs", type=int, default=1, help="Matched fixed training budget for RW and ordinary control")
    ifair.add_argument("--alpha", type=float, default=0.05, help="RR strength; choose on validation only")
    ifair.add_argument("--epsilon", type=float, default=1e-6)
    ifair.add_argument("--export-dir")
    ifair.add_argument("--data-dir")
    ifair.add_argument("--limit-requests", type=int, help="Validation/test smoke-test prefix")
    ifair.add_argument("--limit-train-rows", type=int, help="Training smoke-test prefix")
    fit = commands.add_parser("ifairlrs-fit-rr", help="Freeze RR corrections from a shared validation artifact")
    fit.add_argument("artifact")
    fit.add_argument("--histories", required=True, help="JSON request-ID to unpadded history item IDs")
    fit.add_argument("--group", default="popularity")
    fit.add_argument("--cutoffs", nargs="+", type=int, default=[1, 5, 10, 20])
    fit.add_argument("--alpha", type=float, default=0.05)
    fit.add_argument("--epsilon", type=float, default=1e-6)
    fit.add_argument("--output", required=True)
    apply = commands.add_parser("ifairlrs-apply-rr", help="Apply frozen RR corrections without refitting")
    apply.add_argument("artifact")
    apply.add_argument("--correction", required=True)
    apply.add_argument("--histories")
    apply.add_argument("--topk", nargs="+", type=int, default=[5, 10, 20])
    apply.add_argument("--output", required=True)
    importer = commands.add_parser("import-legacy")
    for name in ("scores", "targets", "request-ids", "groups", "metadata", "output"):
        importer.add_argument("--" + name, required=True)
    importer.add_argument("--format", choices=("compact_grid", "explicit_mask"), required=True)
    importer.add_argument("--candidate-mask")
    evaluator = commands.add_parser("evaluate")
    evaluator.add_argument("artifact")
    evaluator.add_argument("--topk", nargs="+", type=int, default=[5, 10])
    evaluator.add_argument("--output", required=True)
    reranker = commands.add_parser("rerank")
    reranker.add_argument("artifact")
    reranker.add_argument("--config", required=True)
    reranker.add_argument("--output", required=True)
    reranker.add_argument("--limit-requests", type=int, help="Smoke test on the first N requests; not a full evaluation")
    beauty = commands.add_parser("prepare-beauty")
    beauty.add_argument("--output-dir", default="additional_baselines/artifacts/beauty_rkmeans_suffix4")
    beauty.add_argument("--topk", nargs="+", type=int, default=[5, 10, 20])
    beauty.add_argument("--max-dense-bytes", type=int, default=4294967296)
    direct = commands.add_parser("rerank-beauty", help="Read original Beauty files in memory; write only results")
    direct.add_argument("--model", choices=["CPFair", "PMMF"], required=True)
    direct.add_argument("--output", required=True)
    direct.add_argument("--export-dir", help="Original score-export directory; defaults to verified complete-metadata run")
    direct.add_argument("--data-dir", help="Processed Beauty directory; defaults to recommendation/processed_dataset/amazon-beauty")
    direct.add_argument("--topk", nargs="+", type=int, default=[5, 10, 20])
    direct.add_argument("--lambda", dest="fairness_lambda", type=float, help="Default: CPFair=1, PMMF=30")
    direct.add_argument("--learning-rate", type=float, default=0.5)
    direct.add_argument("--gamma", type=float, default=0.2)
    direct.add_argument("--weights", dest="weights_path", help="PMMF group-to-weight JSON; default: explicit all-one weights")
    direct.add_argument("--horizon", type=int, help="Default: whole request sequence")
    direct.add_argument("--order-seed", type=int)
    direct.add_argument("--max-dense-bytes", type=int, default=4294967296)
    direct.add_argument("--limit-requests", type=int, help="Smoke test prefix; not a full evaluation")
    direct.add_argument("--trace", action="store_true")
    args = parser.parse_args()
    if args.command == "sprint-beauty":
        from .ifairlrs_grid import run_beauty
        options = {key: value for key, value in vars(args).items() if key != "command" and value is not None}
        settings = {key: options.pop(key) for key in ("attention_weight", "ffn_weight", "embedding_weight", "temperature", "iterations")}
        print(json.dumps(run_beauty(mode="sprint", sprint_settings=settings, **options), indent=2))
    elif args.command == "ifairlrs-beauty":
        from .ifairlrs_grid import run_beauty
        options = {key: value for key, value in vars(args).items() if key != "command" and value is not None}
        print(json.dumps(run_beauty(**options), indent=2))
    elif args.command == "evaluate-continuation":
        from .continuation_grid import evaluate_selected_continuation
        options = {key: value for key, value in vars(args).items() if key != "command"}
        result = evaluate_selected_continuation(**options)
        print(json.dumps({key: value for key, value in result.items() if key != "metrics"}, indent=2))
    elif args.command == "continue-beauty":
        from .continuation_grid import run_continuation
        options = vars(args).copy()
        options.pop("command")
        settings = {key: options.pop(key) for key in ("attention_weight", "ffn_weight", "embedding_weight", "temperature", "iterations")}
        if args.mode == "sprint":
            options["sprint_settings"] = settings
        options = {key: value for key, value in options.items() if value is not None}
        print(json.dumps(run_continuation(**options), indent=2))
    elif args.command in {"ifairlrs-fit-rr", "ifairlrs-apply-rr"}:
        from .ifairlrs import apply_rr, fit_rr
        from .ifairlrs_grid import write_json
        if Path(args.output).exists():
            raise FileExistsError(f"Output already exists: {args.output}")
        artifact = load(args.artifact)
        histories = json.loads(Path(args.histories).read_text()) if args.histories else None
        if args.command == "ifairlrs-fit-rr":
            result = fit_rr(artifact, histories, args.group, args.cutoffs, args.alpha, args.epsilon)
            result["validation_artifact"] = file_reference(args.artifact)
            result["histories_source"] = file_reference(args.histories)
        else:
            result = apply_rr(artifact, json.loads(Path(args.correction).read_text()), args.topk, histories)
            result["correction_source"] = file_reference(args.correction)
            result["input_artifact"] = file_reference(args.artifact)
        write_json(args.output, result)
    elif args.command == "rerank-beauty":
        from .rerank_beauty import rerank_beauty
        options = vars(args).copy()
        options.pop("command")
        options = {key: value for key, value in options.items() if value is not None}
        report = rerank_beauty(**options)
        print(json.dumps({"output": str(Path(args.output).resolve()), "evaluation_scope": report["evaluation_scope"]}, indent=2))
    elif args.command == "prepare-beauty":
        from .prepare_beauty import prepare
        summary = prepare(args.output_dir, topk=args.topk, max_dense_bytes=args.max_dense_bytes)
        print(json.dumps(summary, indent=2))
    elif args.command == "import-legacy":
        def read(path):
            return json.loads(Path(path).read_text())
        groups = {name: {int(i): g for i, g in mapping.items()} for name, mapping in read(args.groups).items()}
        metadata = read(args.metadata)
        metadata["request_order_source"] = file_reference(args.request_ids)
        metadata["group_mapping_source"] = file_reference(args.groups)
        artifact = import_legacy(args.scores, args.targets, read(args.request_ids), groups, metadata,
                                 export_format=args.format, candidate_mask_path=args.candidate_mask)
        save(artifact, args.output)
    else:
        if Path(args.output).exists():
            raise FileExistsError(f"Output already exists: {args.output}")
        if args.command == "rerank":
            from .run_rerank import run
            config = json.loads(Path(args.config).read_text())
            inputs = load(args.artifact)
            original_count = len(inputs.requests)
            if args.limit_requests is not None:
                if args.limit_requests <= 0 or args.limit_requests > original_count:
                    raise ValueError("limit-requests must be between 1 and the artifact request count")
                inputs.requests = inputs.requests[:args.limit_requests]
            result = run(inputs, config)
            result["evaluation_scope"] = {"kind": "smoke_test_prefix" if args.limit_requests is not None else "full_artifact",
                                          "original_request_count": original_count, "evaluated_request_count": len(inputs.requests)}
            result["config_source"] = file_reference(args.config)
        else:
            result = evaluate(load(args.artifact), args.topk)
        result["input_artifact"] = file_reference(args.artifact)
        with Path(args.output).open("x") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")


if __name__ == "__main__":
    main()
