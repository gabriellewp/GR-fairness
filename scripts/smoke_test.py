"""Run a synthetic candidate-artifact demo without downloading data or models."""
import argparse
import json
from pathlib import Path
from additional_baselines.candidates import ExperimentInputs, Request, file_reference, save
from additional_baselines.evaluation import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/smoke"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    source = args.output_dir / "synthetic_source.json"
    source.write_text(json.dumps({"scope": "synthetic_demo", "description": "Fixture, not a trained model"})+"\n")
    reference = file_reference(source)
    inputs = ExperimentInputs({"checkpoint": reference, "sid_mapping": reference,
        "split": {"name": "test", "source": reference}, "seed": 42,
        "generation": {"beam_width": 3}, "score_convention": "utility",
        "candidate_membership": "synthetic_fixture", "short_list_policy": "return_available"},
        {"brand": {0: "a", 1: "b", 2: "a"}},
        [Request("test:0", [0, 1, 2], [0.9, 0.8, 0.7], [1]),
         Request("test:1", [2, 1, 0], [0.9, 0.8, 0.7], [2])])
    save(inputs, args.output_dir / "inputs.json")
    metrics = evaluate(inputs, [1, 2])
    (args.output_dir / "control_metrics.json").write_text(json.dumps(metrics, indent=2)+"\n")
    print(f"Synthetic inputs and control metrics: {args.output_dir}")


if __name__ == "__main__":
    main()
