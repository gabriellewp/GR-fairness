"""Compare legacy and batched evaluation on a saved Pet SASRec checkpoint.

This uses a validation prefix only, and never trains or selects a checkpoint.
Both paths rank the full catalog with the same sigmoid/score-shift convention.
"""
import argparse
import ast
import csv
import itertools
import importlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--requests", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.requests < 1:
        parser.error("--requests must be positive")
    if args.output.exists():
        raise FileExistsError(args.output)
    import numpy as np
    import torch
    from additional_baselines.ifairlrs_grid import original_module
    module = importlib.import_module("fairdiverse.recommendation.evaluator")
    config = json.loads((ROOT / "experiments_config_rq2_pet_sasrec_fast_cuda.json").read_text())[0]["config"]
    config["device"] = args.device
    model = original_module("sasrec").SASRec(config).to(args.device).eval()
    model.load_state_dict(torch.load(args.checkpoint, map_location=args.device, weights_only=True), strict=True)
    data = ROOT / "recommendation/processed_dataset/amazon-pet-supplies"
    with (data / "amazon-pet-supplies.valid.ranking").open() as stream:
        rows = list(itertools.islice(csv.DictReader(stream, delimiter="\t"), args.requests))
    catalog = torch.arange(config["item_num"])
    histories, candidates, users = [], [], []
    for row in rows:
        history = ast.literal_eval(row["history_behaviors"])[-config["history_length"]:]
        histories.append(history + [-1] * (config["history_length"] - len(history)))
        target = int(row["target_item"])
        candidates.append(torch.cat((torch.tensor([target]), catalog[catalog != target])))
        users.append(int(row["user_id"]))
    batch = (torch.tensor(users), torch.tensor(histories), torch.stack(candidates), torch.ones(len(rows), dtype=torch.long))
    iid2pid = {int(i): int(group) for i, group in json.loads((data / "iid2pid.json").read_text()).items()}
    # The CSV load is unrelated to scoring; keep per-user metric recording in
    # both timed paths but only load the prefix's membership map for this smoke.
    module.load_activity = lambda path: {str(user): {} for user in users}
    reports, elapsed, user_metrics = {}, {}, {}
    serial_scores = []
    serial_predict = model.full_predict

    def capture(*values):
        result = serial_predict(*values)
        serial_scores.append(result[0].detach().cpu().numpy())
        return result

    model.full_predict = capture
    for name, batch_size in (("legacy", 0), ("batched", config["classic_eval_batch_size"])):
        evaluator = module.Ranking_Evaluator.__new__(module.Ranking_Evaluator)
        evaluator.config = dict(config, classic_eval_batch_size=batch_size)
        evaluator.iid2pid = iid2pid
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        start = time.perf_counter()
        reports[name] = evaluator.eval([batch], model)
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        elapsed[name] = time.perf_counter() - start
        user_metrics[name] = evaluator.last_user_metrics
    fast_scores = model.full_predict_batch(
        {"user_ids": batch[0].to(args.device), "history_ids": batch[1].to(args.device)},
        batch[2], config["classic_eval_batch_size"],
    ).numpy()
    max_difference = float(np.max(np.abs(np.stack(serial_scores) - fast_scores)))
    mismatches = sum(not np.array_equal(np.argsort(old + np.float32(1))[::-1][:20],
                       module._legacy_topk_args(new + np.float32(1), 20))
                     for old, new in zip(serial_scores, fast_scores))
    record = dict(checkpoint=str(args.checkpoint.resolve()), split="validation_prefix",
                  requests=len(rows), catalog_size=config["item_num"], device=args.device,
                  elapsed_seconds=elapsed, speedup=elapsed["legacy"] / elapsed["batched"],
                  maximum_score_difference=max_difference, top20_mismatches=mismatches,
                  metrics_equal=reports["legacy"] == reports["batched"],
                  user_metrics_equal=user_metrics["legacy"] == user_metrics["batched"], metrics=reports)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2), flush=True)
    if mismatches or not record["metrics_equal"] or not record["user_metrics_equal"]:
        raise ValueError("Batched evaluator differs from legacy rankings/metrics; inspect smoke report")


if __name__ == "__main__":
    main()
