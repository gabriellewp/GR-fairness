"""Diagnose common-protocol score parity without changing evaluation tolerances."""
import argparse
import ast
import csv
import itertools
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("USE_TF", "0")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--case", type=int, default=7)
    parser.add_argument("--requests", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.requests < 1 or args.output.exists():
        parser.error("Require positive requests and a new output path")
    import numpy as np
    import torch
    from additional_baselines.ifairlrs_grid import original_module, seed_everything
    from additional_baselines.candidates import file_reference

    torch.set_num_threads(4)
    case = json.loads(args.inventory.read_text())["cases"][args.case]
    for ref in [case["checkpoint"], case["config_reference"], *case["inputs"].values()]:
        if file_reference(ref["path"]) != ref:
            raise ValueError(f"Pinned input changed: {ref['path']}")
    seed_everything(case["config"]["seed"])
    config = dict(case["config"], device=args.device)
    model_name = "sasrec" if case["model"] == "SASRec" else "gru4rec"
    model = getattr(original_module(model_name), case["model"])(config).to(args.device)
    model.load_state_dict(torch.load(case["checkpoint"]["path"], map_location=args.device,
                                     weights_only=True), strict=True)
    model.eval()
    with open(case["inputs"]["test"]["path"]) as stream:
        rows = list(itertools.islice(csv.DictReader(stream, delimiter="\t"), args.requests))
    histories = [ast.literal_eval(row["history_behaviors"])[-40:] for row in rows]
    feed = dict(user_ids=torch.tensor([int(r["user_id"]) for r in rows], device=args.device),
                history_ids=torch.tensor([h + [-1] * (40-len(h)) for h in histories], device=args.device))
    diagnostics = []
    with torch.inference_mode():
        embeddings = model.get_user_embedding(feed)
        weights = model.item_embedding.weight[:config["item_num"]]
        logits = embeddings @ weights.T
        for i, row in enumerate(rows):
            single = {k: v[i:i+1] for k, v in feed.items()}
            serial_embedding = model.get_user_embedding(single)
            catalog = torch.arange(config["item_num"], device=args.device)[None]
            legacy = model.full_predict(single, catalog)[0]
            serial_logits = (serial_embedding * weights).sum(-1)
            batched_reduction = (embeddings[i] * weights).sum(-1)
            score = torch.sigmoid(logits[i])
            allowed = 1e-6 + 1e-5 * score.abs()
            failing = (legacy-score).abs() > allowed
            rank = torch.argsort(logits[i], descending=True, stable=True)[:20]
            serial_rank = torch.argsort(serial_logits, descending=True, stable=True)[:20]
            legacy_order = np.argsort(legacy.cpu().numpy() + np.float32(1))[::-1][:20].copy()
            target = int(row["target_item"])
            record = dict(user_id=int(row["user_id"]), target=target,
                max_embedding_difference=float((serial_embedding[0]-embeddings[i]).abs().max()),
                max_sigmoid_score_difference=float((legacy-score).abs().max()),
                failed_score_count=int(failing.sum()),
                max_tolerance_ratio=float(((legacy-score).abs()/allowed).max()),
                max_gemm_vs_elementwise_logit_difference=float((logits[i]-batched_reduction).abs().max()),
                max_serial_vs_batched_elementwise_logit_difference=float((serial_logits-batched_reduction).abs().max()),
                serial_full_predict_matches_serial_reduction=bool(torch.equal(legacy,torch.sigmoid(serial_logits))),
                raw_top20_equal=bool(torch.equal(rank,serial_rank)),
                legacy_vs_raw_top20_equal=bool(np.array_equal(legacy_order,rank.cpu().numpy())),
                raw_top20=rank.cpu().tolist(), serial_raw_top20=serial_rank.cpu().tolist(),
                legacy_top20=legacy_order.tolist())
            diagnostics.append(record)
            print(json.dumps(record), flush=True)
    report = dict(checkpoint=case["checkpoint"], device=args.device, torch_version=torch.__version__,
                  requests=len(rows), diagnostics=diagnostics,
                  cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                  float32_matmul_precision=torch.get_float32_matmul_precision())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
