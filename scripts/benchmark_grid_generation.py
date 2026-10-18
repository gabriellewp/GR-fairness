"""Measure exact GRID generation batching on a fixed validation request sample."""

import argparse
import ast
import csv
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import torch
    import yaml
    from additional_baselines.ifairlrs_grid import original_module, seed_everything
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--requests", type=int, default=128)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    args = parser.parse_args()
    seed_everything(42)
    torch.cuda.set_per_process_memory_fraction(.25)
    config = yaml.safe_load((Path(args.export_dir) / "config.yaml").read_text())
    config["top_k_for_generation"] = 100
    length = config["sequence_length"] // config["num_hierarchies"]
    records = []
    with Path(args.validation).open() as f:
        for row in csv.DictReader(f, delimiter="\t"):
            h = ast.literal_eval(row["history_behaviors"])[-length:]
            records.append((int(row["user_id"]), h + [-1] * (length - len(h))))
            if len(records) == args.requests:
                break
    model = original_module("grid_model").GRIDBaseModel(config).cuda().eval()
    h = torch.tensor([r[1] for r in records], device="cuda")
    u = torch.tensor([r[0] for r in records], device="cuda")
    results, reference = [], None
    for batch in args.batch_sizes:
        try:
            with torch.inference_mode():
                model.generate(h[:batch], u[:batch])
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter()
                generated, probabilities = [], []
                for i in range(0, len(records), batch):
                    ids, p = model.generate(h[i:i+batch], u[i:i+batch])
                    generated.append(ids.cpu())
                    probabilities.append(p.cpu())
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
            ids, p = torch.cat(generated), torch.cat(probabilities)
            if reference is None:
                reference = ids, p
            result = dict(batch_size=batch, requests=len(records), seconds=elapsed,
                requests_per_second=len(records)/elapsed,
                peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
                exact_sid_rows=int((ids == reference[0]).flatten(1).all(1).sum()),
                probabilities_close=bool(torch.allclose(p, reference[1], rtol=1e-4, atol=1e-7)),
                max_probability_difference=float((p-reference[1]).abs().max()))
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            result = dict(batch_size=batch, status="out_of_memory_at_25_percent_cap")
        results.append(result)
        print(json.dumps(result), flush=True)
        Path(args.output).write_text(json.dumps(results, indent=2)+'\n')


if __name__ == "__main__":
    main()
