"""Read existing W&B records for a scoped compute audit; never launch experiments."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
import os
from pathlib import Path
import re
import subprocess


from scripts.paper.archive_paths import optional_source_root, optional_output_root
ROOT = optional_source_root(Path(__file__).resolve().parents[1])
OUT = optional_output_root(ROOT / "docs/compute_cost_audit_20261015")
# Matched to the paths in the parameter inventory and successful output logs.
CONTINUATION_JOBS = {
    "351446": ("amazon-beauty", "rkmeans", "ordinary"),
    "351893": ("amazon-beauty", "rkmeans", "rw"),
    "351894": ("amazon-beauty", "rkmeans", "sprint"),
    "352212": ("amazon-beauty", "rqvae", "ordinary,rw,sprint"),
    "352213": ("amazon-beauty", "rvq", "ordinary,rw,sprint"),
    "352272": ("amazon-sports", "rkmeans", "ordinary,rw,sprint"),
    "352287": ("amazon-sports", "rqvae", "ordinary,rw,sprint"),
    "352288": ("amazon-sports", "rvq", "ordinary,rw,sprint"),
    "352264": ("amazon-toys", "rkmeans", "ordinary,rw,sprint"),
    "352265": ("amazon-toys", "rqvae", "ordinary,rw,sprint"),
    "352387": ("amazon-toys", "rvq", "ordinary,rw,sprint"),
    "352685": ("amazon-pet-supplies", "rkmeans", "ordinary,rw,sprint"),
    "352686": ("amazon-pet-supplies", "rqvae", "ordinary"),
    "352687": ("amazon-pet-supplies", "rvq", "ordinary,rw,sprint"),
    "358525": ("amazon-pet-supplies", "rqvae", "rw"),
    "358526": ("amazon-pet-supplies", "rqvae", "sprint"),
}
CANDIDATE_JOBS = {
    "355047": ("amazon-sports", "rkmeans", "candidates"),
    "355054": ("amazon-toys", "rkmeans", "candidates"),
    "355061": ("amazon-pet-supplies", "rkmeans", "candidates"),
    "357377": ("amazon-beauty", "rqvae", "candidates"),
    "357385": ("amazon-beauty", "rvq", "candidates"),
    "357394": ("amazon-toys", "rqvae", "candidates"),
    "357402": ("amazon-toys", "rvq", "candidates"),
    "357411": ("amazon-sports", "rqvae", "candidates"),
    "357435": ("amazon-pet-supplies", "rqvae", "candidates"),
    "357443": ("amazon-pet-supplies", "rvq", "candidates"),
    "358210": ("amazon-sports", "rvq", "candidates;beam=300"),
}
# These interrupted sessions produced the Pet checkpoints resumed by the final
# source-model runs. The later RKMeans branch prn9loap also starts from step 4200
# and is not an ancestor of the selected 10k checkpoint.
NATIVE_PARENT_RUNS = {"rkmeans": "n9mses7r", "rvq": "8c127whv", "rqvae": "gq9jlgrv"}


def local_records():
    paths = list((ROOT / "wandb").glob("run-*/files/wandb-metadata.json"))
    paths += list((ROOT / "GRID/outputs").glob("**/wandb-metadata.json"))
    records = []
    seen = set()
    for path in sorted(paths):
        if path.resolve() in seen:
            continue
        seen.add(path.resolve())
        metadata = json.loads(path.read_text())
        summary_path = path.with_name("wandb-summary.json")
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        slurm = metadata.get("slurm", {})
        args = metadata.get("args", [])
        job = slurm.get("job_id", "")
        task = slurm.get("array_task_id", "")
        configfile = ""
        for i, arg in enumerate(args[:-1]):
            if arg == "--slurm_job_id":
                job = args[i + 1]
            elif arg == "--slurm_array_task_id":
                task = args[i + 1]
            elif arg == "--config_file":
                configfile = args[i + 1]
        records.append(dict(
            run_id=path.parent.parent.name.rsplit("-", 1)[-1],
            source=str(path.relative_to(ROOT)),
            started_at=metadata.get("startedAt"), program=metadata.get("program"),
            runtime_seconds=summary.get("_wandb", {}).get("runtime", summary.get("_runtime")),
            gpu_count=metadata.get("gpu_count"), gpu=metadata.get("gpu"),
            slurm_job_id=job, slurm_array_task_id=task,
            slurm_array_job_id=slurm.get("array_job_id", ""),
            config_file=configfile,
            experiment_args=[arg for arg in args if arg.startswith(("experiment=", "hydra.run.dir=", "ckpt_path="))],
        ))
    return records


def fetch_reported():
    import wandb

    requested = {}
    with (ROOT / "docs/rq2_comparison_draft_20261011/comparison_results.csv").open() as handle:
        for row in csv.DictReader(handle):
            if row["wandb_url"]:
                requested[row["wandb_url"].rsplit("/", 1)[-1]] = {
                    key: row[key] for key in ("dataset", "tokenizer", "method", "source", "wandb_url")
                }
    api = wandb.Api(timeout=30)

    def fetch(item):
        run_id, identity = item
        run = api.run(f"{os.environ['WANDB_RECOMMENDATION_PROJECT_PATH']}/{run_id}")
        summary = dict(run.summary)
        runtime = summary.get("_wandb", {}).get("runtime", summary.get("_runtime"))
        fields = ("log_name", "model", "rank_model", "dataset", "grid_tokenizer", "seed",
                  "slurm_job_id", "slurm_array_task_id", "grid_fair_finetune_steps", "grid_checkpoint_path")
        return dict(**identity, run_id=run.id, name=run.name, state=run.state,
                    runtime_seconds=runtime, summary_runtime_seconds=summary.get("_runtime"),
                    created_at=run.created_at,
                    config={key: run.config.get(key) for key in fields})

    results = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        for future in as_completed([pool.submit(fetch, item) for item in requested.items()]):
            results.append(future.result())
            if len(results) % 12 == 0:
                print(f"Read {len(results)}/{len(requested)} reported W&B runs", flush=True)
    return sorted(results, key=lambda row: (row["dataset"], row["tokenizer"], row["method"]))


def fetch_native(local):
    import wandb

    inventory = json.loads((ROOT / "docs/experiment_parameters_20261014/parameter_inventory.json").read_text())
    requested = []
    for record in inventory["records"]:
        if record["family"] != "native_generator":
            continue
        config = record["parameters"]
        stage = (ROOT / record["config_sources"][0]).parents[1].resolve()
        name = config["logger"]["wandb"]["name"]
        job = re.search(r"-(\d+)-tiger-training$", name).group(1)
        matches = [row for row in local if row["slurm_job_id"] == job
                   and f"hydra.run.dir={stage}" in row["experiment_args"]
                   and "experiment=tiger_train_flat" in row["experiment_args"]]
        if len(matches) != 1:
            raise ValueError(f"Ambiguous native training execution: {record['id']}")
        identity = dict(dataset=record["dataset"], tokenizer=record["tokenizer"],
                        stage_path=str(stage), config_source=record["config_sources"][0],
                        selected_checkpoint=record["checkpoint"]["path"])
        requested.append(dict(**identity, **matches[0], role="final"))
        if config.get("ckpt_path"):
            parent_id = NATIVE_PARENT_RUNS[record["tokenizer"]]
            parent = next(row for row in local if row["run_id"] == parent_id)
            requested.append(dict(**{**identity, "stage_path": str(Path(config["ckpt_path"]).parents[1]),
                                      "selected_checkpoint": config["ckpt_path"]},
                                  **parent, role="parent"))
    api = wandb.Api(timeout=30)

    def fetch(item):
        run = api.run(os.environ["WANDB_GRID_PROJECT_PATH"] + "/" + item["run_id"])
        summary = dict(run.summary)
        return dict(**item, wandb_url=run.url, name=run.name, state=run.state,
                    online_runtime=summary.get("_wandb", {}).get("runtime", summary.get("_runtime")),
                    summary_runtime=summary.get("_runtime"), global_step=summary.get("trainer/global_step"),
                    created_at=run.created_at)

    with ThreadPoolExecutor(max_workers=6) as pool:
        return list(pool.map(fetch, requested))


def collect_slurm():
    jobs = {**CONTINUATION_JOBS, **CANDIDATE_JOBS}
    command = ["sacct", "-X", "-j", ",".join(jobs), "-n", "-P", "--units=G",
               "--format=JobIDRaw,JobName%80,State,ElapsedRaw,AllocCPUS,AllocTRES%180,Start,End"]
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    (OUT / "slurm_accounting_raw.txt").write_text(result.stdout)
    rows = []
    for line in result.stdout.splitlines():
        job, name, state, elapsed, cpus, tres, start, end = [v.strip() for v in line.split("|")]
        if job not in jobs:
            continue
        match = re.search(r"(?:^|,)gres/gpu=(\d+)(?:,|$)", tres)
        gpu_count = int(match.group(1)) if match else 0
        if state != "COMPLETED" or gpu_count != 1:
            raise ValueError(f"Unexpected accounting record: {line}")
        dataset, tokenizer, modes = jobs[job]
        logs = list((ROOT / "output").glob(f"*{job}*.out"))
        rows.append(dict(job_id=job, dataset=dataset, tokenizer=tokenizer, modes=modes,
                         category="historical_continuation" if job in CONTINUATION_JOBS else "candidate_generation",
                         job_name=name, state=state, elapsed_seconds=int(elapsed), gpu_count=gpu_count,
                         gpu_hours=int(elapsed) * gpu_count / 3600, allocated_cpus=int(cpus),
                         allocated_tres=tres, start=start, end=end,
                         logs=";".join(str(p.relative_to(ROOT)) for p in sorted(logs))))
    if len(rows) != len(jobs):
        raise ValueError("Missing selected Slurm jobs")
    with (OUT / "slurm_gpu_hours.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def summarize(local):
    byid = {row["run_id"]: row for row in local}
    summary = {"audit_date": "2026-10-15", "units": "elapsed seconds times GPU count divided by 3600",
               "categories": [], "limitations": [
                   "This is a partial ledger of the reported experiments, not project-wide compute.",
                   "Fine-tuning runtimes include validation and final test generation; suite jobs also include prerequisite score exports.",
                   "W&B runtimes cover the tracked run interval; Slurm covers the full allocated job interval. They are reported separately.",
                   "GPU hours measure elapsed time associated with GPUs, not GPU utilization or energy.",
                   "The joint-fairness pilot, embeddings/tokenizers, classic baselines, discarded retries, and most reranking CPU jobs are excluded.",
                   "Beauty RKMeans candidate generation shares job 351160 with an earlier RW experiment and has no isolated duration here.",
                   "The replaced Sports RVQ beam-100 candidate job 357419 is excluded from the final-pool generation subtotal.",
               ]}
    native_path = OUT / "native_grid_wandb.json"
    if native_path.exists():
        native = json.loads(native_path.read_text())
        selected = [row for row in native if row["role"] == "final" or row["run_id"] in NATIVE_PARENT_RUNS.values()]
        if len(selected) != 15 or len({row["run_id"] for row in selected}) != 15:
            raise ValueError("Expected 12 final native training sessions and 3 required parent sessions")
        rows = []
        for run in selected:
            if run["gpu_count"] != 1 or run["online_runtime"] is None:
                raise ValueError("Missing native GPU count or recorded runtime")
            if run["role"] == "final" and run["state"] != "finished":
                raise ValueError("Final native training session did not finish")
            rows.append(dict(dataset=run["dataset"], tokenizer=run["tokenizer"], role=run["role"],
                run_id=run["run_id"], state=run["state"], runtime_seconds=run["online_runtime"],
                gpu_count=run["gpu_count"], gpu_hours=run["online_runtime"] * run["gpu_count"] / 3600,
                gpu=run["gpu"], wandb_url=run["wandb_url"], metadata_source=run["source"],
                slurm_job_id=run["slurm_job_id"],
                timing_status="finished W&B interval" if run["state"] == "finished" else "last logged interval; interrupted parent"))
        rows.sort(key=lambda row: (row["dataset"], row["tokenizer"], row["role"]))
        with (OUT / "native_grid_gpu_hours.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        summary["categories"].append(dict(category="native_grid_training_and_evaluation",
            source="W&B stage runtimes + local GPU metadata", count=len(rows), model_count=12,
            gpu_hours=sum(row["gpu_hours"] for row in rows),
            finished_final_session_gpu_hours=sum(row["gpu_hours"] for row in rows if row["role"] == "final"),
            required_parent_recorded_gpu_hours=sum(row["gpu_hours"] for row in rows if row["role"] == "parent"),
            per_dataset={d: sum(row["gpu_hours"] for row in rows if row["dataset"] == d)
                         for d in sorted({row["dataset"] for row in rows})}))
        summary["limitations"].extend([
            "Native GRID includes 12 finished training stages and 3 interrupted Pet parent sessions whose checkpoints were reused; interrupted parents have last-logged runtimes, so this is a partial recorded subtotal.",
            "The unused Pet RKMeans sibling continuation 351146 and empty failed-restart sessions are excluded.",
            "Native training-stage W&B runtimes are used instead of whole pipeline-job times, which can include embeddings, tokenizers, or later failures.",
        ])
    online_path = OUT / "reported_fairdiverse_wandb.json"
    if online_path.exists():
        online = json.loads(online_path.read_text())
        rows = []
        for run in online:
            meta = byid[run["run_id"]]
            if run["state"] != "finished" or meta["gpu_count"] != 1:
                raise ValueError(f"Incomplete or ambiguous reported run: {run['run_id']}")
            rows.append(dict(dataset=run["dataset"], tokenizer=run["tokenizer"], method=run["method"],
                             run_id=run["run_id"], runtime_seconds=run["runtime_seconds"], gpu_count=meta["gpu_count"],
                             gpu_hours=run["runtime_seconds"] * meta["gpu_count"] / 3600,
                             gpu=meta["gpu"], wandb_url=run["wandb_url"], metadata_source=meta["source"]))
        with (OUT / "reported_fairdiverse_gpu_hours.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        summary["categories"].append(dict(category="fairdiverse_training_and_evaluation", source="W&B runtime + local GPU metadata",
            count=len(rows), gpu_hours=sum(r["gpu_hours"] for r in rows),
            per_dataset={d: sum(r["gpu_hours"] for r in rows if r["dataset"] == d) for d in sorted({r["dataset"] for r in rows})}))
    slurm_path = OUT / "slurm_gpu_hours.csv"
    if slurm_path.exists():
        rows = list(csv.DictReader(slurm_path.open()))
        for category in ("historical_continuation", "candidate_generation"):
            selected = [r for r in rows if r["category"] == category]
            summary["categories"].append(dict(category=category, source="Slurm full job allocation",
                count=len(selected), gpu_hours=sum(float(r["gpu_hours"]) for r in selected),
                per_dataset={d: sum(float(r["gpu_hours"]) for r in selected if r["dataset"] == d)
                             for d in sorted({r["dataset"] for r in selected})}))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["categories"], indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--online", action="store_true")
    parser.add_argument("--slurm", action="store_true")
    parser.add_argument("--native-online", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    local = local_records()
    (OUT / "local_wandb_runtime.json").write_text(json.dumps(local, indent=2) + "\n")
    print(f"Indexed {len(local)} local W&B metadata records", flush=True)
    if args.online:
        online = fetch_reported()
        (OUT / "reported_fairdiverse_wandb.json").write_text(json.dumps(online, indent=2) + "\n")
        print(f"Saved {len(online)} reported FairDiverse W&B records", flush=True)
    if args.slurm:
        collect_slurm()
    if args.native_online:
        native = fetch_native(local)
        (OUT / "native_grid_wandb.json").write_text(json.dumps(native, indent=2) + "\n")
    summarize(local)


if __name__ == "__main__":
    main()
