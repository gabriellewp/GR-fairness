"""Sampled validation screening, followed by full confirmation and full test."""

from collections import Counter, defaultdict
import csv
import gc
import json
import math
from pathlib import Path
import random
import time

from .adapters import GenerationRequest
from .batched_generation import generate_grid_batched
from .candidates import file_reference
from .continuation_selection import checkpoint_key, objective_values, selection_policy
from .deadline_continuation import run_deadline_continuation
from .evaluation import evaluate
from .ifairlrs import history_shares, popularity_groups
from .ifairlrs_grid import original_module, read_rows, seed_everything, training_popularity, write_json
from .prepare_beauty import ROOT
from .tune_continuation import choose_winners


def stratified_sample(rows, activity, popularity, size=10000, seed=42):
    strata = defaultdict(list)
    for user, _, target in rows:
        strata[activity[user] + "/" + popularity[target]].append(user)
    size = min(size, len(rows))
    raw = {s: size * len(users) / len(rows) for s, users in strata.items()}
    counts = {s: math.floor(v) for s, v in raw.items()}
    remaining = size - sum(counts.values())
    for s in sorted(strata, key=lambda s: (-(raw[s] - counts[s]), s))[:remaining]:
        counts[s] += 1
    rng = random.Random(seed)
    users = sorted(u for s in sorted(strata) for u in rng.sample(sorted(strata[s]), counts[s]))
    if len(users) != len(set(users)) or len(users) != size:
        raise ValueError("Sampling requires one validation request per user")
    return dict(user_ids=users, seed=seed, sample_size=size, population_rows=len(rows),
                stratification="training activity group x training-popularity group of validation target",
                population_strata={s: len(v) for s, v in strata.items()}, sample_strata=counts)


def prepare(matrix_path, destination):
    matrix_path, destination = Path(matrix_path).resolve(), Path(destination).resolve()
    original = json.loads(matrix_path.read_text())
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    samples = {}
    for dataset in original["datasets"]:
        data = ROOT / "recommendation/processed_dataset" / dataset
        brand = {int(i): str(g) for i, g in json.loads((data / "iid2pid.json").read_text()).items()}
        training = read_rows(data / f"{dataset}.train", training=True, history_length=40, catalog=set(brand))
        popularity = popularity_groups(training_popularity(training), set(brand))
        del training
        rows = read_rows(data / f"{dataset}.valid.ranking", training=False, history_length=40, catalog=set(brand))
        with (data / "user_activity.csv").open() as f:
            activity = {int(r["user_id"]): r["activity_group"] for r in csv.DictReader(f)}
        sample = stratified_sample(rows, activity, popularity)
        sample["source"] = file_reference(data / f"{dataset}.valid.ranking")
        path = destination / f"{dataset}_sample.json"
        write_json(path, sample)
        group_path = destination / f"{dataset}_groups.json"
        write_json(group_path, dict(popularity=popularity, brand=brand))
        samples[dataset] = dict(sample=file_reference(path), groups=file_reference(group_path))
        print(f"Prepared {dataset}: {sample['sample_size']}/{len(rows)} validation users", flush=True)
    cases = []
    # Put the largest three cases first so their full confirmations start early.
    ordered = sorted(original["cases"], key=lambda c: (c["dataset"] != "amazon-pet-supplies", c["index"]))
    for index, case in enumerate(ordered):
        options = dict(case["sweep_options"])
        options["output_dir"] = str(destination / f"{case['dataset']}_{case['tokenizer']}")
        cases.append(dict(index=index, dataset=case["dataset"], tokenizer=case["tokenizer"],
            options=options, trials=case["plan"]["trials"], prepared_inputs=case["inputs"],
            control=case["plan"]["control"], export_config=case["plan"]["export_config"],
            sample=samples[case["dataset"]]["sample"], groups=samples[case["dataset"]]["groups"]))
    source_files = ["additional_baselines/batched_generation.py", "additional_baselines/deadline_continuation.py",
                    "additional_baselines/deadline_pilot.py", "scripts/run_deadline_pilot.py"]
    code = dict(original["code"])
    code.update({p: file_reference(ROOT / p) for p in source_files})
    manifest = dict(cases=cases, case_count=12, trial_count=48, batch_size=16, sample_size=10000,
        full_validation_confirmation=True, full_test=True, seed=42, max_parallel_gpu_tasks=6,
        code=code, source_manifest=file_reference(matrix_path),
        note="Screening uses fixed stratified validation; full claims require full validation and full test. No test-based retuning.")
    path = destination / "manifest.json"
    write_json(path, manifest)
    return path


def verify(ref):
    if file_reference(ref["path"]) != ref:
        raise ValueError(f"Frozen input changed: {ref['path']}")


def evaluate_checkpoint(case, checkpoint, split, *, sample=None, output=None):
    import torch
    import yaml
    verify(checkpoint)
    config = yaml.safe_load(Path(case["export_config"]["path"]).read_text())
    source_control = json.loads(Path(case["control"]["path"]).read_text())
    verify(source_control["identity"]["checkpoint"])
    verify(source_control["identity"]["sid_mapping"])
    config["grid_checkpoint_path"] = source_control["identity"]["checkpoint"]["path"]
    config["sid_path"] = source_control["identity"]["sid_mapping"]["path"]
    config["top_k_for_generation"] = source_control["identity"]["generation_beam_width"]
    groups = {name: {int(i): g for i, g in m.items()}
              for name, m in json.loads(Path(case["groups"]["path"]).read_text()).items()}
    path = case["prepared_inputs"][split]["path"]
    rows = read_rows(path, training=False, history_length=config["sequence_length"] // config["num_hierarchies"],
                     catalog=set(groups["brand"]))
    if sample is not None:
        selected = set(sample["user_ids"])
        rows = [r for r in rows if r[0] in selected]
        if len(rows) != len(selected):
            raise ValueError("Sample user alignment changed")
    seed_everything(42)
    model = original_module("grid_model").GRIDBaseModel(config).cuda().eval()
    if checkpoint != source_control["identity"]["checkpoint"]:
        saved = torch.load(checkpoint["path"], map_location="cpu", weights_only=False)
        model.grid.load_state_dict(saved.get("state_dict", saved), strict=True)
    sids = dict(enumerate(model.tokenizer.sid_table.detach().cpu().tolist()))
    eligible = list(sids)
    histories = {f"{split}:{i}:user:{u}": h for i, (u, h, _) in enumerate(rows)}
    gh = history_shares(histories, histories, groups["popularity"])
    metadata = dict(checkpoint=checkpoint, sid_mapping=source_control["identity"]["sid_mapping"],
        split=dict(name=split, source=case["prepared_inputs"][split]), seed=42,
        generation=dict(beam_width=config["top_k_for_generation"], sequence_length=config["sequence_length"],
                        num_hierarchies=config["num_hierarchies"]),
        score_convention="probability", short_list_policy="return_available")
    parity = None
    if sample is not None:
        probe = rows[:64]
        probe_requests = [GenerationRequest(f"probe:{i}", h, [t], eligible, u)
                          for i, (u, h, t) in enumerate(probe)]
        single = generate_grid_batched(model, probe_requests, sids, groups, metadata, device="cuda", batch_size=1)
        batched = generate_grid_batched(model, probe_requests, sids, groups, metadata, device="cuda", batch_size=16)
        exact = all(a.candidate_ids == b.candidate_ids and single.ranking(a) == batched.ranking(b)
                    for a, b in zip(single.requests, batched.requests))
        a = torch.tensor([v for r in single.requests for v in r.scores])
        b = torch.tensor([v for r in batched.requests for v in r.scores])
        close = a.shape == b.shape and bool(torch.allclose(a, b, rtol=1e-4, atol=1e-7))
        parity = dict(requests=len(probe), exact_rankings=exact, scores_close=close)
        if not exact or not close:
            raise RuntimeError(f"Batched generation parity failed: {parity}")
        print(f"Batch parity passed on {len(probe)} sampled users", flush=True)
        del single, batched
    started = time.monotonic()

    def requests():
        for i, (user, history, target) in enumerate(rows):
            if i and i % 10000 == 0:
                print(f"{split}: generated {i}/{len(rows)} requests; elapsed={time.monotonic()-started:.1f}s", flush=True)
            yield GenerationRequest(f"{split}:{i}:user:{user}", history, [target], eligible, user)

    artifact = generate_grid_batched(model, requests(), sids, groups, metadata, device="cuda", batch_size=16)
    metrics = evaluate(artifact, (5, 10, 20))
    report = dict(metrics=metrics, history_shares=gh, input_metadata=artifact.metadata,
                  scope="fixed_stratified_sample" if sample else "full_split", sample=case["sample"] if sample else None,
                  candidate_export_saved=False, generation_batch_size=16, seconds=time.monotonic()-started)
    report["sampled_batch_parity_check"] = parity
    if output is not None:
        write_json(output, report)
    del artifact, model
    gc.collect()
    torch.cuda.empty_cache()
    return report


def run_case(manifest_path, index):
    import torch
    manifest = json.loads(Path(manifest_path).read_text())
    case = manifest["cases"][index]
    for ref in [*manifest["code"].values(), *case["prepared_inputs"].values(),
                case["control"], case["export_config"], case["sample"], case["groups"]]:
        verify(ref)
    output = Path(case["options"]["output_dir"])
    output.mkdir()
    print(f"Starting deadline pilot: {case['dataset']}/{case['tokenizer']}", flush=True)
    sample = json.loads(Path(case["sample"]["path"]).read_text())
    source_control = json.loads(Path(case["control"]["path"]).read_text())
    baseline = evaluate_checkpoint(case, source_control["selected_checkpoint"], "validation", sample=sample,
                                   output=output / "sampled_ordinary_control.json")
    control = dict(source_control)
    control["identity"] = dict(control["identity"], validation_sample=case["sample"])
    control["selected_validation_metrics"] = baseline["metrics"]
    control["validation_scope"] = "fixed_stratified_sample"
    control["source_control"] = case["control"]
    control["selection_policy"] = "Original full-validation-selected ordinary weights, reevaluated on the same frozen screening sample"
    control_path = output / "sample_control.json"
    write_json(control_path, control)
    identity = source_control["identity"]
    common = dict(control_report=control_path, group="popularity", device="cuda", seed=42,
        max_updates=identity["max_updates"], validation_interval=identity["validation_interval"],
        learning_rate=identity["learning_rate"], optimizer_state=identity["optimizer_state"],
        export_dir=case["options"]["export_dir"], data_dir=case["options"]["data_dir"],
        generation_beam_width=identity["generation_beam_width"], tokenizer=case["tokenizer"], dataset=case["dataset"],
        require_control_improvement=False, selection_objective="both", max_relative_recall_loss=.02,
        evaluate_test=False, validation_sample_path=case["sample"]["path"])
    completed = []
    for settings in case["trials"]:
        options = dict(common)
        options.update({k: v for k, v in settings.items() if k != "name"})
        folder = output / settings["name"]
        report = run_deadline_continuation(output_dir=folder, **options)
        completed.append(dict(settings=settings, run_dir=str(folder), report=report))
        write_json(output / f"completed_{settings['name']}.json", completed[-1])
        gc.collect()
        torch.cuda.empty_cache()
    winners = choose_winners(completed, control_path)
    selection_path = output / "sampled_selection.json"
    write_json(selection_path, dict(winners=winners, trial_count=4, selection_scope="fixed_stratified_sample",
               sample=case["sample"], selection_uses_test=False))
    # Full confirmation gates claims; its outcomes never choose another setting.
    full_policy, full_history = None, None
    if any(w["status"] == "validation_fairness_improved" for w in winners.values()):
        full_baseline = evaluate_checkpoint(case, source_control["selected_checkpoint"], "validation",
                                           output=output / "full_ordinary_validation.json")
        full_history = full_baseline["history_shares"]
        full_policy = selection_policy("both", control_metrics=full_baseline["metrics"],
                                       history_shares=full_history, max_relative_recall_loss=.02)
    confirmations = {}
    for mode, winner in winners.items():
        if winner["status"] != "validation_fairness_improved":
            confirmations[mode] = dict(status="no_sampled_improvement_keep_ordinary")
            continue
        folder = output / f"{mode}_confirmation"
        folder.mkdir()
        full = evaluate_checkpoint(case, winner["selected_checkpoint"], "validation",
                                   output=folder / "full_validation.json")
        key = checkpoint_key(dict(update=winner["selected_update"], metrics=full["metrics"]), full_policy)
        status = "full_validation_confirmed" if key is not None else "failed_full_validation_keep_ordinary"
        confirmations[mode] = dict(status=status, checkpoint=winner["selected_checkpoint"],
            objectives=objective_values(full["metrics"], full_history), full_validation_policy=full_policy)
        if key is not None:
            evaluate_checkpoint(case, winner["selected_checkpoint"], "test", output=folder / "full_test.json")
            confirmations[mode]["test_result"] = file_reference(folder / "full_test.json")
        write_json(folder / "confirmation.json", confirmations[mode])
    write_json(output / "complete.json", dict(dataset=case["dataset"], tokenizer=case["tokenizer"],
        sample_selection=file_reference(selection_path), confirmations=confirmations,
        ordinary_fallback=case["control"], no_test_based_retuning=True))
    print(f"Completed deadline pilot: {case['dataset']}/{case['tokenizer']}", flush=True)
