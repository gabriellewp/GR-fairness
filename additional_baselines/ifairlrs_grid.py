"""Isolated GRID IFairLRS workflow; original model, collator and trainer stay untouched."""

import ast
from collections import Counter
import csv
from dataclasses import replace
import importlib
import hashlib
import json
from pathlib import Path
import random
import sys
from types import ModuleType

from .adapters import GenerationRequest, generate_grid
from .candidates import file_reference, save
from .evaluation import evaluate
from .ifairlrs import apply_rr, fairness, fit_rr, popularity_groups, rw_weights, weighted_loss
from .prepare_beauty import DEFAULT_DATA, DEFAULT_EXPORT, ROOT


def original_module(name):
    prefix = "additional_baselines._grid_recommendation"
    root = ROOT / "fairdiverse/recommendation"
    for package, path in ((prefix, root), (prefix + ".base_model", root / "base_model")):
        if package not in sys.modules:
            module = ModuleType(package)
            module.__path__ = [str(path)]
            module.__package__ = package
            sys.modules[package] = module
    return importlib.import_module(f"{prefix}.base_model.{name}")


def seed_everything(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_rows(path, *, training, history_length, catalog):
    rows = []
    with Path(path).open(newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            user = int(row["user_id:token"] if training else row["user_id"])
            target = int(row["item_id:token"] if training else row["target_item"])
            history = ast.literal_eval(row["history_behaviors"])
            if not isinstance(history, list) or any(type(i) is not int or i not in catalog for i in history) or target not in catalog:
                raise ValueError("History/target item IDs differ from the configured catalog")
            if training and float(row["label:float"]) != 1:
                raise ValueError("GRID RW expects positive sequential training examples")
            rows.append((user, history[-history_length:], target))
    if not rows:
        raise ValueError(f"Empty split: {path}")
    return rows


def training_popularity(rows):
    """Each chronological event once, rather than counting repeated prefixes."""
    counts, seen_users = Counter(), set()
    for user, history, target in rows:
        if user not in seen_users:
            counts.update(history)
            seen_users.add(user)
        counts[target] += 1
    return counts


def replay_batches(rows, *, epochs, seed, history_length, pre_batch_size, batch_size, augment=True):
    """Deterministically replay shuffle + original capped suffix augmentation.

    CPU RNG is isolated from model/dropout RNG. No augmented examples are
    materialized on disk, and counting and both training branches see exactly
    the same examples, batch boundaries and epoch order.
    """
    import torch
    collate = original_module("grid_collate").collate_fn_grid_causal_augmentation
    for epoch in range(epochs):
        generator = torch.Generator(device="cpu").manual_seed(seed + epoch)
        order = torch.randperm(len(rows), generator=generator).tolist()
        # The original collator uses the default CPU RNG. Temporarily lend it
        # the private generator's state, without seeding any accelerator RNG.
        generator_state = generator.get_state()
        for start in range(0, len(order), pre_batch_size):
            samples = []
            for index in order[start:start + pre_batch_size]:
                user, history, target = rows[index]
                padded = history + [-1] * (history_length - len(history))
                samples.append(tuple(torch.tensor(value, dtype=torch.long) for value in (user, padded, target, 0)))
            with torch.random.fork_rng(devices=[]):
                torch.set_rng_state(generator_state)
                batch = collate(samples, max_batch_size=batch_size, padding_value=-1) if augment else tuple(torch.stack([s[j] for s in samples]) for j in range(4))
                generator_state = torch.get_rng_state()
            yield batch


def train_branch(model, batches, mapping, weights, *, device, learning_rate, weight_decay, clip_norm, regularizer=None):
    import torch
    model.to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    count, loss_sum, steps = 0, 0.0, 0
    term_sums = {}
    digest = hashlib.sha256()
    for users, histories, targets, groups in batches:
        for tensor in (users, histories, targets, groups):
            digest.update(str(tuple(tensor.shape)).encode())
            digest.update(tensor.contiguous().numpy().tobytes())
        interaction = {"user_ids": users.to(device), "history_ids": histories.to(device), "item_ids": targets.to(device)}
        optimizer.zero_grad(set_to_none=True)
        if regularizer is None:
            losses = model.compute_loss(interaction)
            sample_weights = losses.new_tensor([weights[mapping[i]] for i in targets.tolist()])
            loss = weighted_loss(losses, sample_weights)
        else:
            loss = regularizer.objective(interaction)
            if loss.ndim != 0 or not torch.isfinite(loss):
                raise ValueError("SPRINT training objective must be a finite scalar")
            if regularizer.enabled:
                for name, value in regularizer.last_terms.items():
                    term_sums[name] = term_sums.get(name, 0.0) + value * len(targets)
        loss.backward()
        if clip_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()
        count += len(targets)
        loss_sum += float(loss.detach()) * len(targets)
        steps += 1
        if steps % 100 == 0:
            print(f"training: {steps} steps, {count} examples", flush=True)
    if not steps:
        raise ValueError("Training plan is empty")
    report = {"optimizer_steps": steps, "examples": count, "mean_training_loss": loss_sum / count,
              "training_plan_sha256": digest.hexdigest()}
    if term_sums:
        report["mean_sprint_unweighted_terms"] = {name: value / count for name, value in term_sums.items()}
    return report


def plan_digest(batches):
    digest = hashlib.sha256()
    for batch in batches:
        for tensor in batch:
            digest.update(str(tuple(tensor.shape)).encode())
            digest.update(tensor.contiguous().numpy().tobytes())
    return digest.hexdigest()


def regenerate(model, rows, split_path, split, groups, checkpoint, sid_path, seed, device, provenance):
    """Beam membership followed by native log likelihood; no shifted-score loss."""
    import torch
    table = model.tokenizer.sid_table.detach().cpu().tolist()
    sids = {i: sid for i, sid in enumerate(table)}
    catalog = list(sids)
    metadata = {"checkpoint": file_reference(checkpoint), "sid_mapping": file_reference(sid_path),
                "split": {"name": split, "source": file_reference(split_path)}, "seed": seed,
                "generation": {"beam_width": model.top_k_for_generation, "sequence_length": model.sequence_length,
                               "num_hierarchies": model.num_hierarchies, "use_prefix_validation": bool(model.config.get("use_prefix_validation", False))},
                "score_convention": "probability", "candidate_membership": "generated_sid_exact_match",
                "short_list_policy": "return_available", "provenance": provenance}
    histories, generated = {}, []
    model.eval()
    print(f"Generating {split}: {len(rows)} requests", flush=True)

    def requests():
        for index, (user, history, target) in enumerate(rows):
            if index and index % 100 == 0:
                print(f"{split}: generated {index}/{len(rows)}", flush=True)
            yield GenerationRequest(f"{split}:{index}:user:{user}", history, [target], catalog, user)

    # One adapter call validates/builds the SID and group mappings once for
    # the split. Request order, eligibility and diagnostic totals are retained.
    artifact = generate_grid(model, requests(), sids, groups, metadata, device=device)
    diagnostics = artifact.metadata["generation_adapter"]
    print(f"Scoring {split}: {len(rows)} requests", flush=True)
    for index, (result, (user, history, target)) in enumerate(zip(artifact.requests, rows)):
        values = []
        # CE summed over SID positions equals negative native log probability.
        with torch.no_grad():
            for start in range(0, len(result.candidate_ids), 32):
                ids = result.candidate_ids[start:start + 32]
                interaction = {"history_ids": torch.tensor([history] * len(ids), dtype=torch.long, device=device),
                               "user_ids": torch.full((len(ids),), user, dtype=torch.long, device=device),
                               "item_ids": torch.tensor(ids, dtype=torch.long, device=device)}
                values.extend((-model.compute_loss(interaction)).double().cpu().tolist())
        generated.append(replace(result, scores=values))
        histories[result.request_id] = history
        if (index + 1) % 100 == 0:
            print(f"{split}: scored {index + 1}/{len(rows)}", flush=True)
    artifact.requests = generated
    artifact.metadata["score_convention"] = "log_probability"
    artifact.metadata["candidate_membership"] = "beam_sid_exact_match_full_catalog"
    artifact.metadata["generation_adapter"] = {"name": "GRID beam + teacher-forced native log SID likelihood",
                                                "distance": "negative_log_sid_probability", "unknown_sid_count": diagnostics["unknown_sid_count"],
                                                "ineligible_count": diagnostics["ineligible_count"], "duplicate_policy": "stable_first_order"}
    return artifact.validate(), histories


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def run_beauty(*, mode, output_dir, group="popularity", device="cuda", seed=42,
               epochs=1, alpha=0.05, epsilon=1e-6, limit_requests=None, limit_train_rows=None,
               export_dir=DEFAULT_EXPORT, data_dir=DEFAULT_DATA, sprint_settings=None, generation_beam_width=None):
    import torch
    import yaml
    if mode not in {"rr", "rw", "sprint"} or group not in {"popularity", "brand"}:
        raise ValueError("Use rr/rw/sprint and popularity/brand")
    if sprint_settings is not None and mode != "sprint":
        raise ValueError("SPRINT settings require sprint mode")
    if mode == "sprint":
        from .sprint_grid import resolve_settings
        sprint_settings = resolve_settings(sprint_settings or {})
    if generation_beam_width is not None and (type(generation_beam_width) is not int or generation_beam_width <= 0):
        raise ValueError("Generation beam width must be a positive integer")
    if type(epochs) is not int or epochs <= 0 or type(seed) is not int:
        raise ValueError("Epochs must be positive and seed integer")
    if mode == "rr":
        import math
        if not math.isfinite(alpha) or alpha < 0 or not math.isfinite(epsilon) or not 0 < epsilon <= 1:
            raise ValueError("Require alpha >= 0 and 0 < epsilon <= 1")
    for limit in (limit_requests, limit_train_rows):
        if limit is not None and (type(limit) is not int or limit <= 0):
            raise ValueError("Smoke-test limits must be positive integers")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; run in the GPU environment or use --device cpu for a smoke test")
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"Output already exists: {destination}")
    # Reuse the established source/mapping/seed validation without copying inputs.
    from .prepare_beauty import load_beauty
    config_path = Path(export_dir) / "config.yaml"
    config = yaml.safe_load(config_path.read_text())
    verified, _, summary = load_beauty(export_dir=export_dir, data_dir=data_dir, topk=(1,),
                                     check_dense_memory=False, tokenizer=config["grid_tokenizer"])
    if generation_beam_width is not None:
        config["top_k_for_generation"] = generation_beam_width
    if config.get("level_loss_weights") is not None or config.get("ranking_loss_weight", 0) != 0:
        raise ValueError("This workflow supports the ordinary summed GRID SID cross entropy objective")
    if config["sequence_length"] != config["history_length"] * config["num_hierarchies"]:
        raise ValueError("RW history accounting requires history_length * num_hierarchies == sequence_length")
    if config.get("causal_group_by_user", False):
        raise ValueError("This workflow uses shuffled interaction rows; grouped-user batching is not supported")
    catalog = set(verified.item_groups["brand"])
    data_dir = Path(data_dir)
    paths = {"train": data_dir / "amazon-beauty.train", "validation": data_dir / "amazon-beauty.valid.ranking",
             "test": data_dir / "amazon-beauty.test.ranking"}
    rows = {split: read_rows(path, training=split == "train", history_length=config["history_length"], catalog=catalog)
            for split, path in paths.items()}
    groups = {"popularity": popularity_groups(training_popularity(rows["train"]), catalog),
              "brand": verified.item_groups["brand"]}
    if limit_train_rows is not None:
        rows["train"] = rows["train"][:limit_train_rows]
    if limit_requests is not None:
        for split in ("validation", "test"):
            rows[split] = rows[split][:limit_requests]
    provenance = {"export_config": file_reference(config_path), "training_data": file_reference(paths["train"]),
                  "group_construction": "5 equal catalog-size popularity bins from unaugmented training events; item-ID tie break",
                  "evaluation_scope": "smoke_test" if limit_requests or limit_train_rows else "full_splits",
                  "limits": {"requests": limit_requests, "train_rows": limit_train_rows},
                  "seed": seed, "initial_checkpoint": verified.metadata["checkpoint"]}
    provenance["generation_beam_width"] = config["top_k_for_generation"]
    provenance["group_sources"] = {"brands": verified.metadata["provenance"]["group_mapping"],
                                  "popularity": file_reference(paths["train"])}
    destination.mkdir(parents=True)
    write_json(destination / "groups.json", groups)
    mapping = groups[group]
    model_class = original_module("grid_model").GRIDBaseModel
    branches = (("ordinary_control", "sprint_grid_encoder_adaptation") if mode == "sprint" else
                ("ordinary_control", f"ifairlrs_rw_grid_{group}_adaptation") if mode == "rw" else ("original",))
    plan = {"epochs": epochs, "seed": seed, "history_length": config["history_length"],
            "pre_batch_size": config.get("causal_pre_batch_size", 32), "batch_size": config["batch_size"],
            "augment": config.get("use_causal_augmentation", False)}
    if not plan["augment"]:
        plan["pre_batch_size"] = config["batch_size"]
    if mode in {"rw", "sprint"}:
        if mode == "rw":
            stats = rw_weights(replay_batches(rows["train"], **plan), mapping)
        plan["sha256"] = plan_digest(replay_batches(rows["train"], **plan))
        provenance["training_plan"] = plan.copy()
        provenance["training_settings"] = {"optimizer": "Adam", "learning_rate": config["learning_rate"],
                                           "weight_decay": config["weight_decay"], "scheduler": None,
                                           "objective": "sum SID-position CE, then mean over examples", "selection": "final_fixed_epoch"}
        if mode == "rw":
            write_json(destination / "rw_weights.json", {**stats, "plan": plan, "group": group,
                                                       "collator": file_reference(ROOT / "fairdiverse/recommendation/base_model/grid_collate.py")})
        else:
            provenance["sprint"] = sprint_settings or {}
            provenance["sprint_sources"] = {
                "adapter": file_reference(Path(__file__).with_name("sprint_grid.py")),
                "shared_helpers": file_reference(ROOT / "fairdiverse/recommendation/base_model/sasrec_sprint.py"),
                "collator": file_reference(ROOT / "fairdiverse/recommendation/base_model/grid_collate.py")}
            provenance["training_settings"]["objective"] += " + encoder SPRINT penalty (adaptation branch only)"
    write_json(destination / "run_config.json", {"mode": mode, "group": group, "seed": seed, "epochs": epochs,
                                               "alpha": alpha, "epsilon": epsilon, "device": device, "provenance": provenance})
    for branch in branches:
        seed_everything(seed)
        print(f"Loading starting checkpoint for {branch}", flush=True)
        model = model_class(config).to(device)
        checkpoint = Path(verified.metadata["checkpoint"]["path"])
        branch_dir = destination / branch
        branch_dir.mkdir()
        regularizer = None
        if mode in {"rw", "sprint"}:
            print(f"Training {branch} for {epochs} fixed epochs", flush=True)
            weights = {g: 1.0 for g in set(mapping.values())}
            if mode == "rw" and branch != "ordinary_control":
                weights = stats["weights"]
            if mode == "sprint" and branch != "ordinary_control":
                from .sprint_grid import GridEncoderSPRINT
                regularizer = GridEncoderSPRINT(model, **(sprint_settings or {}))
                write_json(branch_dir / "sprint_settings.json", regularizer.settings())
            training = train_branch(model, replay_batches(rows["train"], **{k: v for k, v in plan.items() if k != "sha256"}), mapping, weights,
                                    device=device, learning_rate=config["learning_rate"],
                                    weight_decay=config["weight_decay"], clip_norm=0.0, regularizer=regularizer)
            if training["training_plan_sha256"] != plan["sha256"]:
                raise RuntimeError("Training examples differed from the plan used to estimate RW weights")
            checkpoint = branch_dir / "checkpoint.ckpt"
            torch.save({"state_dict": model.grid.state_dict(), "initial_checkpoint": verified.metadata["checkpoint"],
                        "variant": branch, "seed": seed, "plan": plan}, checkpoint)
            write_json(branch_dir / "training.json", training)
            if regularizer is not None:
                torch.save(regularizer.state_dict(), branch_dir / "sprint_power_vectors.pt")
        for split in (("validation", "test") if mode in {"rr", "sprint"} else ("test",)):
            seed_everything(seed)
            artifact, histories = regenerate(model, rows[split], paths[split], split, groups, checkpoint,
                                             verified.metadata["sid_mapping"]["path"], seed, device, provenance)
            save(artifact, branch_dir / f"{split}_candidates.json")
            write_json(branch_dir / f"{split}_histories.json", histories)
            if mode == "rr" and split == "validation":
                frozen = fit_rr(artifact, histories, group, alpha=alpha, epsilon=epsilon)
                frozen["validation_artifact"] = file_reference(branch_dir / "validation_candidates.json")
                frozen["histories_source"] = file_reference(branch_dir / "validation_histories.json")
                write_json(branch_dir / "rr_correction.json", frozen)
            elif mode == "rr":
                write_json(branch_dir / "test_results.json", apply_rr(artifact, frozen, histories=histories))
            else:
                report = {"variant": branch, "metrics": evaluate(artifact, (5, 10, 20)),
                          "input_metadata": artifact.metadata}
                if mode == "sprint":
                    report["group_calibration"] = {name: fairness(artifact, histories, name, (5, 10, 20)) for name in groups}
                else:
                    report["ifairlrs_fairness"] = fairness(artifact, histories, group, (5, 10, 20))
                write_json(branch_dir / f"{split}_results.json", report)
        del regularizer
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    return {"output_dir": str(destination.resolve()), "branches": list(branches), "provenance": provenance}
