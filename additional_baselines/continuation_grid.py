"""Validation-selected Beauty GRID continuation, isolated from native GRID."""

from collections import OrderedDict
import hashlib
import json
import math
from pathlib import Path
import sys

from .adapters import GenerationRequest, generate_grid
from .candidates import file_reference, load, save
from .evaluation import evaluate
from .ifairlrs import history_shares, popularity_groups, rw_weights, weighted_loss
from .continuation_selection import select_checkpoint, selection_policy
from .ifairlrs_grid import original_module, read_rows, seed_everything, training_popularity, write_json
from .prepare_beauty import DEFAULT_DATA, DEFAULT_EXPORT, ROOT, load_beauty


def reconstruct_users(rows):
    """Recover each native full-user row from chronological endpoint examples."""
    users = OrderedDict()
    for user, history, target in rows:
        if user not in users:
            if not history:
                raise ValueError(f"User {user} has no initial training history")
            users[user] = list(history) + [target]
        else:
            sequence = users[user]
            if history != sequence[-len(history):]:
                raise ValueError(f"Nonchronological or inconsistent training prefixes for user {user}")
            sequence.append(target)
    return list(users.items())


def native_batches(users, *, updates, seed, users_per_batch, cap, accumulation, history_length):
    """Replay native full-user causal windows and its per-microbatch random cap."""
    import torch
    if not users or min(len(items) for _, items in users) < 2:
        raise ValueError("Every user needs at least two training items")
    rng = torch.Generator(device="cpu").manual_seed(seed)
    order, cursor = [], 0
    for _ in range(updates * accumulation):
        selected = []
        while len(selected) < users_per_batch:
            if cursor == len(order):
                order = torch.randperm(len(users), generator=rng).tolist()
                cursor = 0
            count = min(users_per_batch - len(selected), len(order) - cursor)
            selected.extend(users[i] for i in order[cursor:cursor + count])
            cursor += count
        windows = []
        for user, items in selected:
            for end in range(2, len(items) + 1):
                for start in range(end - 1):
                    windows.append((user, items[start:end - 1], items[end - 1]))
        if len(windows) > cap:
            indices = set(torch.randperm(len(windows), generator=rng)[:cap].tolist())
            windows = [window for index, window in enumerate(windows) if index in indices]
        yield tuple(torch.tensor(values, dtype=torch.long) for values in (
            [u for u, _, _ in windows],
            [h[-history_length:] + [-1] * (history_length - len(h[-history_length:])) for _, h, _ in windows],
            [t for _, _, t in windows],
            [0] * len(windows),
        ))


def batch_digest(batches):
    digest = hashlib.sha256()
    for batch in batches:
        for tensor in batch:
            digest.update(str(tuple(tensor.shape)).encode())
            digest.update(tensor.contiguous().numpy().tobytes())
    return digest.hexdigest()


def native_schedule(config_path):
    import yaml
    config = yaml.safe_load(Path(config_path).read_text())
    data = config["data_loading"]["train_dataloader_config"]["dataloader"]
    target = data["collate_fn"]["_target_"]
    if not target.endswith(".collate_with_sid_causal_duplicate") or not data["drop_last"]:
        raise ValueError("Unsupported native GRID training collator or batch policy")
    if config["optim"]["scheduler"] is not None:
        raise ValueError("Native scheduler must be absent for this continuation")
    if config["sequence_length"] % config["num_hierarchies"]:
        raise ValueError("Native GRID sequence length must contain whole semantic IDs")
    model_history_length = config["sequence_length"] // config["num_hierarchies"]
    return {"users_per_batch": data["batch_size_per_device"], "cap": data["collate_fn"]["max_batch_size"],
            "accumulation": config["trainer"]["accumulate_grad_batches"],
            "history_length": model_history_length - 1, "model_history_length": model_history_length,
            "native_learning_rate": config["optim"]["optimizer"]["lr"],
            "weight_decay": config["optim"]["optimizer"]["weight_decay"],
            "native_collator": target, "native_checkpoint_metric": config["callbacks"]["model_checkpoint"]["monitor"]}


def resume_optimizer(model, checkpoint_path, *, learning_rate, weight_decay, policy):
    import torch
    if policy not in {"require", "auto", "fresh"}:
        raise ValueError("optimizer policy must be require, auto, or fresh")
    optimizer = torch.optim.Adam(model.grid.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # Lightning serialized config objects need GRID/src available during unpickling.
    grid_path = str(ROOT / "GRID")
    if grid_path not in sys.path:
        sys.path.insert(0, grid_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source = checkpoint.get("optimizer_states", [])
    audit = {"policy": policy, "source_global_step": checkpoint.get("global_step"),
             "source_optimizer_present": bool(source), "resumed": False, "reason": None,
             "learning_rate": learning_rate, "weight_decay": weight_decay}
    if policy == "fresh":
        audit["reason"] = "explicit fresh Adam request"
        return optimizer, audit
    try:
        if len(source) != 1 or len(source[0]["param_groups"]) != 1:
            raise ValueError("Expected one native Adam optimizer and parameter group")
        native = source[0]
        parameters = list(model.grid.parameters())
        ids = native["param_groups"][0]["params"]
        if len(ids) != len(parameters) or set(ids) != set(native["state"]):
            raise ValueError("Native optimizer parameter IDs do not cover the adapter model")
        for param_id, parameter in zip(ids, parameters):
            state = native["state"][param_id]
            for key in ("exp_avg", "exp_avg_sq"):
                if key not in state or state[key].shape != parameter.shape:
                    raise ValueError(f"Native Adam {key} shape differs at parameter {param_id}")
        optimizer.load_state_dict(native)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
            group["weight_decay"] = weight_decay
        audit["resumed"] = True
        audit["source_adam_step"] = sorted({int(s["step"]) for s in native["state"].values()})
    except (KeyError, ValueError, RuntimeError) as error:
        if policy == "require":
            raise RuntimeError(f"Cannot resume native Adam state: {error}") from error
        audit["reason"] = str(error)
        optimizer = torch.optim.Adam(model.grid.parameters(), lr=learning_rate, weight_decay=weight_decay)
    return optimizer, audit


def selected_step(trace, metric="recall@5"):
    if metric != "recall@5" or not trace or trace[0]["update"] != 0:
        raise ValueError("Selection requires zero-step validation and declared recall@5")
    return max(trace, key=lambda row: (row["metrics"]["at_k"]["5"]["recall"], -row["update"]))["update"]


def validate_control(control, identity, *, require_control_improvement=True):
    """Keep matched-run checks when the improvement prerequisite is waived."""
    if control.get("mode") != "ordinary" or control.get("identity") != identity:
        raise ValueError("Ordinary control must have the identical run identity")
    update = control.get("selected_update")
    if type(update) is not int or not 0 <= update <= identity["max_updates"]:
        raise ValueError("Ordinary control has an invalid selected update")
    if require_control_improvement and update == 0:
        raise ValueError("Ordinary control must improve on validation under the identical run identity")


def run_continuation(*, mode, output_dir, control_report=None, group="popularity", device="cuda", seed=42,
                     max_updates=32, validation_interval=8, learning_rate=0.001, optimizer_state="require",
                     export_dir=DEFAULT_EXPORT, data_dir=DEFAULT_DATA, validation_requests=None,
                     train_users=None, generation_beam_width=None, sprint_settings=None, tokenizer="rkmeans",
                     dataset="amazon-beauty", require_control_improvement=True,
                     rw_target="history", rw_strength=1.0, selection_objective="recall",
                     max_relative_recall_loss=0.02, evaluate_test=True):
    import torch
    import yaml
    from .sprint_grid import GridEncoderSPRINT, resolve_settings
    if mode not in {"ordinary", "rw", "sprint"} or group not in {"popularity", "brand"}:
        raise ValueError("Invalid continuation mode/group")
    if any(type(x) is not int or x < 1 for x in (max_updates, validation_interval, seed)):
        raise ValueError("Updates, interval and seed must be positive integers")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Learning rate must be finite and positive")
    for name, value in (("validation_requests", validation_requests), ("train_users", train_users),
                        ("generation_beam_width", generation_beam_width)):
        if value is not None and (type(value) is not int or value < 1):
            raise ValueError(f"{name} must be a positive integer")
    if mode != "ordinary" and not control_report:
        raise ValueError("RW and SPRINT require a matched ordinary control report")
    if rw_target not in {"history", "uniform"} or not math.isfinite(rw_strength) or not 0 <= rw_strength <= 1:
        raise ValueError("Invalid RW target distribution or strength")
    if mode != "rw" and (rw_target != "history" or rw_strength != 1):
        raise ValueError("RW weighting options require RW mode")
    if mode == "ordinary" and selection_objective != "recall":
        raise ValueError("Ordinary controls must maximize validation Recall@5")
    # Validate the declared policy before any model work.
    selection_policy("recall", max_relative_recall_loss=max_relative_recall_loss)
    if selection_objective not in {"recall", "popularity", "brand", "both"}:
        raise ValueError("Unknown validation selection objective")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    verified, _, _ = load_beauty(export_dir=export_dir, data_dir=data_dir, topk=(1,),
                               check_dense_memory=False, tokenizer=tokenizer, dataset=dataset)
    export_dir, data_dir = Path(export_dir), Path(data_dir)
    config = yaml.safe_load((export_dir / "config.yaml").read_text())
    if generation_beam_width is not None:
        config["top_k_for_generation"] = generation_beam_width
    checkpoint_path = Path(verified.metadata["checkpoint"]["path"])
    schedule = native_schedule(checkpoint_path.parent.parent / ".hydra/config.yaml")
    catalog = set(verified.item_groups["brand"])
    paths = {name: data_dir / f"{dataset}.{suffix}" for name, suffix in
             (("train", "train"), ("validation", "valid.ranking"), ("test", "test.ranking"))}
    train_rows = read_rows(paths["train"], training=True, history_length=schedule["model_history_length"], catalog=catalog)
    users = reconstruct_users(train_rows)
    if train_users is not None:
        users = users[:train_users]
    mapping = {"popularity": popularity_groups(training_popularity(train_rows), catalog),
               "brand": verified.item_groups["brand"]}
    rows = {split: read_rows(paths[split], training=False, history_length=schedule["model_history_length"], catalog=catalog)
            for split in (("validation", "test") if evaluate_test else ("validation",))}
    if validation_requests is not None:
        rows["validation"] = rows["validation"][:validation_requests]
    plan = {k: schedule[k] for k in ("users_per_batch", "cap", "accumulation", "history_length")}
    plan.update(updates=max_updates, seed=seed)
    digest = batch_digest(native_batches(users, **plan))
    identity = {"checkpoint": verified.metadata["checkpoint"], "sid_mapping": verified.metadata["sid_mapping"],
                "seed": seed, "plan_sha256": digest, "max_updates": max_updates,
                "validation_interval": validation_interval, "validation_requests": validation_requests,
                "learning_rate": learning_rate, "optimizer_state": optimizer_state,
                "generation_beam_width": config["top_k_for_generation"], "train_users": train_users}
    control = None
    if control_report:
        control = json.loads(Path(control_report).read_text())
        validate_control(control, identity, require_control_improvement=require_control_improvement)
    histories = {f"validation:{i}:user:{user}": history
                 for i, (user, history, _) in enumerate(rows["validation"])}
    gh = history_shares(histories, histories, mapping["popularity"])
    policy = selection_policy(selection_objective, max_relative_recall_loss=max_relative_recall_loss,
                              control_metrics=control["selected_validation_metrics"] if control else None,
                              history_shares=gh)
    stats = rw_weights(native_batches(users, **plan), mapping[group], target_distribution=rw_target,
                       strength=rw_strength) if mode == "rw" else None
    seed_everything(seed)
    model = original_module("grid_model").GRIDBaseModel(config).to(device)
    optimizer, audit = resume_optimizer(model, checkpoint_path, learning_rate=learning_rate,
                                         weight_decay=schedule["weight_decay"], policy=optimizer_state)
    if control_report:
        if control["optimizer_audit"]["resumed"] != audit["resumed"]:
            raise ValueError("Ordinary and fairness branches differ in optimizer resume behavior")
    regularizer = GridEncoderSPRINT(model, **resolve_settings(sprint_settings or {})) if mode == "sprint" else None
    variant = (stats["variant"] if stats else regularizer.settings()["variant"] if regularizer else "ordinary")
    table = model.tokenizer.sid_table.detach().cpu().tolist()
    sids = {i: sid for i, sid in enumerate(table)}
    eligible_ids = list(sids)
    checkpoint_reference = file_reference(checkpoint_path)
    sid_reference = verified.metadata["sid_mapping"]

    def validate(split, checkpoint):
        metadata = {"checkpoint": checkpoint_reference if checkpoint == checkpoint_path else file_reference(checkpoint),
                    "sid_mapping": sid_reference,
                    "split": {"name": split, "source": file_reference(paths[split])}, "seed": seed,
                    "generation": {"beam_width": config["top_k_for_generation"],
                                   "sequence_length": config["sequence_length"], "num_hierarchies": config["num_hierarchies"]},
                    "score_convention": "probability", "short_list_policy": "return_available"}
        def requests():
            for i, (user, history, target) in enumerate(rows[split]):
                if i and i % 1000 == 0:
                    print(f"{split}: generated {i}/{len(rows[split])} requests", flush=True)
                yield GenerationRequest(f"{split}:{i}:user:{user}", history, [target], eligible_ids, user)
        cuda_devices = [torch.cuda.current_device()] if device.startswith("cuda") else []
        with torch.random.fork_rng(devices=cuda_devices):
            artifact = generate_grid(model, requests(), sids, mapping, metadata, device=device)
        return artifact, evaluate(artifact, (5, 10, 20))

    destination.mkdir(parents=True)
    write_json(destination / "run_config.json", {"mode": mode, "identity": identity, "native_schedule": schedule,
               "replay_policy": "full-user causal windows, deterministic user shuffle; native streaming order unavailable",
               "optimizer_audit": audit, "group": group, "sprint_settings": regularizer.settings() if regularizer else None,
               "rw_statistics": stats, "control_report": str(control_report) if control_report else None,
               "control_improvement_required": require_control_improvement,
               "control_selected_update": control["selected_update"] if control_report else None,
               "selection": policy, "evaluate_test": evaluate_test,
               "rw_target": rw_target, "rw_strength": rw_strength})
    # Freeze everything needed to evaluate a winner later, without retraining.
    context_path = destination / "evaluation_context.json"
    write_json(context_path, {"config": config, "item_groups": mapping,
               "test_split": file_reference(paths["test"]), "history_length": schedule["model_history_length"],
               "sid_mapping": sid_reference, "dataset": dataset, "tokenizer": tokenizer, "seed": seed})
    trace = []
    artifact, metrics = validate("validation", checkpoint_path)
    trace.append({"update": 0, "metrics": metrics})
    # The earlier RR export uses the same starting GRID checkpoint and split.
    # Compare its saved rankings when available; score conventions can differ.
    reference_path = ROOT / "additional_baselines/artifacts/beauty_ifairlrs_rr_popularity/original/validation_candidates.json" if dataset == "amazon-beauty" else None
    zero_reference = None
    if validation_requests is None and reference_path is not None and reference_path.exists():
        reference = load(reference_path)
        if reference.metadata.get("checkpoint") == checkpoint_reference and len(reference.requests) == len(artifact.requests):
            reference_metrics = evaluate(reference, (5, 10, 20))
            aligned = all(a.request_id == b.request_id and a.targets == b.targets
                          for a, b in zip(artifact.requests, reference.requests))
            if not aligned:
                raise ValueError("Saved zero-step reference has different request/target alignment")
            zero_reference = {"source": file_reference(reference_path),
                              "candidate_membership_equal": all(set(a.candidate_ids) == set(b.candidate_ids)
                                                                for a, b in zip(artifact.requests, reference.requests)),
                              "recall_at_5_difference": metrics["at_k"]["5"]["recall"] - reference_metrics["at_k"]["5"]["recall"],
                              "coverage_difference": metrics["candidate_target_recall"] - reference_metrics["candidate_target_recall"],
                              "score_conventions": [artifact.metadata["score_convention"], reference.metadata["score_convention"]]}
            print(f"zero-step reference: {zero_reference}", flush=True)
        del reference
    best = select_checkpoint(trace, policy)
    best_update = best["update"] if best else None
    # Step zero is a real selection candidate. Retain its weights so a losing
    # continuation cannot be tested and mislabeled as the source checkpoint.
    best_state = ({name: tensor.detach().cpu().clone() for name, tensor in model.grid.state_dict().items()}
                  if best else None)
    print(f"validation update=0 recall@5={metrics['at_k']['5']['recall']:.6f} coverage={metrics['candidate_target_recall']:.6f}", flush=True)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    count = 0
    training_trace = []
    total_loss_sum, penalty_sum = 0.0, 0.0
    for micro_index, (users_batch, histories, targets, _) in enumerate(native_batches(users, **plan), 1):
        interaction = {"user_ids": users_batch.to(device), "history_ids": histories.to(device), "item_ids": targets.to(device)}
        if mode == "sprint":
            loss = regularizer.objective(interaction)
        else:
            losses = model.compute_loss(interaction)
            if mode == "rw":
                weights = losses.new_tensor([stats["weights"][mapping[group][item]] for item in targets.tolist()])
                loss = weighted_loss(losses, weights)
            else:
                loss = losses.mean()
        if not torch.isfinite(loss):
            raise ValueError("Continuation training objective became nonfinite")
        total_loss_sum += float(loss.detach())
        if mode == "sprint" and regularizer.enabled:
            penalty_sum += sum(regularizer.last_terms[name] * getattr(regularizer, name + "_weight")
                               for name in ("embedding", "attention", "ffn"))
        (loss / schedule["accumulation"]).backward()
        count += len(targets)
        if micro_index % schedule["accumulation"]:
            continue
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        update = micro_index // schedule["accumulation"]
        training_trace.append({"update": update,
            "mean_total_loss": total_loss_sum / schedule["accumulation"],
            "mean_regularization_loss": penalty_sum / schedule["accumulation"],
            "mean_recommendation_loss": (total_loss_sum - penalty_sum) / schedule["accumulation"]})
        total_loss_sum, penalty_sum = 0.0, 0.0
        print(f"training update={update}/{max_updates} examples={count} last_microbatch_loss={float(loss.detach()):.6f}", flush=True)
        if update % validation_interval and update != max_updates:
            continue
        artifact, metrics = validate("validation", checkpoint_path)
        trace.append({"update": update, "metrics": metrics})
        score = metrics["at_k"]["5"]["recall"]
        print(f"validation update={update} recall@5={score:.6f} coverage={metrics['candidate_target_recall']:.6f}", flush=True)
        best = select_checkpoint(trace, policy)
        if best is not None and best["update"] != best_update:
            best_update = best["update"]
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.grid.state_dict().items()}
        model.train()
    best = select_checkpoint(trace, policy)
    if best_update != (best["update"] if best else None):
        raise RuntimeError("Checkpoint selection disagrees with validation trace")
    write_json(destination / "validation_trace.json", trace)
    write_json(destination / "training_trace.json", training_trace)
    if best_state is None:
        report = {"mode": mode, "variant": variant, "identity": identity, "optimizer_audit": audit,
                  "selected_update": None, "selected_checkpoint": None,
                  "selected_validation_metrics": None, "selection": policy,
                  "status": "no_feasible_fairness_checkpoint", "training_examples": count,
                  "evaluation_context": file_reference(context_path), "test_evaluated": False}
        write_json(destination / "continuation.json", report)
        return report
    selected_checkpoint = checkpoint_path
    model.grid.load_state_dict(best_state)
    if best_update > 0:
        selected_checkpoint = destination / "selected_checkpoint.ckpt"
        torch.save({"state_dict": best_state, "initial_checkpoint": verified.metadata["checkpoint"],
                    "selected_update": best_update, "mode": mode, "identity": identity}, selected_checkpoint)
    report = {"mode": mode, "variant": variant, "identity": identity, "optimizer_audit": audit, "selected_update": best_update,
              "selected_weights_restored": True,
              "zero_step_reference": zero_reference,
              "selection_metric": f"validation/{selection_objective}@5", "selection_policy": policy["rule"],
              "selection": policy, "evaluation_context": file_reference(context_path),
              "selected_validation_metrics": next(row["metrics"] for row in trace if row["update"] == best_update),
              "training_examples": count, "selected_checkpoint": file_reference(selected_checkpoint),
              "status": ("fairness_improved" if selection_objective != "recall" else
                         "improved" if best_update else "no_continuation_improvement"),
              "test_evaluated": False}
    if not evaluate_test or (best_update == 0 and mode == "ordinary"):
        write_json(destination / "continuation.json", report)
        return report
    test_artifact, test_metrics = validate("test", selected_checkpoint)
    save(test_artifact, destination / "test_candidates.json")
    write_json(destination / "test_results.json", {"mode": mode, "variant": variant, "selected_update": best_update,
                                                     "metrics": test_metrics, "input_metadata": test_artifact.metadata})
    report["test_evaluated"] = True
    write_json(destination / "continuation.json", report)
    return report


def evaluate_selected_continuation(run_dir, output_dir, *, device="cuda"):
    """Evaluate an already frozen checkpoint; never train or select on test."""
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    folder = Path(run_dir).resolve()
    report = json.loads((folder / "continuation.json").read_text())
    if report["selected_checkpoint"] is None:
        raise ValueError("Run has no feasible selected checkpoint")
    context_ref = report["evaluation_context"]
    if file_reference(context_ref["path"]) != context_ref:
        raise ValueError("Frozen evaluation context changed")
    context = json.loads(Path(context_ref["path"]).read_text())
    for ref in (report["identity"]["checkpoint"], report["selected_checkpoint"], context["test_split"], context["sid_mapping"]):
        if file_reference(ref["path"]) != ref:
            raise ValueError("Frozen evaluation input changed")
    config = dict(context["config"])
    config["grid_checkpoint_path"] = report["identity"]["checkpoint"]["path"]
    config["sid_path"] = context["sid_mapping"]["path"]
    seed_everything(context["seed"])
    model = original_module("grid_model").GRIDBaseModel(config).to(device)
    if report["selected_checkpoint"] != report["identity"]["checkpoint"]:
        import torch
        saved = torch.load(report["selected_checkpoint"]["path"], map_location="cpu", weights_only=False)
        model.grid.load_state_dict(saved.get("state_dict", saved), strict=True)
    mapping = {name: {int(i): g for i, g in groups.items()}
               for name, groups in context["item_groups"].items()}
    table = model.tokenizer.sid_table.detach().cpu().tolist()
    sids = dict(enumerate(table))
    rows = read_rows(context["test_split"]["path"], training=False,
                     history_length=context["history_length"], catalog=set(sids))
    metadata = {"checkpoint": report["selected_checkpoint"], "sid_mapping": context["sid_mapping"],
                "split": {"name": "test", "source": context["test_split"]}, "seed": context["seed"],
                "generation": {"beam_width": config["top_k_for_generation"],
                               "sequence_length": config["sequence_length"], "num_hierarchies": config["num_hierarchies"]},
                "score_convention": "probability", "short_list_policy": "return_available"}
    eligible_ids = list(sids)
    requests = (GenerationRequest(f"test:{i}:user:{user}", history, [target], eligible_ids, user)
                for i, (user, history, target) in enumerate(rows))
    artifact = generate_grid(model, requests, sids, mapping, metadata, device=device)
    result = {"mode": report["mode"], "variant": report["variant"], "selected_update": report["selected_update"],
              "metrics": evaluate(artifact, (5, 10, 20)), "input_metadata": artifact.metadata,
              "frozen_selection": file_reference(folder / "continuation.json")}
    destination.mkdir(parents=True)
    save(artifact, destination / "test_candidates.json")
    write_json(destination / "test_results.json", result)
    return result
