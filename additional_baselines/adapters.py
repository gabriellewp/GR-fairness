"""Strict legacy import and fresh generation through an existing GRID model."""

from copy import deepcopy
from dataclasses import dataclass
import math

from .candidates import ExperimentInputs, Request, file_reference


def import_legacy(scores_path, targets_path, request_ids, item_groups, metadata,
                  *, export_format, candidate_mask_path=None):
    """Import known probability+1 exports without guessing eligibility.

    compact_grid: stored CSR entries are the export's eligible candidates.
    explicit_mask: a separate binary sparse matrix defines eligibility.
    Dense probability+1 scores alone cannot recover generated membership.
    """
    from scipy.sparse import load_npz

    if export_format not in {"compact_grid", "explicit_mask"}:
        raise ValueError("Unknown export format; regenerate or provide an explicit candidate mask")
    if export_format == "explicit_mask" and candidate_mask_path is None:
        raise ValueError("Dense exports require an explicit candidate mask; regenerate if unavailable")
    if export_format == "compact_grid" and candidate_mask_path is not None:
        raise ValueError("Use explicit_mask format when supplying a candidate mask")
    def read_sparse(path):
        matrix = load_npz(path)
        stored_count = matrix.nnz
        matrix = matrix.tocsr()
        matrix.sum_duplicates()
        if matrix.nnz != stored_count:
            raise ValueError("Duplicate sparse entries are ambiguous; regenerate")
        return matrix

    scores = read_sparse(scores_path)
    targets = read_sparse(targets_path)
    if scores.shape != targets.shape or len(request_ids) != scores.shape[0]:
        raise ValueError("Scores, targets, and explicit request ID order must align")
    if any(value != 1 for value in targets.data):
        raise ValueError("Ground truth must contain binary positive entries only")
    if candidate_mask_path is not None:
        mask = read_sparse(candidate_mask_path)
        if mask.shape != scores.shape or any(v != 1 for v in mask.data):
            raise ValueError("Candidate mask must be binary sparse data aligned with scores")
    else:
        mask = scores
    meta = deepcopy(metadata)
    if meta.get("score_convention") != "probability":
        raise ValueError("Legacy adapter supports probability+1 exports only")
    meta["candidate_membership"] = (
        "legacy_compact_stored_positive_probability_candidates"
        if export_format == "compact_grid" else "explicit_candidate_mask"
    )
    meta["legacy_import"] = {
        "scores": file_reference(scores_path), "targets": file_reference(targets_path),
        "export_format": export_format, "stored_score_offset": 1.0,
        "precision_warning": "Float32 addition may have lost small probabilities; native log probabilities are preferable for distance corrections.",
    }
    if candidate_mask_path is not None:
        meta["legacy_import"]["candidate_mask"] = file_reference(candidate_mask_path)
    requests = []
    for row, request_id in enumerate(request_ids):
        score_row = scores.getrow(row)
        stored = dict(zip(score_row.indices.tolist(), score_row.data.tolist()))
        ids = mask.getrow(row).indices.tolist()
        if set(ids) - stored.keys():
            raise ValueError("Eligible candidate has no explicitly stored score")
        probabilities = [float(stored[item]) - 1.0 for item in ids]
        if any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities):
            raise ValueError("Scores do not match the declared probability+1 format")
        if export_format == "compact_grid" and any(p == 0 for p in probabilities):
            raise ValueError("Probability lost during shifted export; regenerate candidates")
        requests.append(Request(request_id, ids, probabilities, targets.getrow(row).indices.tolist()))
    return ExperimentInputs(meta, item_groups, requests).validate()


@dataclass
class GenerationRequest:
    request_id: str
    history_ids: list[int]
    targets: list[int]
    eligible_ids: list[int]
    user_id: int | None = None


def generate_grid(model, requests, item_to_sid, item_groups, metadata, *, device="cpu"):
    """Generate a fresh artifact from a loaded GRIDBaseModel, without retraining.

    Explicit SID membership preserves generated candidates even if probability
    underflows to zero. Targets are never supplied to generation. The caller
    provides the exact checkpoint SID mapping and eligible evaluation catalog.
    """
    import torch

    meta = deepcopy(metadata)
    if meta.get("score_convention") != "probability":
        raise ValueError("GRID.generate returns probabilities; declare probability scores")
    sid_to_item = {}
    for item, sid in item_to_sid.items():
        if type(item) is not int or item < 0 or not sid or any(type(token) is not int or token < 0 for token in sid):
            raise ValueError("SID mapping requires nonnegative integer item IDs and token lists")
        key = tuple(sid)
        if key in sid_to_item:
            raise ValueError("Ambiguous SID mapping; disambiguate collisions before generation")
        sid_to_item[key] = item
    if hasattr(model, "tokenizer"):
        actual_sids = model.tokenizer.sid_table.detach().cpu().tolist()
        if any(item >= len(actual_sids) or list(sid) != actual_sids[item] for item, sid in item_to_sid.items()):
            raise ValueError("Supplied SID mapping differs from the loaded GRID tokenizer")
    for setting, attribute in (("beam_width", "top_k_for_generation"), ("sequence_length", "sequence_length"), ("num_hierarchies", "num_hierarchies")):
        if hasattr(model, attribute):
            actual = getattr(model, attribute)
            if setting in meta["generation"] and meta["generation"][setting] != actual:
                raise ValueError(f"Generation metadata disagrees with loaded model: {setting}")
            meta["generation"][setting] = actual
    meta["candidate_membership"] = "generated_sid_exact_match_intersect_explicit_eligible_ids"
    meta["generation_adapter"] = {
        "name": "GRIDBaseModel.generate", "duplicate_policy": "max_probability_stable_first_order",
        "unknown_sid_policy": "drop_and_count", "unknown_sid_count": 0,
        "ineligible_count": 0,
    }
    output = []
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for request in requests:
                eligible = set(request.eligible_ids)
                if len(eligible) != len(request.eligible_ids) or eligible - item_to_sid.keys():
                    raise ValueError("Eligible IDs must be unique and present in SID mapping")
                history = torch.tensor([request.history_ids], dtype=torch.long, device=device)
                users = None if request.user_id is None else torch.tensor([request.user_id], dtype=torch.long, device=device)
                sids, probabilities = model.generate(history, users)
                if sids.ndim != 3 or probabilities.ndim != 2 or sids.shape[:2] != probabilities.shape or sids.shape[0] != 1:
                    raise ValueError("Unexpected GRID generation output shapes")
                candidates = {}
                for sid, probability in zip(sids[0].tolist(), probabilities[0].tolist()):
                    if not math.isfinite(probability) or not 0 <= probability <= 1:
                        raise ValueError("GRID returned an invalid probability")
                    item = sid_to_item.get(tuple(sid))
                    if item is None:
                        meta["generation_adapter"]["unknown_sid_count"] += 1
                    elif item not in eligible:
                        meta["generation_adapter"]["ineligible_count"] += 1
                    else:
                        candidates[item] = max(candidates.get(item, 0), probability)
                output.append(Request(request.request_id, list(candidates), list(candidates.values()), list(request.targets)))
    finally:
        model.train(was_training)
    return ExperimentInputs(meta, item_groups, output).validate()
