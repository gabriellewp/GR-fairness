"""Versioned, explicit candidate contract, using integer catalog item IDs."""

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path


def file_reference(path):
    """Fingerprint an input without copying or modifying it."""
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "sha256": digest.hexdigest()}


@dataclass
class Request:
    request_id: str
    candidate_ids: list[int]
    scores: list[float]
    targets: list[int]


@dataclass
class ExperimentInputs:
    metadata: dict
    item_groups: dict[str, dict[int, str]]
    requests: list[Request]
    schema_version: int = 1

    def validate(self):
        if self.schema_version != 1:
            raise ValueError("Unsupported schema_version")
        required = {"checkpoint", "split", "sid_mapping", "seed", "generation",
                    "score_convention", "candidate_membership", "short_list_policy"}
        if required - self.metadata.keys():
            raise ValueError(f"Missing metadata: {sorted(required - self.metadata.keys())}")
        for key in ("checkpoint", "sid_mapping"):
            _reference(self.metadata[key], key)
        split = self.metadata["split"]
        if not isinstance(split, dict) or not split.get("name"):
            raise ValueError("split needs a name and a fingerprinted source")
        _reference(split.get("source"), "split.source")
        if type(self.metadata["seed"]) is not int:
            raise ValueError("seed must be an integer")
        if not isinstance(self.metadata["generation"], dict) or not self.metadata["generation"]:
            raise ValueError("generation must record the actual generation settings")
        if self.metadata["score_convention"] not in {
            "probability", "log_probability", "negative_log_probability", "utility"
        }:
            raise ValueError("Unknown score_convention")
        if not isinstance(self.metadata["candidate_membership"], str) or not self.metadata["candidate_membership"]:
            raise ValueError("candidate_membership must describe eligibility")
        if self.metadata["short_list_policy"] != "return_available":
            raise ValueError("Only return_available is supported: never pad with catalog items")
        if not self.item_groups or not self.requests:
            raise ValueError("Provide group mappings and at least one request")
        for name, mapping in self.item_groups.items():
            if not isinstance(name, str) or not name or not mapping:
                raise ValueError("Group mappings need nonempty names and contents")
            for item, group in mapping.items():
                _item(item)
                if not isinstance(group, str) or not group:
                    raise ValueError("Group IDs must be nonempty strings")
        seen = set()
        for request in self.requests:
            if not isinstance(request.request_id, str) or not request.request_id or request.request_id in seen:
                raise ValueError("request_id must be a unique nonempty string")
            seen.add(request.request_id)
            if len(request.candidate_ids) != len(request.scores):
                raise ValueError("Candidate IDs and scores must have equal lengths")
            for ids in (request.candidate_ids, request.targets):
                for item in ids:
                    _item(item)
                if len(set(ids)) != len(ids):
                    raise ValueError("Duplicate candidate or target IDs")
            if not request.targets:
                raise ValueError("Each held-out request needs at least one target")
            for mapping in self.item_groups.values():
                if (any(item not in mapping for item in request.candidate_ids)
                        or any(item not in mapping for item in request.targets)):
                    raise ValueError("Missing group mapping for candidate or target")
            for score in request.scores:
                if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
                    raise ValueError("Scores must be finite numbers")
                convention = self.metadata["score_convention"]
                if convention == "probability" and not 0 <= score <= 1:
                    raise ValueError("Probability outside [0, 1]")
                if convention == "log_probability" and score > 0:
                    raise ValueError("Log probability must be nonpositive")
                if convention == "negative_log_probability" and score < 0:
                    raise ValueError("Negative log probability must be nonnegative")
        return self

    def ranking(self, request):
        # Stable ties preserve explicitly recorded candidate order.
        ascending = self.metadata["score_convention"] == "negative_log_probability"
        order = sorted(range(len(request.scores)), key=lambda i: request.scores[i], reverse=not ascending)
        return [request.candidate_ids[i] for i in order]


def _item(item):
    if type(item) is not int or item < 0:
        raise ValueError("Item IDs must be nonnegative integers")


def _reference(value, name):
    if not isinstance(value, dict) or not isinstance(value.get("path"), str) or not value["path"]:
        raise ValueError(f"{name} needs path and sha256")
    sha = value.get("sha256", "")
    if not isinstance(sha, str) or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
        raise ValueError(f"{name} needs a SHA-256 fingerprint")


def save(inputs, path):
    """Create a new artifact; never overwrite an existing experiment."""
    inputs.validate()
    payload = json.dumps(asdict(inputs), allow_nan=False, indent=2)
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(payload + "\n")


def load(path):
    with Path(path).open(encoding="utf-8") as stream:
        data = json.load(stream)
    data["requests"] = [Request(**record) for record in data["requests"]]
    data["item_groups"] = {
        name: {int(item): group for item, group in mapping.items()}
        for name, mapping in data["item_groups"].items()
    }
    return ExperimentInputs(**data).validate()
