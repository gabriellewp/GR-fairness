"""Activity-group evaluation, independent of model and candidate generation.

Run ``python -m fairdiverse.recommendation.user_fairness --help`` for cached data.
"""

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def fingerprint(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path.resolve()), "sha256": digest.hexdigest()}


def write_activity(counts, output_dir, provenance):
    """Freeze approximate tertiles of positive counts, without splitting ties.

    Select distinct count boundaries nearest the 1/3 and 2/3 cumulative
    populations. Collapsed boundaries yield fewer groups; zero is cold-start.
    Counts must describe original positive training events, not expanded rows.
    """
    normalized = {}
    for user, count in counts.items():
        if isinstance(count, bool) or int(count) != count or count < 0:
            raise ValueError("Activity counts must be nonnegative integers")
        if str(user) in normalized:
            raise ValueError("Duplicate normalized user ID")
        normalized[str(user)] = int(count)
    if not normalized:
        raise ValueError("No users supplied")
    frequencies = Counter(c for c in normalized.values() if c > 0)
    values = sorted(frequencies)
    cumulative = np.cumsum([frequencies[v] for v in values])
    boundaries = sorted({
        values[min(range(len(values) - 1), key=lambda i: abs(cumulative[i] - cumulative[-1] * q))]
        for q in (1 / 3, 2 / 3)
    }) if len(values) > 1 else []
    labels = {0: ["warm"], 1: ["low", "high"], 2: ["low", "medium", "high"]}[len(boundaries)]
    rows = []
    for user, count in sorted(normalized.items()):
        group = "cold_start" if count == 0 else labels[sum(count > b for b in boundaries)]
        rows.append({"user_id": user, "train_interaction_count": count, "activity_group": group})
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "user_activity.csv"
    _write_csv(path, rows)
    metadata = {
        "schema_version": 1, "definition": "original positive training interactions",
        "grouping": "nearest cumulative tertiles; ties kept together; zero separate",
        "upper_boundaries": boundaries, "group_counts": dict(Counter(r["activity_group"] for r in rows)),
        "provenance": provenance, "mapping": {**fingerprint(path), "path": path.name},
    }
    (output_dir / "user_activity_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return rows


def load_activity(path):
    metadata_path = Path(path).with_name("user_activity_metadata.json")
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata["mapping"]["sha256"] != fingerprint(path)["sha256"]:
            raise ValueError("Activity mapping does not match its metadata fingerprint")
    with Path(path).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    mapping = {}
    for row in rows:
        user = row["user_id"]
        count = int(row["train_interaction_count"])
        group = row["activity_group"]
        if not user or user in mapping or count < 0 or group not in {"low", "medium", "high", "warm", "cold_start"}:
            raise ValueError("Invalid or duplicate activity mapping row")
        if (count == 0) != (group == "cold_start"):
            raise ValueError("Zero-activity users must be in cold_start")
        mapping[user] = row
    if not mapping:
        raise ValueError("Empty activity mapping")
    return mapping


def ranking_metrics(ranking, targets, cutoffs):
    """Binary relevance at actual returned positions; absent targets are misses."""
    ranking, targets = list(ranking), set(targets)
    if not targets:
        raise ValueError("Evaluation requires at least one target")
    if len(ranking) != len(set(ranking)):
        raise ValueError("Ranking contains duplicate item IDs")
    result = {}
    for k in cutoffs:
        if type(k) is not int or k <= 0:
            raise ValueError("Cutoffs must be positive integers")
        hits = [item in targets for item in ranking[:k]]
        ideal = sum(1 / np.log2(i + 2) for i in range(min(k, len(targets))))
        result[f"ndcg@{k}"] = float(sum(hit / np.log2(i + 2) for i, hit in enumerate(hits)) / ideal)
        result[f"recall@{k}"] = sum(hits) / len(targets)
    return result


def summarize(rows, activity, bootstrap_samples=0, seed=42):
    """Average requests within users, then users within groups.

    Bootstrap users within fixed groups. Intervals describe evaluation-user
    uncertainty, not training-seed variation or paired model comparisons.
    """
    rows = list(rows)
    if not rows:
        raise ValueError("No evaluation rows")
    if type(bootstrap_samples) is not int or bootstrap_samples < 0:
        raise ValueError("bootstrap_samples must be a nonnegative integer")
    metrics = sorted(k for k in rows[0] if k.startswith(("ndcg@", "recall@")))
    if not metrics:
        raise ValueError("No NDCG/Recall metrics")
    by_user, seen = defaultdict(list), set()
    for row in rows:
        user, request = str(row["user_id"]), str(row["request_id"])
        if user not in activity:
            raise ValueError(f"Missing activity mapping for user {user}")
        if (user, request) in seen:
            raise ValueError(f"Duplicate evaluation request: {user}/{request}")
        seen.add((user, request))
        if sorted(k for k in row if k.startswith(("ndcg@", "recall@"))) != metrics:
            raise ValueError("Inconsistent metric columns")
        vector = np.asarray([float(row[m]) for m in metrics])
        if not np.isfinite(vector).all() or (vector < 0).any() or (vector > 1).any():
            raise ValueError("Metrics must be finite values in [0, 1]")
        by_user[user].append(vector)
    groups, per_user = defaultdict(list), []
    for user in sorted(by_user):
        vector = np.mean(by_user[user], axis=0)
        group = activity[user]["activity_group"]
        groups[group].append(vector)
        per_user.append({"user_id": user, "activity_group": group,
                         "train_interaction_count": int(activity[user]["train_interaction_count"]),
                         "request_count": len(by_user[user]), **dict(zip(metrics, vector.tolist()))})
    arrays = {g: np.asarray(v) for g, v in groups.items()}

    def statistics(data):
        means = {g: v.mean(axis=0) for g, v in data.items()}
        matrix = np.asarray(list(means.values()))
        stats = {"overall": np.concatenate(list(data.values())).mean(axis=0),
                 "gap": matrix.max(axis=0) - matrix.min(axis=0),
                 "worst_group": matrix.min(axis=0), **{f"group/{g}": v for g, v in means.items()}}
        if "low" in means and "high" in means:
            stats["high_minus_low"] = means["high"] - means["low"]
        return stats

    stats = statistics(arrays)
    summary = {
        "schema_version": 1, "aggregation": "mean requests per user, then mean users",
        "request_count": len(rows), "user_count": len(per_user),
        "mapped_users_without_evaluation": len(set(activity) - set(by_user)),
        "group_user_counts": {g: len(v) for g, v in arrays.items()},
        "unevaluated_groups": sorted({r["activity_group"] for r in activity.values()} - set(groups)),
        "gap_scope": "all evaluated groups, including cold_start if present",
        "statistics": {name: dict(zip(metrics, values.tolist())) for name, values in stats.items()},
    }
    if bootstrap_samples:
        rng = np.random.default_rng(seed)
        draws = defaultdict(list)
        for _ in range(bootstrap_samples):
            sample = {g: v[rng.integers(len(v), size=len(v))] for g, v in arrays.items()}
            for name, values in statistics(sample).items():
                draws[name].append(values)
        summary["bootstrap"] = {"samples": bootstrap_samples, "seed": seed,
                                "method": "stratified user percentile", "confidence": 0.95}
        summary["confidence_intervals"] = {
            name: {m: [float(x) for x in bounds[:, i]] for i, m in enumerate(metrics)}
            for name, values in draws.items()
            for bounds in [np.quantile(values, [0.025, 0.975], axis=0)]
        }
    return per_user, summary


def _write_csv(path, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_report(rows, activity_path, output_dir, bootstrap_samples=1000, seed=42, metadata=None):
    rows = list(rows)
    users, summary = summarize(rows, load_activity(activity_path), bootstrap_samples, seed)
    summary["activity_mapping"] = fingerprint(activity_path)
    activity_metadata = Path(activity_path).with_name("user_activity_metadata.json")
    if activity_metadata.exists():
        summary["activity_metadata"] = json.loads(activity_metadata.read_text())
    summary["evaluation"] = metadata or {}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "per_request_metrics.csv", rows)
    _write_csv(output_dir / "per_user_metrics.csv", users)
    _write_csv(output_dir / "user_fairness_comparison.csv", [
        {"model": summary["evaluation"].get("model", "unspecified"), "statistic": name,
         "metric": metric, "value": value}
        for name, values in summary["statistics"].items() for metric, value in values.items()
    ])
    (output_dir / "user_fairness_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Build mapping from canonical processed data")
    prepare.add_argument("--data-dir", required=True, type=Path)
    prepare.add_argument("--dataset", required=True)
    prepare.add_argument("--repository-root", type=Path, default=Path.cwd())
    report = commands.add_parser("report", help="Report exported per-request metrics (any model)")
    report.add_argument("--metrics", required=True, type=Path)
    report.add_argument("--activity", required=True, type=Path)
    report.add_argument("--output-dir", required=True, type=Path)
    report.add_argument("--bootstrap-samples", type=int, default=1000)
    report.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.command == "report":
        with args.metrics.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        save_report(rows, args.activity, args.output_dir, args.bootstrap_samples, args.seed,
                    {"metrics_source": fingerprint(args.metrics)})
    else:
        prepare_activity(args.data_dir, args.dataset, args.repository_root)


def prepare_activity(data_dir, dataset, repository_root):
    """Backfill cached datasets. GRID counts come from unexpanded TFRecords."""
    import yaml

    data_dir = Path(data_dir)
    manifest_path = data_dir / "grid_data_manifest.json"
    if manifest_path.exists():
        from .grid_data_bridge import _read_tfrecords, _source_fingerprint
        manifest = json.loads(manifest_path.read_text())
        source = Path(repository_root) / manifest["source_dir"]
        if _source_fingerprint(source) != manifest["source_fingerprint"]:
            raise ValueError("GRID source changed since the processed split was built")
        sequences = dict(_read_tfrecords(source, "training"))
        counts = {str(i): len(sequences[u]) for i, u in enumerate(sorted(sequences))}
        provenance = {"kind": "grid_training_sequences", "manifest": fingerprint(manifest_path),
                      "user_id_mapping": "sorted (encoded GRID ID, occurrence) to zero-based FairDiverse IDs"}
    else:
        with (data_dir / "process_config.yaml").open() as stream:
            config = yaml.safe_load(stream)
        if config.get("grid_data_source") == "embedded_tfrecords":
            raise ValueError("Missing GRID manifest; cannot infer original counts from expanded rows")
        train_path = data_dir / f"{dataset}.train"
        counts = Counter()
        with train_path.open(newline="") as stream:
            for row in csv.DictReader(stream, delimiter="\t"):
                counts[row[config["user_id"]]] += int(float(row[config["label_id"]]) == 1)
        for split in ("valid", "test"):
            with (data_dir / f"{dataset}.{split}.ranking").open(newline="") as stream:
                for row in csv.DictReader(stream, delimiter="\t"):
                    counts.setdefault(row["user_id"], 0)
        provenance = {"kind": "positive_training_rows", "training": fingerprint(train_path),
                      "dataset": dataset, "user_id_mapping": "processed dataset IDs"}
    return write_activity(counts, data_dir, provenance)


if __name__ == "__main__":
    main()
