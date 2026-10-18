#!/usr/bin/env python3
"""Download and prepare supported McAuley Amazon 5-core datasets.

Despite the historical filename, this is the shared preparation entry point
for the 2014 Sports/Toys datasets and the 2023 Pet Supplies benchmark.
"""

import argparse
import ast
import csv
import gzip
import json
import shutil
import sys
import urllib.request
from pathlib import Path
from typing import Dict, Optional, Set, Tuple


LEGACY_BASE_URL = "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles"
CATEGORIES = {
    "sports": {
        "release": "2014", "base_url": LEGACY_BASE_URL,
        "reviews": "reviews_Sports_and_Outdoors_5.json.gz",
        "ratings": "ratings_Sports_and_Outdoors.csv",
        "metadata": "meta_Sports_and_Outdoors.json.gz",
        "filtered_metadata": "meta_Sports_and_Outdoors_5core.json.gz",
    },
    "toys": {
        "release": "2014", "base_url": LEGACY_BASE_URL,
        "reviews": "reviews_Toys_and_Games_5.json.gz",
        "ratings": "ratings_Toys_and_Games.csv",
        "metadata": "meta_Toys_and_Games.json.gz",
        "filtered_metadata": "meta_Toys_and_Games_5core.json.gz",
    },
    "pet-supplies": {
        "release": "2023", "reviews": "Pet_Supplies.csv.gz",
        "reviews_url": ("https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/"
                        "benchmark/5core/rating_only/Pet_Supplies.csv.gz"),
        "metadata": "meta_Pet_Supplies.jsonl.gz",
        "metadata_url": ("https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/"
                         "raw/meta_categories/meta_Pet_Supplies.jsonl.gz"),
        "filtered_metadata": "meta_Pet_Supplies_5core.jsonl.gz",
        "expected_stats": {"users": 594800, "items": 114625,
                           "interactions": 5277315},
    },
}
DEFAULT_DATA_DIRS = {
    "2014": Path("data/raw/amazon-mcauley-2014"),
    "2023": Path("data/raw/amazon-mcauley-2023"),
}


def download(url: str, destination: Path, force: bool = False) -> None:
    if destination.exists() and not force:
        print(f"Using existing {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    print(f"Downloading {url} -> {destination}")
    try:
        with urllib.request.urlopen(url) as response, partial.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
        partial.replace(destination)
    except Exception:
        if partial.exists():
            partial.unlink()
        raise


def read_record(line: str) -> dict:
    try:
        record = json.loads(line)
    except json.JSONDecodeError:
        record = ast.literal_eval(line)
    if not isinstance(record, dict):
        raise ValueError("record is not an object")
    return record


def collect_review_stats_2014(path: Path) -> Tuple[Set[str], int, int]:
    items = set()  # type: Set[str]
    users = set()  # type: Set[str]
    interactions = 0
    with gzip.open(path, "rt", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            try:
                review = read_record(line)
                item = str(review.get("asin", "") or "").strip()
                user = str(review.get("reviewerID", "") or "").strip()
                if item and user:
                    items.add(item)
                    users.add(user)
                    interactions += 1
            except (ValueError, SyntaxError) as error:
                print(f"Warning: skipping malformed review line {line_number}: {error}",
                      file=sys.stderr)
    return items, len(users), interactions


def collect_review_stats_2023(path: Path) -> Tuple[Set[str], int, int]:
    items = set()  # type: Set[str]
    users = set()  # type: Set[str]
    interactions = 0
    with gzip.open(path, "rt", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        required = {"user_id", "parent_asin", "rating", "timestamp"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"5-core CSV is missing columns: {sorted(missing)}")
        for row in reader:
            user = (row.get("user_id") or "").strip()
            item = (row.get("parent_asin") or "").strip()
            if user and item:
                users.add(user)
                items.add(item)
                interactions += 1
    return items, len(users), interactions


def _filter_metadata(source_path: Path, output_path: Path, items: Set[str],
                     item_field: str, legacy_syntax: bool,
                     deduplicate: bool) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_name(output_path.name + ".part")
    kept_items = set()  # type: Set[str]
    kept_rows = 0
    try:
        with gzip.open(source_path, "rt", encoding="utf-8") as source, gzip.open(
            partial, "wt", encoding="utf-8"
        ) as output:
            for line_number, line in enumerate(source, start=1):
                try:
                    record = read_record(line) if legacy_syntax else json.loads(line)
                    if not isinstance(record, dict):
                        raise ValueError("record is not an object")
                except (json.JSONDecodeError, ValueError, SyntaxError) as error:
                    print(f"Warning: skipping malformed metadata line {line_number}: {error}",
                          file=sys.stderr)
                    continue
                item = str(record.get(item_field, "") or "").strip()
                if item in items and (not deduplicate or item not in kept_items):
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
                    kept_items.add(item)
                    kept_rows += 1
        partial.replace(output_path)
    except Exception:
        if partial.exists():
            partial.unlink()
        raise
    return len(kept_items) if deduplicate else kept_rows


def filter_metadata_2014(source_path: Path, output_path: Path,
                         items: Set[str]) -> int:
    return _filter_metadata(source_path, output_path, items, "asin", True, False)


def filter_metadata_2023(source_path: Path, output_path: Path,
                         items: Set[str]) -> int:
    return _filter_metadata(source_path, output_path, items,
                            "parent_asin", False, True)


# Backward-compatible names used by existing callers and tests.
collect_review_stats = collect_review_stats_2014
filter_metadata = filter_metadata_2014


def _source_url(config: Dict, kind: str) -> str:
    return str(config.get(kind + "_url") or
               (str(config["base_url"]).rstrip("/") + "/" + str(config[kind])))


def prepare_category(category: str, data_dir: Optional[Path] = None,
                     force: bool = False, verify: bool = True) -> Dict:
    config = CATEGORIES[category]
    release = str(config["release"])
    target_dir = data_dir or DEFAULT_DATA_DIRS[release]
    source_kinds = ["reviews", "metadata"]
    if "ratings" in config:
        source_kinds.insert(1, "ratings")
    for kind in source_kinds:
        download(_source_url(config, kind), target_dir / str(config[kind]), force)

    reviews_path = target_dir / str(config["reviews"])
    metadata_path = target_dir / str(config["metadata"])
    filtered_path = target_dir / str(config["filtered_metadata"])
    if release == "2014":
        items, users, interactions = collect_review_stats_2014(reviews_path)
        metadata_items = filter_metadata_2014(metadata_path, filtered_path, items)
    else:
        items, users, interactions = collect_review_stats_2023(reviews_path)
        metadata_items = filter_metadata_2023(metadata_path, filtered_path, items)

    actual = {"users": users, "items": len(items), "interactions": interactions}
    expected = config.get("expected_stats")
    if verify and expected and actual != expected:
        raise ValueError(f"Unexpected {category} 5-core statistics: {actual}; expected {expected}")
    if verify and release == "2023" and metadata_items != len(items):
        raise ValueError(
            f"Metadata covers {metadata_items} of {len(items)} benchmark items; "
            "refusing to create a dataset whose interaction counts would shrink")

    stats = {
        "category": category, "release": release, "users": users,
        "items": len(items), "interactions": interactions,
        "metadata_items_written": metadata_items,
        "review_items_without_metadata": len(items) - metadata_items,
        "reviews_path": str(reviews_path), "metadata_path": str(filtered_path),
    }
    if "ratings" in config:
        stats["ratings_path"] = str(target_dir / str(config["ratings"]))
    (target_dir / f"{category}_5core_source_stats.json").write_text(
        json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2))
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("categories", nargs="*", metavar="CATEGORY",
                        help="Categories to prepare (default: sports toys; supported: "
                             + ", ".join(sorted(CATEGORIES)) + ")")
    parser.add_argument("--data-dir", type=Path,
                        help="Override the release-specific destination directory")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--skip-count-verification", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    # Preserve the old no-argument behavior; Pet Supplies is selected explicitly.
    categories = args.categories or ["sports", "toys"]
    invalid = sorted(set(categories) - set(CATEGORIES))
    if invalid:
        raise SystemExit("Unknown category {} (choose from: {})".format(
            ", ".join(invalid), ", ".join(sorted(CATEGORIES))))
    for category in categories:
        prepare_category(category, args.data_dir, args.force_download,
                         not args.skip_count_verification)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
