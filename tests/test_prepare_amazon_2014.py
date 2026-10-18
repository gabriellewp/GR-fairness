import gzip
import json
from pathlib import Path
from typing import List

from scripts.prepare_amazon import collect_review_stats, filter_metadata


def write_gzip(path: Path, lines: List[str]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as output:
        output.writelines(line + "\n" for line in lines)


def test_collect_review_stats_and_filter_legacy_metadata(tmp_path):
    reviews = tmp_path / "reviews.json.gz"
    metadata = tmp_path / "meta.json.gz"
    filtered = tmp_path / "meta_5core.json.gz"
    write_gzip(
        reviews,
        [
            json.dumps({"reviewerID": "u1", "asin": "a1"}),
            json.dumps({"reviewerID": "u1", "asin": "a2"}),
            json.dumps({"reviewerID": "u2", "asin": "a1"}),
        ],
    )
    write_gzip(
        metadata,
        [
            "{'asin': 'a1', 'title': 'Legacy syntax'}",
            json.dumps({"asin": "a2", "title": "JSON syntax"}),
            "{'asin': 'unused', 'title': 'Not reviewed'}",
        ],
    )

    asins, users, interactions = collect_review_stats(reviews)
    kept = filter_metadata(metadata, filtered, asins)

    assert (asins, users, interactions, kept) == ({"a1", "a2"}, 2, 3, 2)
    with gzip.open(filtered, "rt", encoding="utf-8") as source:
        records = [json.loads(line) for line in source]
    assert [record["asin"] for record in records] == ["a1", "a2"]
