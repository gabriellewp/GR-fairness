#!/usr/bin/env python3
"""Verify the metadata-mode contract of a preprocessed GRID dataset."""

import argparse
import glob
import json
from pathlib import Path

import tensorflow as tf


EXPECTED_FIELDS = {
    "description": ["description"],
    "complete_metadata": ["title", "categories", "brand", "description"],
}
COMPLETE_MARKERS = ("Title: ", "; Categories: ", "; Brand: ", "; Description: ")


def verify(data_dir: str, expected_mode: str) -> int:
    root = Path(data_dir)
    manifest_path = root / "split_config.json"
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)

    actual_mode = manifest.get("metadata_text_mode")
    if actual_mode != expected_mode:
        raise ValueError(
            f"manifest mode is {actual_mode!r}, expected {expected_mode!r}"
        )
    if manifest.get("metadata_text_fields") != EXPECTED_FIELDS[expected_mode]:
        raise ValueError(
            "manifest metadata_text_fields are missing or do not match "
            f"{EXPECTED_FIELDS[expected_mode]!r}"
        )
    if manifest.get("metadata_text_format_version") != 1:
        raise ValueError("unsupported or missing metadata_text_format_version")

    files = sorted(glob.glob(str(root / "items" / "*.tfrecord.gz")))
    if not files:
        raise ValueError(f"no item TFRecords found below {root / 'items'}")

    count = 0
    for raw in tf.data.TFRecordDataset(files, compression_type="GZIP"):
        example = tf.train.Example.FromString(raw.numpy())
        values = example.features.feature["text"].bytes_list.value
        if len(values) != 1:
            raise ValueError(f"item record {count} does not contain exactly one text value")
        text = values[0].decode("utf-8")
        is_complete = text.startswith(COMPLETE_MARKERS[0]) and all(
            marker in text for marker in COMPLETE_MARKERS[1:]
        )
        if expected_mode == "complete_metadata" and not is_complete:
            raise ValueError(
                f"item record {count} is not labelled complete metadata: {text[:120]!r}"
            )
        if expected_mode == "description" and is_complete:
            raise ValueError(f"item record {count} contains complete metadata in description mode")
        count += 1

    print(
        f"Verified {count} item records: metadata_text_mode={expected_mode}, "
        f"fields={EXPECTED_FIELDS[expected_mode]}"
    )
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir")
    parser.add_argument(
        "expected_mode", choices=["description", "complete_metadata"]
    )
    args = parser.parse_args()
    verify(args.data_dir, args.expected_mode)
