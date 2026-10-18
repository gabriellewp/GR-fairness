"""Build FairDiverse tables directly from embedded GRID temporal-LOO TFRecords."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
import struct
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

import yaml
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory


BRIDGE_VERSION = 3
def _required_outputs(dataset: str) -> tuple[str, ...]:
    return (
        f"{dataset}.train",
        f"{dataset}.valid.CTR",
        f"{dataset}.test.CTR",
        f"{dataset}.valid.ranking",
        f"{dataset}.test.ranking",
        "dataset_stats.json",
        "iid2item.json",
        "iid2pid.json",
        "iid2text.json",
        "process_config.yaml",
        "user_activity.csv",
        "user_activity_metadata.json",
    )


def _example_message_class():
    """Return a protobuf class compatible with tensorflow.train.Example.

    FairDiverse only needs to read GRID's existing TFRecords. Requiring the
    complete TensorFlow runtime for that small operation made otherwise valid
    FairDiverse environments fail during startup. Define TensorFlow's stable
    Example wire schema directly with protobuf instead.
    """
    file_descriptor = descriptor_pb2.FileDescriptorProto()
    file_descriptor.name = "fairdiverse_tensorflow_example.proto"
    file_descriptor.package = "tensorflow"
    file_descriptor.syntax = "proto3"

    def repeated_scalar(name, scalar_type, *, packed=False):
        message = file_descriptor.message_type.add()
        message.name = name
        field = message.field.add()
        field.name = "value"
        field.number = 1
        field.label = descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED
        field.type = scalar_type
        field.options.packed = packed

    repeated_scalar("BytesList", descriptor_pb2.FieldDescriptorProto.TYPE_BYTES)
    repeated_scalar(
        "FloatList", descriptor_pb2.FieldDescriptorProto.TYPE_FLOAT, packed=True
    )
    repeated_scalar(
        "Int64List", descriptor_pb2.FieldDescriptorProto.TYPE_INT64, packed=True
    )

    feature = file_descriptor.message_type.add()
    feature.name = "Feature"
    feature.oneof_decl.add().name = "kind"
    for number, (name, type_name) in enumerate(
        (
            ("bytes_list", ".tensorflow.BytesList"),
            ("float_list", ".tensorflow.FloatList"),
            ("int64_list", ".tensorflow.Int64List"),
        ),
        start=1,
    ):
        field = feature.field.add()
        field.name = name
        field.number = number
        field.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
        field.type = descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE
        field.type_name = type_name
        field.oneof_index = 0

    features = file_descriptor.message_type.add()
    features.name = "Features"
    entry = features.nested_type.add()
    entry.name = "FeatureEntry"
    entry.options.map_entry = True
    key = entry.field.add()
    key.name = "key"
    key.number = 1
    key.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    key.type = descriptor_pb2.FieldDescriptorProto.TYPE_STRING
    value = entry.field.add()
    value.name = "value"
    value.number = 2
    value.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    value.type = descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE
    value.type_name = ".tensorflow.Feature"
    feature_map = features.field.add()
    feature_map.name = "feature"
    feature_map.number = 1
    feature_map.label = descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED
    feature_map.type = descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE
    feature_map.type_name = ".tensorflow.Features.FeatureEntry"

    example = file_descriptor.message_type.add()
    example.name = "Example"
    example_features = example.field.add()
    example_features.name = "features"
    example_features.number = 1
    example_features.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    example_features.type = descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE
    example_features.type_name = ".tensorflow.Features"

    pool = descriptor_pool.DescriptorPool()
    pool.Add(file_descriptor)
    return message_factory.GetMessageClass(
        pool.FindMessageTypeByName("tensorflow.Example")
    )


_EXAMPLE_MESSAGE = _example_message_class()


def _iter_tfrecord_examples(paths: list[Path]):
    """Yield Example messages from gzip-compressed TFRecord files."""
    for path in paths:
        with gzip.open(path, "rb") as stream:
            record_index = 0
            while True:
                header = stream.read(12)  # uint64 length plus masked CRC32C
                if not header:
                    break
                if len(header) != 12:
                    raise ValueError(f"Truncated TFRecord header in {path}")
                record_length = struct.unpack("<Q", header[:8])[0]
                payload = stream.read(record_length)
                footer = stream.read(4)  # masked payload CRC32C
                if len(payload) != record_length or len(footer) != 4:
                    raise ValueError(
                        f"Truncated TFRecord payload {record_index} in {path}"
                    )
                example = _EXAMPLE_MESSAGE()
                example.ParseFromString(payload)
                yield example
                record_index += 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_files(source_dir: Path) -> list[Path]:
    files = [
        source_dir / "item_id_mapping.json",
        source_dir / "split_config.json",
        source_dir / "validation_items.json",
        source_dir / "test_items.json",
    ]
    for split in ("items", "training", "validation", "testing"):
        files.extend(sorted((source_dir / split).glob("*.tfrecord.gz")))
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete embedded GRID dataset: {missing}")
    return files


def _source_is_complete(source_dir: Path) -> bool:
    try:
        return bool(_source_files(source_dir))
    except FileNotFoundError:
        return False


def _ensure_grid_tfrecords(
    repository_root: Path, source_dir: Path, config: dict
) -> str:
    """Create the canonical embedded TFRecords once when they do not exist."""
    if _source_is_complete(source_dir):
        return f"reused embedded GRID TFRecords from {source_dir}"

    if source_dir.exists() and any(source_dir.iterdir()):
        raise RuntimeError(
            f"Embedded GRID data directory is partial or corrupt: {source_dir}. "
            "It was left untouched; move it aside or complete it before retrying."
        )

    metadata_path = Path(config["dataset_path"]).resolve()
    reviews_path = Path(config["reviews_path"]).resolve()
    if metadata_path.parent != reviews_path.parent:
        raise ValueError(
            "GRID preprocessing expects metadata and interactions in one input "
            f"directory, got {metadata_path.parent} and {reviews_path.parent}"
        )
    for path in (metadata_path, reviews_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing raw Amazon input required by GRID: {path}")

    source_dir.parent.mkdir(parents=True, exist_ok=True)
    preprocess_script = repository_root / "GRID/preprocess_amazon_beauty.py"
    with tempfile.TemporaryDirectory(
        prefix="amazon-grid-tfrecords-", dir=source_dir.parent
    ) as temporary_parent:
        temporary_output = Path(temporary_parent) / "temporal_loo"
        preprocess_python = os.environ.get("GRID_PYTHON", sys.executable)
        command = [
            preprocess_python,
            str(preprocess_script),
            "--input-dir",
            str(metadata_path.parent),
            "--output-dir",
            str(temporary_output),
            "--metadata-file",
            metadata_path.name,
            "--interactions-file",
            reviews_path.name,
            "--min-seq-length",
            "4",
            "--split-strategy",
            "temporal_loo",
            "--train-ratio",
            "0.8",
            "--val-ratio",
            "0.1",
            "--seed",
            "42",
        ]
        environment = os.environ.copy()
        environment.setdefault("PYTHONHASHSEED", "42")
        subprocess.run(command, check=True, cwd=repository_root / "GRID", env=environment)
        if not _source_is_complete(temporary_output):
            raise RuntimeError("GRID preprocessing completed without all required outputs")
        if source_dir.exists():
            # Only an empty directory reaches this point; partial data is never removed.
            source_dir.rmdir()
        os.replace(temporary_output, source_dir)
    return f"built embedded GRID TFRecords in {source_dir}"


def _source_fingerprint(source_dir: Path) -> str:
    digest = hashlib.sha256()
    for path in _source_files(source_dir):
        digest.update(str(path.relative_to(source_dir)).encode())
        digest.update(_sha256(path).encode())
    return digest.hexdigest()


def _read_tfrecords(
    source_dir: Path, split: str
) -> list[tuple[tuple[int, int], list[int]]]:
    """Keep hash-colliding users distinct by their occurrence in each split.

    GRID encodes nonnumeric IDs with hash(user_id) % 2**31, which is not
    unique. Temporal-LOO writes users in the same order in each split. Pair
    the encoded ID with its occurrence rather than overwriting or dropping
    a record; _build_outputs still verifies every user's split histories.
    """
    paths = sorted((source_dir / split).glob("*.tfrecord.gz"))
    rows = []
    occurrences = Counter()
    for example in _iter_tfrecord_examples(paths):
        features = example.features.feature
        user_id = int(features["user_id"].int64_list.value[0])
        sequence = [int(value) for value in features["sequence_data"].int64_list.value]
        user_key = (user_id, occurrences[user_id])
        occurrences[user_id] += 1
        rows.append((user_key, sequence))
    return rows


def _read_item_texts(source_dir: Path) -> dict[int, str]:
    paths = sorted((source_dir / "items").glob("*.tfrecord.gz"))
    result = {}
    for example in _iter_tfrecord_examples(paths):
        features = example.features.feature
        item_id = int(features["id"].int64_list.value[0])
        raw_text = features["text"].bytes_list.value[0]
        result[item_id] = raw_text.decode("utf-8", errors="replace")
    return result


def _provider_mapping(config: dict, item_to_iid: dict[str, int]) -> dict[int, int]:
    metadata_path = Path(config["dataset_path"])
    opener = gzip.open if metadata_path.suffix == ".gz" else open
    item_brand = {}
    with opener(metadata_path, "rt", encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            item = str(record.get("asin") or record.get("parent_asin") or "")
            if item in item_to_iid:
                item_brand[item] = str(
                    record.get("brand") or record.get("store") or ""
                ).strip() or "no_brand"

    missing = sorted(set(item_to_iid) - set(item_brand))
    if missing:
        raise ValueError(f"Metadata is missing {len(missing)} GRID items: {missing[:5]}")

    counts = Counter(item_brand.values())
    threshold = int(config["group_aggregation_threshold"])
    effective_brand = {
        item: brand if counts[brand] >= threshold else "__OTHER__"
        for item, brand in item_brand.items()
    }
    brands = sorted(set(effective_brand.values()))
    brand_to_pid = {brand: index for index, brand in enumerate(brands)}
    return {
        iid: brand_to_pid[effective_brand[item]]
        for item, iid in item_to_iid.items()
    }


def _write_tsv(path: Path, fieldnames: list[str], rows) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _merged_process_config(run_config: dict) -> dict:
    root = Path(__file__).resolve().parent
    dataset = run_config.get("dataset")
    if not dataset or Path(dataset).name != dataset:
        raise ValueError(f"Invalid dataset name: {dataset!r}")
    dataset_config = root / "properties/dataset" / f"{dataset}.yaml"
    if not dataset_config.is_file():
        raise FileNotFoundError(f"Missing dataset configuration: {dataset_config}")
    with (root / "properties/dataset.yaml").open() as stream:
        config = yaml.safe_load(stream)
    with dataset_config.open() as stream:
        config.update(yaml.safe_load(stream))
    config.update(run_config)
    config.update(
        {
            "history_field": "history_behaviors",
            "grid_compact_ranking": True,
            "grid_data_source": "embedded_tfrecords",
            "reprocess": False,
        }
    )
    return config


def _build_outputs(source_dir: Path, output_dir: Path, run_config: dict) -> dict:
    config = _merged_process_config(run_config)
    dataset = config["dataset"]
    item_to_iid = {
        str(item): int(iid)
        for item, iid in json.loads((source_dir / "item_id_mapping.json").read_text()).items()
    }
    iid_to_item = {iid: item for item, iid in item_to_iid.items()}
    item_count = len(item_to_iid)
    if set(iid_to_item) != set(range(item_count)):
        raise ValueError("GRID item mapping is not contiguous from zero")

    training = dict(_read_tfrecords(source_dir, "training"))
    validation = dict(_read_tfrecords(source_dir, "validation"))
    testing = dict(_read_tfrecords(source_dir, "testing"))
    if not training or training.keys() != validation.keys() or training.keys() != testing.keys():
        raise ValueError("GRID temporal-LOO splits do not contain identical users")

    for user_id, train_sequence in training.items():
        val_sequence = validation[user_id]
        test_sequence = testing[user_id]
        if val_sequence[:-1] != train_sequence or test_sequence[:-2] != train_sequence:
            raise ValueError(f"GRID split history mismatch for encoded user {user_id}")
        if test_sequence[-2] != val_sequence[-1]:
            raise ValueError(f"GRID validation target missing from test history for user {user_id}")

    iid_to_pid = _provider_mapping(config, item_to_iid)
    group_count = len(set(iid_to_pid.values()))
    iid_to_text = _read_item_texts(source_dir)
    if set(iid_to_text) != set(range(item_count)):
        raise ValueError("GRID item TFRecords and item mapping differ")

    uid_field = config["user_id"]
    iid_field = config["item_id"]
    pid_field = config["group_id"]
    label_field = "label:float"
    user_to_iid = {encoded_user: index for index, encoded_user in enumerate(sorted(training))}
    from .user_fairness import write_activity
    write_activity(
        {user_to_iid[user]: len(sequence) for user, sequence in training.items()},
        output_dir,
        {"kind": "grid_training_sequences", "source_fingerprint": _source_fingerprint(source_dir),
         "user_id_mapping": "sorted (encoded GRID ID, occurrence) to zero-based FairDiverse IDs"},
    )

    def train_rows():
        for encoded_user in sorted(training):
            user_id = user_to_iid[encoded_user]
            sequence = training[encoded_user]
            for target_index in range(1, len(sequence)):
                target = sequence[target_index]
                yield {
                    uid_field: user_id,
                    iid_field: target,
                    "history_behaviors": sequence[:target_index],
                    label_field: 1,
                    pid_field: iid_to_pid[target],
                    config["timestamp"]: 0.0,
                }

    interaction_fields = [
        uid_field,
        iid_field,
        "history_behaviors",
        label_field,
        pid_field,
        config["timestamp"],
    ]
    _write_tsv(output_dir / f"{dataset}.train", interaction_fields, train_rows())

    def evaluation_rows(split_rows: dict[tuple[int, int], list[int]]):
        for encoded_user in sorted(split_rows):
            sequence = split_rows[encoded_user]
            target = sequence[-1]
            yield {
                uid_field: user_to_iid[encoded_user],
                iid_field: target,
                "history_behaviors": sequence[:-1],
                label_field: 1,
                pid_field: iid_to_pid[target],
                config["timestamp"]: 0.0,
            }

    _write_tsv(
        output_dir / f"{dataset}.valid.CTR",
        interaction_fields,
        evaluation_rows(validation),
    )
    _write_tsv(
        output_dir / f"{dataset}.test.CTR",
        interaction_fields,
        evaluation_rows(testing),
    )

    compact_fields = ["user_id", "history_behaviors", "target_item"]
    for name, split_rows in (("valid", validation), ("test", testing)):
        rows = (
            {
                "user_id": user_to_iid[encoded_user],
                "history_behaviors": sequence[:-1],
                "target_item": sequence[-1],
            }
            for encoded_user, sequence in sorted(split_rows.items())
        )
        _write_tsv(output_dir / f"{dataset}.{name}.ranking", compact_fields, rows)

    config.update(
        {
            "item_num": item_count,
            "user_num": len(training),
            "group_num": group_count,
            "label_id": label_field,
        }
    )
    (output_dir / "dataset_stats.json").write_text(
        json.dumps(
            {"item_num": item_count, "user_num": len(training), "group_num": group_count}
        )
    )
    (output_dir / "iid2item.json").write_text(
        json.dumps({str(iid): iid_to_item[iid] for iid in range(item_count)})
    )
    (output_dir / "iid2pid.json").write_text(
        json.dumps({str(iid): iid_to_pid[iid] for iid in range(item_count)})
    )
    (output_dir / "iid2text.json").write_text(
        json.dumps({str(iid): iid_to_text[iid] for iid in range(item_count)})
    )
    with (output_dir / "process_config.yaml").open("w") as stream:
        yaml.safe_dump(config, stream)
    return config


def ensure_grid_fairdiverse_data(run_config: dict) -> str:
    """Reuse or atomically rebuild FairDiverse data from the canonical GRID files."""
    repository_root = Path(__file__).resolve().parents[2]
    dataset = run_config.get("dataset")
    if not dataset or Path(dataset).name != dataset:
        raise ValueError(f"Invalid dataset name: {dataset!r}")
    source_dir = (repository_root / run_config["grid_data_dir"]).resolve()
    embedded_grid_root = (repository_root / "GRID").resolve()
    if not source_dir.is_relative_to(embedded_grid_root):
        raise ValueError(f"grid_data_dir must be below embedded {embedded_grid_root}")

    full_config = _merged_process_config(run_config)
    tfrecord_state = _ensure_grid_tfrecords(
        repository_root, source_dir, full_config
    )

    split_config = json.loads((source_dir / "split_config.json").read_text())
    if split_config.get("strategy") != "temporal_loo":
        raise ValueError(f"Expected temporal_loo GRID data, got {split_config}")

    output_dir = repository_root / "recommendation" / "processed_dataset" / dataset
    required_outputs = _required_outputs(dataset)
    manifest_path = output_dir / "grid_data_manifest.json"
    source_fingerprint = _source_fingerprint(source_dir)
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if (
            manifest.get("bridge_version") == BRIDGE_VERSION
            and manifest.get("source_fingerprint") == source_fingerprint
            and all((output_dir / name).is_file() for name in required_outputs)
        ):
            return (
                f"{tfrecord_state}; reused canonical GRID-derived "
                f"FairDiverse data from {output_dir}"
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="grid-fairdiverse-", dir=output_dir.parent) as tmp:
        temporary_dir = Path(tmp)
        config = _build_outputs(source_dir, temporary_dir, run_config)
        manifest = {
            "bridge_version": BRIDGE_VERSION,
            "source_dir": str(source_dir.relative_to(repository_root)),
            "source_fingerprint": source_fingerprint,
            "split_strategy": "temporal_loo",
            "item_num": config["item_num"],
            "user_num": config["user_num"],
            "group_num": config["group_num"],
        }
        (temporary_dir / "grid_data_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
        for name in (*required_outputs, "grid_data_manifest.json"):
            os.replace(temporary_dir / name, output_dir / name)
    return (
        f"{tfrecord_state}; built canonical GRID-derived FairDiverse data "
        f"in {output_dir}"
    )
