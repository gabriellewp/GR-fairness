import json

import pytest

from fairdiverse.recommendation import grid_data_bridge as bridge


def test_provider_mapping_supports_amazon_2023_metadata(tmp_path):
    metadata = tmp_path / "metadata.jsonl"
    records = [
        {"parent_asin": "a", "store": "Brand A"},
        {"asin": "b", "brand": "Brand A", "store": "ignored"},
        {"parent_asin": "c", "store": "Brand C"},
    ]
    metadata.write_text("\n".join(json.dumps(record) for record in records))
    providers = bridge._provider_mapping(
        {"dataset_path": str(metadata), "group_aggregation_threshold": 1},
        {"a": 0, "b": 1, "c": 2},
    )
    assert providers[0] == providers[1]
    assert providers[0] != providers[2]


def _write_split(root, split, rows):
    import gzip
    import struct

    directory = root / split
    directory.mkdir()
    with gzip.open(directory / "000.tfrecord.gz", "wb") as stream:
        for user, sequence in rows:
            example = bridge._EXAMPLE_MESSAGE()
            example.features.feature["user_id"].int64_list.value.append(user)
            example.features.feature["sequence_data"].int64_list.value.extend(sequence)
            payload = example.SerializeToString()
            stream.write(struct.pack("<Q", len(payload)) + b"\0" * 4)
            stream.write(payload + b"\0" * 4)


@pytest.mark.parametrize("misaligned", [False, True])
def test_hash_collisions_preserve_users_and_check_split_alignment(tmp_path, monkeypatch, misaligned):
    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "output"
    output.mkdir()
    (source / "item_id_mapping.json").write_text(json.dumps({str(i): i for i in range(5)}))
    _write_split(source, "training", [(7, [0, 1]), (8, [1, 2]), (7, [2, 3])])
    validation = [(7, [0, 1, 2]), (8, [1, 2, 3]), (7, [2, 3, 4])]
    if misaligned:
        validation[0], validation[2] = validation[2], validation[0]
    _write_split(source, "validation", validation)
    _write_split(source, "testing", [(7, [0, 1, 2, 3]), (8, [1, 2, 3, 4]), (7, [2, 3, 4, 0])])
    training = dict(bridge._read_tfrecords(source, "training"))
    assert training == {(7, 0): [0, 1], (8, 0): [1, 2], (7, 1): [2, 3]}
    monkeypatch.setattr(bridge, "_provider_mapping", lambda *args: {i: 0 for i in range(5)})
    monkeypatch.setattr(bridge, "_read_item_texts", lambda *args: {i: str(i) for i in range(5)})
    monkeypatch.setattr(bridge, "_source_fingerprint", lambda *args: "test")
    if misaligned:
        with pytest.raises(ValueError, match="split history mismatch"):
            bridge._build_outputs(source, output, {"dataset": "amazon-pet-supplies"})
    else:
        config = bridge._build_outputs(source, output, {"dataset": "amazon-pet-supplies"})
        assert config["user_num"] == 3
        assert json.loads((output / "dataset_stats.json").read_text())["user_num"] == 3
