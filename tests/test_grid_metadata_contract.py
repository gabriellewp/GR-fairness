import gzip
import importlib.util
import json
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "GRID" / "preprocess_amazon_beauty.py"


def _load_preprocessor():
    spec = importlib.util.spec_from_file_location("grid_preprocess", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _metadata_file(tmp_path):
    path = tmp_path / "metadata.json.gz"
    item = {
        "asin": "item-1",
        "title": "Example title",
        "categories": [["Beauty", "Skin Care"]],
        "brand": "Example brand",
        "description": "Example description",
    }
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps(item) + "\n")
    return path


def test_description_mode_uses_only_description(tmp_path):
    module = _load_preprocessor()
    item = module.load_item_metadata(
        str(_metadata_file(tmp_path)), "description"
    )["item-1"]

    assert item["text"] == "Example description"


def test_complete_metadata_mode_uses_all_declared_fields(tmp_path):
    module = _load_preprocessor()
    item = module.load_item_metadata(
        str(_metadata_file(tmp_path)), "complete_metadata"
    )["item-1"]

    assert item["text"] == (
        "Title: Example title; Categories: [['Beauty', 'Skin Care']]; "
        "Brand: Example brand; Description: Example description"
    )


def test_complete_alias_has_identical_content(tmp_path):
    module = _load_preprocessor()
    path = str(_metadata_file(tmp_path))

    assert module.load_item_metadata(path, "complete")["item-1"]["text"] == (
        module.load_item_metadata(path, "complete_metadata")["item-1"]["text"]
    )


def test_amazon_2023_parent_asin_metadata_is_supported(tmp_path):
    module = _load_preprocessor()
    path = tmp_path / "metadata.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "parent_asin": "pet-item-1",
            "title": "Pet item",
            "description": ["Pet description", "Second sentence"],
        }) + "\n")

    items = module.load_item_metadata(str(path), "description")

    assert items["pet-item-1"]["text"] == "Pet description Second sentence"


def test_amazon_2023_compressed_interaction_csv_is_supported(tmp_path):
    module = _load_preprocessor()
    path = tmp_path / "Pet_Supplies.csv.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write("user_id,parent_asin,rating,timestamp\n")
        handle.write("pet-user-1,pet-item-1,5.0,1700000000\n")

    interactions = module.load_interactions(str(path))

    assert interactions.to_dict("records") == [{
        "user_id": "pet-user-1",
        "item_id": "pet-item-1",
        "rating": 5.0,
        "timestamp": 1700000000,
    }]
