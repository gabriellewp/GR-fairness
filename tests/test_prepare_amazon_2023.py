import gzip
import json

from fairdiverse.recommendation.dataset_loaders import McAuleyRawLoader
from scripts.prepare_amazon import (
    collect_review_stats_2023 as collect_review_stats,
    filter_metadata_2023 as filter_metadata,
)


def test_pet_supplies_2023_stats_metadata_and_loader(tmp_path):
    reviews = tmp_path / "Pet_Supplies.csv.gz"
    metadata = tmp_path / "meta.jsonl.gz"
    filtered = tmp_path / "meta_5core.jsonl.gz"
    with gzip.open(reviews, "wt", encoding="utf-8", newline="") as output:
        output.write("user_id,parent_asin,rating,timestamp\n")
        output.write("u1,p1,5.0,1000\n")
        output.write("u1,p2,4.0,2000\n")
        output.write("u2,p1,3.0,3000\n")
    with gzip.open(metadata, "wt", encoding="utf-8") as output:
        output.write(json.dumps({
            "parent_asin": "p1", "title": "Dog toy", "store": "PetCo",
            "main_category": "Pet Supplies", "categories": ["Dogs"],
            "details": {"Brand": "Fallback"},
        }) + "\n")
        output.write(json.dumps({
            "parent_asin": "p2", "title": "Cat toy", "store": None,
            "main_category": "Pet Supplies", "categories": ["Cats"],
            "details": {"Brand": "CatBrand"},
        }) + "\n")
        output.write(json.dumps({"parent_asin": "unused", "title": "Unused"}) + "\n")

    items, users, interactions = collect_review_stats(reviews)
    assert (items, users, interactions) == ({"p1", "p2"}, 2, 3)
    assert filter_metadata(metadata, filtered, items) == 2

    config = {
        "reviews_path": str(reviews), "dataset_path": str(filtered),
        "user_id": "user_id:token", "item_id": "item_id:token",
        "label_id": "rating:float", "timestamp": "timestamp:float",
        "group_id": "brand:token", "text_id": "title:token",
    }
    loader = McAuleyRawLoader(config)
    interactions_df = loader.load_interactions()
    items_df = loader.load_items().set_index("item_id:token")

    assert len(interactions_df) == 3
    assert set(interactions_df["item_id:token"]) == {"p1", "p2"}
    assert items_df.loc["p1", "brand:token"] == "PetCo"
    assert items_df.loc["p2", "brand:token"] == "CatBrand"
    assert "Dogs" in items_df.loc["p1", "categories:token_seq"]
