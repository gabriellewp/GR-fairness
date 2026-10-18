"""Prepare canonical Pet Supplies tables before submitting continuation suites."""
import json
from pathlib import Path
from fairdiverse.recommendation.grid_data_bridge import ensure_grid_fairdiverse_data


def main():
    root = Path(__file__).resolve().parents[1]
    entries = json.loads((root / "configs/reproduction/experiments_config_pet_continuation_exports.json").read_text())
    config = dict(entries[0]["config"], dataset="amazon-pet-supplies")
    print(ensure_grid_fairdiverse_data(config), flush=True)


if __name__ == "__main__":
    main()
