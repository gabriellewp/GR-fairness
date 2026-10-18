"""Build FairDiverse tables from the matching GRID temporal-LOO data."""
import argparse
from fairdiverse.recommendation.grid_data_bridge import ensure_grid_fairdiverse_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["amazon-beauty", "amazon-sports", "amazon-toys", "amazon-pet-supplies"], required=True)
    parser.add_argument("--grid-data-dir", required=True)
    args = parser.parse_args()
    print(ensure_grid_fairdiverse_data(dict(dataset=args.dataset, grid_data_dir=args.grid_data_dir,
        grid_data_source="embedded_tfrecords", metadata_text_mode="complete_metadata")))


if __name__ == "__main__":
    main()
