import run_experiment
import pytest


def test_selection_preserves_explicit_ids_after_pruning():
    experiments = [{"exp_id": 57}, {"exp_id": 65}]
    assert run_experiment.select_experiment(experiments, 65) is experiments[1]
    with pytest.raises(ValueError):
        run_experiment.select_experiment(experiments, 1)


def test_selection_supports_legacy_indices():
    experiments = [{"config": {}}, {"config": {}}]
    assert run_experiment.select_experiment(experiments, 1) is experiments[1]


def test_selection_rejects_duplicate_ids():
    with pytest.raises(ValueError):
        run_experiment.select_experiment([{"exp_id": 57}, {"exp_id": 57}], 57)


def test_auto_grid_log_name_contains_dataset_and_provenance():
    config = {
        "log_name": "auto",
        "dataset": "amazon-sports",
        "grid_tokenizer": "rkmeans",
        "embedding_model": "google/flan-t5-xl",
        "metadata_text_mode": "complete_metadata",
        "rank_model": "SDRO",
    }

    run_experiment.resolve_auto_log_name(config)

    assert config["log_name"] == (
        "grid-independent-amazon-sports-rkmeans-flan-t5-xl-"
        "complete-metadata-SDRO"
    )


def test_explicit_grid_log_name_gets_missing_dataset():
    config = {
        "log_name": "grid-independent-rkmeans-cf-sasrec-description-SDRO",
        "dataset": "amazon-beauty",
    }

    run_experiment.resolve_auto_log_name(config)

    assert config["log_name"] == (
        "grid-independent-amazon-beauty-rkmeans-cf-sasrec-description-SDRO"
    )
