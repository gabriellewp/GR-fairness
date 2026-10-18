from pathlib import Path

from fairdiverse.recommendation.grid_data_bridge import (
    _merged_process_config,
    _required_outputs,
)


ROOT = Path(__file__).resolve().parents[1]


def test_grid_bridge_uses_dataset_specific_configuration_and_outputs():
    for dataset in ("amazon-beauty", "amazon-sports", "amazon-toys"):
        config = _merged_process_config({"dataset": dataset})
        outputs = _required_outputs(dataset)

        assert config["dataset"] == dataset
        assert f"{dataset}.train" in outputs
        assert f"{dataset}.valid.CTR" in outputs
        assert f"{dataset}.test.ranking" in outputs
        assert not any(
            name.startswith("amazon-beauty.")
            for name in outputs
            if dataset != "amazon-beauty"
        )


def test_grid_bridge_output_directory_is_not_hard_coded_to_beauty():
    source = (ROOT / "fairdiverse/recommendation/grid_data_bridge.py").read_text()

    assert '"processed_dataset" / dataset' in source
    assert 'processed_dataset/amazon-beauty' not in source


def test_embedded_grid_run_can_reach_trainer_without_processed_directory():
    source = (ROOT / "run_experiment.py").read_text()

    assert "and not builds_from_embedded_grid" in source
    assert "config.get('grid_data_source') == 'embedded_tfrecords'" in source
