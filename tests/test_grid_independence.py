from pathlib import Path

import yaml
import json


ROOT = Path(__file__).resolve().parents[1]


def test_pipeline_never_references_external_grid_checkout():
    script = (ROOT / "GRID" / "run_independent_pipeline.sh").read_text()
    assert 'GRID_ROOT="/' not in script
    assert 'elif [[ -n "${SLURM_JOB_ID:-}" ]]' in script
    assert 'GRID_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"' in script
    assert 'FAIRDIVERSE_GRID_ROOT' in script
    assert "#SBATCH --partition=gpu" in script
    assert "#SBATCH --gres=gpu:1" in script
    assert "#SBATCH --chdir=/" not in script
    assert '--tokenizer)' in script
    assert '--embedding-path)' in script
    assert '--embedding-name)' in script
    assert '--data-dir)' in script
    assert '--output-run-name)' in script
    assert 'TOKENIZER_TRAIN_EXPERIMENT="rkmeans_train_flat"' in script
    assert 'TOKENIZER_TRAIN_EXPERIMENT="rvq_train_flat"' in script
    assert 'TOKENIZER_TRAIN_EXPERIMENT="rqvae_train_flat"' in script
    assert 'OUTPUT_DIR="${RUN_OUTPUT_ROOT}/${TOKENIZER}"' in script
    assert 'TOKENIZER_HIERARCHIES="${TOKENIZER_HIERARCHIES:-3}"' in script
    assert 'TIGER_HIERARCHIES="${TIGER_HIERARCHIES:-4}"' in script
    assert 'TIGER_SEQUENCE_LENGTH="${TIGER_SEQUENCE_LENGTH:-160}"' in script
    assert 'num_hierarchies="${TOKENIZER_HIERARCHIES}"' in script
    assert 'num_hierarchies="${TIGER_HIERARCHIES}"' in script
    assert 'sequence_length="${TIGER_SEQUENCE_LENGTH}"' in script
    assert 'if [[ ! -d "${checkpoint_dir}" ]]' in script
    assert "-name merged_predictions_tensor.pt -print -quit" in script
    assert 'GRID_USE_WANDB="${GRID_USE_WANDB:-false}"' in script
    assert "lightning.pytorch.loggers.wandb.WandbLogger" in script
    assert 'set_wandb_stage "tokenizer-training"' in script
    assert 'set_wandb_stage "tiger-training"' in script
    assert "WANDB_API_KEY=" not in script
    assert "HF_KEY=" not in script


def test_pipeline_supports_amazon_2023_pet_supplies():
    script = (ROOT / "GRID/run_independent_pipeline.sh").read_text()
    assert "pet-supplies|pet_supplies)" in script
    assert 'DATASET_SLUG="pet_supplies"' in script
    assert 'amazon-mcauley-2023' in script
    assert 'meta_Pet_Supplies_5core.jsonl.gz' in script
    assert 'Pet_Supplies.csv.gz' in script


def test_embedded_configs_match_verified_schedule():
    with (ROOT / "GRID/configs/experiment/rkmeans_train_flat.yaml").open() as f:
        rkmeans = yaml.safe_load(f)
    with (ROOT / "GRID/configs/experiment/tiger_train_flat.yaml").open() as f:
        tiger = yaml.safe_load(f)
    with (ROOT / "GRID/configs/experiment/tiger_inference_flat.yaml").open() as f:
        tiger_inference = yaml.safe_load(f)

    assert rkmeans["trainer"]["max_steps"] == 30
    assert rkmeans["trainer"]["max_epochs"] == 10
    assert rkmeans["trainer"]["devices"] == -1
    assert tiger["trainer"]["max_steps"] == 320000
    assert tiger["trainer"]["max_epochs"] == 10
    assert tiger["trainer"]["devices"] == -1
    assert tiger["sequence_length"] == 160
    assert tiger_inference["sequence_length"] == 160
    train_dataset = tiger["data_loading"]["train_dataloader_config"]["dataloader"]["dataset_config"]
    assert train_dataset["min_sequence_length"] == '${math_eval:"2 * ${model.num_hierarchies}"}'
    preprocessing = train_dataset["preprocessing_functions"]
    assert any(step.get("_target_", "").endswith("filter_sequence_length_row") for step in preprocessing)


def test_rvq_optimizer_is_instantiable():
    with (ROOT / "GRID/configs/experiment/rvq_train_flat.yaml").open() as f:
        rvq = yaml.safe_load(f)
    optimizer = rvq["optim"]["optimizer"]
    assert optimizer["_target_"] == "torch.optim.Adam"
    assert optimizer["_partial_"] is True
    assert optimizer["lr"] == 0.001


def test_rqvae_inference_architecture_matches_training():
    with (ROOT / "GRID/configs/experiment/rqvae_train_flat.yaml").open() as f:
        training = yaml.safe_load(f)
    with (ROOT / "GRID/configs/experiment/rqvae_inference_flat.yaml").open() as f:
        inference = yaml.safe_load(f)

    for component in ("encoder", "decoder"):
        assert inference["model"][component]["input_dim"] == training["model"][component]["input_dim"]
        assert inference["model"][component]["hidden_dim_list"] == training["model"][component]["hidden_dim_list"]
        assert inference["model"][component]["output_dim"] == training["model"][component]["output_dim"]
    assert inference["model"]["quantization_layer"]["n_features"] == training["model"]["quantization_layer"]["n_features"]


def test_fairdiverse_adapter_points_to_embedded_grid():
    source = (ROOT / "fairdiverse/recommendation/base_model/grid_model.py").read_text()
    with (ROOT / "fairdiverse/recommendation/properties/models/GRID.yaml").open() as f:
        model_defaults = yaml.safe_load(f)
    assert '_REPOSITORY_ROOT / "GRID"' in source
    assert '"../../../../GRID"' not in source
    assert "SemanticIDEncoderDecoder" in source
    assert "must live below embedded" in source
    assert 'config.get("sequence_length", 160)' in source
    assert model_defaults["sequence_length"] == 160


def test_native_grid_model_import_does_not_require_tensorflow_iterator():
    interfaces = (
        ROOT / "GRID/src/data/loading/components/interfaces.py"
    ).read_text()
    iterators = (
        ROOT / "GRID/src/data/loading/components/iterators.py"
    ).read_text()
    assert "if TYPE_CHECKING:" in interfaces
    assert "from src.data.loading.components.iterators import RawDataIterator" in interfaces
    assert "except ModuleNotFoundError:" in iterators
    assert "def _require_tensorflow():" in iterators


def test_fairdiverse_loads_native_grid_checkpoint():
    source = (ROOT / "fairdiverse/recommendation/base_model/grid_model.py").read_text()
    assert 'checkpoint_dir.glob("checkpoint_*.ckpt")' in source
    assert "sid_inference_dir = sid_path.parents[1]" in source
    assert "weights_only=False" in source




def test_grid_fair_finetune_is_checkpoint_initialized_and_finally_validated():
    trainer = (ROOT / "fairdiverse/recommendation/trainer.py").read_text()
    model = (ROOT / "fairdiverse/recommendation/base_model/grid_model.py").read_text()
    assert "grid_fair_finetune requires a successfully loaded native checkpoint" in trainer
    assert "max_steps must equal" in trainer
    assert "last_validation_optimizer_step != optimizer_step" in trainer
    assert "Strictly loaded native checkpoint" in model


def test_reweight_score_forward_does_not_retain_a_second_autograd_graph():
    trainer = (ROOT / "fairdiverse/recommendation/trainer.py").read_text()
    score_forward = "with torch.no_grad():\n                            scores = self.Model(feed_user_dict, feed_item_ids)"
    assert score_forward in trainer


def test_grid_adapter_separates_t5_and_collision_suffix_vocabularies():
    model = (ROOT / "fairdiverse/recommendation/base_model/grid_model.py").read_text()
    assert "self.sid_vocabulary_size = max(" in model
    assert "int(codebooks.max().item()) + 1" in model
    assert "vocab_size=self.codebook_size" in model
    assert "embed_tokens=nn.Embedding(self.codebook_size, d_model)" in model
    assert "num_embeddings_per_hierarchy=self.sid_vocabulary_size" in model


def test_slurm_runner_contains_no_plaintext_service_tokens():
    script = (ROOT / "sbatch_run_all_experiments.sh").read_text()
    assert "WANDB_API_KEY=" not in script
    assert "HF_KEY=" not in script
    assert "pip install" not in script
    assert "scripts/slurm/submit_experiments.sh" in script


def test_grid_preprocessing_is_serialized_for_slurm_arrays():
    source = (ROOT / "fairdiverse/recommendation/trainer.py").read_text()
    assert "fcntl.LOCK_EX" in source
    assert ".grid-preprocess.lock" in source
    assert "ensure_grid_fairdiverse_data(self.train_config)" in source


def test_grid_bridge_uses_tfrecord_splits_without_raw_interaction_fallback():
    source = (ROOT / "fairdiverse/recommendation/grid_data_bridge.py").read_text()
    assert '_read_tfrecords(source_dir, "training")' in source
    assert '_read_tfrecords(source_dir, "validation")' in source
    assert '_read_tfrecords(source_dir, "testing")' in source
    assert "val_sequence[:-1] != train_sequence" in source
    assert "test_sequence[:-2] != train_sequence" in source
    assert "tensorflow as tf" not in source
    assert "descriptor_pb2.FileDescriptorProto" in source
    assert "_iter_tfrecord_examples(paths)" in source
    assert "preprocess_amazon_beauty.py" in source
    assert '"--split-strategy"' in source
    assert '"temporal_loo"' in source
    assert '"--min-seq-length"' in source
    assert "source_dir.exists() and any(source_dir.iterdir())" in source


def test_grid_bridge_decodes_tfrecord_without_tensorflow(tmp_path):
    import gzip
    import struct
    from fairdiverse.recommendation.grid_data_bridge import _iter_tfrecord_examples

    # Independent protobuf wire fixture for a tensorflow.train.Example:
    # Example.features -> Features.feature map -> Feature.int64_list.
    def field(number, payload):
        return bytes([(number << 3) | 2, len(payload)]) + payload

    def entry(key, values):
        feature = field(3, field(1, bytes(values)))
        return field(1, field(1, key.encode()) + field(2, feature))

    payload = field(1, entry("user_id", [7]) + entry("sequence_data", [0, 1, 2]))
    path = tmp_path / "fixture.tfrecord.gz"
    with gzip.open(path, "wb") as stream:
        # This parser reads framing, leaving CRC verification to the producer.
        stream.write(struct.pack("<Q", len(payload)) + bytes(4) + payload + bytes(4))
    examples = list(_iter_tfrecord_examples([path]))
    assert len(examples) == 1
    features = examples[0].features.feature
    assert list(features["user_id"].int64_list.value) == [7]
    assert list(features["sequence_data"].int64_list.value) == [0, 1, 2]
