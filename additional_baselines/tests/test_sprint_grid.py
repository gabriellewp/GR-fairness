from copy import deepcopy

import pytest
import torch

from additional_baselines.ifairlrs_grid import original_module, train_branch
from additional_baselines.sprint_grid import GridEncoderSPRINT


@pytest.fixture
def model(tmp_path, monkeypatch):
    module = original_module("grid_model")
    sid = tmp_path / "sid.pt"
    torch.save(torch.arange(6).expand(4, -1).clone(), sid)
    monkeypatch.setattr(module, "_local_grid_artifact", lambda path, **kwargs: sid)
    config = {"num_hierarchies": 4, "codebook_size": 8, "item_num": 6, "sid_path": str(sid),
              "embedding_size": 8, "n_heads": 2, "d_kv": 3, "n_layers": 1,
              "inner_size": 12, "hidden_dropout_prob": 0.0, "sequence_length": 8,
              "top_k_for_generation": 3}
    torch.manual_seed(42)
    return module.GRIDBaseModel(config)


def interaction():
    return {"user_ids": torch.tensor([0, 1]), "history_ids": torch.tensor([[0, -1], [1, 2]]),
            "item_ids": torch.tensor([3, 4])}


def test_native_sprint_gradients_and_no_checkpoint_pollution(model):
    regularizer = GridEncoderSPRINT(model)
    original_keys = set(model.grid.state_dict())
    rng = torch.get_rng_state().clone()
    loss = regularizer.objective(interaction())
    assert torch.equal(rng, torch.get_rng_state())
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    block = model.grid.encoder.encoder.encoder.block[0]
    for parameter in [model.grid.item_sid_embedding_table_encoder.weight, block.layer[0].SelfAttention.v.weight,
                      block.layer[0].SelfAttention.o.weight] + regularizer.ffn_matrices(block):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    assert set(model.grid.state_dict()) == original_keys
    assert regularizer.state_dict()
    assert regularizer.attentions == [] and regularizer.valid_mask is None
    assert not block.layer[0].SelfAttention._forward_hooks
    assert not block.layer[0].SelfAttention._forward_pre_hooks


def test_zero_weights_exact_loss_and_update_parity(model):
    ordinary = deepcopy(model)
    regularizer = GridEncoderSPRINT(model, attention_weight=0, ffn_weight=0, embedding_weight=0)
    assert torch.equal(regularizer.objective(interaction()), ordinary.compute_loss(interaction()).mean())
    batch = (interaction()["user_ids"], interaction()["history_ids"], interaction()["item_ids"], torch.zeros(2, dtype=torch.long))
    kwargs = dict(device="cpu", learning_rate=0.001, weight_decay=0.0001, clip_norm=0)
    a = train_branch(ordinary, [batch], {3: "a", 4: "a"}, {"a": 1}, **kwargs)
    b = train_branch(model, [batch], {3: "a", 4: "a"}, {"a": 1}, regularizer=regularizer, **kwargs)
    assert a == b
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in ordinary.state_dict().items())


def test_expanded_mask_and_attention_only_gradients(model):
    regularizer = GridEncoderSPRINT(model, embedding_weight=0, ffn_weight=0)
    with regularizer.capture():
        model.compute_loss(interaction())
    assert len(regularizer.attentions) == 1
    mask = regularizer.valid_mask
    assert mask[0].sum() < mask[1].sum()
    assert mask[0].sum() >= 4  # Item zero is real, with all four SID tokens valid.
    assert mask.shape[1] == regularizer.attentions[0].shape[-1]
    regularizer.penalty().backward()
    attention = model.grid.encoder.encoder.encoder.block[0].layer[0].SelfAttention
    assert attention.v.weight.grad.abs().sum() > 0
    assert attention.o.weight.grad.abs().sum() > 0


def test_capture_cleanup_after_failure(model):
    regularizer = GridEncoderSPRINT(model)
    with pytest.raises(RuntimeError):
        with regularizer.capture():
            raise RuntimeError("failure")
    assert not model.grid.encoder._forward_pre_hooks
    assert not model.grid.encoder.encoder.encoder.block[0].layer[0].SelfAttention._forward_hooks


@pytest.mark.parametrize("settings", [{"attention_weight": -1}, {"ffn_weight": float("nan")},
                                      {"temperature": 0}, {"iterations": 0}])
def test_invalid_settings(model, settings):
    with pytest.raises(ValueError):
        GridEncoderSPRINT(model, **settings)
