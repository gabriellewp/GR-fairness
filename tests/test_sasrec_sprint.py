import torch

from fairdiverse.recommendation.base_model.sasrec_sprint import SASRecSPRINT


def _config():
    return {
        "user_num": 2,
        "item_num": 6,
        "embedding_size": 4,
        "n_layers": 1,
        "n_heads": 1,
        "inner_size": 8,
        "hidden_dropout_prob": 0.0,
        "attn_dropout_prob": 0.0,
        "hidden_act": "gelu",
        "layer_norm_eps": 1e-12,
        "initializer_range": 0.02,
        "history_length": 4,
        "padding_item_id": -1,
        "sprint_attention_weight": 1.0,
        "sprint_ffn_weight": 0.001,
        "sprint_embedding_weight": 0.1,
        "sprint_attention_temperature": 4.0,
        "sprint_spectral_norm_iters": 1,
    }


def test_sprint_loss_is_finite_and_backpropagates():
    model = SASRecSPRINT(_config())
    interaction = {
        "history_ids": torch.tensor([[0, 2, -1, -1], [1, 3, 4, -1]]),
        "item_ids": torch.tensor([3, 0]),
    }

    loss = model.compute_loss(interaction)

    assert loss.shape == (2,)
    assert torch.isfinite(loss).all()
    loss.mean().backward()
    assert model.item_embedding.weight.grad is not None
    assert torch.isfinite(model.item_embedding.weight.grad).all()


def test_plain_sasrec_transformer_api_remains_unchanged():
    model = SASRecSPRINT(_config()).eval()
    user_embedding = model.get_user_embedding(
        {"history_ids": torch.tensor([[0, 2, -1, -1]])}
    )
    assert user_embedding.shape == (1, 4)
