import torch

from fairdiverse.recommendation.base_model.sasrec import SASRec


def _config():
    return {
        "user_num": 2,
        "item_num": 5,
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
    }


def test_sasrec_accepts_negative_padding_and_real_item_zero():
    model = SASRec(_config()).eval()
    interaction = {
        "history_ids": torch.tensor([[0, 2, -1, -1], [1, 3, 4, -1]]),
        "item_ids": torch.tensor([3, 0]),
    }

    loss = model.compute_loss(interaction)

    assert loss.shape == (2,)
    assert torch.isfinite(loss).all()
    assert model.item_embedding.num_embeddings == 6
    assert torch.count_nonzero(model.item_embedding.weight[5]) == 0


def test_sasrec_uses_last_real_position_not_right_padding():
    model = SASRec(_config()).eval()
    short = {"history_ids": torch.tensor([[0, 2, -1, -1]])}
    extended = {"history_ids": torch.tensor([[0, 2, 3, -1]])}

    short_embedding = model.get_user_embedding(short)
    extended_embedding = model.get_user_embedding(extended)

    assert short_embedding.shape == extended_embedding.shape == (1, 4)
    assert not torch.equal(short_embedding, extended_embedding)
