"""SASRec with SPRINT spectral regularization.

This is a native FairDiverse adaptation of the public GenRec SPRINT baseline:
https://github.com/Tiny-Snow/GenRec
"""

import torch

from .sasrec import SASRec


def _spectral_norm(module, weight, name, iterations=1, eps=1e-12):
    """Estimate a matrix spectral norm while retaining gradients for ``weight``."""
    if weight.ndim != 2:
        raise ValueError("SPRINT spectral norm expects a matrix")

    rows, columns = weight.shape
    u_name = f"_{name}_sprint_u"
    v_name = f"_{name}_sprint_v"
    if not hasattr(module, u_name):
        u = torch.randn(rows, device=weight.device, dtype=weight.dtype)
        v = torch.randn(columns, device=weight.device, dtype=weight.dtype)
        module.register_buffer(u_name, u / u.norm().clamp_min(eps))
        module.register_buffer(v_name, v / v.norm().clamp_min(eps))

    u = getattr(module, u_name).detach().clone()
    v = getattr(module, v_name).detach().clone()
    for _ in range(iterations):
        u = weight @ v
        u = u / u.norm().clamp_min(eps)
        v = weight.transpose(0, 1) @ u
        v = v / v.norm().clamp_min(eps)

    with torch.no_grad():
        getattr(module, u_name).copy_(u)
        getattr(module, v_name).copy_(v)
    return (weight @ v).norm()


def _attention_norm(attention_probs, valid_mask, temperature):
    """Smooth the attention matrix one-norm upper bound for every head."""
    pair_mask = valid_mask[:, None, :, None] & valid_mask[:, None, None, :]
    attention_probs = attention_probs.masked_fill(~pair_mask, 0.0)
    column_sums = attention_probs.sum(dim=-2).permute(1, 0, 2).flatten(1)
    valid_columns = valid_mask.flatten()
    if not valid_columns.any():
        return column_sums.new_zeros(column_sums.size(0))
    return torch.logsumexp(
        column_sums[:, valid_columns] * temperature, dim=-1
    ) / temperature


class SASRecSPRINT(SASRec):
    """Fairness-aware SASRec baseline from GenRec's SPRINT implementation."""

    def __init__(self, config):
        super().__init__(config)
        self.sprint_attention_weight = float(config["sprint_attention_weight"])
        self.sprint_ffn_weight = float(config["sprint_ffn_weight"])
        self.sprint_embedding_weight = float(config["sprint_embedding_weight"])
        self.sprint_attention_temperature = float(
            config["sprint_attention_temperature"]
        )
        self.sprint_spectral_norm_iters = int(config["sprint_spectral_norm_iters"])

    def _encode_with_attentions(self, user_dict):
        users = user_dict["history_ids"]
        valid_mask = users.ne(self.padding_item_id)
        safe_users = users.masked_fill(~valid_mask, self.padding_embedding_id)
        item_embeddings = self.item_embedding(safe_users)
        positions = torch.arange(
            self.history_length, dtype=torch.long, device=users.device
        ).repeat(users.size(0), 1)
        hidden = self.dropout(
            self.LayerNorm(item_embeddings + self.position_embedding(positions))
        )
        attention_mask = (~valid_mask).to(hidden.dtype)[:, None, None, :] * -10000.0
        layers, attentions = self.trm_encoder(
            hidden,
            attention_mask,
            output_all_encoded_layers=True,
            output_attentions=True,
        )
        last_real = valid_mask.long().sum(dim=1).sub(1).clamp_min(0)
        return self.gather_indexes(layers[-1], last_real), attentions, valid_mask

    def sprint_regularization(self, attentions, valid_mask):
        iterations = self.sprint_spectral_norm_iters
        embedding_norm = _spectral_norm(
            self,
            self.item_embedding.weight[: self.config["item_num"]],
            "item_embedding",
            iterations,
        ).log1p()

        attention_loss = embedding_norm.new_zeros(())
        ffn_loss = embedding_norm.new_zeros(())
        head_dim = self.hidden_size // self.n_heads
        for layer_index, (layer, attention_probs) in enumerate(
            zip(self.trm_encoder.layer, attentions)
        ):
            attention = layer.multi_head_attention
            attention_weight_norm = _attention_norm(
                attention_probs, valid_mask, self.sprint_attention_temperature
            )
            output_norm = _spectral_norm(
                layer, attention.dense.weight, f"layer_{layer_index}_attn_out", iterations
            )
            value_weights = attention.value.weight.view(
                self.n_heads, head_dim, self.hidden_size
            )
            weighted_value_norm = embedding_norm.new_zeros(())
            for head_index in range(self.n_heads):
                value_norm = _spectral_norm(
                    layer,
                    value_weights[head_index],
                    f"layer_{layer_index}_value_{head_index}",
                    iterations,
                )
                weighted_value_norm = (
                    weighted_value_norm
                    + attention_weight_norm[head_index] * value_norm.square()
                )
            attention_loss = attention_loss + (
                weighted_value_norm.sqrt() * output_norm
            ).log1p()

            feed_forward = layer.feed_forward
            first_norm = _spectral_norm(
                layer, feed_forward.dense_1.weight, f"layer_{layer_index}_ffn_1", iterations
            )
            second_norm = _spectral_norm(
                layer, feed_forward.dense_2.weight, f"layer_{layer_index}_ffn_2", iterations
            )
            ffn_loss = ffn_loss + (first_norm * second_norm).log1p()

        attention_loss = attention_loss / self.n_layers
        ffn_loss = ffn_loss / self.n_layers
        return (
            self.sprint_attention_weight * attention_loss
            + self.sprint_ffn_weight * ffn_loss
            + self.sprint_embedding_weight * embedding_norm
        )

    def compute_loss(self, interaction):
        user_embedding, attentions, valid_mask = self._encode_with_attentions(interaction)
        item_embeddings = self.item_embedding.weight[: self.config["item_num"]]
        logits = user_embedding @ item_embeddings.transpose(0, 1)
        recommendation_loss = self.loss(logits, interaction["item_ids"])
        return recommendation_loss + self.sprint_regularization(attentions, valid_mask)
