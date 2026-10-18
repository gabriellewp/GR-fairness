"""Encoder-only SPRINT adaptation for the embedded GRID/TIGER model."""

from contextlib import contextmanager
import math

import torch
from torch import nn

from .ifairlrs_grid import original_module


DEFAULT_SETTINGS = {"attention_weight": 1.0, "ffn_weight": 0.001, "embedding_weight": 0.1,
                    "temperature": 4.0, "iterations": 1}


def resolve_settings(settings):
    if set(settings) - DEFAULT_SETTINGS.keys():
        raise ValueError("Unknown SPRINT setting")
    result = {**DEFAULT_SETTINGS, **settings}
    for key in ("attention_weight", "ffn_weight", "embedding_weight"):
        if not math.isfinite(result[key]) or result[key] < 0:
            raise ValueError("SPRINT penalty weights must be finite and nonnegative")
    if not math.isfinite(result["temperature"]) or result["temperature"] <= 0 or type(result["iterations"]) is not int or result["iterations"] < 1:
        raise ValueError("SPRINT needs positive temperature and positive integer iterations")
    return result


class GridEncoderSPRINT(nn.Module):
    """Regularize SID embeddings and T5 encoder attention/FFN blocks.

    Power-iteration buffers live here, outside the native GRID checkpoint.
    Hooks collect attention probabilities only during the training objective.
    """

    def __init__(self, model, *, attention_weight=1.0, ffn_weight=0.001,
                 embedding_weight=0.1, temperature=4.0, iterations=1):
        super().__init__()
        resolve_settings(dict(attention_weight=attention_weight, ffn_weight=ffn_weight,
                              embedding_weight=embedding_weight, temperature=temperature, iterations=iterations))
        # Do not register model as a child: its parameters remain owned by GRID.
        object.__setattr__(self, "model", model)
        self.attention_weight = attention_weight
        self.ffn_weight = ffn_weight
        self.embedding_weight = embedding_weight
        self.temperature = temperature
        self.iterations = iterations
        self.blocks = list(model.grid.encoder.encoder.encoder.block)
        if not self.blocks:
            raise ValueError("GRID encoder has no T5 blocks")
        for block in self.blocks:
            attention = block.layer[0].SelfAttention
            if not all(hasattr(attention, name) for name in ("v", "o", "n_heads", "key_value_proj_dim")):
                raise ValueError("Unsupported T5 attention architecture")
            self.ffn_matrices(block)
        helpers = original_module("sasrec_sprint")
        self._spectral = helpers._spectral_norm
        self._attention_bound = helpers._attention_norm
        self.attentions = []
        self.valid_mask = None
        self.last_terms = {}

    @staticmethod
    def ffn_matrices(block):
        ffn = block.layer[-1]
        if hasattr(ffn, "DenseReluDense"):
            dense = ffn.DenseReluDense
            if not hasattr(dense, "wi") or not hasattr(dense, "wo"):
                raise ValueError("Gated T5 FFNs need a separate SPRINT adaptation")
            return [dense.wi.weight, dense.wo.weight]
        if hasattr(ffn, "mlp") and hasattr(ffn.mlp, "model"):
            modules = list(ffn.mlp.model)
            if any(not isinstance(m, (nn.Linear, nn.ReLU, nn.Dropout)) for m in modules):
                raise ValueError("Unsupported GRID multi-layer FFN activation/layout")
            weights = [m.weight for m in modules if isinstance(m, nn.Linear)]
            if len(weights) >= 2:
                return weights
        raise ValueError("Unsupported GRID encoder FFN architecture")

    @property
    def enabled(self):
        return any((self.attention_weight, self.ffn_weight, self.embedding_weight))

    def norm(self, weight, name):
        # Random power-vector initialization must not change model/dropout RNG.
        if not hasattr(self, f"_{name}_sprint_u"):
            devices = [weight.device.index if weight.device.index is not None else torch.cuda.current_device()] if weight.is_cuda else []
            with torch.random.fork_rng(devices=devices):
                return self._spectral(self, weight, name, self.iterations)
        return self._spectral(self, weight, name, self.iterations)

    @contextmanager
    def capture(self):
        self.attentions = []
        self.valid_mask = None
        handles = []

        def mask_hook(module, args, kwargs):
            mask = kwargs.get("attention_mask")
            if mask is None:
                raise ValueError("GRID encoder must supply its expanded SID/SEP attention mask")
            self.valid_mask = mask.bool()

        def request_attentions(module, args, kwargs):
            kwargs = dict(kwargs)
            kwargs["output_attentions"] = True
            return args, kwargs

        def collect(module, args, output):
            self.attentions.append(output[-1])

        try:
            if self.attention_weight:
                handles.append(self.model.grid.encoder.register_forward_pre_hook(mask_hook, with_kwargs=True))
                for block in self.blocks:
                    attention = block.layer[0].SelfAttention
                    handles.append(attention.register_forward_pre_hook(request_attentions, with_kwargs=True))
                    handles.append(attention.register_forward_hook(collect))
            yield
        finally:
            for handle in handles:
                handle.remove()

    def penalty(self):
        grid = self.model.grid
        zero = grid.item_sid_embedding_table_encoder.weight.new_zeros(())
        embedding, attention_loss, ffn_loss = zero, zero, zero
        if self.embedding_weight:
            embedding = self.norm(grid.item_sid_embedding_table_encoder.weight, "sid_embedding").log1p()
        if self.attention_weight and (len(self.attentions) != len(self.blocks) or self.valid_mask is None):
            raise RuntimeError("SPRINT did not capture all encoder attention maps")
        for index, block in enumerate(self.blocks):
            if self.attention_weight:
                attention = block.layer[0].SelfAttention
                probabilities = self.attentions[index]
                if probabilities.ndim != 4 or probabilities.shape[-2:] != (self.valid_mask.shape[1], self.valid_mask.shape[1]):
                    raise RuntimeError("Unexpected T5 encoder attention shape")
                bound = self._attention_bound(probabilities, self.valid_mask, self.temperature)
                values = attention.v.weight.reshape(attention.n_heads, attention.key_value_proj_dim, -1)
                weighted_values = zero
                for head in range(attention.n_heads):
                    value_norm = self.norm(values[head], f"encoder_{index}_value_{head}")
                    weighted_values = weighted_values + bound[head] * value_norm.square()
                output_norm = self.norm(attention.o.weight, f"encoder_{index}_output")
                # A clamped square root has finite gradients for all-padding maps.
                attention_loss = attention_loss + (weighted_values.clamp_min(1e-12).sqrt() * output_norm).log1p()
            if self.ffn_weight:
                product = zero.new_ones(())
                for matrix_index, weight in enumerate(self.ffn_matrices(block)):
                    product = product * self.norm(weight, f"encoder_{index}_ffn_{matrix_index}")
                ffn_loss = ffn_loss + product.log1p()
        attention_loss = attention_loss / len(self.blocks)
        ffn_loss = ffn_loss / len(self.blocks)
        self.last_terms = {"embedding": float(embedding.detach()), "attention": float(attention_loss.detach()),
                           "ffn": float(ffn_loss.detach())}
        return self.embedding_weight * embedding + self.attention_weight * attention_loss + self.ffn_weight * ffn_loss

    def objective(self, interaction):
        if not self.enabled:
            return self.model.compute_loss(interaction).mean()
        try:
            with self.capture():
                losses = self.model.compute_loss(interaction)
            return losses.mean() + self.penalty()
        finally:
            # Release the graph; the returned loss retains what backward needs.
            self.attentions = []
            self.valid_mask = None

    def settings(self):
        return {"variant": "sprint_grid_encoder_adaptation", "scope": "encoder_self_attention_and_ffn_plus_sid_embedding",
                "attention_weight": self.attention_weight, "ffn_weight": self.ffn_weight,
                "embedding_weight": self.embedding_weight, "temperature": self.temperature,
                "spectral_norm_iterations": self.iterations, "shared_embedding_policy": "encoder_table_once",
                "ffn_policy": "product_of_all_linear_spectral_norms_per_encoder_block",
                "attention_maps": "native_post_dropout_maps_with_expanded_valid_token_mask"}
