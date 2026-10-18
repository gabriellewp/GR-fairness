from __future__ import annotations

import inspect
import sys
from functools import partial
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .grid_tokenizer import SIDTokenizer


_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_GRID_ROOT = (_REPOSITORY_ROOT / "GRID").resolve()
if not (_GRID_ROOT / "src" / "__init__.py").is_file():
    raise RuntimeError(f"Embedded GRID implementation not found at {_GRID_ROOT}")
if str(_GRID_ROOT) not in sys.path:
    sys.path.insert(0, str(_GRID_ROOT))

from transformers import T5EncoderModel
from transformers.models.t5.modeling_t5 import T5Config, T5Stack
# Match GRID's own entrypoint import order. Its utility package initializes
# shared interface types before tiger_generation_model is imported.
import src.utils  # noqa: F401, E402
from src.models.modules.semantic_id.tiger_generation_model import (
    SemanticIDEncoderDecoder,
)

# Fail loudly if another checkout populated the generic src package first.
_NATIVE_MODEL_FILE = Path(inspect.getfile(SemanticIDEncoderDecoder)).resolve()
if not _NATIVE_MODEL_FILE.is_relative_to(_GRID_ROOT):
    raise RuntimeError(
        "GRID was imported from outside this source checkout: "
        f"{_NATIVE_MODEL_FILE}. Expected a module below {_GRID_ROOT}."
    )


def _local_grid_artifact(path_value: str, *, must_exist: bool = True) -> Path:
    """Resolve an artifact and reject dependencies on another GRID checkout."""
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = _REPOSITORY_ROOT / path
    path = path.resolve()
    if "description" in path.parts:
        raise ValueError("Description-only GRID artifacts are excluded; use complete_metadata")
    if not path.is_relative_to(_GRID_ROOT):
        raise ValueError(
            f"GRID artifact must live below embedded {_GRID_ROOT}, got {path}"
        )
    if must_exist and not path.is_file():
        raise FileNotFoundError(
            f"Missing local GRID artifact: {path}. Run "
            "GRID/run_independent_pipeline.sh to generate it."
        )
    return path


class GRIDBaseModel(nn.Module):
    """FairDiverse adapter around GRID's native SemanticIDEncoderDecoder.

    The generative model, teacher-forced loss, and beam search come from the
    implementation under this repository's GRID/src. This class only translates
    FairDiverse batches and exposes the FairDiverse model interface.
    """

    type = ["sequential"]
    IR_type = ["ranking", "retrieval"]

    def __init__(self, config: dict) -> None:
        super().__init__()
        if config.get("metadata_text_mode", "complete_metadata") != "complete_metadata":
            raise ValueError("This release requires complete_metadata GRID checkpoints")
        self.config = config
        self.num_hierarchies = int(config["num_hierarchies"])
        self.codebook_size = int(config["codebook_size"])
        self.item_num = int(config["item_num"])
        self.padding_item_id = int(config.get("padding_item_id", -1))
        self.sequence_length = int(config.get("sequence_length", 160))
        self.top_k_for_generation = int(config.get("top_k_for_generation", 10))

        sid_path = _local_grid_artifact(config["sid_path"])
        self.tokenizer = SIDTokenizer(
            str(sid_path), self.num_hierarchies, self.item_num
        )
        codebooks = self.tokenizer.sid_table.transpose(0, 1).contiguous()

        # The standalone suffix-four pipeline has three learned codebook levels
        # followed by a deterministic collision-disambiguation level.  The
        # learned/T5 vocabulary remains ``codebook_size`` (normally 256), but a
        # collision group can require a suffix larger than 255.  Native GRID
        # trains its SID heads and embedding table using max(SID) + 1 in that
        # case.  Keep those two widths separate so the FairDiverse model has the
        # exact same shapes as the native checkpoint.
        sid_min = int(codebooks.min().item())
        if sid_min < 0:
            raise ValueError(
                f"Semantic IDs must be non-negative, found minimum {sid_min}"
            )
        self.sid_vocabulary_size = max(
            self.codebook_size, int(codebooks.max().item()) + 1
        )

        d_model = int(config.get("embedding_size", 128))
        encoder_config = T5Config(
            vocab_size=self.codebook_size,
            d_model=d_model,
            num_heads=int(config.get("n_heads", 6)),
            dropout_rate=float(config.get("hidden_dropout_prob", 0.15)),
            d_ff=int(config.get("inner_size", 1024)),
            d_kv=int(config.get("d_kv", 64)),
            num_layers=int(config.get("n_layers", 4)),
        )
        decoder_config = T5Config(
            vocab_size=self.codebook_size,
            d_model=d_model,
            num_heads=int(config.get("n_heads", 6)),
            dropout_rate=float(config.get("hidden_dropout_prob", 0.15)),
            d_ff=int(config.get("inner_size", 1024)),
            d_kv=int(config.get("d_kv", 64)),
            num_layers=int(config.get("n_layers", 4)),
            is_decoder=True,
            is_encoder_decoder=False,
        )

        self.grid = SemanticIDEncoderDecoder(
            huggingface_model=T5EncoderModel(encoder_config),
            postprocessor=None,
            aggregator=None,
            optimizer=partial(
                torch.optim.Adam,
                lr=float(config.get("learning_rate", 1.0e-3)),
                weight_decay=float(config.get("weight_decay", 1.0e-4)),
            ),
            scheduler=None,
            loss_function=nn.CrossEntropyLoss(reduction="none"),
            evaluator=None,
            weight_tying=bool(config.get("weight_tying", True)),
            compile=False,
            decoder=T5Stack(
                decoder_config,
                embed_tokens=nn.Embedding(self.codebook_size, d_model),
            ),
            feature_to_model_input_map={
                "sequence_data": "input_ids",
                "user_id": "user_id",
            },
            codebooks=codebooks,
            embedding_dim=d_model,
            num_hierarchies=self.num_hierarchies,
            num_embeddings_per_hierarchy=self.sid_vocabulary_size,
            num_user_bins=config.get("num_user_bins"),
            mlp_layers=config.get("mlp_layers", 2),
            should_check_prefix=bool(config.get("use_prefix_validation", False)),
            should_add_sep_token=True,
            top_k_for_generation=self.top_k_for_generation,
        )

        checkpoint_path = config.get("grid_checkpoint_path")
        self.loaded_checkpoint_path: Optional[str] = None
        if checkpoint_path:
            if checkpoint_path == "auto":
                tokenizer_name = config.get(
                    "grid_tokenizer", config.get("tokenizers", "rkmeans")
                )
                if tokenizer_name not in {"rkmeans", "rvq", "rqvae"}:
                    raise ValueError(
                        f"Unsupported grid_tokenizer for auto checkpoint: {tokenizer_name}"
                    )
                # Keep the checkpoint coupled to the SID artifact namespace.
                # A SID produced at
                #   <run>/<tokenizer>/step2b_sid_inference/pickle/<file>
                # must load its checkpoint from the sibling step3 directory.
                # This matters when the same tokenizer is trained from multiple
                # embedding sources (for example T5-XL and CF-SASRec).
                sid_inference_dir = sid_path.parents[1]
                tokenizer_dir = sid_inference_dir.parent
                if (
                    sid_inference_dir.name == "step2b_sid_inference"
                    and tokenizer_dir.name == tokenizer_name
                ):
                    checkpoint_dir = (
                        tokenizer_dir
                        / f"step3_training_suffix{self.num_hierarchies}"
                        / "checkpoints"
                    )
                else:
                    raise ValueError(
                        "grid_checkpoint_path=auto requires sid_path layout "
                        "<run>/<tokenizer>/step2b_sid_inference/pickle/<file>; "
                        f"got {sid_path}"
                    )
                candidates = list(checkpoint_dir.glob("checkpoint_*.ckpt"))
                if not candidates:
                    raise FileNotFoundError(
                        f"No local GRID checkpoints found below {checkpoint_dir}"
                    )
                checkpoint_path = str(
                    max(candidates, key=lambda path: path.stat().st_mtime)
                )
            checkpoint = torch.load(
                _local_grid_artifact(checkpoint_path),
                map_location="cpu",
                weights_only=False,
            )
            state_dict = checkpoint.get("state_dict", checkpoint)
            self.grid.load_state_dict(state_dict, strict=True)
            self.loaded_checkpoint_path = str(_local_grid_artifact(checkpoint_path))
            print(f"[GRID] Strictly loaded native checkpoint: {self.loaded_checkpoint_path}")

        self._last_generated_sids: Optional[torch.Tensor] = None

    def _native_batch(
        self, history_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert item histories to GRID's padded, flattened SID representation."""
        item_mask = history_ids.ne(self.padding_item_id)
        safe_ids = history_ids.masked_fill(~item_mask, 0)
        self.tokenizer.to(history_ids.device)
        sid_table = self.tokenizer.sid_table
        sids = sid_table[safe_ids].reshape(history_ids.shape[0], -1)
        mask = (
            item_mask.unsqueeze(-1)
            .expand(-1, -1, self.num_hierarchies)
            .reshape(history_ids.shape[0], -1)
        )

        current_length = sids.shape[1]
        if current_length > self.sequence_length:
            sids = sids[:, -self.sequence_length :]
            mask = mask[:, -self.sequence_length :]
        elif current_length < self.sequence_length:
            amount = self.sequence_length - current_length
            sids = torch.cat(
                [sids, sids.new_full((sids.shape[0], amount), -1)], dim=1
            )
            mask = torch.cat(
                [mask, mask.new_zeros((mask.shape[0], amount))], dim=1
            )

        return sids.masked_fill(~mask, 0), mask.long()

    def _target_sids(self, item_ids: torch.Tensor) -> torch.Tensor:
        self.tokenizer.to(item_ids.device)
        return self.tokenizer.sid_table[item_ids.long()]

    def _decoder_outputs(
        self,
        history_ids: torch.Tensor,
        item_ids: torch.Tensor,
        user_ids: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        input_ids, attention_mask = self._native_batch(history_ids)
        target_sids = self._target_sids(item_ids)
        output = self.grid(
            attention_mask_encoder=attention_mask,
            input_ids=input_ids,
            user_id=user_ids,
            future_ids=target_sids,
        )
        return output[:, :-1], target_sids

    def compute_loss(self, interaction: dict) -> torch.Tensor:
        output, target_sids = self._decoder_outputs(
            interaction["history_ids"],
            interaction["item_ids"],
            interaction.get("user_ids"),
        )
        loss = output.new_zeros(output.shape[0])
        for hierarchy in range(self.num_hierarchies):
            logits = self.grid.decoder.decoder_mlp[hierarchy](
                output[:, hierarchy]
            )
            loss = loss + nn.functional.cross_entropy(
                logits, target_sids[:, hierarchy].long(), reduction="none"
            )
        return loss

    def forward(self, user_dict: dict, item_ids: torch.Tensor) -> torch.Tensor:
        """Differentiable native sequence likelihood for fairness reweighting."""
        output, target_sids = self._decoder_outputs(
            user_dict["history_ids"], item_ids, user_dict.get("user_ids")
        )
        log_probability = output.new_zeros(output.shape[0])
        for hierarchy in range(self.num_hierarchies):
            logits = self.grid.decoder.decoder_mlp[hierarchy](
                output[:, hierarchy]
            )
            log_probability += nn.functional.log_softmax(logits, dim=-1).gather(
                1, target_sids[:, hierarchy].long().unsqueeze(1)
            ).squeeze(1)
        return log_probability.exp()

    def full_scores(self, user_dict: dict, items: torch.Tensor) -> torch.Tensor:
        """Native target-sequence likelihood for every user/candidate pair."""
        batch_size = user_dict["history_ids"].shape[0]
        scores = [
            self(user_dict, item_id.expand(batch_size))
            for item_id in items
        ]
        return torch.stack(scores, dim=1)

    def full_ranking(
        self, user_dict: dict, items: torch.Tensor, k: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.topk(self.full_scores(user_dict, items), k=k, dim=-1)

    def generate(
        self,
        history_ids: torch.Tensor,
        user_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        input_ids, attention_mask = self._native_batch(history_ids)
        generated_sids, probabilities = self.grid.generate(
            attention_mask=attention_mask,
            input_ids=input_ids,
            user_id=user_ids,
        )
        self._last_generated_sids = generated_sids.detach()
        return generated_sids, probabilities

    def full_predict(self, user_dict: dict, items: torch.Tensor) -> torch.Tensor:
        """Score candidates by exact match with native GRID beam predictions."""
        if items.dim() == 2:
            items = items.squeeze(0)
        generated_sids, probabilities = self.generate(
            user_dict["history_ids"], user_dict.get("user_ids")
        )
        candidate_sids = self._target_sids(items)
        matches = (
            candidate_sids.unsqueeze(1) == generated_sids[0].unsqueeze(0)
        ).all(dim=-1)
        first_match = matches.float().argmax(dim=1)
        scores = probabilities[0, first_match] * matches.any(dim=1)
        return scores.unsqueeze(0)

    def full_predict_batch(
        self, user_dict: dict, items: torch.Tensor, generation_batch_size: int
    ) -> torch.Tensor:
        """Generate in microbatches and match complete SIDs using a catalog index.

        Return CPU scores in each request's original candidate order. Keeping
        all zero-scored catalog items preserves the legacy ranking/tie rules;
        duplicate catalog SIDs receive the first matching beam's probability,
        exactly as in ``full_predict``. The index is not model/checkpoint state.
        """
        if self.training:
            raise ValueError("Batched GRID prediction requires evaluation mode")
        if generation_batch_size < 1:
            raise ValueError("generation_batch_size must be positive")
        histories = user_dict["history_ids"]
        users = user_dict.get("user_ids")
        if items.ndim != 2 or items.shape[0] != histories.shape[0]:
            raise ValueError("Expected one candidate row per history")
        if not hasattr(self, "_catalog_sid_index"):
            index = {}
            for item, sid in enumerate(self.tokenizer.sid_table.cpu().tolist()):
                index.setdefault(tuple(sid), []).append(item)
            self._catalog_sid_index = index

        catalog_scores = torch.zeros(histories.shape[0], self.item_num)
        generated_batches = []
        for start in range(0, histories.shape[0], generation_batch_size):
            end = start + generation_batch_size
            generated, probabilities = self.generate(
                histories[start:end], None if users is None else users[start:end]
            )
            generated = generated.cpu()
            probabilities = probabilities.cpu()
            generated_batches.append(generated)
            for offset, (beams, scores) in enumerate(
                zip(generated.tolist(), probabilities.tolist())
            ):
                seen = set()
                for sid, probability in zip(beams, scores):
                    key = tuple(sid)
                    if key in seen:
                        continue
                    seen.add(key)
                    item_ids = self._catalog_sid_index.get(key)
                    if item_ids is not None:
                        catalog_scores[start + offset, item_ids] = probability
        self._last_generated_sids = torch.cat(generated_batches)
        return catalog_scores.gather(1, items.cpu().long())

    def beam_target_diagnostics(self, target_items: torch.Tensor, batch_index: int = 0) -> dict:
        if self._last_generated_sids is None:
            return {}
        target_sids = self._target_sids(target_items)
        beams = self._last_generated_sids[batch_index].to(target_sids.device)
        result = {
            "beam_target_coverage": float(
                (target_sids.unsqueeze(1) == beams.unsqueeze(0))
                .all(dim=-1)
                .any()
                .item()
            )
        }
        for level in range(1, self.num_hierarchies + 1):
            result[f"beam_prefix_coverage@{level}"] = float(
                (target_sids[:, None, :level] == beams[None, :, :level])
                .all(dim=-1)
                .any()
                .item()
            )
        return result
