import torch
import torch.nn as nn
import numpy as np
from .abstract_model import AbstractBaseModel
from .layers import TransformerEncoder

r"""
GRU4Rec
################################################

Reference:
    Wang-Cheng Kang et al. "Self-Attentive Sequential Recommendation." in ICDM 2018.

"""

class SASRec(AbstractBaseModel):
    def __init__(self, config):
        super().__init__(config)
        # self.user_embedding = nn.Embedding(num_users, embedding_dim)
        # self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.type = ["sequential"]
        self.IR_type = ["retrieval", "ranking"]

        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]
        self.hidden_size = config["embedding_size"]  # same as embedding_size
        self.inner_size = config[
            "inner_size"
        ]  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.attn_dropout_prob = config["attn_dropout_prob"]
        self.hidden_act = config["hidden_act"]
        self.layer_norm_eps = float(config["layer_norm_eps"])

        self.initializer_range = config["initializer_range"]

        self.history_length = config['history_length']

        # FairDiverse's legacy datasets use item 0 as padding, but the
        # GRID-aligned catalog uses every ID in [0, item_num) and pads with -1.
        # Embedding cannot consume a negative index, and mapping -1 back to 0
        # would turn a real catalog item into padding.  Reserve one private row
        # after the catalog for sequence padding instead; candidate/target IDs
        # remain unchanged.
        self.padding_item_id = int(config.get('padding_item_id', 0))
        self.padding_embedding_id = int(config['item_num'])
        self.item_embedding = nn.Embedding(
            int(config['item_num']) + 1,
            self.hidden_size,
            padding_idx=self.padding_embedding_id,
        )

        self.position_embedding = nn.Embedding(self.history_length, self.hidden_size)
        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        self.sigmoid = nn.Sigmoid()
        self.loss = nn.CrossEntropyLoss(reduction='none')
        self.apply(self._init_weights)
        with torch.no_grad():
            self.item_embedding.weight[self.padding_embedding_id].zero_()


    # def forward(self, user_ids, item_ids):
    #     user_embeds = self.user_embedding(user_ids)
    #     item_embeds = self.item_embedding(item_ids)
    #     dot_product = (user_embeds * item_embeds).sum(1)
    #     return self.sigmoid(dot_product)


    def compute_loss(self, interaction):

        #user = interaction['user_ids']
        #history_ids = interaction['history_ids']
        pos_items = interaction['item_ids']
        user_emb = self.get_user_embedding(interaction)
        # The final embedding row is sequence padding, not a recommendable item.
        test_item_emb = self.item_embedding.weight[:self.config['item_num']]
        logits = torch.matmul(user_emb, test_item_emb.transpose(0, 1))
        loss = self.loss(logits, pos_items)
        return loss

    def get_user_embedding(self, user_dict):
        users = user_dict['history_ids']
        item_mask = users.ne(self.padding_item_id)
        safe_users = users.masked_fill(~item_mask, self.padding_embedding_id)
        item_seq_emb = self.item_embedding(safe_users)
        ids = torch.arange(
            self.history_length, dtype=torch.long, device=users.device
        )
        position_ids = ids.repeat(users.size(0), 1)
        position_embedding = self.position_embedding(position_ids)
        input_emb = item_seq_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        # MultiHeadAttention expects an additive mask broadcastable to
        # [batch, heads, query_length, key_length].
        attention_mask = (~item_mask).to(input_emb.dtype)[:, None, None, :] * -10000.0
        trm_output = self.trm_encoder(
            input_emb, attention_mask, output_all_encoded_layers=True
        )
        output = trm_output[-1]
        # Histories are right-padded, so the last tensor position is frequently
        # padding.  Represent the user with the last real interaction instead.
        last_real_index = item_mask.long().sum(dim=1).sub(1).clamp_min(0)
        seq_output = self.gather_indexes(output, last_real_index)
        #seq_output = self.gather_indexes(gru_output, self.config['history_length'] - 1)
        return seq_output



    def forward(self, user_dict, item_id):
        ###here user_id denotes the historical ids
        user_embeds = self.get_user_embedding(user_dict)
        item_embeds = self.item_embedding(item_id)
        dot_product = (user_embeds * item_embeds).sum(1)
        return self.sigmoid(dot_product)

    @torch.no_grad()
    def full_predict_batch(self, user_dict, items, generation_batch_size):
        """Score users together, retaining full_predict's sigmoid and row order.

        Chunked elementwise dot products retain the legacy score reduction;
        a GEMM could change rounding near tied scores. Catalog embeddings are
        shared across users rather than gathered afresh for every request.
        Returned scores are on CPU, as in GRID's evaluator batching interface.
        """
        if self.training:
            raise ValueError("Batched SASRec prediction requires evaluation mode")
        if generation_batch_size < 1:
            raise ValueError("generation_batch_size must be positive")
        chunk_size = int(self.config.get("classic_eval_item_chunk_size", 2048))
        if chunk_size < 1:
            raise ValueError("classic_eval_item_chunk_size must be positive")
        histories = user_dict["history_ids"]
        if items.ndim != 2 or items.shape[0] != histories.shape[0]:
            raise ValueError("Expected one candidate row per history")
        if items.numel() and (items.min() < 0 or items.max() >= self.config["item_num"]):
            raise ValueError("Candidates must be real catalog item IDs")
        result = torch.empty(items.shape, dtype=self.item_embedding.weight.dtype)
        for start in range(0, histories.shape[0], generation_batch_size):
            end = min(start + generation_batch_size, histories.shape[0])
            feed = {key: value[start:end] for key, value in user_dict.items()}
            users = self.get_user_embedding(feed)
            scores = users.new_empty((end - start, self.config["item_num"]))
            for offset in range(0, self.config["item_num"], chunk_size):
                embeddings = self.item_embedding.weight[offset:min(
                    offset + chunk_size, self.config["item_num"])]
                scores[:, offset:offset + len(embeddings)] = self.sigmoid(
                    (users[:, None, :] * embeddings[None, :, :]).sum(-1)
                )
            result[start:end] = scores.gather(1, items[start:end].to(users.device)).cpu()
        return result
