import torch
import torch.nn as nn
import numpy as np
from .abstract_model import AbstractBaseModel


r"""
GRU4Rec
################################################

Reference:
    Yong Kiam Tan et al. "Improved Recurrent Neural Networks for Session-based Recommendations." in DLRS 2016.

"""

class GRU4Rec(AbstractBaseModel):
    def __init__(self, config):
        super().__init__(config)
        # self.user_embedding = nn.Embedding(num_users, embedding_dim)
        # self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.type = ["sequential"]
        self.IR_type = ["retrieval", "ranking"]

        # Catalog item zero is real in GRID-aligned datasets. Keep padding in
        # a private row without changing candidate or target IDs.
        self.padding_item_id = int(config.get('padding_item_id', 0))
        self.padding_embedding_id = int(config['item_num'])
        self.item_embedding = nn.Embedding(
            self.padding_embedding_id + 1, config['embedding_size'],
            padding_idx=self.padding_embedding_id,
        )

        self.emb_dropout = nn.Dropout(self.config['dropout_prob'])
        self.gru_layers = nn.GRU(
            input_size=self.config['embedding_size'],
            hidden_size=self.config['hidden_size'],
            num_layers=self.config['num_layers'],
            bias=False,
            batch_first=True,
        )
        self.dense = nn.Linear(self.config['hidden_size'], self.config['embedding_size'])

        self.sigmoid = nn.Sigmoid()
        self.loss = nn.CrossEntropyLoss(reduction='none')
        self.apply(self._init_weights)
        with torch.no_grad():
            self.item_embedding.weight[self.padding_embedding_id].zero_()



    def compute_loss(self, interaction):

        #user = interaction['user_ids']
        #history_ids = interaction['history_ids']
        pos_items = interaction['item_ids']
        user_emb = self.get_user_embedding(interaction)
        test_item_emb = self.item_embedding.weight[:self.padding_embedding_id]
        logits = torch.matmul(user_emb, test_item_emb.transpose(0, 1))
        loss = self.loss(logits, pos_items)
        return loss

    def get_user_embedding(self, user_dict):
        users = user_dict['history_ids']
        item_mask = users.ne(self.padding_item_id)
        safe_users = users.masked_fill(~item_mask, self.padding_embedding_id)
        item_seq_emb = self.item_embedding(safe_users)
        item_seq_emb_dropout = self.emb_dropout(item_seq_emb)
        gru_output, _ = self.gru_layers(item_seq_emb_dropout)
        gru_output = self.dense(gru_output)
        # Shared loaders right-pad histories. Select the last real timestep,
        # so adding padding cannot change the sequence representation.
        positions = torch.arange(users.size(1), device=users.device)
        last_position = positions.unsqueeze(0).expand_as(users).masked_fill(
            ~item_mask, -1
        ).max(dim=1).values
        seq_output = self.gather_indexes(gru_output, last_position.clamp_min(0))
        seq_output = seq_output.masked_fill(last_position.lt(0).unsqueeze(1), 0)
        #seq_output = self.gather_indexes(gru_output, self.config['history_length'] - 1)
        return seq_output



    def forward(self, user_dict, item_id):
        ###here user_id denotes the historical ids
        user_embeds = self.get_user_embedding(user_dict)
        item_embeds = self.item_embedding(item_id)
        dot_product = (user_embeds * item_embeds).sum(1)
        return self.sigmoid(dot_product)

