import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler
import random
import json
import os
import ast
from .utils import Build_Adjecent_Matrix, convert_keys_values_to_int


def _build_padded_history_tensor(rows, pad_len, padding_value=0):
    """Trim to the most recent interactions, then right-pad with the configured padding value."""
    histories = []
    for row in rows:
        history = list(row)
        if len(history) > pad_len:
            history = history[-pad_len:]
        if len(history) < pad_len:
            history = history + [padding_value] * (pad_len - len(history))
        histories.append(torch.tensor(history, dtype=torch.long))
    return torch.stack(histories)


class AbstractDataset(Dataset):
    def __init__(self, df, config):
        """
            Abstract dataset utilized for training

            :param df: The input DataFrame.
            :param config: The merged config file.
        """


        self.config = config
        self.uid_field, self.iid_field, self.pid_field, self.label_field = config['user_id'], config['item_id'], config['group_id'], config['label_id']
        history_column = 'history_behaviors'

        self.M, self.iid2pid = Build_Adjecent_Matrix(config)

        self.user_ids = torch.tensor(df[self.uid_field].values, dtype=torch.long)
        self.item_ids = torch.tensor(df[self.iid_field].values, dtype=torch.long)
        self.label = torch.tensor(df[self.label_field].values, dtype=torch.float32)
        self.group_id = torch.tensor(df[self.pid_field].values, dtype=torch.long)
        pad_len = config.get('history_length') or max((len(row) for row in df[history_column]), default=1)
        padding_value = config.get('padding_item_id', 0)
        self.history_ids = _build_padded_history_tensor(df[history_column], pad_len, padding_value)
        #self.history_ids = torch.tensor(df[history_column].values, dtype=torch.long)


    def __len__(self):
        return len(self.label)

class PointWiseDataset(AbstractDataset):

    def __getitem__(self, idx):
        """
           getitem function for pointwise dataset

           :param idx: The index of the sample
           :return: The utilized features of the dataset
        """
        return self.user_ids[idx], self.history_ids[idx], self.item_ids[idx], self.group_id[idx], self.label[idx]

class SequentialDataset(AbstractDataset):

    def __init__(self, df, config):
        """
          SequentialDataset initialization:we only utilize the positive label and sample some neg labels

          :param df: The input DataFrame.
          :param config: The merged config file.
       """
        filter_df = df[df[config['label_id']] == 1]
        super().__init__(filter_df, config)

    def __getitem__(self, idx):
        """
           getitem function for sequential dataset

           :param idx: The index of the sample
           :return: The utilized features of the sequential dataset
        """
        return self.user_ids[idx], self.history_ids[idx], self.item_ids[idx], self.group_id[idx]


class UserGroupedBatchSampler(Sampler):
    """Batch complete per-user interaction rows while shuffling users each epoch."""

    def __init__(self, user_ids, users_per_batch, seed=0, drop_last=False):
        if users_per_batch <= 0:
            raise ValueError("users_per_batch must be positive")

        self.users_per_batch = users_per_batch
        self.drop_last = drop_last
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)

        self.user_indices = {}
        for index, user_id in enumerate(user_ids.tolist()):
            self.user_indices.setdefault(user_id, []).append(index)
        self.users = list(self.user_indices)

    def __iter__(self):
        order = torch.randperm(len(self.users), generator=self.generator).tolist()
        for start in range(0, len(order), self.users_per_batch):
            user_positions = order[start:start + self.users_per_batch]
            if self.drop_last and len(user_positions) < self.users_per_batch:
                break

            batch_indices = []
            for position in user_positions:
                batch_indices.extend(self.user_indices[self.users[position]])
            yield batch_indices

    def __len__(self):
        user_count = len(self.users)
        if self.drop_last:
            return user_count // self.users_per_batch
        return (user_count + self.users_per_batch - 1) // self.users_per_batch


class PairWiseDataset(AbstractDataset):
    def __init__(self, df, config):
        """
           PairWiseDataset initialization: here we construct the positive samples for each items

           :param df: The input DataFrame.
           :param config: The merged config file.
        """
        filter_df = df[df[config['label_id']] == 1]
        super().__init__(filter_df, config)
        self.user2pos = {}
        for row in filter_df[[self.uid_field, self.iid_field, self.pid_field]].itertuples(index=True):
            #print(row)
            user_id, item_id, pid = row._1, row._2, row._3
            if user_id not in self.user2pos.keys():
                self.user2pos[user_id] = []
            self.user2pos[user_id].append(item_id)
        for key in self.user2pos.keys():
            self.user2pos[key] = list(set(self.user2pos[key]))


    def __getitem__(self, idx):
        """
          getitem function for sequential dataset with negative sampling operations

          :param idx: The index of the sample
          :return: The utilized features of the sequential dataset
       """
        user_id = int(self.user_ids[idx].numpy())
        neg_item = 0
        while neg_item in self.user2pos[user_id]:
            neg_item = random.randint(0, self.config['item_num']-1)

        neg_group = torch.tensor(self.iid2pid[neg_item], dtype=torch.long)
        neg_item = torch.tensor(neg_item, dtype=torch.long)

        return self.user_ids[idx], self.history_ids[idx], self.item_ids[idx], neg_item, self.group_id[idx], neg_group
        #return self.user_ids[idx], self.item_ids[idx], self.label[idx], self.group_id[idx]

class RankingTestDataset(Dataset):
    def __init__(self, df, config):
        with open(os.path.join("recommendation", "processed_dataset", config['dataset'], "iid2pid.json"), "r") as file:
            self.item2pid = json.load(file)
        self.user_ids = torch.tensor(df["user_id"].values, dtype=torch.long)
        self.compact_full_catalog = bool(
            config.get('grid_compact_ranking', False) and 'target_item' in df.columns
        )
        if self.compact_full_catalog:
            self.target_items = torch.tensor(df["target_item"].values, dtype=torch.long)
            self.item_num = int(config['item_num'])
            self.items = None
            self.pos_length = torch.ones(len(df), dtype=torch.long)
        else:
            df['items'] = df['items'].apply(lambda x: np.array(ast.literal_eval(x)))
            self.items = torch.stack([torch.tensor(row, dtype=torch.long) for row in df["items"]])
            self.pos_length = torch.tensor(df["pos_length"].values, dtype=torch.long)
        pad_len = config.get('history_length') or max((len(row) for row in df["history_behaviors"]), default=1)
        padding_value = config.get('padding_item_id', 0)
        self.history_ids = _build_padded_history_tensor(df["history_behaviors"], pad_len, padding_value)

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        """
             getitem function for RankingDataset dataset under test dataset since in test set we already finish negative samples

             :param idx: The index of the sample
             :return: The utilized features of the RankingDataset
        """
        if self.compact_full_catalog:
            target = self.target_items[idx]
            catalog = torch.arange(self.item_num, dtype=torch.long)
            items = torch.cat((target.view(1), catalog[catalog != target]))
        else:
            items = self.items[idx]
        return self.user_ids[idx], self.history_ids[idx], items, self.pos_length[idx]
