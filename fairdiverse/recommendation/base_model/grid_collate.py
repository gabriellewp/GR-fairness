"""
Causal sub-sequence data augmentation for GRID training in FairDiverseSimulation.

Mirrors the original GRID's ``collate_with_sid_causal_duplicate`` strategy from:
  GRID/src/data/loading/components/collate_functions.py

Key difference vs original GRID:
  - Original GRID operates on pre-converted SID token sequences (item × H tokens per item).
  - FairDiverseSimulation operates on raw item-ID sequences; SID conversion happens
    inside GRIDBaseModel at forward time via SIDTokenizer.encode().
  So the augmentation here works at the item-ID level, not the SID-token level.
  Across the per-interaction rows for an L-item user sequence, this produces
  the same O(L²) contiguous windows as standalone GRID.
"""

from __future__ import annotations

from typing import List, Tuple

import torch


def collate_fn_grid_causal_augmentation(
    batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    max_batch_size: int = 512,
    padding_value: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate causal sub-sequences from per-interaction histories for GRID training.

    For each (user, history, target, group) sample, create every suffix of the
    history and keep the row's target fixed. FairDiverse already materializes
    one row per chronological endpoint, so varying only the start position
    reproduces GRID's contiguous windows without duplicating older targets.

        end_indices  = range(2*H, len+1, H)       # original: SID tokens
        start_indices = range(0, end-2*H+1, H)

    Translated to item-IDs (H=1 per item in this representation):

        for each row history=[i0, ..., iL-1], target=iL:
          for start in [0, ..., L-1]:
            yield history=[start, ..., iL-1], target=iL

    Args:
        batch: List of (user_id, history_ids, item_id, group_id) tuples,
               one per sample.  Each element is a scalar/1-D LongTensor from
               SequentialDataset.__getitem__.
        max_batch_size: Maximum number of augmented sequences to return.
                        If the total generated count exceeds this, a random
                        subset is selected.  Mirrors original GRID's
                        ``max_batch_size=512`` parameter.
    Returns:
        Tuple of stacked tensors (user_ids, history_ids, item_ids, group_ids)
        with batch dimension = min(total_generated, max_batch_size).
    """
    augmented_users: List[torch.Tensor] = []
    augmented_histories: List[torch.Tensor] = []
    augmented_items: List[torch.Tensor] = []
    augmented_groups: List[torch.Tensor] = []

    for user_id, history_ids, item_id, group_id in batch:
        # history_ids: 1-D LongTensor of shape (history_length,)
        # Padding, if present, uses padding_value. Real item id 0 is valid and
        # must not be treated as padding.
        L = history_ids.shape[0]

        # Find the last non-padding position so that we only augment over real items.
        # For full-length training data this is usually L; for shorter padded
        # histories we still handle it correctly.
        non_padding = (history_ids != padding_value).nonzero(as_tuple=False)
        if non_padding.numel() == 0:
            # GRID requires at least a history item and a target item.
            continue

        # Histories are right-padded, so real items occupy [0, actual_len).
        last_real_pos = non_padding[-1, 0].item()  # last non-padding index
        actual_len = last_real_pos + 1

        real_history = history_ids[:actual_len]

        # Across FairDiverse's per-interaction rows, these suffixes are exactly
        # GRID's (start, end) windows, with each endpoint represented once.
        for start_item in range(actual_len):
            sub = real_history[start_item:]
            if sub.shape[0] < L:
                pad_len = L - sub.shape[0]
                sub = torch.cat([sub, sub.new_full((pad_len,), padding_value)])

            augmented_users.append(user_id)
            augmented_histories.append(sub)
            augmented_items.append(item_id)
            augmented_groups.append(group_id)

    total = len(augmented_users)
    if total == 0:
        # Fallback: return original batch unstacked (shouldn't happen in practice)
        user_ids   = torch.stack([b[0] for b in batch])
        hist_ids   = torch.stack([b[1] for b in batch])
        item_ids   = torch.stack([b[2] for b in batch])
        group_ids  = torch.stack([b[3] for b in batch])
        return user_ids, hist_ids, item_ids, group_ids

    # Random sampling if the pool exceeds max_batch_size.
    # Uses the same strategy as original GRID's randint-based selection.
    if total > max_batch_size:
        selected = torch.randperm(total, dtype=torch.long)[:max_batch_size].tolist()
        augmented_users     = [augmented_users[i]     for i in selected]
        augmented_histories = [augmented_histories[i] for i in selected]
        augmented_items     = [augmented_items[i]     for i in selected]
        augmented_groups    = [augmented_groups[i]    for i in selected]

    return (
        torch.stack(augmented_users),     # (B, )
        torch.stack(augmented_histories), # (B, history_length)
        torch.stack(augmented_items),     # (B, )
        torch.stack(augmented_groups),    # (B, )
    )
