from __future__ import annotations

import pickle
import numpy as np
import torch


class SIDTokenizer:
    """
    Maps integer item IDs to hierarchical Semantic ID (SID) token sequences
    produced by GRID's RQ-KMeans / RQ-VAE offline step.

    Expected sid_path file formats (auto-detected by extension):
    - .pt  : torch.Tensor in FairDiverse layout (num_items, num_hierarchies),
             standalone GRID clean layout (num_hierarchies, num_items), or
             standalone GRID raw layout (num_hierarchies + 1, num_items),
             whose final row is the duplicate/collision indicator
             OR dict {item_id -> List[int]}
    - .pkl / .pickle : same variants as above, serialised with pickle

    The table is stored as a CPU LongTensor and moved to the target device
    on demand via ``SIDTokenizer.to(device)``.
    """

    def __init__(self, sid_path: str, num_hierarchies: int, num_items: int):
        self.num_hierarchies = num_hierarchies
        self.num_items = num_items
        self.sid_table = self._load(sid_path, num_hierarchies, num_items)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _load(sid_path: str, num_hierarchies: int, num_items: int) -> torch.Tensor:
        """Load SID mapping and return a (num_items, num_hierarchies) LongTensor."""
        if sid_path.endswith(".pt"):
            data = torch.load(sid_path, map_location="cpu", weights_only=True)
        elif sid_path.endswith(".pkl") or sid_path.endswith(".pickle"):
            with open(sid_path, "rb") as f:
                data = pickle.load(f)
        else:
            raise ValueError(
                f"Unsupported SID file extension for '{sid_path}'. "
                "Expected .pt, .pkl, or .pickle"
            )

        # --- convert to (num_items, num_hierarchies) LongTensor ---
        if isinstance(data, torch.Tensor):
            table = data.long()
        elif isinstance(data, np.ndarray):
            table = torch.from_numpy(data).long()
        elif isinstance(data, dict):
            table = torch.zeros(num_items, num_hierarchies, dtype=torch.long)
            for item_id, tokens in data.items():
                table[int(item_id)] = torch.tensor(tokens, dtype=torch.long)
        else:
            raise ValueError(
                f"Unsupported SID data type: {type(data)}. "
                "Expected Tensor, ndarray, or dict."
            )

        # Native standalone GRID inference is hierarchy-first. Its raw tensor
        # contains one additional duplicate/collision-indicator row. Normalize
        # both native layouts before applying FairDiverse's item-first checks.
        if table.ndim == 2 and table.shape == (num_hierarchies, num_items):
            table = table.transpose(0, 1).contiguous()
        elif table.ndim == 2 and table.shape == (num_hierarchies + 1, num_items):
            table = table[:num_hierarchies].transpose(0, 1).contiguous()

        if table.shape != (num_items, num_hierarchies):
            # Allow the table to be larger (padded datasets); truncate if needed.
            if table.shape[0] >= num_items and table.shape[1] == num_hierarchies:
                table = table[:num_items]
            else:
                raise ValueError(
                    f"SID table shape mismatch: got {tuple(table.shape)}, "
                    f"expected ({num_items}, {num_hierarchies})."
                )

        return table

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def encode(self, item_ids: torch.Tensor) -> torch.Tensor:
        """
        Return the SID token sequences for the given item IDs.

        :param item_ids: LongTensor of arbitrary shape (...,)
        :return: LongTensor of shape (..., num_hierarchies)
        """
        return self.sid_table[item_ids]

    def to(self, device):
        """Move the SID table to ``device`` in-place (idempotent)."""
        self.sid_table = self.sid_table.to(device)
        return self
