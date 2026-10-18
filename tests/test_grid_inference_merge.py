import importlib.util
import sys
from pathlib import Path

import pytest
import torch


MODULE_PATH = (
    Path(__file__).parents[1] / "GRID" / "src" / "utils" / "tensor_utils.py"
)
GRID_ROOT = Path(__file__).parents[1] / "GRID"


def _merge(rows):
    if str(GRID_ROOT) not in sys.path:
        sys.path.insert(0, str(GRID_ROOT))
    spec = importlib.util.spec_from_file_location("grid_tensor_utils", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.merge_list_of_keyed_tensors_to_single_tensor(
        rows, index_key="id", value_key="embedding"
    )


def test_merge_requires_one_unique_row_per_index():
    rows = [
        {"id": 0, "embedding": torch.tensor([1.0, 2.0])},
        {"id": 0, "embedding": torch.tensor([3.0, 4.0])},
    ]

    with pytest.raises(ValueError, match="duplicate id"):
        _merge(rows)


def test_merge_requires_contiguous_indices():
    rows = [
        {"id": 0, "embedding": torch.tensor([1.0, 2.0])},
        {"id": 2, "embedding": torch.tensor([3.0, 4.0])},
    ]

    with pytest.raises(ValueError, match="must be exactly"):
        _merge(rows)


def test_merge_preserves_index_order():
    rows = [
        {"id": 1, "embedding": torch.tensor([3.0, 4.0])},
        {"id": 0, "embedding": torch.tensor([1.0, 2.0])},
    ]

    assert torch.equal(
        _merge(rows), torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    )
