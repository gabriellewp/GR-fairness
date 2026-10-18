#!/usr/bin/env python3
"""Convert GRID tokenizer inference output to a FairDiverse ``sids.pt`` file.

GRID writes semantic IDs as a tensor with shape ``(H + 1, N)``.  The first
``H`` rows contain learned hierarchical codes and the final row contains a
collision/deduplication code.  By default this script retains all ``H + 1``
levels so every item has a unique SID, arranged as a ``torch.long`` tensor
with shape ``(N, H + 1)``.
"""

import argparse
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        type=Path,
        help="GRID merged_predictions_tensor.pt produced by tokenizer inference",
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Destination .pt file (for example GRID/outputs/amazon-beauty/sids.pt)",
    )
    parser.add_argument(
        "--num-hierarchies",
        type=int,
        default=3,
        help="Expected number of learned SID hierarchy levels (default: 3)",
    )
    parser.add_argument(
        "--drop-deduplication-level",
        action="store_true",
        help=(
            "Write only the learned hierarchy levels. This may create duplicate "
            "SIDs and is not recommended for recommendation experiments."
        ),
    )
    return parser.parse_args()


def convert(
    input_path: Path,
    output_path: Path,
    num_hierarchies: int,
    drop_deduplication_level: bool,
) -> None:
    if num_hierarchies < 1:
        raise ValueError("--num-hierarchies must be at least 1")
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    grid_sids = torch.load(input_path, map_location="cpu", weights_only=True)
    if not isinstance(grid_sids, torch.Tensor):
        raise TypeError(
            f"Expected a torch.Tensor, but {input_path} contains "
            f"{type(grid_sids).__name__}"
        )
    if grid_sids.ndim != 2:
        raise ValueError(f"Expected a 2-D tensor, got shape {tuple(grid_sids.shape)}")
    expected_rows = num_hierarchies + 1
    if grid_sids.shape[0] != expected_rows:
        raise ValueError(
            "Unexpected GRID tensor shape "
            f"{tuple(grid_sids.shape)}: expected ({expected_rows}, N) for "
            f"{num_hierarchies} hierarchy rows plus one collision row"
        )

    output_rows = num_hierarchies if drop_deduplication_level else expected_rows
    sids = grid_sids[:output_rows].transpose(0, 1).contiguous().long()
    unique_count = torch.unique(sids, dim=0).shape[0]
    duplicate_count = sids.shape[0] - unique_count

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(sids, output_path)

    print(f"Input:  {input_path} {tuple(grid_sids.shape)} {grid_sids.dtype}")
    print(f"Output: {output_path} {tuple(sids.shape)} {sids.dtype}")
    print(f"Unique SIDs: {unique_count}/{sids.shape[0]}")
    if duplicate_count:
        print(
            f"WARNING: output contains {duplicate_count} duplicate SIDs. "
            "Retain the deduplication level for training."
        )


def main() -> None:
    args = parse_args()
    convert(
        args.input,
        args.output,
        args.num_hierarchies,
        args.drop_deduplication_level,
    )


if __name__ == "__main__":
    main()
