"""Append GRID's collision-disambiguation hierarchy to a SID tensor."""

import argparse
import hashlib
from pathlib import Path

import torch


def deduplicate_sids(sids: torch.Tensor) -> torch.Tensor:
    if sids.ndim != 2:
        raise ValueError(f"Expected a 2-D SID tensor, got shape {tuple(sids.shape)}")

    _, inverse_indices, counts = torch.unique(
        sids, dim=0, return_inverse=True, return_counts=True
    )
    collision_tokens = torch.zeros_like(inverse_indices)

    for duplicate_index in torch.where(counts > 1)[0]:
        item_indices = torch.where(inverse_indices == duplicate_index)[0]
        collision_tokens[item_indices] = torch.arange(
            1, item_indices.numel() + 1, dtype=collision_tokens.dtype
        )

    return torch.cat((sids.long(), collision_tokens.unsqueeze(1)), dim=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--codebook-size", type=int, default=256)
    args = parser.parse_args()

    sids = torch.load(args.input, map_location="cpu", weights_only=True)
    deduplicated = deduplicate_sids(sids)
    max_collision_token = int(deduplicated[:, -1].max())
    if max_collision_token >= args.codebook_size:
        raise ValueError(
            f"Collision token {max_collision_token} exceeds codebook size "
            f"{args.codebook_size}"
        )
    if torch.unique(deduplicated, dim=0).shape[0] != deduplicated.shape[0]:
        raise RuntimeError("Deduplicated SID tensor still contains collisions")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(deduplicated, args.output)
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(
        f"Saved {tuple(deduplicated.shape)} collision-free SIDs to {args.output}\n"
        f"max_collision_token={max_collision_token}\nsha256={digest}"
    )


if __name__ == "__main__":
    main()
