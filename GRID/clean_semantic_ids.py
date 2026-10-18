#!/usr/bin/env python3
"""Prepare tokenizer SIDs for TIGER while preserving native H x N layout."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--tokenizer-hierarchies", type=int, default=3)
    parser.add_argument("--tiger-hierarchies", type=int, default=4)
    parser.add_argument("--codebook-size", type=int, default=256)
    args = parser.parse_args()

    tensor = torch.load(args.input, map_location="cpu", weights_only=False)
    if not isinstance(tensor, torch.Tensor) or tensor.ndim != 2:
        raise ValueError(f"Expected a 2-D tensor, got {type(tensor).__name__}")
    if args.tiger_hierarchies not in {
        args.tokenizer_hierarchies,
        args.tokenizer_hierarchies + 1,
    }:
        raise ValueError(
            "TIGER hierarchies must equal the learned tokenizer hierarchies "
            "or include exactly one deterministic suffix"
        )
    expected_raw_rows = args.tokenizer_hierarchies + 1
    if tensor.shape[0] != expected_raw_rows:
        raise ValueError(
            f"Expected {expected_raw_rows} rows ({args.tokenizer_hierarchies} learned "
            f"plus one suffix), got {tuple(tensor.shape)}"
        )

    clean = tensor[: args.tiger_hierarchies].contiguous().long()
    learned = clean[: args.tokenizer_hierarchies]
    if learned.min().item() < 0 or learned.max().item() >= args.codebook_size:
        raise ValueError(
            f"Learned SID token range [{learned.min().item()}, {learned.max().item()}] is outside "
            f"the codebook [0, {args.codebook_size})"
        )
    if args.tiger_hierarchies == args.tokenizer_hierarchies + 1:
        learned_rows = learned.transpose(0, 1)
        _, inverse_indices, counts = torch.unique(
            learned_rows, dim=0, return_inverse=True, return_counts=True
        )
        expected_suffix = torch.zeros(clean.shape[1], dtype=torch.long)
        for duplicate_index in torch.where(counts > 1)[0]:
            item_indices = torch.where(inverse_indices == duplicate_index)[0]
            expected_suffix[item_indices] = torch.arange(
                1, item_indices.numel() + 1, dtype=torch.long
            )
        if not torch.equal(clean[-1].cpu(), expected_suffix):
            raise ValueError(
                "Raw collision row is not the deterministic suffix implied by "
                "the learned SID levels and canonical item order"
            )
        unique_count = torch.unique(clean.transpose(0, 1), dim=0).shape[0]
        if unique_count != clean.shape[1]:
            raise ValueError(
                f"Collision suffix did not produce unique SIDs: {unique_count}/{clean.shape[1]}"
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(clean, args.output)
    print(
        f"Saved TIGER semantic IDs {tuple(clean.shape)} to {args.output} "
        f"({args.tokenizer_hierarchies} learned + "
        f"{args.tiger_hierarchies - args.tokenizer_hierarchies} suffix)"
    )


if __name__ == "__main__":
    main()
