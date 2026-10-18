#!/usr/bin/env python3
"""
Extract item embeddings from a trained SASRec (or BPR/MF) checkpoint
and save them as a .pt file compatible with GRID/sbatch_grid_step23.sh --embed_path.

Usage:
    python3 extract_cf_embeddings.py --dataset amazon-beauty --model SASRec-base
    python3 extract_cf_embeddings.py --dataset steam --model SASRec-base
    python3 extract_cf_embeddings.py --dataset amazon-beauty --model SASRec-base --normalize
"""

import argparse
import glob
import json
import os
import sys

import torch
import torch.nn.functional as F


def find_checkpoint(dataset: str, model: str) -> str:
    pattern = f"fairdiverse/recommendation/log/*_{dataset}-{model}/best_model.pth"
    matches = glob.glob(pattern)
    if not matches:
        print(f"ERROR: No checkpoint found matching: {pattern}")
        sys.exit(1)
    # Date directory names are not zero-padded, so lexicographic ordering can
    # select June 9 over June 24. Use checkpoint modification time instead.
    path = max(matches, key=os.path.getmtime)
    print(f"Using checkpoint: {path}")
    return path


def extract(
    dataset: str,
    model: str,
    normalize: bool,
    output_dir: str,
    checkpoint_path: str = None,
    item_mapping_path: str = None,
) -> str:
    ckpt_path = checkpoint_path if checkpoint_path else find_checkpoint(dataset, model)
    if checkpoint_path:
        print(f"Using checkpoint: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    if "item_embedding.weight" not in state_dict:
        print(f"ERROR: 'item_embedding.weight' not found in checkpoint. Keys: {list(state_dict.keys())[:10]}")
        sys.exit(1)

    embeddings = state_dict["item_embedding.weight"]  # (item_num, embedding_size)
    print(f"Extracted item embeddings: shape={embeddings.shape}, dtype={embeddings.dtype}")

    if item_mapping_path:
        with open(item_mapping_path) as stream:
            expected_items = len(json.load(stream))
        if embeddings.ndim != 2:
            raise ValueError(f"Expected a 2-D item embedding table, got {embeddings.shape}")
        if embeddings.shape[0] == expected_items + 1:
            # SASRec reserves its final embedding row for sequence padding;
            # catalog item IDs remain [0, expected_items). GRID must never
            # tokenize that private padding row.
            embeddings = embeddings[:expected_items]
            print("Removed SASRec's final private padding-embedding row.")
        elif embeddings.shape[0] != expected_items:
            raise ValueError(
                f"SASRec checkpoint has {embeddings.shape[0]} embedding rows, "
                f"but {item_mapping_path} contains {expected_items} items"
            )
        print(f"Verified embedding rows against {item_mapping_path}: {expected_items} items")

    if normalize:
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        print("Normalized embeddings to unit length.")

    os.makedirs(output_dir, exist_ok=True)
    suffix = "_norm" if normalize else ""
    model_tag = model.lower().replace("-", "_")
    out_path = os.path.join(output_dir, f"embeddings_cf_{model_tag}{suffix}.pt")

    # Save as dict matching GRID's expected format (merged_predictions_tensor.pt)
    torch.save(embeddings, out_path)
    print(f"Saved to: {out_path}")
    print(f"\nTo use with GRID tokenizer:")
    print(f"  sbatch --array=0-2 GRID/sbatch_grid_step23.sh \\")
    print(f"    --dataset {dataset} \\")
    print(f"    --embed_model cf_{model_tag}{suffix} \\")
    print(f"    --embed_path {os.path.abspath(out_path)}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="amazon-beauty", help="Dataset name")
    parser.add_argument("--model", default="SASRec-base", help="Model log suffix (e.g. SASRec-base, BPR-base)")
    parser.add_argument("--normalize", action="store_true", help="L2-normalize embeddings (cosine similarity)")
    parser.add_argument("--checkpoint_path", default=None, help="Direct path to best_model.pth (bypasses auto-detection)")
    parser.add_argument("--output_dir", default=None, help="Output directory (default: GRID/logs/inference/runs/cf_<model>/<dataset>/pickle/)")
    parser.add_argument("--item_mapping_path", default=None,
                        help="Optional GRID item_id_mapping.json used to validate row count")
    args = parser.parse_args()

    out_dir = args.output_dir or f"GRID/logs/inference/runs/cf_{args.model.lower().replace('-','_')}/{args.dataset}/pickle"
    extract(
        args.dataset,
        args.model,
        args.normalize,
        out_dir,
        checkpoint_path=args.checkpoint_path,
        item_mapping_path=args.item_mapping_path,
    )
