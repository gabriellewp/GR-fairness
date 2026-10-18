#!/usr/bin/env python3
"""
prepare_grid_sids.py
====================
Bridge script: FairDiverseSimulation processed dataset → GRID Semantic ID (SID) file.

What it does
------------
1. Reads ``iid2text.json``  from a FairDiverseSimulation processed dataset directory.
   Format: {str(internal_item_id): "item text ..."}
2. Generates dense embeddings for every item using a sentence-transformer model.
3. Runs Residual KMeans (H levels, sklearn MiniBatchKMeans) to learn codebooks and
   assign a H-digit SID tuple to every item.
4. Saves:
   - ``<output_dir>/sids.pt``          — LongTensor (item_num, H), row i = SID of item i
   - ``<output_dir>/sids_info.json``   — metadata (codebook_size, num_hierarchies, …)

The resulting ``sids.pt`` is the ``sid_path`` you set in the GRID model config.

Usage
-----
# Steam dataset (already pre-processed):
python prepare_grid_sids.py --dataset steam

# Amazon Beauty dataset (already pre-processed):
python prepare_grid_sids.py --dataset amazon-beauty

# Custom options:
python prepare_grid_sids.py \\
    --dataset steam \\
    --num_hierarchies 3 \\
    --codebook_size 256 \\
    --embed_model sentence-transformers/all-MiniLM-L6-v2 \\
    --output_dir GRID/outputs/steam \\
    --device cuda

Dependencies
------------
    pip install sentence-transformers scikit-learn torch tqdm
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_item_texts(dataset_dir: str) -> dict:
    """Return {item_id(int): text(str)} from iid2text.json."""
    path = os.path.join(dataset_dir, "iid2text.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"iid2text.json not found in {dataset_dir}.\n"
            "Run FairDiverseSimulation preprocessing first:\n"
            "   python fairdiverse/main.py  (with reprocess: True)"
        )
    with open(path) as f:
        raw = json.load(f)
    # keys may be strings; convert to int
    return {int(k): v for k, v in raw.items()}


def load_dataset_stats(dataset_dir: str) -> dict:
    path = os.path.join(dataset_dir, "dataset_stats.json")
    with open(path) as f:
        return json.load(f)


def embed_items(
    item_texts: dict,
    item_num: int,
    model_name: str,
    device: str,
    batch_size: int = 256,
) -> np.ndarray:
    """
    Embed all items and return a float32 numpy array of shape (item_num, embed_dim).
    Items with no text entry are left as zero vectors.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("ERROR: sentence-transformers not installed.")
        print("       pip install sentence-transformers")
        sys.exit(1)

    print(f"Loading embedding model: {model_name} ...")
    model = SentenceTransformer(model_name, device=device)

    # Build ordered list; item 0 is typically a padding placeholder
    ids_sorted = sorted(item_texts.keys())
    texts = [item_texts[i] for i in ids_sorted]

    print(f"Embedding {len(texts)} items (batch_size={batch_size}) ...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # unit sphere → better for cosine / dot-product
    )  # (len(texts), D)

    # Place into (item_num, D) array indexed by item_id
    D = embeddings.shape[1]
    table = np.zeros((item_num, D), dtype=np.float32)
    for idx, iid in enumerate(ids_sorted):
        if iid < item_num:
            table[iid] = embeddings[idx]

    return table  # (item_num, D)


def residual_kmeans(
    embeddings: np.ndarray,  # (N, D)
    num_hierarchies: int,
    codebook_size: int,
    max_iter: int = 300,
    random_state: int = 42,
) -> np.ndarray:
    """
    Residual KMeans: assign each item an H-tuple (c_0, c_1, ..., c_{H-1}).

    Level 0  clusters the original embeddings.
    Level h  clusters the residuals after subtracting level h-1 centroids.

    Returns
    -------
    sids : np.ndarray of shape (N, H), dtype int64
    """
    try:
        from sklearn.cluster import MiniBatchKMeans
    except ImportError:
        print("ERROR: scikit-learn not installed.")
        print("       pip install scikit-learn")
        sys.exit(1)

    N = embeddings.shape[0]
    sids = np.zeros((N, num_hierarchies), dtype=np.int64)
    residuals = embeddings.copy()

    for h in range(num_hierarchies):
        print(f"  Level {h}: fitting MiniBatchKMeans (k={codebook_size}) ...")
        km = MiniBatchKMeans(
            n_clusters=codebook_size,
            max_iter=max_iter,
            random_state=random_state,
            n_init=3,
            batch_size=min(4096, N),
            verbose=0,
        )
        labels = km.fit_predict(residuals)        # (N,)
        sids[:, h] = labels

        # subtract assigned centroid → residual for next level
        centroids = km.cluster_centers_           # (codebook_size, D)
        residuals = residuals - centroids[labels]

        inertia = km.inertia_
        print(f"    inertia={inertia:.4f}")

    return sids


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate GRID Semantic IDs from FairDiverseSimulation item text."
    )
    parser.add_argument(
        "--dataset", required=True,
        help="Dataset name, e.g. 'steam' or 'amazon-beauty'. "
             "Must have a processed directory under "
             "fairdiverse/recommendation/processed_dataset/<dataset>/",
    )
    parser.add_argument(
        "--num_hierarchies", type=int, default=3,
        help="Number of RQ codebook levels (H). Default: 3",
    )
    parser.add_argument(
        "--codebook_size", type=int, default=256,
        help="Number of centroids per codebook level. Default: 256",
    )
    parser.add_argument(
        "--embed_model", default="sentence-transformers/all-MiniLM-L6-v2",
        help="HuggingFace sentence-transformer model for item embeddings.",
    )
    parser.add_argument(
        "--embed_batch_size", type=int, default=256,
        help="Batch size for embedding inference.",
    )
    parser.add_argument(
        "--output_dir", default=None,
        help="Where to save sids.pt. Defaults to GRID/outputs/<dataset>/",
    )
    parser.add_argument(
        "--device", default="cpu",
        help="Device for embedding model: 'cpu', 'cuda', 'cuda:0', etc.",
    )
    parser.add_argument(
        "--max_iter", type=int, default=300,
        help="Max iterations for MiniBatchKMeans.",
    )
    parser.add_argument(
        "--random_state", type=int, default=42,
    )
    parser.add_argument(
        "--save_embeddings", action="store_true",
        help="Also save the raw embedding tensor as embeddings_<model>.pt in output_dir. "
             "Useful to reuse embeddings for RVQ/RQ-VAE via the GRID pipeline.",
    )
    args = parser.parse_args()

    # ── Resolve paths ────────────────────────────────────────────────────
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(
        script_dir,
        "fairdiverse", "recommendation", "processed_dataset", args.dataset,
    )
    if not os.path.isdir(dataset_dir):
        print(f"ERROR: dataset directory not found: {dataset_dir}")
        print(f"Available datasets: steam, amazon-beauty")
        sys.exit(1)

    output_dir = args.output_dir or os.path.join(script_dir, "GRID", "outputs", args.dataset)
    os.makedirs(output_dir, exist_ok=True)

    # ── Load metadata ─────────────────────────────────────────────────────
    stats = load_dataset_stats(dataset_dir)
    item_num = stats["item_num"]
    print(f"\nDataset : {args.dataset}")
    print(f"item_num: {item_num}  |  H={args.num_hierarchies}  |  C={args.codebook_size}")
    print(f"Output  : {output_dir}\n")

    # ── Embed ─────────────────────────────────────────────────────────────
    item_texts = load_item_texts(dataset_dir)
    print(f"Loaded {len(item_texts)} item texts.")

    embeddings = embed_items(
        item_texts,
        item_num=item_num,
        model_name=args.embed_model,
        device=args.device,
        batch_size=args.embed_batch_size,
    )  # (item_num, D)
    print(f"Embedding shape: {embeddings.shape}\n")

    # ── Residual KMeans ───────────────────────────────────────────────────
    print(f"Running Residual KMeans ({args.num_hierarchies} levels) ...")
    sids = residual_kmeans(
        embeddings=embeddings,
        num_hierarchies=args.num_hierarchies,
        codebook_size=args.codebook_size,
        max_iter=args.max_iter,
        random_state=args.random_state,
    )  # (item_num, H)

    # ── Save ──────────────────────────────────────────────────────────────
    sids_tensor = torch.from_numpy(sids).long()  # (item_num, H)
    sids_path = os.path.join(output_dir, "sids.pt")
    torch.save(sids_tensor, sids_path)
    print(f"\n✓ Saved SID tensor {tuple(sids_tensor.shape)} → {sids_path}")

    if args.save_embeddings:
        embed_path = os.path.join(output_dir, f"embeddings_{args.embed_model.split('/')[-1]}.pt")
        torch.save(torch.from_numpy(embeddings), embed_path)
        print(f"✓ Saved embeddings {tuple(embeddings.shape)} → {embed_path}")

    info = {
        "dataset": args.dataset,
        "item_num": item_num,
        "num_hierarchies": args.num_hierarchies,
        "codebook_size": args.codebook_size,
        "embed_model": args.embed_model,
        "sids_path": sids_path,
        "embeddings_path": os.path.join(output_dir, f"embeddings_{args.embed_model.split('/')[-1]}.pt") if args.save_embeddings else None,
    }
    info_path = os.path.join(output_dir, "sids_info.json")
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"✓ Saved metadata → {info_path}")

    print("\n--- Next step ---")
    print(f"Set in your In-processing.yaml (or experiment config):")
    print(f"   model: GRID")
    print(f"   data_type: sequential")
    print(f"   sid_path: \"{sids_path}\"")
    print(f"   num_hierarchies: {args.num_hierarchies}")
    print(f"   codebook_size: {args.codebook_size}")
    print(f"   dataset: {args.dataset}")
    print()


if __name__ == "__main__":
    main()
