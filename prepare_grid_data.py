#!/usr/bin/env python3
"""
prepare_grid_data.py
====================
Unified data-preparation script for GRID experiments in FairDiverseSimulation.

Merges the functionality of:
  - prepare_steam_grid_data.py  (TFRecord conversion for the GRID T5 embedding pipeline)
  - prepare_grid_sids.py        (rkmeans SID generation via sentence-transformers)

Modes
-----
  tfrecords   Convert iid2text.json → gzipped TFRecord shards.
              Requires tensorflow (GRID venv).

  sids        Embed items with a sentence-transformer model, then run
              Residual KMeans to produce a SID tensor (N, H).
              Requires sentence-transformers + scikit-learn (FairDiverse venv).

  all         Run tfrecords first, then sids. Requires both environments.

Usage
-----
  # TFRecords only (uses GRID venv for tensorflow):
  python prepare_grid_data.py --dataset steam --mode tfrecords

  # SIDs only (rkmeans, uses FairDiverse venv):
  python prepare_grid_data.py --dataset steam --mode sids

  # Both in one go (if both environments' packages are available):
  python prepare_grid_data.py --dataset steam --mode all

  # Custom embedding model and output name:
  python prepare_grid_data.py \\
      --dataset steam --mode sids \\
      --embed_model sentence-transformers/all-MiniLM-L6-v2 \\
      --sid_name sids_rkmeans_minilm.pt \\
      --save_embeddings

Dependencies
------------
  tfrecords mode: tensorflow
  sids mode:      sentence-transformers, scikit-learn, torch, tqdm
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import torch
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Shared helpers
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
    return {int(k): v for k, v in raw.items()}


def load_dataset_stats(dataset_dir: str) -> dict:
    path = os.path.join(dataset_dir, "dataset_stats.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"dataset_stats.json not found in {dataset_dir}.")
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# TFRecord mode
# ---------------------------------------------------------------------------

def write_tfrecords(item_texts: dict, output_dir: str, shard_size: int = 256) -> None:
    """Write item_id + text as gzipped TFRecord shards."""
    try:
        import tensorflow as tf
    except ImportError:
        print("ERROR: tensorflow not installed. Activate the GRID venv first:")
        print("   source .venv/bin/activate")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    # Remove stale TFRecord shards from previous runs so the inference pipeline
    # does not pick up old files that belong to a different item set.
    import glob as _glob
    stale = _glob.glob(os.path.join(output_dir, "*.tfrecord.gz"))
    for f in stale:
        os.remove(f)

    ids_sorted = sorted(item_texts.keys())
    n = len(ids_sorted)
    num_shards = math.ceil(n / shard_size)

    print(f"Writing {n} items into {num_shards} TFRecord shards → {output_dir}")

    options = tf.io.TFRecordOptions(compression_type="GZIP")

    for shard_idx in range(num_shards):
        shard_path = os.path.join(output_dir, f"data_{shard_idx}.tfrecord.gz")
        chunk = ids_sorted[shard_idx * shard_size : (shard_idx + 1) * shard_size]

        with tf.io.TFRecordWriter(shard_path, options=options) as writer:
            for iid in chunk:
                text = item_texts[iid]
                feature = {
                    "id": tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[iid])
                    ),
                    "text": tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[text.encode("utf-8")])
                    ),
                }
                example = tf.train.Example(
                    features=tf.train.Features(feature=feature)
                )
                writer.write(example.SerializeToString())

        if (shard_idx + 1) % 5 == 0 or shard_idx == num_shards - 1:
            print(f"  ✓ {shard_idx + 1}/{num_shards} shards written")

    print(f"\n✓ TFRecords written to {output_dir}")


# ---------------------------------------------------------------------------
# SID mode
# ---------------------------------------------------------------------------

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

    ids_sorted = sorted(item_texts.keys())
    texts = [item_texts[i] for i in ids_sorted]

    print(f"Embedding {len(texts)} items (batch_size={batch_size}) ...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )  # (len(texts), D)

    D = embeddings.shape[1]
    table = np.zeros((item_num, D), dtype=np.float32)
    for idx, iid in enumerate(ids_sorted):
        if iid < item_num:
            table[iid] = embeddings[idx]

    return table  # (item_num, D)


def residual_kmeans(
    embeddings: np.ndarray,
    num_hierarchies: int,
    codebook_size: int,
    max_iter: int = 300,
    random_state: int = 42,
) -> np.ndarray:
    """
    Residual KMeans: assign each item an H-tuple SID.

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
        labels = km.fit_predict(residuals)
        sids[:, h] = labels

        centroids = km.cluster_centers_
        residuals = residuals - centroids[labels]

        print(f"    inertia={km.inertia_:.4f}")

    return sids


def run_sids(args, dataset_dir: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    stats = load_dataset_stats(dataset_dir)
    item_num = stats["item_num"]
    print(f"\nDataset : {args.dataset}")
    print(f"item_num: {item_num}  |  H={args.num_hierarchies}  |  C={args.codebook_size}")
    print(f"Output  : {output_dir}\n")

    item_texts = load_item_texts(dataset_dir)
    print(f"Loaded {len(item_texts)} item texts.")

    embeddings = embed_items(
        item_texts,
        item_num=item_num,
        model_name=args.embed_model,
        device=args.device,
        batch_size=args.embed_batch_size,
    )
    print(f"Embedding shape: {embeddings.shape}\n")

    if args.save_embeddings:
        embed_fname = f"embeddings_{args.embed_model.split('/')[-1]}.pt"
        embed_path = os.path.join(output_dir, embed_fname)
        torch.save(torch.from_numpy(embeddings), embed_path)
        print(f"✓ Saved embeddings {tuple(embeddings.shape)} → {embed_path}")

    print(f"Running Residual KMeans ({args.num_hierarchies} levels) ...")
    sids = residual_kmeans(
        embeddings=embeddings,
        num_hierarchies=args.num_hierarchies,
        codebook_size=args.codebook_size,
        max_iter=args.max_iter,
        random_state=args.random_state,
    )

    sids_tensor = torch.from_numpy(sids).long()
    sids_path = os.path.join(output_dir, args.sid_name)
    torch.save(sids_tensor, sids_path)
    print(f"\n✓ Saved SID tensor {tuple(sids_tensor.shape)} → {sids_path}")

    info = {
        "dataset": args.dataset,
        "item_num": item_num,
        "num_hierarchies": args.num_hierarchies,
        "codebook_size": args.codebook_size,
        "embed_model": args.embed_model,
        "sid_name": args.sid_name,
        "sids_path": sids_path,
        "embeddings_path": os.path.join(output_dir, f"embeddings_{args.embed_model.split('/')[-1]}.pt")
        if args.save_embeddings
        else None,
    }
    info_fname = args.sid_name.replace(".pt", "_info.json")
    info_path = os.path.join(output_dir, info_fname)
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"✓ Saved metadata → {info_path}")

    print("\n--- Next step ---")
    print(f"Set in your experiment config:")
    print(f"   model: GRID")
    print(f"   data_type: sequential")
    print(f"   sid_path: \"{sids_path}\"")
    print(f"   num_hierarchies: {args.num_hierarchies}")
    print(f"   codebook_size: {args.codebook_size}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Unified GRID data preparation: TFRecords and/or rkmeans SIDs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Shared ──────────────────────────────────────────────────────────────
    parser.add_argument(
        "--dataset", required=True,
        help="Dataset name, e.g. 'steam' or 'amazon-beauty'.",
    )
    parser.add_argument(
        "--mode", choices=["tfrecords", "sids", "all"], default="sids",
        help=(
            "tfrecords: generate TFRecord shards (needs tensorflow / GRID venv).\n"
            "sids: generate rkmeans SID tensor (needs sentence-transformers).\n"
            "all: run both steps."
        ),
    )
    parser.add_argument(
        "--dataset_dir", default=None,
        help="Override path to the processed dataset directory. "
             "Default: fairdiverse/recommendation/processed_dataset/<dataset>/",
    )
    parser.add_argument(
        "--output_dir", default=None,
        help="Override output directory. Default: GRID/outputs/<dataset>/",
    )

    # ── TFRecords options ────────────────────────────────────────────────────
    tfrecords_group = parser.add_argument_group("TFRecords options (--mode tfrecords or all)")
    tfrecords_group.add_argument(
        "--tfrecords_dir", default=None,
        help="Directory for TFRecord shards. "
             "Default: GRID/data/<dataset>/items/",
    )
    tfrecords_group.add_argument(
        "--shard_size", type=int, default=256,
        help="Items per TFRecord shard.",
    )

    # ── SID options ──────────────────────────────────────────────────────────
    sids_group = parser.add_argument_group("SID options (--mode sids or all)")
    sids_group.add_argument(
        "--num_hierarchies", type=int, default=3,
        help="Number of RQ codebook levels (H).",
    )
    sids_group.add_argument(
        "--codebook_size", type=int, default=256,
        help="Number of centroids per codebook level.",
    )
    sids_group.add_argument(
        "--embed_model", default="sentence-transformers/all-MiniLM-L6-v2",
        help="HuggingFace sentence-transformer model name.",
    )
    sids_group.add_argument(
        "--embed_batch_size", type=int, default=256,
        help="Batch size for embedding inference.",
    )
    sids_group.add_argument(
        "--device", default="cpu",
        help="Device for embedding: 'cpu', 'cuda', 'cuda:0', etc.",
    )
    sids_group.add_argument(
        "--max_iter", type=int, default=300,
        help="Max iterations for MiniBatchKMeans.",
    )
    sids_group.add_argument(
        "--random_state", type=int, default=42,
    )
    sids_group.add_argument(
        "--sid_name", default="sids.pt",
        help="Output filename for the SID tensor inside output_dir.",
    )
    sids_group.add_argument(
        "--save_embeddings", action="store_true",
        help="Also save the raw embedding tensor as embeddings_<model>.pt. "
             "Useful to reuse embeddings for RVQ/RQ-VAE via the GRID pipeline.",
    )

    args = parser.parse_args()

    # ── Resolve base paths ──────────────────────────────────────────────────
    script_dir = os.path.dirname(os.path.abspath(__file__))

    dataset_dir = args.dataset_dir or os.path.join(
        script_dir, "fairdiverse", "recommendation", "processed_dataset", args.dataset
    )
    if not os.path.isdir(dataset_dir):
        print(f"ERROR: dataset directory not found: {dataset_dir}")
        sys.exit(1)

    output_dir = args.output_dir or os.path.join(script_dir, "GRID", "outputs", args.dataset)
    tfrecords_dir = args.tfrecords_dir or os.path.join(
        script_dir, "GRID", "data", args.dataset, "items"
    )

    # ── Dispatch ────────────────────────────────────────────────────────────
    if args.mode in ("tfrecords", "all"):
        print("\n========== [TFRecords] ==========")
        item_texts = load_item_texts(dataset_dir)
        print(f"Loaded {len(item_texts)} items from {dataset_dir}")
        write_tfrecords(item_texts, tfrecords_dir, shard_size=args.shard_size)

        print("\n--- Next step (TFRecords) ---")
        print("Run T5-XL embedding inference:")
        print(f"  cd GRID && source .venv/bin/activate")
        print(f"  python -m src.inference experiment=sem_embeds_inference_flat \\")
        print(f"      data_dir={os.path.join(script_dir, 'GRID', 'data', args.dataset)} \\")
        print(f"      id=t5xl/{args.dataset}")

    if args.mode in ("sids", "all"):
        print("\n========== [SIDs / rkmeans] ==========")
        run_sids(args, dataset_dir, output_dir)


if __name__ == "__main__":
    main()
