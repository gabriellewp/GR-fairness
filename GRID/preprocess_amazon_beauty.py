#!/usr/bin/env python3
"""
Convert supported Amazon recommendation datasets to TFRecord format for GRID.

Raw data sources:
- meta_Beauty.json.gz: Item metadata (title, description)
- ratings_Beauty.csv: User ratings (user_id, product_id, rating, timestamp), or
- reviews_Beauty_5.json.gz: legacy 5-core review interactions
- Pet_Supplies.csv.gz: Amazon 2023 5-core interactions with a header

Output structure:
  output_dir/
  ├── training/
  │   ├── user_sequences_*.tfrecord.gz
  │   └── item_metadata_*.tfrecord.gz
  ├── validation/
  │   └── user_sequences_*.tfrecord.gz
  ├── testing/
  │   └── user_sequences_*.tfrecord.gz
  └── items/
      └── item_metadata_*.tfrecord.gz
"""

import argparse
import gzip
import json
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _metadata_text(value) -> str:
    """Normalize scalar and Amazon 2023 list-valued metadata fields."""
    if isinstance(value, list):
        return " ".join(str(part) for part in value if part)
    return str(value or "")


def load_item_metadata(
    metadata_path: str,
    text_mode: str = "description",
) -> Dict[str, Dict]:
    """Load item metadata and construct the text used for item embeddings.

    ``description`` uses only the description field. ``complete_metadata``
    combines title, categories, brand, and description into labelled text.
    ``complete`` is accepted as a backwards-compatible CLI alias.
    """
    if text_mode == "complete":
        text_mode = "complete_metadata"
    if text_mode not in {"description", "complete_metadata"}:
        raise ValueError(
            "Unknown metadata text mode "
            f"{text_mode!r}; expected description or complete_metadata"
        )

    logger.info(
        "Loading item metadata from %s (text mode: %s)", metadata_path, text_mode
    )
    items = {}
    
    with gzip.open(metadata_path, 'rt', encoding='utf-8') as f:
        for line_num, line in enumerate(tqdm(f, desc="Loading metadata")):
            try:
                item = json.loads(line)
                # The 2014 release uses ``asin``; Amazon 2023 metadata uses
                # ``parent_asin`` to match the benchmark interaction files.
                item_id = item.get('asin') or item.get('parent_asin')
                if item_id:
                    title = _metadata_text(item.get('title'))
                    description = _metadata_text(item.get('description'))[:1000]

                    if text_mode == "description":
                        text = description
                    else:
                        # Amazon 2023 calls the brand/store field ``store``.
                        brand = _metadata_text(item.get('brand') or item.get('store'))
                        categories = str(item.get('categories') or '')
                        text = (
                            f"Title: {title}; Categories: {categories}; "
                            f"Brand: {brand}; Description: {description}"
                        )
                    
                    items[item_id] = {
                        'title': title,
                        'description': description,
                        'text': text,
                    }
            except json.JSONDecodeError:
                if line_num % 10000 == 0:
                    logger.warning(f"Failed to parse line {line_num}")
                continue
    
    logger.info(f"Loaded {len(items)} items")
    return items


def load_interactions(interactions_path: str) -> pd.DataFrame:
    """Load interactions from ratings CSV or McAuley review JSON lines."""
    logger.info(f"Loading interactions from {interactions_path}")

    if interactions_path.endswith((".json.gz", ".jsonl.gz")):
        records = []
        with gzip.open(interactions_path, "rt", encoding="utf-8") as f:
            for line in tqdm(f, desc="Loading review interactions"):
                review = json.loads(line)
                records.append(
                    (
                        review.get("reviewerID"),
                        review.get("asin"),
                        review.get("overall"),
                        review.get("unixReviewTime"),
                    )
                )

        df = pd.DataFrame.from_records(
            records,
            columns=["user_id", "item_id", "rating", "timestamp"],
        ).dropna(subset=["user_id", "item_id", "timestamp"])
        logger.info(f"Loaded {len(df)} review interactions")
        return df
    
    # Try to detect delimiter and whether this is the Amazon 2023 CSV schema.
    opener = gzip.open if interactions_path.endswith(".gz") else open
    with opener(interactions_path, 'rt', encoding='utf-8', newline='') as f:
        first_line = f.readline()
    
    delimiter = ',' if ',' in first_line else '\t'
    logger.info(f"Detected delimiter: {repr(delimiter)}")
    
    first_columns = [column.strip() for column in first_line.strip().split(delimiter)]
    has_header = "user_id" in first_columns and (
        "parent_asin" in first_columns or "item_id" in first_columns
    )
    if has_header:
        df = pd.read_csv(interactions_path, delimiter=delimiter)
    else:
        # Legacy McAuley ratings format has no header.
        df = pd.read_csv(
            interactions_path,
            delimiter=delimiter,
            header=None,
            names=['user_id', 'item_id', 'rating', 'timestamp'],
        )
    
    # Rename columns to standard names if needed
    column_mapping = {
        'UserId': 'user_id',
        'ProductId': 'item_id',
        'Rating': 'rating',
        'Timestamp': 'timestamp',
        'user_id': 'user_id',
        'product_id': 'item_id',
        'parent_asin': 'item_id',
        'rating': 'rating',
        'timestamp': 'timestamp',
    }
    
    df = df.rename(columns=column_mapping)
    df = df[['user_id', 'item_id', 'rating', 'timestamp']].copy()
    
    logger.info(f"Loaded {len(df)} rating interactions")
    return df


def load_ratings(ratings_path: str) -> pd.DataFrame:
    """Backward-compatible alias for older callers."""
    return load_interactions(ratings_path)


def create_user_sequences(
    ratings_df: pd.DataFrame,
    min_seq_length: int = 2,
) -> Dict[str, List[str]]:
    """Create user interaction sequences from ratings, sorted by timestamp."""
    logger.info("Creating user interaction sequences...")
    
    user_sequences = defaultdict(list)
    
    # Sort by user and timestamp
    ratings_df_sorted = ratings_df.sort_values(['user_id', 'timestamp'])
    
    for user_id, group in tqdm(ratings_df_sorted.groupby('user_id'), desc="Creating sequences"):
        items = group['item_id'].tolist()
        
        if len(items) >= min_seq_length:
            user_sequences[user_id] = items
    
    logger.info(f"Created sequences for {len(user_sequences)} users")
    return dict(user_sequences)


def split_data_random(
    user_sequences: Dict[str, List[str]],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str]]:
    """Split users into train/val/test sets using random 80/10/10 split."""
    logger.info("Splitting data into train/val/test (RANDOM 80/10/10 split)...")
    
    np.random.seed(seed)
    user_ids = list(user_sequences.keys())
    np.random.shuffle(user_ids)
    
    n_total = len(user_ids)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_users = user_ids[:n_train]
    val_users = user_ids[n_train:n_train + n_val]
    test_users = user_ids[n_train + n_val:]
    
    logger.info(f"Train: {len(train_users)} | Val: {len(val_users)} | Test: {len(test_users)}")
    
    return train_users, val_users, test_users


def split_data_temporal_loo(
    user_sequences: Dict[str, List[str]],
    min_train_sequence_length: int = 2,
) -> Tuple[Dict[str, List[str]], Dict[str, str], Dict[str, str]]:
    """
    Split user sequences using temporal leave-one-out evaluation.
    
    For each user:
    - Training: all items except last 2
    - Validation: second-to-last item
    - Testing: last item
    
    Returns:
    - train_sequences: Dict mapping user_id -> shortened item list
    - val_items: Dict mapping user_id -> validation item
    - test_items: Dict mapping user_id -> test item
    """
    logger.info("Splitting data using TEMPORAL LEAVE-ONE-OUT...")
    
    train_sequences = {}
    val_items = {}
    test_items = {}
    
    min_original_sequence_length = min_train_sequence_length + 2
    users_skipped = 0
    
    for user_id, items in user_sequences.items():
        if len(items) >= min_original_sequence_length:
            # Hold out last 2 items while keeping enough training history.
            train_sequences[user_id] = items[:-2]
            val_items[user_id] = items[-2]
            test_items[user_id] = items[-1]
        else:
            # Shorter histories cannot support both holdouts and the TIGER objective.
            users_skipped += 1
    
    logger.info(f"Train users: {len(train_sequences)}")
    logger.info(f"  - With validation+test: {len([u for u in val_items if val_items[u] is not None])}")
    logger.info(
        f"Skipped users with <{min_original_sequence_length} items: {users_skipped}"
    )
    
    return train_sequences, val_items, test_items


def build_temporal_loo_evaluation_sequences(
    train_sequences: Dict[str, List[str]],
    val_items: Dict[str, str],
    test_items: Dict[str, str],
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """Build causal validation and test sequences for temporal LOO.

    ``NextKTokenMasking`` removes the final item (represented by K semantic-ID
    tokens) and uses it as the target.  Validation therefore needs the observed
    training history followed by the validation item.  Test additionally makes
    the validation item observable before appending the test target.
    """
    validation_sequences = {
        user_id: [*history, val_items[user_id]]
        for user_id, history in train_sequences.items()
    }
    test_sequences = {
        user_id: [*validation_sequences[user_id], test_items[user_id]]
        for user_id in train_sequences
    }
    return validation_sequences, test_sequences


def encode_sequence_example(
    user_id: str,
    item_sequence: List[str],
    item_id_map: Dict[str, int],
) -> tf.train.Example:
    """Create a TFRecord example for a user sequence."""
    
    # Map item strings to integers
    sequence_ids = []
    for item_id in item_sequence:
        if item_id in item_id_map:
            sequence_ids.append(item_id_map[item_id])
    
    if not sequence_ids:
        return None
    
    # Create feature dict
    feature = {
        'user_id': tf.train.Feature(
            int64_list=tf.train.Int64List(value=[int(user_id) if user_id.isdigit() else hash(user_id) % (2**31)])
        ),
        'sequence_data': tf.train.Feature(
            int64_list=tf.train.Int64List(value=sequence_ids)
        ),
    }
    
    return tf.train.Example(features=tf.train.Features(feature=feature))


def encode_item_example(
    item_id: str,
    item_info: Dict,
    item_id_map: Dict[str, int],
) -> tf.train.Example:
    """Create a TFRecord example for item metadata."""
    
    if item_id not in item_id_map:
        return None
    
    text = item_info.get('text', item_info.get('title', ''))
    if not text:
        text = item_info.get('description', '')
    
    feature = {
        'id': tf.train.Feature(
            int64_list=tf.train.Int64List(value=[item_id_map[item_id]])
        ),
        'text': tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[text.encode('utf-8')])
        ),
    }
    
    return tf.train.Example(features=tf.train.Features(feature=feature))


def write_tfrecords(
    output_dir: str,
    split_name: str,
    examples: List[tf.train.Example],
    records_per_file: int = 10000,
):
    """Write examples to TFRecord files."""
    
    split_dir = Path(output_dir) / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    
    file_count = 0
    example_count = 0
    
    for file_idx in range(0, len(examples), records_per_file):
        output_file = split_dir / f"data_{file_idx // records_per_file:05d}.tfrecord.gz"
        
        with tf.io.TFRecordWriter(str(output_file), options=tf.io.TFRecordOptions(
            compression_type='GZIP'
        )) as writer:
            for example in examples[file_idx:file_idx + records_per_file]:
                if example is not None:
                    writer.write(example.SerializeToString())
                    example_count += 1
        
        file_count += 1
        logger.info(f"Wrote {example_count} examples to {output_file.name}")
    
    logger.info(f"Created {file_count} TFRecord files for split '{split_name}'")
    return example_count


def main(args):
    """Main preprocessing pipeline."""
    # Store only canonical mode names in paths, logs, and provenance manifests.
    if args.metadata_text_mode == "complete":
        args.metadata_text_mode = "complete_metadata"
    
    logger.info("Starting Amazon dataset preprocessing...")
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # ============================================================
    # Step 1: Load data
    # ============================================================
    metadata_path = os.path.join(args.input_dir, args.metadata_file)
    interactions_path = os.path.join(args.input_dir, args.interactions_file)
    
    if not os.path.exists(metadata_path):
        logger.error(f"Metadata file not found: {metadata_path}")
        return False
    if not os.path.exists(interactions_path):
        logger.error(f"Interactions file not found: {interactions_path}")
        return False
    
    items_dict = load_item_metadata(metadata_path, args.metadata_text_mode)
    ratings_df = load_interactions(interactions_path)
    
    # ============================================================
    # Step 2: Filter items that have metadata
    # ============================================================
    logger.info(f"Filtering to items with metadata...")
    ratings_df = ratings_df[ratings_df['item_id'].isin(items_dict.keys())].copy()
    logger.info(f"Interactions after metadata filtering: {len(ratings_df)}")
    
    # ============================================================
    # Step 3: Create user sequences
    # ============================================================
    user_sequences = create_user_sequences(
        ratings_df,
        min_seq_length=args.min_seq_length,
    )
    
    # ============================================================
    # Step 4: Split users based on strategy
    # ============================================================
    if args.split_strategy == "random_split":
        train_users, val_users, test_users = split_data_random(
            user_sequences,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )
        train_sequences = {u: user_sequences[u] for u in train_users}
        val_sequences = {u: user_sequences[u] for u in val_users}
        test_sequences = {u: user_sequences[u] for u in test_users}
        
    elif args.split_strategy == "temporal_loo":
        train_sequences, val_items_dict, test_items_dict = split_data_temporal_loo(
            user_sequences,
            min_train_sequence_length=2,
        )
        val_sequences, test_sequences = build_temporal_loo_evaluation_sequences(
            train_sequences,
            val_items_dict,
            test_items_dict,
        )
    else:
        raise ValueError(f"Unknown split_strategy: {args.split_strategy}")
    
    # ============================================================
    # Step 5: Create item ID mapping
    # ============================================================
    logger.info("Creating item ID mapping...")
    item_ids = sorted(items_dict.keys())
    item_id_map = {item_id: idx for idx, item_id in enumerate(item_ids)}
    
    # Save mapping for future reference
    mapping_path = Path(args.output_dir) / "item_id_mapping.json"
    with open(mapping_path, 'w') as f:
        # Store as string keys for JSON compatibility
        mapping_data = {item_id: int(idx) for item_id, idx in item_id_map.items()}
        json.dump(mapping_data, f)
    logger.info(f"Saved item ID mapping to {mapping_path}")
    
    # Save split strategy metadata
    split_config_path = Path(args.output_dir) / "split_config.json"
    split_config = {
        "strategy": args.split_strategy,
        "metadata_text_mode": args.metadata_text_mode,
        "metadata_text_fields": (
            ["description"]
            if args.metadata_text_mode == "description"
            else ["title", "categories", "brand", "description"]
        ),
        "metadata_text_format_version": 1,
        "seed": args.seed,
        "train_ratio": args.train_ratio if args.split_strategy == "random_split" else None,
        "val_ratio": args.val_ratio if args.split_strategy == "random_split" else None,
        "min_seq_length": args.min_seq_length,
    }
    with open(split_config_path, 'w') as f:
        json.dump(split_config, f, indent=2)
    logger.info(f"Saved split configuration to {split_config_path}")
    
    # ============================================================
    # Step 6: Convert sequences to TFRecords
    # ============================================================
    logger.info("Converting sequences to TFRecords...")
    
    if args.split_strategy == "random_split":
        # Original random split approach
        for split_name, sequences in [
            ('training', train_sequences),
            ('validation', val_sequences),
            ('testing', test_sequences),
        ]:
            examples = []
            for user_id, items in tqdm(sequences.items(), desc=f"Encoding {split_name} sequences"):
                example = encode_sequence_example(
                    user_id,
                    items,
                    item_id_map,
                )
                if example:
                    examples.append(example)
            
            write_tfrecords(
                args.output_dir,
                split_name,
                examples,
                records_per_file=args.records_per_file,
            )
    
    elif args.split_strategy == "temporal_loo":
        # Each evaluation row includes its target as the final item. TIGER's
        # NextKTokenMasking removes that final semantic-ID group before scoring.
        for split_name, sequences in [
            ('training', train_sequences),
            ('validation', val_sequences),
            ('testing', test_sequences),
        ]:
            examples = []
            for user_id, items in tqdm(
                sequences.items(), desc=f"Encoding {split_name} sequences"
            ):
                example = encode_sequence_example(user_id, items, item_id_map)
                if example:
                    examples.append(example)

            write_tfrecords(
                args.output_dir,
                split_name,
                examples,
                records_per_file=args.records_per_file,
            )
        
        # Save validation items mapping
        val_mapping = {}
        for user_id, item_id in val_items_dict.items():
            if item_id is not None and item_id in item_id_map:
                val_mapping[str(user_id)] = {
                    'item_id': item_id,
                    'item_id_encoded': item_id_map[item_id]
                }
        
        val_mapping_path = Path(args.output_dir) / "validation_items.json"
        with open(val_mapping_path, 'w') as f:
            json.dump(val_mapping, f)
        logger.info(f"Saved validation items mapping to {val_mapping_path} ({len(val_mapping)} items)")
        
        # Save test items mapping
        test_mapping = {}
        for user_id, item_id in test_items_dict.items():
            if item_id is not None and item_id in item_id_map:
                test_mapping[str(user_id)] = {
                    'item_id': item_id,
                    'item_id_encoded': item_id_map[item_id]
                }
        
        test_mapping_path = Path(args.output_dir) / "test_items.json"
        with open(test_mapping_path, 'w') as f:
            json.dump(test_mapping, f)
        logger.info(f"Saved test items mapping to {test_mapping_path} ({len(test_mapping)} items)")
        

    # ============================================================
    # Step 7: Convert item metadata to TFRecords
    # ============================================================
    logger.info("Converting item metadata to TFRecords...")
    
    examples = []
    for item_id, item_info in tqdm(items_dict.items(), desc="Encoding items"):
        example = encode_item_example(item_id, item_info, item_id_map)
        if example:
            examples.append(example)
    
    write_tfrecords(
        args.output_dir,
        'items',
        examples,
        records_per_file=args.records_per_file,
    )
    
    # ============================================================
    # Summary
    # ============================================================
    logger.info("=" * 70)
    logger.info("Preprocessing complete!")
    logger.info("=" * 70)
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Split strategy: {args.split_strategy}")
    logger.info(f"Total items: {len(items_dict)}")
    logger.info(f"Total original sequences: {len(user_sequences)}")
    
    if args.split_strategy == "random_split":
        logger.info(f"  - Training users: {len(train_sequences)}")
        logger.info(f"  - Validation users: {len(val_sequences)}")
        logger.info(f"  - Testing users: {len(test_sequences)}")
    elif args.split_strategy == "temporal_loo":
        logger.info(f"  - Training sequences (shortened): {len(train_sequences)}")
        logger.info(f"  - Validation items (held-out): {len(val_items_dict)}")
        logger.info(f"  - Test items (held-out): {len(test_items_dict)}")
    
    logger.info("")
    logger.info("To run GRID with this data:")
    logger.info(f"  sbatch run_amazon_beauty_pipeline.sbatch data_dir={args.output_dir}")
    
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert Amazon Beauty raw data to TFRecord format for GRID'
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        required=True,
        help='Path to raw Amazon Beauty data directory'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/amazon_beauty_tfrecords',
        help='Output directory for TFRecord files'
    )
    parser.add_argument(
        '--metadata-file',
        type=str,
        default='meta_Beauty_5core.json.gz',
        help='Name of metadata file (gzipped JSON)'
    )
    parser.add_argument(
        '--metadata-text-mode',
        type=str,
        default='description',
        choices=['description', 'complete_metadata', 'complete'],
        help=(
            "Text used for item embeddings: description uses only the description "
            "field; complete_metadata combines title, categories, brand, and "
            "description (complete is a compatibility alias)"
        ),
    )
    parser.add_argument(
        '--interactions-file', '--ratings-file',
        dest='interactions_file',
        type=str,
        default='ratings_Beauty.csv',
        help='Name of ratings CSV or McAuley review JSON-lines file'
    )
    parser.add_argument(
        '--min-seq-length',
        type=int,
        default=2,
        help='Minimum sequence length for user interactions'
    )
    parser.add_argument(
        '--split-strategy',
        type=str,
        default='random_split',
        choices=['random_split', 'temporal_loo'],
        help='Data splitting strategy: random_split (80/10/10) or temporal_loo (leave-one-out with temporal ordering)'
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.8,
        help='Proportion of data for training (only for random_split strategy)'
    )
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.1,
        help='Proportion of data for validation (only for random_split strategy)'
    )
    parser.add_argument(
        '--records-per-file',
        type=int,
        default=10000,
        help='Number of records per TFRecord file'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    args = parser.parse_args()
    success = main(args)
    exit(0 if success else 1)
