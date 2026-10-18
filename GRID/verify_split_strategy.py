#!/usr/bin/env python3
"""
Verification script for temporal LOO setup.

This script checks if the data was preprocessed correctly with the chosen split strategy.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional


def infer_split_strategy(data_dir: Path) -> Optional[str]:
    """Infer the split strategy from the existing directory layout."""
    if (data_dir / "validation_items.json").exists() and (
        data_dir / "test_items.json"
    ).exists():
        return "temporal_loo"

    if all((data_dir / name).exists() for name in ["training", "validation", "testing"]):
        return "random_split"

    return None


def write_inferred_split_config(data_dir: Path, strategy: str) -> dict:
    """Write a minimal split_config.json for legacy outputs."""
    config = {
        "strategy": strategy,
        "seed": None,
        "train_ratio": None,
        "val_ratio": None,
        "min_seq_length": None,
        "source": "inferred_from_existing_layout",
    }
    with open(data_dir / "split_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"  ✓ Recreated missing split_config.json as {strategy}")
    return config


def verify_random_split(data_dir: Path) -> bool:
    """Verify random 80/10/10 split structure."""
    print("\n📋 Verifying RANDOM 80/10/10 Split...")
    
    required_dirs = ['training', 'validation', 'testing', 'items']
    required_files = ['item_id_mapping.json', 'split_config.json']
    
    # Check directories
    for dir_name in required_dirs:
        dir_path = data_dir / dir_name
        if not dir_path.exists():
            print(f"  ❌ Missing directory: {dir_name}")
            return False
        tfrecords = list(dir_path.glob("*.tfrecord.gz"))
        if not tfrecords:
            print(f"  ❌ No TFRecord files in {dir_name}")
            return False
        print(f"  ✓ {dir_name}: {len(tfrecords)} TFRecord files")
    
    # Check files
    for file_name in required_files:
        file_path = data_dir / file_name
        if not file_path.exists():
            print(f"  ❌ Missing file: {file_name}")
            return False
        print(f"  ✓ {file_name}")
    
    # Verify split config
    with open(data_dir / 'split_config.json', 'r') as f:
        config = json.load(f)
        if config.get('strategy') != 'random_split':
            print(f"  ❌ Wrong strategy in split_config: {config.get('strategy')}")
            return False
        print(f"  ✓ Strategy: {config['strategy']}")
        print(f"    - Train ratio: {config.get('train_ratio', 'N/A')}")
        print(f"    - Val ratio: {config.get('val_ratio', 'N/A')}")
        print(f"    - Seed: {config.get('seed', 'N/A')}")
    
    return True


def verify_temporal_loo(data_dir: Path) -> bool:
    """Verify temporal leave-one-out split structure."""
    print("\n📋 Verifying TEMPORAL LEAVE-ONE-OUT Split...")
    required_min_seq_length = 4
    
    required_files = [
        'item_id_mapping.json',
        'split_config.json',
        'validation_items.json',
        'test_items.json',
    ]
    # TIGER consumes TFRecords for every stage. The JSON files are useful
    # metadata, but cannot by themselves drive validation or testing.
    required_dirs = ['training', 'validation', 'testing', 'items']
    
    # Check files
    for file_name in required_files:
        file_path = data_dir / file_name
        if not file_path.exists():
            print(f"  ❌ Missing file: {file_name}")
            return False
        print(f"  ✓ {file_name}")
    
    # Check directories
    for dir_name in required_dirs:
        dir_path = data_dir / dir_name
        if not dir_path.exists():
            print(f"  ❌ Missing directory: {dir_name}")
            return False
        tfrecords = list(dir_path.glob("*.tfrecord.gz"))
        if not tfrecords:
            print(f"  ❌ No TFRecord files in {dir_name}")
            return False
        print(f"  ✓ {dir_name}: {len(tfrecords)} TFRecord files")
    
    # Verify split config
    with open(data_dir / 'split_config.json', 'r') as f:
        config = json.load(f)
        if config.get('strategy') != 'temporal_loo':
            print(f"  ❌ Wrong strategy in split_config: {config.get('strategy')}")
            return False
        print(f"  ✓ Strategy: {config['strategy']}")
        min_seq_length = config.get('min_seq_length')
        print(f"    - Min seq length: {min_seq_length if min_seq_length is not None else 'N/A'}")
        if min_seq_length is None or min_seq_length < required_min_seq_length:
            print(
                "  ❌ Temporal LOO data was generated with too-short histories. "
                f"Expected min_seq_length >= {required_min_seq_length} so "
                "TIGER training keeps at least two items after validation/test holdout."
            )
            return False
    
    # Check validation items
    with open(data_dir / 'validation_items.json', 'r') as f:
        val_items = json.load(f)
        print(f"  ✓ Validation items: {len(val_items)} users have held-out validation items")
    
    # Check test items
    with open(data_dir / 'test_items.json', 'r') as f:
        test_items = json.load(f)
        print(f"  ✓ Test items: {len(test_items)} users have held-out test items")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Verify temporal LOO or random split setup'
    )
    parser.add_argument(
        'data_dir',
        type=str,
        help='Path to preprocessed data directory'
    )
    
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    
    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        sys.exit(1)
    
    # Check split_config to determine which strategy
    config_path = data_dir / 'split_config.json'
    if not config_path.exists():
        strategy = infer_split_strategy(data_dir)
        if strategy is None:
            print(f"❌ split_config.json not found in {data_dir}")
            print("   Make sure you ran preprocess_amazon_beauty.py")
            sys.exit(1)

        print(f"⚠️  split_config.json not found in {data_dir}")
        print(f"   Inferred split strategy from existing files: {strategy}")
        config = write_inferred_split_config(data_dir, strategy)
    else:
        with open(config_path, 'r') as f:
            config = json.load(f)

    strategy = config.get('strategy', 'unknown')
    
    print(f"\n{'='*60}")
    print(f"Data Directory: {data_dir}")
    print(f"Split Strategy: {strategy}")
    print(f"{'='*60}")
    
    if strategy == 'random_split':
        success = verify_random_split(data_dir)
    elif strategy == 'temporal_loo':
        success = verify_temporal_loo(data_dir)
    else:
        print(f"❌ Unknown strategy: {strategy}")
        success = False
    
    print(f"\n{'='*60}")
    if success:
        print("✅ Verification PASSED! Data is ready for training.")
        print("\nYou can now run:")
        launch_script = (
            "complete_pipeline_5core.sh"
            if "amazon_beauty_5core_tfrecords" in str(data_dir)
            else "complete_pipeline.sh"
        )
        print(f"  sbatch {launch_script}")
    else:
        print("❌ Verification FAILED! Check the errors above.")
    print(f"{'='*60}\n")
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
