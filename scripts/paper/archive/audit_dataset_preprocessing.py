"""Count the four 5-core sources and record SHA-256 identities without exporting IDs."""
import csv
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

import yaml

from scripts.paper.archive_paths import source_root, output_root
ROOT = source_root()
OUTPUT = output_root()
DATASETS = [('amazon-beauty', 'Beauty', 2014),
            ('amazon-sports', 'Sports & Outdoors', 2014),
            ('amazon-toys', 'Toys & Games', 2014),
            ('amazon-pet-supplies', 'Pet Supplies', 2023)]


def identity(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    try:
        name = str(path.relative_to(ROOT))
    except ValueError:
        name = str(path)
    return {'path': name, 'bytes': path.stat().st_size, 'sha256': digest.hexdigest()}


def main():
    records = []
    for dataset, label, release in DATASETS:
        directory = ROOT / 'recommendation/processed_dataset' / dataset
        config = yaml.safe_load((directory / 'process_config.yaml').read_text())
        manifest = json.loads((directory / 'grid_data_manifest.json').read_text())
        source = Path(config['reviews_path'])
        users, items = Counter(), Counter()
        with gzip.open(source, 'rt', encoding='utf-8', newline='') as stream:
            rows = (json.loads(line) for line in stream) if release == 2014 else csv.DictReader(stream)
            for row in rows:
                user = row['reviewerID'] if release == 2014 else row['user_id']
                item = row['asin'] if release == 2014 else row['parent_asin']
                users[user] += 1
                items[item] += 1
        total = sum(users.values())
        with gzip.open(config['dataset_path'], 'rt', encoding='utf-8') as stream:
            metadata_ids = {str(r.get('asin') or r.get('parent_asin'))
                            for r in map(json.loads, stream)}
        with (directory / 'user_activity.csv').open() as stream:
            activity = [int(row['train_interaction_count']) for row in csv.DictReader(stream)]
        assert len(users) == manifest['user_num'] == len(activity)
        assert len(items) == manifest['item_num']
        assert min(users.values()) >= 5 and min(items.values()) >= 5
        assert not set(items) - metadata_ids
        assert total == sum(activity) + 2 * len(users)
        data_dir = ROOT / manifest['source_dir']
        source_paths = [data_dir / name for name in ('item_id_mapping.json', 'split_config.json',
                                                   'validation_items.json', 'test_items.json')]
        for split in ('items', 'training', 'validation', 'testing'):
            source_paths.extend(sorted((data_dir / split).glob('*.tfrecord.gz')))
        source_ids = [identity(p) for p in source_paths]
        fingerprint = hashlib.sha256()
        for path, info in zip(source_paths, source_ids):
            fingerprint.update(str(path.relative_to(data_dir)).encode())
            fingerprint.update(info['sha256'].encode())
        matches = fingerprint.hexdigest() == manifest['source_fingerprint']
        record = {'dataset': dataset, 'display_name': label, 'release': release,
                  'users': len(users), 'items': len(items), 'interactions': total,
                  'train_interactions': sum(activity), 'validation_interactions': len(users),
                  'test_interactions': len(users), 'minimum_user_interactions': min(users.values()),
                  'minimum_item_interactions': min(items.values()), 'missing_metadata_items': 0,
                  'recorded_grid_source_fingerprint': manifest['source_fingerprint'],
                  'recomputed_grid_source_fingerprint': fingerprint.hexdigest(),
                  'grid_source_fingerprint_matches': matches,
                  'raw_inputs': [identity(source), identity(Path(config['dataset_path']))],
                  'processed_files': [identity(directory / name) for name in
                      ('grid_data_manifest.json', 'process_config.yaml', 'dataset_stats.json',
                       'iid2item.json', 'iid2pid.json', 'user_activity.csv',
                       f'{dataset}.valid.ranking', f'{dataset}.test.ranking')],
                  'grid_source_files': source_ids}
        slug = {'amazon-beauty': 'Beauty', 'amazon-sports': 'Sports_and_Outdoors',
                'amazon-toys': 'Toys_and_Games', 'amazon-pet-supplies': 'Pet_Supplies'}[dataset]
        legacy_url = 'https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/'
        modern_url = 'https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/'
        record['interaction_source_url'] = (legacy_url + f'reviews_{slug}_5.json.gz'
            if release == 2014 else modern_url + 'benchmark/5core/rating_only/Pet_Supplies.csv.gz')
        record['unfiltered_metadata_source_url'] = (legacy_url + f'meta_{slug}.json.gz'
            if release == 2014 else modern_url + 'raw/meta_categories/meta_Pet_Supplies.jsonl.gz')
        record['metadata_note'] = 'Hashed local metadata is a project-filtered catalog subset.'
        record['licence_status'] = 'Unverified: no explicit dataset licence found on the inspected primary source pages/card.'
        records.append(record)
        print(f'{label}: users={len(users):,}, items={len(items):,}, interactions={total:,}; '
              f'5-core verified; source fingerprint matches={matches}', flush=True)
    output = OUTPUT / 'DATASET_PREPROCESSING_AUDIT.json'
    output.write_text(json.dumps({'audit_date': '2026-10-09',
        'count_definition': 'Rows in released 5-core interaction files before holdout and prefix augmentation',
        'scope': 'Verifies aggregate source counts, minimum degrees, metadata coverage, and GRID source fingerprints; does not establish row-level raw-to-TFRecord identity.',
        'datasets': records}, indent=2) + '\n')
    with (OUTPUT / 'DATASET_STATISTICS_5CORE.csv').open('w', newline='') as stream:
        keys = ['dataset', 'release', 'users', 'items', 'interactions', 'train_interactions',
                'validation_interactions', 'test_interactions']
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(records)


if __name__ == '__main__':
    main()
