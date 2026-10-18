"""
Dataset loader class hierarchy for process_dataset.py.

Subclass DatasetLoader and override load_interactions / load_items / load_metadata
to support new raw data formats without touching the core pipeline.

Classes:
  DatasetLoader      — abstract base
  RecBoleLoader      — reads pre-converted RecBole TSV files (default)
  McAuleyRawLoader   — reads McAuley raw JSON/GZ files directly

Helper:
  _detect_price_col  — resolves price:float vs price:token column name
"""

import ast
import gzip
import json
import os

import pandas as pd


def _normalize_mcauley_record(record: dict) -> dict:
    """Normalize a McAuley metadata record so 'category' is always a flat list.

    Handles both:
    - 2018 format: 'category' is already a flat list, 'main_cat' is a string
    - 2014 format: 'categories' is a nested list [[...]] or a stringified
                   nested list "[['Beauty', 'Makeup', ...]]"

    After normalization 'category' is always a flat, deduplicated list of
    strings, so callers never need to know which year the data came from.
    """
    # If 'category' already has content, nothing to do
    existing = record.get('category')
    if existing and isinstance(existing, list) and any(existing):
        return record

    raw = record.get('categories', [])
    if isinstance(raw, str):
        try:
            raw = ast.literal_eval(raw)
        except Exception:
            raw = []

    flat = []
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, list):
                flat.extend(c for c in item if c)
            elif isinstance(item, str) and item:
                flat.append(item)

    if flat:
        # Return a copy so the original cache entry is not mutated
        record = {**record, 'category': list(dict.fromkeys(flat))}
    return record


class DatasetLoader:
    """Base class for dataset loading. Subclasses override load_interactions,
    load_items, and load_metadata to support different source formats."""

    def __init__(self, process_config):
        self.config = process_config

    def load_interactions(self) -> pd.DataFrame:
        """Return DataFrame with user_id, item_id, label, timestamp columns."""
        raise NotImplementedError

    def load_items(self) -> pd.DataFrame:
        """Return DataFrame with item_id, group_id, text_id and optional
        categories:token_seq, price:float / price:token, brand:token columns."""
        raise NotImplementedError

    def load_metadata(self) -> dict:
        """Return dict of item_id -> raw metadata record for feature/description
        enrichment. Return empty dict if not available."""
        return {}


class RecBoleLoader(DatasetLoader):
    """Loads interactions and items from pre-converted RecBole-format TSV files."""

    def __init__(self, process_config, dataset):
        super().__init__(process_config)
        dataset_path = "steam-nonmerged" if dataset == "steam" else dataset
        base = process_config.get("recbole_data_dir", "data/raw/RecBole")
        self.inter_file    = os.path.join(base, dataset_path, dataset_path + ".inter")
        self.provider_file = os.path.join(base, dataset_path, dataset_path + ".item")

    def load_interactions(self) -> pd.DataFrame:
        cfg = self.config
        print("Checking actual columns in inter_file...")
        test_inter = pd.read_csv(self.inter_file, delimiter='\t', nrows=0)
        print(f"Inter file columns: {test_inter.columns.tolist()}")
        frames = pd.read_csv(
            self.inter_file, delimiter='\t',
            dtype={cfg["item_id"]: str, cfg["user_id"]: str, cfg["timestamp"]: float},
            usecols=[cfg["user_id"], cfg["item_id"], cfg["label_id"], cfg["timestamp"]],
        )
        return frames.dropna()

    def load_items(self) -> pd.DataFrame:
        cfg = self.config
        text_col       = cfg["text_id"]
        categories_col = "categories:token_seq"
        price_float    = "price:float"
        price_token    = "price:token"
        brand_col      = "brand:token"

        item_file_cols = pd.read_csv(self.provider_file, delimiter='\t', nrows=0).columns.tolist()
        print("\nChecking actual columns in provider_file...")
        print(f"Provider file columns: {item_file_cols}")
        configured_iid = cfg["item_id"]

        # Detect item ID column
        if configured_iid in item_file_cols:
            item_file_id = configured_iid
            print(f"Item file ID column matches config: '{item_file_id}'")
        else:
            inter_iids = set(
                pd.read_csv(self.inter_file, delimiter='\t', usecols=[configured_iid], dtype=str)
                [configured_iid].dropna()
            )
            best_col, best_overlap = item_file_cols[0], 0
            for col in item_file_cols:
                try:
                    sample = pd.read_csv(
                        self.provider_file, delimiter='\t', usecols=[col], dtype=str, nrows=5000
                    )[col].dropna()
                    overlap = len(set(sample) & inter_iids)
                    if overlap > best_overlap:
                        best_overlap, best_col = overlap, col
                except Exception:
                    pass
            item_file_id = best_col
            print(f"Auto-detected item file ID column: '{item_file_id}' (overlap={best_overlap})")

        usecols    = [item_file_id, cfg["group_id"], text_col]
        dtype_dict = {item_file_id: str, cfg["group_id"]: str, text_col: str}
        for col in [categories_col, price_token, brand_col]:
            if col in item_file_cols:
                usecols.append(col)
                dtype_dict[col] = str
        if price_float in item_file_cols:
            usecols.append(price_float)  # read as numeric (no dtype override)

        provider_frames = pd.read_csv(self.provider_file, delimiter='\t', usecols=usecols, dtype=dtype_dict)
        if item_file_id != configured_iid:
            provider_frames.rename(columns={item_file_id: configured_iid}, inplace=True)
        return provider_frames.dropna(subset=[configured_iid, cfg["group_id"], text_col])

    def load_metadata(self) -> dict:
        meta_path = self.config.get('dataset_path', '')
        if not meta_path:
            return {}
        print(f"Loading item metadata from {meta_path} ...")
        meta = {}
        open_fn = gzip.open if meta_path.endswith('.gz') else open
        with open_fn(meta_path, 'rt', encoding='utf-8') as mf:
            for line in mf:
                try:
                    record = json.loads(line)
                    meta[record['asin']] = _normalize_mcauley_record(record)
                except Exception:
                    pass
        print(f"  -> {len(meta)} metadata records loaded")
        return meta


class McAuleyRawLoader(DatasetLoader):
    """Loads interactions and item metadata directly from McAuley raw JSON/JSONL
    files — no pre-converted RecBole files required.

    Required config keys:
      reviews_path  — path to reviews file (.json.gz or plain .jsonl)
                      McAuley 2014/2018 JSON format:
                      {reviewerID, asin, overall, unixReviewTime}; or Amazon
                      Reviews 2023 5-core CSV format:
                      {user_id, parent_asin, rating, timestamp}
      dataset_path  — path to metadata file (.json.gz)
                      McAuley 2014/2018 format uses ``asin``; 2023 metadata
                      uses ``parent_asin``, ``store``, ``main_category``, and
                      ``categories``.
    """

    def __init__(self, process_config):
        super().__init__(process_config)
        self.reviews_path = process_config['reviews_path']
        self.meta_path    = process_config.get('dataset_path', '')
        self._meta_cache  = None  # lazy-loaded

    def _load_meta_records(self) -> dict:
        if self._meta_cache is not None:
            return self._meta_cache
        if not self.meta_path:
            self._meta_cache = {}
            return self._meta_cache
        print(f"Loading McAuley metadata from {self.meta_path} ...")
        meta = {}
        open_fn = gzip.open if self.meta_path.endswith('.gz') else open
        with open_fn(self.meta_path, 'rt', encoding='utf-8') as f:
            for line in f:
                try:
                    rec  = json.loads(line)
                    asin = str(
                        rec.get('asin', '') or rec.get('parent_asin', '') or ''
                    ).strip()
                    if asin:
                        meta[asin] = rec
                except Exception:
                    pass
        print(f"  -> {len(meta)} metadata records loaded")
        # Normalize category format so callers always see 'category' as a flat list
        self._meta_cache = {asin: _normalize_mcauley_record(rec) for asin, rec in meta.items()}
        return self._meta_cache

    def load_interactions(self) -> pd.DataFrame:
        cfg         = self.config
        meta        = self._load_meta_records()
        valid_asins = set(meta.keys())
        print(f"Loading reviews from {self.reviews_path} ...")
        # The official 2023 5-core benchmark is a CSV/CSV.GZ file.  Reading it
        # directly avoids expanding 5.3M Pet Supplies rows into JSON first.
        if self.reviews_path.endswith(('.csv', '.csv.gz')):
            frames = pd.read_csv(
                self.reviews_path,
                usecols=['user_id', 'parent_asin', 'rating', 'timestamp'],
                dtype={'user_id': str, 'parent_asin': str},
            ).dropna()
            frames = frames[frames['parent_asin'].isin(valid_asins)]
            frames = frames.rename(columns={
                'user_id': cfg['user_id'],
                'parent_asin': cfg['item_id'],
                'rating': cfg['label_id'],
                'timestamp': cfg['timestamp'],
            })
            frames[cfg['label_id']] = frames[cfg['label_id']].astype(float)
            frames[cfg['timestamp']] = frames[cfg['timestamp']].astype(float)
            print(f"  -> {len(frames)} valid interactions loaded")
            return frames

        rows = []
        open_fn = gzip.open if self.reviews_path.endswith('.gz') else open
        with open_fn(self.reviews_path, 'rt', encoding='utf-8') as f:
            for line in f:
                try:
                    rec      = json.loads(line)
                    asin     = str(rec.get('asin', '') or '').strip()
                    reviewer = str(rec.get('reviewerID', '') or '').strip()
                    rating   = rec.get('overall')
                    ts       = rec.get('unixReviewTime')
                    if asin and reviewer and rating is not None and ts is not None and asin in valid_asins:
                        rows.append({
                            cfg['user_id']:   reviewer,
                            cfg['item_id']:   asin,
                            cfg['label_id']:  float(rating),
                            cfg['timestamp']: float(ts),
                        })
                except Exception:
                    pass
        print(f"  -> {len(rows)} valid interactions loaded")
        return pd.DataFrame(rows)

    def load_items(self) -> pd.DataFrame:
        cfg  = self.config
        meta = self._load_meta_records()
        rows = []
        for asin, rec in meta.items():
            title    = str(rec.get('title', '')    or '').strip()
            details  = rec.get('details', {}) or {}
            if not isinstance(details, dict):
                details = {}
            # Amazon Reviews 2023 no longer has a top-level brand field.  Its
            # closest provider is ``store``; details.Brand is a useful fallback.
            brand    = str(
                rec.get('brand', '') or rec.get('store', '') or
                details.get('Brand', '') or 'no_brand'
            ).strip()
            price    = str(rec.get('price', '')    or '').strip()
            main_cat = str(
                rec.get('main_cat', '') or rec.get('main_category', '') or ''
            ).strip()
            cats     = rec.get('category', []) or rec.get('categories', []) or []
            if not isinstance(cats, list):
                cats = []
            all_cats = ([main_cat] if main_cat else []) + [c for c in cats if c]
            cats_str = ', '.join(f"'{c}'" for c in all_cats) if all_cats else ''
            rows.append({
                cfg['item_id']:         asin,
                cfg['group_id']:        brand,
                cfg['text_id']:         title,
                'categories:token_seq': cats_str,
                'price:token':          price,
                'brand:token':          brand,
            })
        df = pd.DataFrame(rows)
        return df.dropna(subset=[cfg['item_id'], cfg['group_id'], cfg['text_id']])

    def load_metadata(self) -> dict:
        """Raw metadata records reused for feature/description enrichment."""
        return self._load_meta_records()


def _detect_price_col(available_cols):
    """Return (price_col, has_price) preferring price:float over price:token."""
    if "price:float" in available_cols:
        return "price:float", True
    if "price:token" in available_cols:
        return "price:token", True
    return None, False
