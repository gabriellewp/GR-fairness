"""Paths supplied by the archive audit launcher; source records remain read only."""
import os
from pathlib import Path


def optional_source_root(default):
    return source_root() if os.environ.get('GR_FAIRNESS_ARCHIVE_ROOT') else Path(default)


def optional_output_root(default):
    return output_root() if os.environ.get('GR_FAIRNESS_ARCHIVE_OUTPUT') else Path(default)


def source_root():
    value = os.environ.get('GR_FAIRNESS_ARCHIVE_ROOT')
    if not value:
        raise ValueError('Use python -m scripts.paper.run_archive with --source-root and --output-dir')
    root = Path(value).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    return root


def output_root():
    root = source_root()
    output = Path(os.environ['GR_FAIRNESS_ARCHIVE_OUTPUT']).resolve()
    if output.is_relative_to(root):
        raise ValueError('Archive outputs must be outside the source artifact workspace')
    output.mkdir(parents=True, exist_ok=True)
    return output
