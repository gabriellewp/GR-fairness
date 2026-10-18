import sys
from pathlib import Path

import numpy as np


GRID_ROOT = Path(__file__).parents[1] / "GRID"
sys.path.insert(0, str(GRID_ROOT))

from src.data.loading.components.pre_processing import convert_bytes_to_string


def test_convert_bytes_to_string_decodes_utf8_byte_arrays():
    row = {"text": np.asarray(["dog’s café".encode("utf-8")])}

    converted = convert_bytes_to_string(
        row,
        dataset_config=None,
        features_to_apply=["text"],
    )

    assert converted["text"].tolist() == ["dog’s café"]


def test_convert_bytes_to_string_decodes_utf8_object_arrays():
    row = {"text": np.asarray(["naïve pet".encode("utf-8")], dtype=object)}

    converted = convert_bytes_to_string(
        row,
        dataset_config=None,
        features_to_apply=["text"],
    )

    assert converted["text"].tolist() == ["naïve pet"]
