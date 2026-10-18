"""Check recovered item membership against the historical dense evaluator."""
import numpy as np
import pytest
from scipy.sparse import csr_matrix

from additional_baselines.popularity_fairness_analysis import (
    dense_top10, metrics, positions_to_items, sparse_top10,
)


@pytest.mark.parametrize('catalog', [13, 50, 500])
def test_top10_matches_full_catalog_quicksort(catalog):
    rng = np.random.default_rng(12)
    values = rng.uniform(1, 2, (100, catalog)).astype('f4')
    values[0] = 1  # Fully collapsed model: tie order affects group exposure.
    values[1] = np.round(values[1], 1)
    values[2] = 1
    values[2, :5] = 2  # Top ten contains absent-score catalog items.
    # Values distinguishable in float64 can tie in the recorded float32 scores.
    values = values.astype('f8')
    values[3, :15 if catalog > 15 else catalog] = 1 + 1e-9
    expected = np.argsort(values.astype('f4'), axis=1)[:, ::-1][:, :10]
    np.testing.assert_array_equal(dense_top10(values), expected)

    targets = rng.integers(0, catalog, 100)
    stored = np.where(rng.uniform(size=values.shape) < .3, values, 1)

    class Collector:
        def __init__(self):
            self.items = []

        def add(self, items, _targets):
            self.items.extend(items)

    aggregate = Collector()
    sparse_top10(csr_matrix(np.where(stored == 1, 0, stored)), targets, aggregate, batch=17)
    expected = []
    for row, target in zip(stored, targets):
        dense = np.r_[row[target], np.delete(row, target)].astype('f4')
        positions = np.argsort(dense)[::-1][:10]
        # Independent positive-first-position -> actual item conversion.
        catalog_order = [int(target)] + [i for i in range(catalog) if i != target]
        expected.append([catalog_order[i] for i in positions])
    # Exposure requires identical item sets; NDCG requires identical target
    # ranks. Ordering a tie entirely among other top-ten items is immaterial.
    actual = np.asarray(aggregate.items)
    expected = np.asarray(expected)
    np.testing.assert_array_equal(np.sort(actual, axis=1), np.sort(expected, axis=1))
    np.testing.assert_array_equal(actual == targets[:, None], expected == targets[:, None])


def test_equality_and_calibration_evaluate_different_targets():
    histories = [.05, .05, .1, .2, .6]
    aligned = metrics(histories, histories)
    equal = metrics([.2]*5, histories)
    assert equal['popularity_gini_at_10'] == pytest.approx(0)
    assert aligned['popularity_gini_at_10'] > 0
    assert aligned['mgu_at_10'] == aligned['dgu_at_10'] == 0
    assert equal['mgu_at_10'] > 0 and equal['dgu_at_10'] > 0
