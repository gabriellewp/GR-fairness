"""Checks for paired statistics, exact bootstrap compression, and user alignment."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy import stats

from additional_baselines import paired_activity as analysis


class PairedActivityTests(unittest.TestCase):
    def test_exact_user_bootstrap_for_two_user_population(self):
        # Resampling two users produces 0, 1, or 2 successes. Its empirical
        # bootstrap distribution is exactly Binomial(2, .5)/2.
        control = np.array([0, 0], dtype=np.uint8)
        method = np.array([0, 1], dtype=np.uint8)
        actual = analysis.bootstrap_changes(control, method, 200, np.random.default_rng(17))
        expected = np.random.default_rng(17).multinomial(2, [.5, .5], size=200)[:, 1] / 2
        np.testing.assert_array_equal(actual[:, 0], expected)
        np.testing.assert_array_equal(actual[:, 1], expected)

    def test_bootstrap_preserves_joint_metric_dependence(self):
        control = np.array([0, 0, 0, 0], dtype=np.uint8)
        method = np.array([2, 0, 2, 0], dtype=np.uint8)
        actual = analysis.bootstrap_changes(control, method, 200, np.random.default_rng(12))
        np.testing.assert_allclose(actual[:, 1], actual[:, 0] / np.log2(3))

    def test_null_and_direct_activity_contrast(self):
        groups = np.array(['low']*5 + ['high']*5)
        control = np.array([0, 1, 2, 0, 3]*2, dtype=np.uint8)
        effects, contrasts = analysis.effects_for_method(control, control, groups, 1000, np.random.default_rng(1))
        self.assertTrue(all(r['mean_change'] == r['ci_low'] == r['ci_high'] == 0 for r in effects))
        self.assertTrue(all(r['p_raw'] == 1 for r in effects))
        self.assertTrue(all(r['contrast'] == r['ci_low'] == r['ci_high'] == 0 for r in contrasts))
        method = control.copy()
        method[0] = 1
        effects, contrasts = analysis.effects_for_method(control, method, groups, 1000, np.random.default_rng(1))
        recall = next(r for r in contrasts if r['metric'] == 'recall@10')
        self.assertAlmostEqual(recall['contrast'], .2)
        self.assertAlmostEqual(recall['p_raw'], stats.ttest_ind([1, 0, 0, 0, 0], [0]*5, equal_var=False).pvalue)

    def test_paired_test_and_holm(self):
        differences = np.array([0, 1, -1, .5, 0, 1])
        self.assertAlmostEqual(analysis.paired_pvalue(differences), stats.ttest_rel(differences, np.zeros(6)).pvalue)
        np.testing.assert_allclose(analysis.holm([.01, .04, .03]), [.03, .06, .06])
        self.assertIsNone(analysis.paired_pvalue(np.array([1.])))

    def test_candidate_alignment_and_stable_ties(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'candidates.json'
            metadata = dict(split={'name': 'test', 'source': {'path': 'split', 'sha256': 'x'}},
                            seed=42, score_convention='probability')
            row = dict(request_id='test:0:user:5', candidate_ids=[9, 8], scores=[.5, .5], targets=[8])
            path.write_text(json.dumps(dict(metadata=metadata, requests=[row])))
            population = dict(users=['5'], rids=['test:0:user:5'], targets=np.array([8]))
            spec = dict(path=str(path), expected={'recall@10': 1., 'ndcg@10': 1/np.log2(3)})
            case = dict(split=metadata['split']['source'])
            ranks, _, _, _ = analysis.candidates(spec, population, case)
            self.assertEqual(ranks.tolist(), [2])
            population['rids'] = ['test:0:user:6']
            with self.assertRaisesRegex(ValueError, 'user/request mismatch'):
                analysis.candidates(spec, population, case)

    def test_incomplete_collection_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp)/'manifest.json'
            manifest.write_text(json.dumps(dict(cases=[dict(index=0, dataset='missing', tokenizer='rvq')])))
            with self.assertRaises(FileNotFoundError):
                analysis.collect(manifest, plots=False)


if __name__ == '__main__':
    unittest.main()
