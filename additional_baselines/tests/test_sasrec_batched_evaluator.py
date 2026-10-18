"""Parity of batched SASRec scoring, top-K ties, metrics and score exports."""
import importlib
import unittest
from unittest.mock import patch

import numpy as np
import torch

from additional_baselines.ifairlrs_grid import original_module

evaluator_module = importlib.import_module("fairdiverse.recommendation.evaluator")
SASRec = original_module("sasrec").SASRec


def evaluator(config):
    result = evaluator_module.Ranking_Evaluator.__new__(evaluator_module.Ranking_Evaluator)
    result.config = config
    result.iid2pid = {i: i % 3 for i in range(config["item_num"])}
    return result


class BatchedSASRecTests(unittest.TestCase):
    def test_topk_matches_full_numpy_sort_with_cutoff_and_saturated_ties(self):
        rng = np.random.default_rng(42)
        for size in (5, 100, 114625):
            samples = [rng.random(size).astype(np.float32), np.ones(size, dtype=np.float32),
                       rng.integers(0, 5, size=size).astype(np.float32),
                       rng.random(size).astype(np.float32) + np.float32(1)]
            for scores in samples:
                for k in (5, 10, 20):
                    np.testing.assert_array_equal(evaluator_module._legacy_topk_args(scores, k),
                                                  np.argsort(scores)[::-1][:k])

    def test_sasrec_batched_scores_match_serial_in_arbitrary_candidate_order(self):
        config = dict(user_num=3, item_num=53, embedding_size=8, n_layers=1, n_heads=2,
                      inner_size=16, hidden_dropout_prob=0., attn_dropout_prob=0., hidden_act="gelu",
                      layer_norm_eps=1e-12, initializer_range=.02, history_length=4,
                      padding_item_id=-1, classic_eval_item_chunk_size=7)
        torch.manual_seed(42)
        model = SASRec(config).eval()
        feed = {"user_ids": torch.arange(3), "history_ids": torch.tensor(
            [[0, 2, -1, -1], [3, 4, 5, -1], [8, 9, 10, 11]])}
        items = torch.stack([torch.randperm(53)[:31] for _ in range(3)])
        with torch.no_grad():
            serial = torch.cat([model.full_predict({key: value[i:i+1] for key, value in feed.items()},
                                                   items[i:i+1]) for i in range(3)])
            for batch_size in (1, 2, 3):
                batched = model.full_predict_batch(feed, items, batch_size)
                torch.testing.assert_close(batched, serial, rtol=1e-6, atol=1e-7)
        with self.assertRaisesRegex(ValueError, "catalog"):
            model.full_predict_batch(feed, torch.full_like(items, 53), 3)

    def test_evaluator_matches_metrics_user_fairness_and_csr_exports(self):
        torch.manual_seed(7)
        scores = torch.rand(4, 100)
        scores[1] = .5  # Catalog-wide ties must retain the original sort order.
        scores[2, :25] = 1.  # Saturated sigmoid scores tied across top-20.

        class Model(torch.nn.Module):
            IR_type = ["retrieval"]

            def full_predict(self, users, items):
                return scores[users["user_ids"]].gather(1, items)

            def full_predict_batch(self, users, items, batch_size):
                return self.full_predict(users, items)

        batch = (torch.arange(4), torch.zeros(4, 4, dtype=torch.long),
                 torch.stack([torch.randperm(100) for _ in range(4)]), torch.tensor([1, 2, 3, 1]))
        config = dict(model="SASRec", item_num=100, device="cpu", group_num=3, topk=[5, 10, 20],
                      fairness_type="Exposure", decimals=8, mmf_eval_ratio=.5, user_fairness=True,
                      user_activity_path="unused.csv")
        activity = {str(i): {} for i in range(4)}
        for fairness_type in ("Exposure", "Relevance"):
            for store_scores in (False, True):
                with self.subTest(fairness_type=fairness_type, store_scores=store_scores):
                    serial = evaluator(dict(config, fairness_type=fairness_type))
                    fast = evaluator(dict(config, fairness_type=fairness_type, classic_eval_batch_size=4))
                    with patch.object(evaluator_module, "load_activity", return_value=activity):
                        expected = serial.eval([batch], Model(), store_scores=store_scores)
                        actual = fast.eval([batch], Model(), store_scores=store_scores)
                    self.assertEqual(serial.last_user_metrics, fast.last_user_metrics)
                    if store_scores:
                        self.assertEqual(expected[0], actual[0])
                        np.testing.assert_array_equal(expected[1].toarray(), actual[1].toarray())
                        np.testing.assert_array_equal(serial.last_ground_truth.toarray(), fast.last_ground_truth.toarray())
                    else:
                        self.assertEqual(expected, actual)


if __name__ == "__main__":
    unittest.main()
