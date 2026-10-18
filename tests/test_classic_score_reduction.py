"""The common ranking protocol must retain single-user score arithmetic."""
import unittest

import torch

from scripts.rq2_classic_evaluation import raw_catalog_logits


class CancellationModel:
    def __init__(self):
        generator = torch.Generator().manual_seed(42)
        self.item_embedding = torch.nn.Embedding(53, 100)
        with torch.no_grad():
            self.item_embedding.weight.copy_(torch.randn(53, 100, generator=generator) * 20)
            self.item_embedding.weight[1].copy_(self.item_embedding.weight[0])
        self.users = torch.randn(3, 100, generator=generator)
        self.batch_sizes = []

    def get_user_embedding(self, feed):
        self.batch_sizes.append(len(feed["user_ids"]))
        return self.users[feed["user_ids"]]


class SerialReductionTests(unittest.TestCase):
    def test_matches_single_user_dot_products_with_catalog_chunks_and_ties(self):
        model = CancellationModel()
        feed = dict(user_ids=torch.arange(3), history_ids=torch.zeros(3, 4, dtype=torch.long))
        expected = torch.stack([(user * model.item_embedding.weight).sum(-1) for user in model.users])
        with torch.no_grad():
            for chunk in (1, 7, 53, 64):
                actual = raw_catalog_logits(model, feed, 53, "serial-reduction", chunk)
                self.assertTrue(torch.equal(actual, expected))
                self.assertTrue(torch.equal(torch.sigmoid(actual), torch.sigmoid(expected)))
                self.assertTrue(torch.equal(
                    torch.argsort(actual, descending=True, stable=True),
                    torch.argsort(expected, descending=True, stable=True)))
        self.assertEqual(set(model.batch_sizes), {1})

    def test_rejects_invalid_scoring_settings(self):
        for method, chunk in (("unknown", 7), ("serial-reduction", 0)):
            with self.assertRaises(ValueError):
                raw_catalog_logits(None, {}, 53, method, chunk)


if __name__ == "__main__":
    unittest.main()
