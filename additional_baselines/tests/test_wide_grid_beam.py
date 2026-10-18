"""Exercise native beam expansion without loading a trained checkpoint."""
import ast
import itertools
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Union
from unittest.mock import patch

import torch

from additional_baselines.prepare_beauty import ROOT


class Cache:
    def __init__(self, *args):
        pass

    def reorder_cache(self, indices):
        assert indices.ndim == 1


def native_methods():
    path = ROOT / "GRID/src/models/modules/semantic_id/tiger_generation_model.py"
    tree = ast.parse(path.read_text())
    methods = {node.name: node for node in ast.walk(tree)
               if isinstance(node, ast.FunctionDef) and node.name in {"generate", "_beam_search_one_step"}}
    namespace = {"torch": torch, "Union": Union, "EncoderDecoderCache": Cache, "DynamicCache": Cache}
    for node in methods.values():
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def model(width, vocabulary, batch_size):
    methods = native_methods()

    class Model:
        top_k_for_generation = width
        num_embeddings_per_hierarchy = vocabulary
        num_hierarchies = 3
        should_check_prefix = False
        generate = methods["generate"]
        _beam_search_one_step = methods["_beam_search_one_step"]

        def encoder_forward_pass(self, **kwargs):
            encoded = torch.arange(batch_size).float().reshape(batch_size, 1, 1)
            return encoded, torch.ones(batch_size, 1)

        def decoder_forward_pass(self, future_ids, encoder_output, attention_mask_for_encoder, **kwargs):
            beams = 1 if future_ids is None else future_ids.shape[0] // batch_size
            assert encoder_output.shape[0] == attention_mask_for_encoder.shape[0] == batch_size * beams
            assert encoder_output[:, 0, 0].tolist() == [b for b in range(batch_size) for _ in range(beams)]
            self.level = 0 if future_ids is None else future_ids.shape[1]
            return torch.zeros(batch_size * beams, 1, 1), Cache()

    instance = Model()
    # Different distributions at each level avoid most product-probability ties.
    logits = [torch.linspace(-1, 1 + level * .3, vocabulary) for level in range(3)]
    instance.decoder = SimpleNamespace(decoder_mlp=[
        lambda hidden, values=values: values.expand(hidden.shape[0], -1)
        for values in logits
    ])
    return instance, logits


class WideGridBeamTests(unittest.TestCase):
    def test_narrow_and_wide_beams_match_exhaustive_probabilities(self):
        for width in (2, 8):
            with self.subTest(width=width):
                instance, logits = model(width, vocabulary=3, batch_size=2)
                ids, probabilities = instance.generate(torch.ones(2, 1), torch.zeros(2, 1, dtype=torch.long))
                distributions = [torch.softmax(values, dim=-1) for values in logits]
                expected = sorted((float(distributions[0][a] * distributions[1][b] * distributions[2][c])
                                   for a, b, c in itertools.product(range(3), repeat=3)), reverse=True)[:width]
                self.assertEqual(tuple(ids.shape), (2, width, 3))
                for batch in range(2):
                    self.assertEqual(len({tuple(sid) for sid in ids[batch].tolist()}), width)
                    self.assertTrue(torch.allclose(probabilities[batch], torch.tensor(expected)))

    def test_beam_300_expands_from_256_tokens_with_aligned_encoder_batches(self):
        instance, _ = model(width=300, vocabulary=256, batch_size=2)
        ids, probabilities = instance.generate(torch.ones(2, 1), torch.zeros(2, 1, dtype=torch.long))
        self.assertEqual(tuple(ids.shape), (2, 300, 3))
        self.assertEqual(tuple(probabilities.shape), (2, 300))
        self.assertTrue(torch.isfinite(probabilities).all())
        self.assertTrue((probabilities[:, :-1] >= probabilities[:, 1:]).all())

    def test_candidate_preparation_rejects_invalid_generation_options(self):
        from additional_baselines.prepare_rerank_candidates import run
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "candidates"
            for kwargs in ({"generation_beam_width": 0}, {"generation_beam_width": True},
                           {"require_min_candidates": -1}):
                with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, "positive integer"):
                    run("amazon-sports", output, tokenizer="rvq", **kwargs)
            self.assertFalse(output.exists())

    def test_candidate_generation_uses_override_and_gates_incomplete_splits(self):
        from additional_baselines import prepare_rerank_candidates as preparation
        from additional_baselines import ifairlrs_grid, prepare_beauty
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            export = root / "export"
            export.mkdir()
            config = "sequence_length: 160\nnum_hierarchies: 4\ntop_k_for_generation: 100\n"
            (export / "config.yaml").write_text(config)
            data = root / "recommendation/processed_dataset/amazon-sports"
            data.mkdir(parents=True)
            (data / "amazon-sports.train").write_text("training")
            verified = SimpleNamespace(item_groups={"brand": {0: "brand"}}, metadata={
                "checkpoint": {"path": "checkpoint"}, "sid_mapping": {"path": "sids"}})

            class Model:
                def __init__(self, settings):
                    self.top_k_for_generation = settings["top_k_for_generation"]

                def to(self, device):
                    return self

            for count in (19, 20):
                output = root / str(count)

                def regenerate(instance, *args):
                    self.assertEqual(instance.top_k_for_generation, 300)
                    return SimpleNamespace(metadata={}, requests=[SimpleNamespace(candidate_ids=list(range(count)))]), {}

                with patch.object(preparation, "ROOT", root), \
                     patch.object(preparation, "source_export", return_value=export), \
                     patch.object(prepare_beauty, "load_beauty", return_value=(verified, None, None)), \
                     patch.object(ifairlrs_grid, "original_module", return_value=SimpleNamespace(GRIDBaseModel=Model)), \
                     patch.object(ifairlrs_grid, "read_rows", return_value=[(0, [], 0)]), \
                     patch.object(ifairlrs_grid, "training_popularity", return_value={0: 1}), \
                     patch.object(ifairlrs_grid, "regenerate", side_effect=regenerate), \
                     patch.object(preparation, "save", side_effect=lambda artifact, path: path.write_text("candidates")):
                    options = dict(tokenizer="rvq", generation_beam_width=300, require_min_candidates=20)
                    if count == 19:
                        with self.assertRaisesRegex(ValueError, "preparation is incomplete"):
                            preparation.run("amazon-sports", output, **options)
                        self.assertFalse((output / "complete.json").exists())
                    else:
                        preparation.run("amazon-sports", output, **options)
                        self.assertTrue((output / "complete.json").exists())
                self.assertEqual((export / "config.yaml").read_text(), config)


if __name__ == "__main__":
    unittest.main()
