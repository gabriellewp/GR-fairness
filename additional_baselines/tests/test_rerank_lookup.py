"""Compare matrix-free production runs against unchanged dense backends."""

from dataclasses import replace
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from additional_baselines.candidates import ExperimentInputs, Request, file_reference
from additional_baselines.rerank import FairDiverseReranker
from additional_baselines.rerank.legacy import load_class


class LookupParityTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "source"
        path.write_text("fixture")
        reference = file_reference(path)
        self.metadata = {"checkpoint": reference, "sid_mapping": reference,
                         "split": {"name": "validation", "source": reference}, "seed": 42,
                         "generation": {"beam_width": 30}, "score_convention": "log_probability",
                         "candidate_membership": "fixture", "short_list_policy": "return_available"}

    def fixture(self, seed, tied=False):
        rng = np.random.default_rng(seed)
        groups = {10 * i + 1: str(i % 3) for i in range(64)}
        requests = []
        for row in range(8):
            ids = rng.choice(list(groups), size=24, replace=False).tolist()
            scores = (rng.integers(-3, 1, size=24) if tied else -rng.uniform(0, 5, size=24)).tolist()
            requests.append(Request(f"r{row}", ids, scores, [ids[0]]))
        return ExperimentInputs(self.metadata.copy(), {"brand": groups}, requests)

    def compare(self, model, inputs, k, **options):
        reranker = FairDiverseReranker(model=model, group="brand", fairness_lambda=options.pop("fairness_lambda", 1),
                                      weights={"0": 1., "1": 1., "2": 1.} if model == "PMMF" else None,
                                      **options)
        states = []
        module = sys.modules[load_class(model).__module__]
        original = getattr(module, "compute_next_dual", None)
        def record(*args):
            value = original(*args)
            states.append(value.copy())
            return value
        with patch.object(module, "compute_next_dual", record, create=True):
            dense = replace(reranker, matrix_mode="dense").rerank(inputs, k, trace=True)
            reference_states = states.copy()
            states.clear()
            lookup = reranker.rerank(inputs, k, trace=True)
        self.assertEqual(lookup.rankings, dense.rankings)
        self.assertEqual(lookup.request_order, dense.request_order)
        self.assertEqual(lookup.trace, dense.trace)
        self.assertEqual(lookup.diagnostics, dense.diagnostics)
        if original:
            np.testing.assert_allclose(states, reference_states, rtol=1e-10, atol=1e-10)

    def test_rankings_states_ties_cutoffs_horizons_and_order(self):
        for model in ("CPFair", "PMMF"):
            for seed in range(3):
                for tied in (False, True):
                    for k, horizon in ((1, None), (5, None), (5, 3)):
                        with self.subTest(model=model, seed=seed, tied=tied, k=k, horizon=horizon):
                            self.compare(model, self.fixture(seed, tied), k, horizon=horizon, order_seed=42,
                                         fairness_lambda=0 if tied else 1)

    def test_no_dense_matrices_and_request_independent_memory_estimate(self):
        inputs = self.fixture(1)
        allocation = np.full
        shapes = []
        def spy(shape, *args, **kwargs):
            shapes.append(shape)
            return allocation(shape, *args, **kwargs)
        reranker = FairDiverseReranker(model="CPFair", group="brand", fairness_lambda=1, max_dense_bytes=5000)
        with patch("additional_baselines.rerank.fairdiverse.np.full", spy):
            first = reranker.rerank(inputs, 5)
        self.assertTrue(all(isinstance(shape, int) for shape in shapes))
        with self.assertRaisesRegex(ValueError, "array estimate"):
            replace(reranker, matrix_mode="dense").rerank(inputs, 5)
        inputs.requests += [replace(request, request_id=request.request_id + "copy") for request in inputs.requests]
        second = reranker.rerank(inputs, 5)
        self.assertEqual(first.config["estimated_array_bytes"], second.config["estimated_array_bytes"])

    def test_budget_overshoot_matches_and_exhausted_provider_is_rejected(self):
        inputs = ExperimentInputs(self.metadata.copy(), {"brand": {0: "a", 1: "a", 2: "b", 3: "b"}},
            [Request("r0", [0, 1], [0.9, 0.8], [0]), Request("r1", [2, 3], [0.9, 0.8], [2])])
        inputs.metadata["score_convention"] = "utility"
        reranker = FairDiverseReranker(model="PMMF", group="brand", fairness_lambda=1, weights={"a": 0.3, "b": 1})
        lookup = reranker.rerank(inputs, 2, trace=True)
        dense = replace(reranker, matrix_mode="dense").rerank(inputs, 2, trace=True)
        self.assertEqual(lookup.rankings, dense.rankings)
        self.assertEqual(lookup.trace, dense.trace)
        self.assertEqual(lookup.diagnostics, dense.diagnostics)
        inputs.requests[1] = Request("r1", [0, 1], [0.9, 0.8], [0])
        for mode in ("lookup", "dense"):
            with self.assertRaisesRegex(ValueError, "exhausted provider"):
                replace(reranker, matrix_mode=mode).rerank(inputs, 2)
