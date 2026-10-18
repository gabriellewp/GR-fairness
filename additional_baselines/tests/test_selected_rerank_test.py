import json
from types import SimpleNamespace

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from additional_baselines.candidates import ExperimentInputs, Request, file_reference, save
from additional_baselines import selected_rerank_test as selected


def make_inputs(tmp_path):
    dataset = "amazon-sports"
    data = tmp_path / "recommendation/processed_dataset" / dataset
    data.mkdir(parents=True)
    source = tmp_path / "candidates"
    source.mkdir()
    (data / "user_activity.csv").write_text("user_id,activity_group\n7,low\n")
    reference = tmp_path / "weights"
    reference.write_text("original")
    ref = file_reference(reference)
    for split, suffix in (("validation", "valid"), ("test", "test")):
        path = data / f"{dataset}.{suffix}.ranking"
        path.write_text("user_id\ttarget_item\n7\t0\n")
        metadata = {"checkpoint": ref, "sid_mapping": ref, "seed": 42,
                    "generation": {"beam_width": 20}, "score_convention": "log_probability",
                    "candidate_membership": "explicit", "short_list_policy": "return_available",
                    "split": {"name": split, "source": file_reference(path)}}
        rid = f"{split}:0:user:7"
        artifact = ExperimentInputs(metadata,
            {"brand": {i: str(i % 2) for i in range(20)}, "popularity": {i: str(i % 2) for i in range(20)}},
            [Request(rid, list(range(20)), [-float(i + 1) for i in range(20)], [0])])
        save(artifact, source / f"{split}_candidates.json")
        (source / f"{split}_histories.json").write_text(json.dumps({rid: [0, 1]}))

    class FakeReranker:
        def __init__(self, **kwargs):
            pass

        def rerank(self, artifact, k):
            return SimpleNamespace(rankings={r.request_id: artifact.ranking(r)[:k] for r in artifact.requests})

    return dataset, source, FakeReranker


class SelectedRerankTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.dataset, self.source, fake = make_inputs(self.root)
        for replacement in (patch.object(selected, "ROOT", self.root),
                            patch.object(selected, "FairDiverseReranker", fake)):
            replacement.start()
            self.addCleanup(replacement.stop)

    def test_all_methods_fit_rr_only_on_validation(self):
        output = self.root / "results"
        selected.run(1, 30, output, self.dataset, self.source, rr_alpha=0.05)
        self.assertTrue(all((output / f"{m}_test_results.json").exists() for m in ("cpfair", "pmmf", "rr")))
        correction = json.loads((output / "rr_correction.json").read_text())
        self.assertEqual(correction["validation_metadata"]["split"]["name"], "validation")
        report = json.loads((output / "rr_test_results.json").read_text())
        self.assertEqual(set(report["rankings"]), {"test:0:user:7"})

    def test_reject_mismatched_checkpoint_before_test(self):
        path = self.source / "test_candidates.json"
        artifact = json.loads(path.read_text())
        artifact["metadata"]["checkpoint"]["sha256"] = "0" * 64
        path.write_text(json.dumps(artifact))
        output = self.root / "results"
        with self.assertRaisesRegex(ValueError, "checkpoint differs"):
            selected.run(1, 30, output, self.dataset, self.source, rr_alpha=0.05)
        self.assertFalse(output.exists())

    def test_rr_only_does_not_repeat_completed_cpfair_pmmf_tests(self):
        output = self.root / "rr_only"
        selected.run(0, 0, output, self.dataset, self.source, rr_alpha=.02,
                     methods=["rr"], selection_provenance={"max_relative_recall_loss": .05})
        self.assertTrue((output / "rr_test_results.json").exists())
        self.assertFalse((output / "cpfair_test_results.json").exists())
        self.assertFalse((output / "pmmf_test_results.json").exists())
        selection = json.loads((output / "selection.json").read_text())
        self.assertEqual(selection["methods"], ["rr"])
        self.assertEqual(selection["validation_selection"]["max_relative_recall_loss"], .05)
