import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from additional_baselines import report_current_diagnostics as reporter
from additional_baselines.candidates import ExperimentInputs, Request, file_reference, save
from additional_baselines.evaluation import evaluate


class CurrentDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.dataset = "amazon-toys"
        data = self.root / "recommendation/processed_dataset" / self.dataset
        data.mkdir(parents=True)
        split = data / (self.dataset+".test.ranking")
        split.write_text("user_id\ttarget_item\n7\t21\n8\t0\n")
        activity = data / "user_activity.csv"
        activity.write_text("user_id,train_interaction_count,activity_group\n7,1,low\n8,9,high\n")
        (data / "user_activity_metadata.json").write_text(json.dumps(dict(
            mapping=file_reference(activity), provenance=dict(source_fingerprint="fixture"))))
        (data / "grid_data_manifest.json").write_text(json.dumps(dict(source_fingerprint="fixture")))
        ref = file_reference(split)
        self.inputs = ExperimentInputs(dict(checkpoint=ref, sid_mapping=ref, seed=42,
            generation=dict(beam_width=20), score_convention="utility", candidate_membership="fixture",
            short_list_policy="return_available", split=dict(name="test", source=ref)),
            {"brand": {i: str(i%2) for i in range(22)}},
            [Request("test:0:user:7", [1], [1.], [21]),
             Request("test:1:user:8", list(range(21)), [float(21-i) for i in range(21)], [0])])
        self.candidates = self.root / "test_candidates.json"
        save(self.inputs, self.candidates)
        self.patcher = patch.object(reporter, "ROOT", self.root)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def test_candidate_misses_retained_and_conditional_metrics_separated(self):
        result = reporter.report(self.dataset, self.candidates, self.root/"report", bootstrap_samples=0)
        diagnostics = result["candidate_diagnostics"]
        self.assertEqual(diagnostics["overall"]["target_coverage"], .5)
        self.assertIsNone(diagnostics["low"]["conditional_metrics"]["recall@5"])
        self.assertEqual(diagnostics["high"]["conditional_metrics"]["recall@5"], 1.)
        self.assertEqual(result["activity"]["statistics"]["overall"]["recall@5"], .5)

    def test_independent_cutoff_allocations_are_preserved(self):
        ranks = {str(k): {r.request_id: self.inputs.ranking(r) for r in self.inputs.requests} for k in (5,10,20)}
        ranks["5"]["test:1:user:8"] = list(reversed(range(21)))
        metrics = {str(k): evaluate(self.inputs, (k,), ranks[str(k)])["at_k"][str(k)] for k in (5,10,20)}
        path = self.root / "cpfair_test_results.json"
        path.write_text(json.dumps(dict(input_metadata=self.inputs.metadata, rankings_by_k=ranks,
            metrics=dict(at_k=metrics, candidate_target_recall=.5))))
        result = reporter.report(self.dataset, self.candidates, self.root/"report", path,
                                 method="cpfair", bootstrap_samples=0)
        overall = result["activity"]["statistics"]["overall"]
        self.assertEqual(overall["recall@5"], 0.)
        self.assertEqual(overall["recall@10"], .5)

    def test_reject_wrong_selected_checkpoint_before_reporting(self):
        with self.assertRaisesRegex(ValueError, "checkpoint hash differs"):
            reporter.report(self.dataset, self.candidates, self.root/"report", bootstrap_samples=0,
                expected_checkpoint_reference=dict(self.inputs.metadata["checkpoint"], sha256="wrong"))
        self.assertFalse((self.root/"report").exists())


if __name__ == "__main__":
    unittest.main()
