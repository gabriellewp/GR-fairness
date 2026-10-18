import importlib.util
from pathlib import Path
import unittest

from additional_baselines.candidates import ExperimentInputs, Request
from additional_baselines.evaluation import evaluate


def artifact(requests):
    reference = {"path": "/tmp/fixture", "sha256": "a" * 64}
    metadata = dict(checkpoint=reference, sid_mapping=reference,
                    split=dict(name="test", source=reference), seed=42,
                    generation=dict(beam_width=100), score_convention="probability",
                    candidate_membership="fixture", short_list_policy="return_available")
    return ExperimentInputs(metadata, {"brand": {0: "a", 1: "b", 2: "c"}}, requests)


def module_from_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ROOT = Path(__file__).resolve().parents[2]
checker = module_from_file("rq1_checker_test", ROOT / "scripts/check_rq1_grid_agreement.py")
CUTOFFS = checker.CUTOFFS
common_ranking_metrics = checker.common_ranking_metrics
fairdiverse_common_metrics = checker.fairdiverse_common_metrics
merge_metrics = checker.merge_metrics


class AgreementTest(unittest.TestCase):
    def setUp(self):
        self.rows = [Request("empty", [], [], [2]), Request("hit", [1, 0], [.7, .3], [1]),
                     Request("miss", [0], [.8], [2]), Request("second", [0, 1], [.9, .8], [1])]

    def test_chunking_preserves_global_exposure_and_missing_targets(self):
        expected = evaluate(artifact(self.rows), CUTOFFS)
        actual = merge_metrics([evaluate(artifact(self.rows[:1]), CUTOFFS),
                                evaluate(artifact(self.rows[1:]), CUTOFFS)])
        self.assertEqual(actual["request_count"], 4)
        for k in map(str, CUTOFFS):
            self.assertEqual(actual["at_k"][k]["groups"]["brand"]["exposure_counts"],
                             expected["at_k"][k]["groups"]["brand"]["exposure_counts"])
            self.assertAlmostEqual(actual["at_k"][k]["ndcg"], expected["at_k"][k]["ndcg"])
            self.assertAlmostEqual(actual["at_k"][k]["groups"]["brand"]["gini"],
                                   expected["at_k"][k]["groups"]["brand"]["gini"])
        self.assertEqual(actual["at_k"]["5"]["recall"], .5)

    def test_three_metric_implementations_agree_on_common_rankings(self):
        native = module_from_file("rq1_native_test", ROOT / "GRID/src/components/eval_metrics.py")
        fd = module_from_file("rq1_fd_test", ROOT / "fairdiverse/recommendation/metric.py")
        inputs = artifact(self.rows)
        expected = evaluate(inputs, CUTOFFS)
        for actual in [common_ranking_metrics(inputs, {"ndcg": native.NDCG, "recall": native.Recall}),
                       fairdiverse_common_metrics(inputs, fd)]:
            for k in map(str, CUTOFFS):
                for name in ["ndcg", "recall"]:
                    self.assertAlmostEqual(actual[k][name], expected["at_k"][k][name], places=6)


if __name__ == "__main__":
    unittest.main()
