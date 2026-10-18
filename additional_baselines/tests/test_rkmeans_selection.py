import unittest
import csv
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from additional_baselines.rkmeans_selected_tests import select
from additional_baselines import rkmeans_selected_tests as selected
from additional_baselines.candidates import file_reference


class ValidationSelectionTest(unittest.TestCase):
    def setUp(self):
        self.rows = [dict(k=5, parameters="{}", recall=.1, mgu=.5),
                     dict(k=5, parameters='{"alpha":0}', recall=.1, mgu=.5),
                     dict(k=5, parameters='{"alpha":0.02}', recall=.096, mgu=.2),
                     dict(k=5, parameters='{"alpha":0.1}', recall=.09, mgu=.1),
                     dict(k=10, parameters='{"alpha":0.1}', recall=.2, mgu=0)]

    def test_budget_excludes_excessive_loss_and_other_cutoffs(self):
        self.assertEqual(select(self.rows, "mgu", .05)["parameters"], {"alpha": .02})
        self.assertEqual(select(self.rows, "mgu", .01)["parameters"], {"alpha": 0})

    def test_tied_fairness_prefers_higher_recall(self):
        self.rows.append(dict(k=5, parameters='{"alpha":0.03}', recall=.097, mgu=.2))
        self.assertEqual(select(self.rows, "mgu", .05)["parameters"], {"alpha": .03})

    def test_prepare_accepts_absolute_repository_and_artifact_relative_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifacts = root / "additional_baselines/artifacts"
            candidate_dir = artifacts / "candidates"
            candidate_dir.mkdir(parents=True)
            split = root / "recommendation/processed_dataset/amazon-sports/amazon-sports.valid.ranking"
            split.parent.mkdir(parents=True)
            split.write_text("validation split")
            candidates = candidate_dir / "validation_candidates.json"
            candidates.write_text("candidate artifact")
            histories = candidate_dir / "validation_histories.json"
            histories.write_text("validation histories")
            source = artifacts / "sweep/merged"
            source.mkdir(parents=True)
            (source / "manifest.json").write_text(json.dumps({
                "scope": "full_validation", "method": "rr", "fairness_axis": "mgu",
                "sources": {"candidates": file_reference(candidates),
                            "split": file_reference(split), "histories": file_reference(histories)},
            }))
            with (source / "validation_tradeoff.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(self.rows[0]))
                writer.writeheader()
                writer.writerows(self.rows)
            with patch.object(selected, "ROOT", root), patch.object(selected, "ARTIFACTS", artifacts):
                for directory in (source, source.relative_to(root), source.relative_to(artifacts)):
                    with self.subTest(directory=str(directory)):
                        _, selections = selected.prepare("amazon-sports", .05, candidate_dir,
                                                         sources={"rr": directory})
                        self.assertEqual(selections["rr"]["parameters"], {"alpha": .02})
                        self.assertEqual(selections["rr"]["manifest"], file_reference(source / "manifest.json"))


if __name__ == "__main__":
    unittest.main()
