import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from additional_baselines import tune_beauty_reranking as tuner
from additional_baselines.candidates import load
from additional_baselines.merge_rerank_tuning import merge
from additional_baselines.tests.test_selected_rerank_test import make_inputs


class TuningArrayTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        dataset, self.source, _ = make_inputs(self.root)
        data = self.root / "recommendation/processed_dataset" / dataset
        self.options = dict(candidates=self.source / "validation_candidates.json",
                            histories_path=self.source / "validation_histories.json",
                            split=data / f"{dataset}.valid.ranking", activity=data / "user_activity.csv",
                            strengths=[0, 0.1], floors=[1e-6, 1e-3])

    def shards(self, method):
        tasks = self.root / f"{method}_tasks"
        for index in range(len(tuner.task_plan(method, self.options["strengths"], self.options["floors"]))):
            tuner.run(method, tasks / f"task_{index}", task_index=index, **self.options)
        return tasks

    def test_array_matches_serial_metrics_and_global_frontier(self):
        for method in ("cpfair", "pmmf", "rr"):
            with self.subTest(method=method):
                serial, merged = self.root / f"{method}_serial", self.root / f"{method}_merged"
                tuner.run(method, serial, **self.options)
                merge(self.shards(method), merged)
                for filename in ("validation_tradeoff.csv", "pareto_frontier.csv", "validation_rows.json"):
                    self.assertEqual((serial / filename).read_text(), (merged / filename).read_text())
                manifest = json.loads((merged / "manifest.json").read_text())
                self.assertIsNone(manifest["horizon"])
                self.assertEqual(len(manifest["task_resources"]), manifest["task_count"])

    def test_default_plan_and_rr_combined_cutoffs(self):
        self.assertEqual(len(tuner.task_plan("cpfair")), 12)
        self.assertEqual(len(tuner.task_plan("pmmf")), 12)
        self.assertEqual(len(tuner.task_plan("rr")), 13)
        fit = tuner.fit_rr
        with patch.object(tuner, "fit_rr", wraps=fit) as spy:
            tuner.run("rr", self.root / "rr_one", task_index=1, **self.options)
        self.assertEqual(spy.call_count, 1)
        self.assertEqual(spy.call_args.args[3], (1, 5, 10, 20))

    def test_rr_loads_candidates_once_and_preserves_full_history_check_for_smoke(self):
        with patch.object(tuner, "load", wraps=load) as spy:
            tuner.run("rr", self.root / "rr_load", task_index=0, limit_requests=1, **self.options)
        self.assertEqual(spy.call_count, 1)
        histories = self.source / "validation_histories.json"
        histories.write_text(json.dumps({"validation:0:user:7": [0, 1], "extra": [0]}))
        with self.assertRaisesRegex(ValueError, "histories do not cover"):
            tuner.run("rr", self.root / "bad_histories", task_index=0, limit_requests=1, **self.options)

    def test_merge_rejects_missing_task_mixed_sources_and_conflicting_controls(self):
        tasks = self.shards("cpfair")
        part = tasks / "task_0"
        part.rename(tasks / "hidden_0")
        with self.assertRaisesRegex(ValueError, "incomplete"):
            merge(tasks, self.root / "incomplete")
        (tasks / "hidden_0").rename(part)
        manifest_path = part / "manifest.json"
        original = json.loads(manifest_path.read_text())
        changed = copy.deepcopy(original)
        changed["sources"]["candidates"]["sha256"] = "0" * 64
        manifest_path.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, "different inputs"):
            merge(tasks, self.root / "mixed")
        manifest_path.write_text(json.dumps(original))
        rows_path = part / "validation_rows.json"
        rows = json.loads(rows_path.read_text())
        rows[0]["recall"] = 0.2
        rows_path.write_text(json.dumps(rows))
        with self.assertRaisesRegex(ValueError, "Control metrics differ"):
            merge(tasks, self.root / "controls")
        self.assertFalse((self.root / "incomplete").exists())
        self.assertFalse((self.root / "mixed").exists())
        self.assertFalse((self.root / "controls").exists())

    def test_validation_checks_membership_without_scanning_keys_per_request(self):
        inputs = load(self.source / "validation_candidates.json")
        class NoKeys(dict):
            def keys(self):
                raise AssertionError("Do not scan catalog keys for each request")
        inputs.item_groups = {name: NoKeys(mapping) for name, mapping in inputs.item_groups.items()}
        inputs.validate()
        del inputs.item_groups["brand"][0]
        with self.assertRaisesRegex(ValueError, "Missing group mapping"):
            inputs.validate()

    def test_invalid_task_fails_before_candidate_loading(self):
        with patch.object(tuner, "load", side_effect=AssertionError("Unexpected load")):
            with self.assertRaisesRegex(ValueError, "Task index"):
                tuner.run("cpfair", self.root / "invalid", task_index=999, **self.options)

    def test_submission_schedules_array_and_dependent_merge_without_real_jobs(self):
        bin_dir = self.root / "bin"
        bin_dir.mkdir()
        sbatch = bin_dir / "sbatch"
        sbatch.write_text('#!/bin/bash\nprintf "%s\\n" "$*" >> "$SUBMIT_LOG"\n'
                          'if [[ "$*" == *--array=* ]]; then echo 123; else echo 124; fi\n')
        sbatch.chmod(0o700)
        log = self.root / "submission.log"
        result = subprocess.run(["bash", str(tuner.ROOT / "additional_baselines/submit_rerank_tuning_array.sh"),
                                 "pmmf", "amazon-pet-supplies", str(self.source), "--dependency=afterok:99"],
                                env={**os.environ, "GRID_RERANK_PYTHON": sys.executable,
                                     "PATH": str(bin_dir) + ":" + os.environ["PATH"], "SUBMIT_LOG": str(log)},
                                capture_output=True, text=True, check=True)
        array, merge_job = log.read_text().splitlines()
        self.assertIn("--array=0-11%1", array)
        self.assertIn("--mem=96G", array)
        self.assertIn("--dependency=afterok:99", array)
        self.assertIn("--dependency=afterok:123", merge_job)
        self.assertIn("/tasks", merge_job)
        self.assertIn("/merged", merge_job)
        self.assertIn("Array: 123", result.stdout)


if __name__ == "__main__":
    unittest.main()
