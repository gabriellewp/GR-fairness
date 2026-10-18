"""Regression checks executable without pytest: python -m unittest this.module."""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import torch


class SelectedWeightsTest(unittest.TestCase):
    def test_source_winner_restores_source_before_test(self):
        self.check_selection(0)

    def test_intermediate_winner_restores_best_before_test(self):
        self.check_selection(1)

    def check_selection(self, winner):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        tmp_path = Path(temporary.name)
        def setattr_for_test(obj, name, value):
            context = patch.object(obj, name, value)
            context.start()
            self.addCleanup(context.stop)
        monkeypatch = SimpleNamespace(setattr=setattr_for_test)
        """Exercise selection through training and test, including a losing final update."""
        from additional_baselines import continuation_grid as runner, sprint_grid

        checkpoint = tmp_path / "source.ckpt"
        checkpoint.write_bytes(b"source")
        export = tmp_path / "export"
        export.mkdir()
        (export / "config.yaml").write_text("top_k_for_generation: 20\nsequence_length: 4\nnum_hierarchies: 1\n")
        for suffix in ("train", "valid.ranking", "test.ranking"):
            (tmp_path / f"toy.{suffix}").write_text("fixture")
        reference = runner.file_reference(checkpoint)
        verified = SimpleNamespace(metadata={"checkpoint": reference, "sid_mapping": reference},
                                   item_groups={"brand": {i: str(i) for i in range(5)}})
        schedule = dict(users_per_batch=1, cap=8, accumulation=1, history_length=1,
                        model_history_length=2, weight_decay=0)
        monkeypatch.setattr(runner, "load_beauty", lambda **kw: (verified, None, None))
        monkeypatch.setattr(runner, "native_schedule", lambda path: schedule)
        monkeypatch.setattr(runner, "read_rows", lambda path, **kw: [(0, [0], 1)])

        class Model(torch.nn.Module):
            def __init__(self, config):
                super().__init__()
                self.grid = torch.nn.Linear(1, 1, bias=False)
                self.grid.weight.data.fill_(1)
                self.tokenizer = SimpleNamespace(sid_table=torch.arange(5).reshape(5, 1))

        monkeypatch.setattr(runner, "original_module", lambda name: SimpleNamespace(GRIDBaseModel=Model))
        monkeypatch.setattr(runner, "resume_optimizer", lambda model, *a, **kw:
                            (torch.optim.SGD(model.grid.parameters(), lr=.1), {"resumed": True}))

        class Regularizer:
            def __init__(self, model, **kw):
                self.model = model

            def objective(self, interaction):
                return self.model.grid.weight.sum()

            def settings(self):
                return {}

        monkeypatch.setattr(sprint_grid, "GridEncoderSPRINT", Regularizer)
        captured = []

        def generate(model, requests, sids, mapping, metadata, **kw):
            captured.append((metadata["split"]["name"], float(model.grid.weight.item())))
            return SimpleNamespace(metadata=metadata)

        monkeypatch.setattr(runner, "generate_grid", generate)
        scores = iter([.5, .4, .3] if winner == 0 else [.5, .6, .3])
        monkeypatch.setattr(runner, "evaluate", lambda artifact, cutoffs:
                            {"at_k": {"5": {"recall": next(scores, .5)}}, "candidate_target_recall": 1})
        monkeypatch.setattr(runner, "save", lambda artifact, path: path.write_text("fixture"))
        identity = dict(checkpoint=reference, sid_mapping=reference, seed=42,
                        plan_sha256=runner.batch_digest(runner.native_batches([(0, [0, 1])], updates=2,
                            seed=42, users_per_batch=1, cap=8, accumulation=1, history_length=1)),
                        max_updates=2, validation_interval=1, validation_requests=None,
                        learning_rate=.001, optimizer_state="require", generation_beam_width=20, train_users=None)
        control = tmp_path / "control.json"
        control.write_text(json.dumps(dict(mode="ordinary", identity=identity, selected_update=1,
                                          optimizer_audit={"resumed": True})))
        report = runner.run_continuation(mode="sprint", output_dir=tmp_path / "run", control_report=control,
            device="cpu", max_updates=2, validation_interval=1, export_dir=export, data_dir=tmp_path, dataset="toy")
        self.assertEqual(report["selected_update"], winner)
        self.assertEqual(captured[-1][0], "test")
        self.assertAlmostEqual(captured[-1][1], 1 - .1 * winner, places=6)
        self.assertEqual(report["selected_checkpoint"] == reference, winner == 0)

if __name__ == "__main__":
    unittest.main()
