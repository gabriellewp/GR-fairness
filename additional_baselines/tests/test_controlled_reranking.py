from copy import deepcopy
import csv
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from additional_baselines.candidates import ExperimentInputs, Request, file_reference, save
from additional_baselines.controlled_reranking import validate_pool
from additional_baselines import controlled_reranking as runner
from additional_baselines import rerank


def pool():
    return ExperimentInputs(dict(split=dict(name="test",source="fixture"), sid_mapping="same",
        generation=dict(beam_width=100), seed=42, checkpoint="base", candidate_membership="same",
        score_convention="log_probability"), {"brand": {0:"a",1:"b"}},
        [Request("test:0:user:7",[0],[-1.],[1])])


class ControlledPoolTests(unittest.TestCase):
    def test_continuation_is_loaded_strictly_after_native_initialization(self):
        import torch
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root/"GRID/source.ckpt"
            source.parent.mkdir()
            selected = root/"additional_baselines/selected_checkpoint.ckpt"
            selected.parent.mkdir()
            expected = torch.nn.Linear(2,2)
            with torch.no_grad():
                expected.weight.fill_(3.)
            torch.save(dict(state_dict=expected.state_dict()),selected)
            class Model(torch.nn.Module):
                def __init__(self,config):
                    super().__init__()
                    if config["grid_checkpoint_path"] != str(source):
                        raise ValueError("Constructor must receive native GRID source")
                    self.grid = torch.nn.Linear(2,2)
            model = runner.load_selected_model(Model,{},selected,source,"cpu")
            torch.testing.assert_close(model.grid.weight,expected.weight)
            self.assertEqual(model.loaded_checkpoint_path,str(selected.resolve()))
            torch.save(dict(state_dict={"wrong_key":torch.zeros(1)}),selected)
            with self.assertRaises(RuntimeError):
                runner.load_selected_model(Model,{},selected,source,"cpu")

    def test_checkpoint_and_candidates_may_change(self):
        base = pool()
        changed = deepcopy(base)
        changed.metadata["checkpoint"] = "rw"
        changed.requests[0].candidate_ids = [1]
        validate_pool(changed,base)

    def test_reject_generation_or_score_drift(self):
        for key, value in (("generation",dict(beam_width=20)),("score_convention","probability")):
            changed = pool()
            changed.metadata[key] = value
            with self.assertRaisesRegex(ValueError,"differs in "+key):
                validate_pool(changed,pool())

    def test_reject_user_target_and_group_drift(self):
        changed = pool()
        changed.requests[0].targets = [0]
        with self.assertRaisesRegex(ValueError,"order or targets"):
            validate_pool(changed,pool())
        changed = pool()
        changed.item_groups["brand"][0] = "b"
        with self.assertRaisesRegex(ValueError,"group definitions"):
            validate_pool(changed,pool())

    def test_four_pools_use_same_setting_and_independent_calls(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "recommendation/processed_dataset" / runner.DATASET
            data.mkdir(parents=True)
            split = data / (runner.DATASET+".test.ranking")
            split.write_text("user_id\ttarget_item\n7\t21\n8\t0\n")
            (data / "user_activity.csv").write_text("user_id,train_interaction_count,activity_group\n7,1,low\n8,9,high\n")
            ref = file_reference(split)
            cases = []
            for index,name in enumerate(("base","ordinary","rw","sprint")):
                folder = root / f"pools/case_{index}/test_full"
                folder.mkdir(parents=True)
                artifact = ExperimentInputs(dict(split=dict(name="test",source=ref), checkpoint=ref,
                    sid_mapping=ref, generation=dict(beam_width=100), seed=42,
                    candidate_membership="same", score_convention="log_probability", short_list_policy="return_available"),
                    {"brand": {i:str(i%2) for i in range(22)}},
                    [Request("test:0:user:7", list(range(21)),[-float(i+1) for i in range(21)],[21]),
                     Request("test:1:user:8", list(range(21)),[-float(i+1) for i in range(21)],[0])])
                save(artifact,folder/"test_candidates.json")
                (folder/"complete.json").write_text(json.dumps(dict(scope="full_test",candidates=file_reference(folder/"test_candidates.json"))))
                cases.append(dict(name=name,checkpoint=ref))
            inventory = root / "inventory.json"
            inventory.write_text("fixture")
            frozen = dict(cases=cases,selection=dict(parameters={"lambda":1.}),protocol={})
            calls = []
            class FakeReranker:
                def __init__(self, **settings):
                    calls.append(settings)
                def rerank(self, inputs, k):
                    return SimpleNamespace(rankings={r.request_id: inputs.ranking(r)[:k] for r in inputs.requests},config={})
            with patch.object(runner,"ROOT",root), patch.object(runner,"checked_inventory",return_value=frozen), patch.object(rerank,"FairDiverseReranker",FakeReranker):
                runner.evaluate_pools(inventory,root/"pools",root/"results",0)
            self.assertEqual(len(calls),12)
            self.assertTrue(all(c["fairness_lambda"]==1. for c in calls))
            with (root/"results/controlled_comparison.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows),24)
            self.assertEqual({r["pool"] for r in rows},{"base","ordinary","rw","sprint"})


if __name__ == "__main__":
    unittest.main()
