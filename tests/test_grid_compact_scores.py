import numpy as np
import torch

from fairdiverse.recommendation.evaluator import Ranking_Evaluator


class GeneratedCandidateModel:
    IR_type = ["retrieval"]

    def eval(self):
        return self

    def full_predict(self, user_dict, items):
        # Items 0 and 2 were generated; zero marks a catalogue item that was
        # absent from the beam.
        return torch.tensor([[0.4, 0.0, 0.2, 0.0]], dtype=torch.float32)


def test_compact_grid_score_export_contains_only_generated_items():
    evaluator = Ranking_Evaluator.__new__(Ranking_Evaluator)
    evaluator.config = {
        "model": "GRID",
        "grid_native_metrics": False,
        "grid_compact_ranking": True,
        "beam_diagnostics": False,
        "use_beam_search_eval": True,
        "topk": [1],
        "group_num": 2,
        "fairness_type": "Exposure",
        "mmf_eval_ratio": 0.5,
        "decimals": 4,
        "device": "cpu",
        "item_num": 4,
    }
    evaluator.iid2pid = {0: 0, 1: 0, 2: 1, 3: 1}
    batch = (
        torch.tensor([0]),
        torch.tensor([[0, 0]]),
        torch.tensor([[0, 1, 2, 3]]),
        torch.tensor([1]),
    )

    _, scores = evaluator.eval([batch], GeneratedCandidateModel(), store_scores=True)
    scores = scores.tocsr()

    assert scores.nnz == 2
    assert scores.indices.tolist() == [0, 2]
    np.testing.assert_allclose(scores.data, [1.4, 1.2])
    ground_truth = evaluator.last_ground_truth.tocsr()
    assert ground_truth.shape == scores.shape
    assert ground_truth.getrow(0).indices.tolist() == [0]


def test_score_export_builds_csr_rows_without_explicit_row_coordinates():
    evaluator = Ranking_Evaluator.__new__(Ranking_Evaluator)
    evaluator.config = {
        "model": "GRID",
        "grid_native_metrics": False,
        "grid_compact_ranking": False,
        "beam_diagnostics": False,
        "use_beam_search_eval": True,
        "topk": [1],
        "group_num": 2,
        "fairness_type": "Exposure",
        "mmf_eval_ratio": 0.5,
        "decimals": 4,
        "device": "cpu",
        "item_num": 4,
    }
    evaluator.iid2pid = {0: 0, 1: 0, 2: 1, 3: 1}
    batch = (
        torch.tensor([0, 1]),
        torch.tensor([[0, 0], [0, 0]]),
        torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]]),
        torch.tensor([1, 1]),
    )

    _, scores = evaluator.eval([batch], GeneratedCandidateModel(), store_scores=True)

    assert scores.format == "csr"
    assert scores.indptr.tolist() == [0, 4, 8]
    assert scores.getrow(0).indices.tolist() == [0, 1, 2, 3]
    assert scores.getrow(1).indices.tolist() == [3, 2, 1, 0]
    np.testing.assert_allclose(scores.getrow(1).data, [1.4, 1.0, 1.2, 1.0])
