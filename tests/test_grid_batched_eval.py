import numpy as np
import pytest
import torch

from additional_baselines.ifairlrs_grid import original_module
from fairdiverse.recommendation.evaluator import Ranking_Evaluator
from fairdiverse.recommendation.rank_model.FairDual import FairDual


@pytest.fixture
def model(tmp_path, monkeypatch):
    module = original_module("grid_model")
    path = tmp_path / "sid.pt"
    torch.save(torch.tensor([[0, 0, 0, 0], [0, 0, 0, 0], [1, 2, 3, 0],
                             [2, 3, 4, 0], [3, 4, 5, 0], [4, 5, 6, 0]]), path)
    monkeypatch.setattr(module, "_local_grid_artifact", lambda *args, **kwargs: path)
    torch.manual_seed(42)
    result = module.GRIDBaseModel({
        "num_hierarchies": 4, "codebook_size": 8, "item_num": 6,
        "sid_path": str(path), "embedding_size": 8, "n_heads": 2, "d_kv": 3,
        "n_layers": 1, "inner_size": 12, "hidden_dropout_prob": 0.0,
        "sequence_length": 8, "top_k_for_generation": 3,
    })
    return result.eval()


def requests():
    return {"user_ids": torch.tensor([0, 1, 2, 3, 4]),
            "history_ids": torch.tensor([[0, -1], [1, 2], [2, 3], [3, -1], [4, 5]])}


def fixed_generation(model, monkeypatch):
    def generate(histories, users=None):
        rows, probabilities = [], []
        for history in histories:
            target = int(history[0])
            sid = model.tokenizer.sid_table[target].tolist()
            # Include repeated beams, an invalid SID, a zero probability, and
            # probabilities that round to ties after the existing +1 offset.
            rows.append([sid, sid, [7, 7, 7, 7], [4, 5, 6, 0]])
            probabilities.append([1e-9 if target == 0 else 0.4, 0.8, 0.3, 0.0])
        generated = torch.tensor(rows)
        model._last_generated_sids = generated
        return generated, torch.tensor(probabilities)
    monkeypatch.setattr(model, "generate", generate)


@pytest.mark.parametrize("microbatch", [1, 2, 8])
def test_indexed_matching_preserves_duplicates_unknowns_zeroes_and_candidate_order(model, monkeypatch, microbatch):
    fixed_generation(model, monkeypatch)
    users = requests()
    items = torch.tensor([[0, 1, 2, 3, 4, 5], [5, 4, 3, 2, 1, 0],
                          [2, 0, 5, 4, 3, 1], [3, 1, 4, 5, 0, 2], [4, 5, 0, 1, 2, 3]])
    legacy = torch.cat([model.full_predict({k: v[i:i+1] for k, v in users.items()}, items[i:i+1])
                        for i in range(len(items))])
    keys = set(model.state_dict())
    batched = model.full_predict_batch(users, items, microbatch)
    assert torch.equal(legacy, batched)
    assert model._last_generated_sids.shape == (5, 4, 4)
    assert set(model.state_dict()) == keys


@pytest.mark.parametrize("microbatch", [1, 2, 8])
def test_native_beam_generation_batch_parity(model, microbatch):
    users = requests()
    items = torch.arange(6).repeat(5, 1)
    beams, scores = [], []
    with torch.no_grad():
        for i in range(5):
            scores.append(model.full_predict({k: v[i:i+1] for k, v in users.items()}, items[i:i+1]))
            beams.append(model._last_generated_sids.clone())
        actual = model.full_predict_batch(users, items, microbatch)
    assert torch.equal(torch.cat(beams), model._last_generated_sids)
    torch.testing.assert_close(torch.cat(scores), actual, rtol=1e-5, atol=1e-8)


def evaluator(batched, *, native=False, compact=True):
    result = Ranking_Evaluator.__new__(Ranking_Evaluator)
    result.config = {
        "model": "GRID", "grid_native_metrics": native,
        "grid_native_diagnostics": True, "grid_compact_ranking": compact,
        "grid_eval_generation_batch_size": 2 if batched else 0,
        "topk": [1, 3, 5], "group_num": 3, "fairness_type": "Exposure",
        "mmf_eval_ratio": 0.5, "decimals": 8, "device": "cpu", "item_num": 6,
    }
    result.iid2pid = {i: i // 2 for i in range(6)}
    return result


@pytest.mark.parametrize("compact", [False, True])
def test_metrics_exports_and_reused_native_diagnostics_match_legacy(model, monkeypatch, compact):
    fixed_generation(model, monkeypatch)
    user = requests()
    items = torch.tensor([[0, 1, 2, 3, 4, 5], [5, 4, 3, 2, 1, 0],
                          [2, 0, 5, 4, 3, 1], [3, 1, 4, 5, 0, 2], [4, 5, 0, 1, 2, 3]])
    batch = (user['user_ids'], user['history_ids'], items, torch.tensor([1, 2, 1, 1, 1]))
    legacy, fast = evaluator(False, compact=compact), evaluator(True, compact=compact)
    expected, expected_scores = legacy.eval([batch], model, store_scores=True)
    actual, scores = fast.eval([batch], model, store_scores=True)
    assert actual == expected
    assert np.array_equal(scores.indptr, expected_scores.indptr)
    assert np.array_equal(scores.indices, expected_scores.indices)
    assert np.array_equal(scores.data, expected_scores.data)
    assert (fast.last_ground_truth != legacy.last_ground_truth).nnz == 0
    assert fast.last_grid_native_result == evaluator(False, native=True).eval([batch], model)


@pytest.mark.parametrize("size", [1, 8, 64])
def test_sparse_fairdual_projection_matches_original(size):
    ranker = FairDual.__new__(FairDual)
    ranker.config = {}
    rng = np.random.default_rng(42)
    rho = rng.uniform(0.3, 2.0, size)
    value = rng.normal(-0.3, 0.7, size)
    order = np.argsort(value * rho)
    value, rho = value[order], rho[order]
    expected = ranker.map_layer(value, rho, 0.5)
    ranker.config['fairdual_sparse_projection'] = True
    actual = ranker.map_layer(value, rho, 0.5)
    np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-4)
    assert np.cumsum(actual * rho).min() >= -0.5002
    problem = ranker._projection_problem
    again = ranker.map_layer(value * 1.1, rho, 0.5)
    assert ranker._projection_problem is problem
    ranker.config['fairdual_sparse_projection'] = False
    np.testing.assert_allclose(again, ranker.map_layer(value * 1.1, rho, 0.5), rtol=2e-4, atol=2e-4)


def test_sparse_fairdual_projection_scales_to_pet_group_count():
    import cvxpy as cp

    ranker = FairDual.__new__(FairDual)
    ranker.config = {'fairdual_sparse_projection': True}
    size = 2046
    value = np.sort(np.random.default_rng(42).normal(size=size))
    actual = ranker.map_layer(value, np.ones(size), 2.0)
    assert np.isfinite(actual).all()
    assert np.cumsum(actual).min() >= -2.002
    data = ranker._projection_problem.get_problem_data(cp.OSQP)[0]
    # Prefix constraints should have a linear-size canonical representation,
    # rather than the original triangular matrix with millions of entries.
    assert data['A'].nnz + data['F'].nnz < 20 * size
