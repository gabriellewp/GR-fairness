import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from fairdiverse.recommendation.base_model.gru4rec import GRU4Rec


def config(padding=-1):
    return dict(user_num=2, item_num=5, embedding_size=4, hidden_size=8,
                num_layers=1, dropout_prob=0.0, padding_item_id=padding)


def test_negative_padding_and_item_zero_train_without_padding_class():
    model = GRU4Rec(config())
    interaction = dict(history_ids=torch.tensor([[0, 2, -1, -1], [1, 3, 4, -1]]),
                       item_ids=torch.tensor([3, 0]))
    loss = model.compute_loss(interaction)
    expected = F.cross_entropy(
        model.get_user_embedding(interaction) @ model.item_embedding.weight[:5].T,
        interaction['item_ids'], reduction='none',
    )
    torch.testing.assert_close(loss, expected)
    assert loss.shape == (2,)
    assert torch.isfinite(loss).all()
    loss.mean().backward()
    assert model.item_embedding.weight.grad[0].abs().sum() > 0
    assert model.item_embedding.weight.grad[5].abs().sum() == 0
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer.step()
    assert torch.count_nonzero(model.item_embedding.weight[5]) == 0


@pytest.mark.parametrize('padding,history', [(-1, [0, 2]), (0, [1, 2])])
def test_right_padding_does_not_change_predictions(padding, history):
    model = GRU4Rec(config(padding)).eval()
    short = {'history_ids': torch.tensor([history])}
    padded = {'history_ids': torch.tensor([history + [padding] * 4])}
    torch.testing.assert_close(model.get_user_embedding(short),
                               model.get_user_embedding(padded))
    candidates = torch.arange(5).unsqueeze(0)
    torch.testing.assert_close(model.full_predict(short, candidates),
                               model.full_predict(padded, candidates))


def test_mixed_history_lengths_and_empty_history():
    model = GRU4Rec(config()).eval()
    histories = torch.tensor([[0, 2, -1], [1, 2, 3], [-1, -1, -1]])
    result = model.get_user_embedding({'history_ids': histories})
    for i, history in enumerate(([0, 2], [1, 2, 3])):
        torch.testing.assert_close(result[i:i+1], model.get_user_embedding(
            {'history_ids': torch.tensor([history])}))
    assert torch.count_nonzero(result[2]) == 0


def test_complete_metadata_experiments_cover_all_four_datasets():
    root = Path(__file__).resolve().parents[1]
    experiments = json.loads((root / 'configs/reproduction/experiments_config_gru4rec_complete_metadata.json').read_text())
    assert [e['dataset'] for e in experiments] == [
        'amazon-beauty', 'amazon-sports', 'amazon-toys', 'amazon-pet-supplies']
    assert [e['exp_id'] for e in experiments] == list(range(4))
    for experiment in experiments:
        c = experiment['config']
        assert c['model'] == 'gru4rec'
        assert c['data_type'] == 'sequential'
        assert c['padding_item_id'] == -1
        assert c['grid_compact_ranking'] is True
        assert c['store_scores'] is False
        assert c['reprocess'] is False
        assert c['expected_grid_source_dir'].endswith('/temporal_loo/complete_metadata')


@pytest.mark.parametrize('source,strategy,reprocess', [
    ('GRID/data/beauty/description', 'temporal_loo', False),
    ('GRID/data/beauty/complete_metadata', 'random', False),
    ('GRID/data/beauty/complete_metadata', 'temporal_loo', True),
])
def test_matched_run_rejects_wrong_split_before_training(tmp_path, monkeypatch,
                                                       source, strategy, reprocess):
    from fairdiverse.recommendation.trainer import RecTrainer

    monkeypatch.chdir(tmp_path)
    directory = tmp_path / 'recommendation/processed_dataset/amazon-beauty'
    directory.mkdir(parents=True)
    (directory / 'grid_data_manifest.json').write_text(json.dumps({
        'source_dir': source, 'split_strategy': strategy,
    }))
    trainer = RecTrainer(dict(
        dataset='amazon-beauty', reprocess=reprocess,
        expected_grid_source_dir='GRID/data/beauty/complete_metadata',
    ))
    with pytest.raises(ValueError):
        trainer.train()
