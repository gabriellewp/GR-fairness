"""Guard the frozen scientific inputs and archive output isolation."""
import json
from pathlib import Path
import shutil

import pytest

from scripts.paper import archive_paths, reproduce

ROOT = Path(__file__).resolve().parents[1]


def test_changed_scientific_input_is_rejected_before_rendering(tmp_path):
    inputs = tmp_path/'inputs'
    shutil.copytree(ROOT/'paper/inputs', inputs)
    data = inputs/'comparison_results.csv'
    data.write_text(data.read_text().replace('0.08375441577605867', '0.9999999999999999', 1))
    with pytest.raises(ValueError, match='Frozen paper input changed'):
        reproduce.verify_inputs(inputs)


def test_archive_cannot_write_into_source_workspace(tmp_path, monkeypatch):
    source = tmp_path/'historical'
    source.mkdir()
    monkeypatch.setenv('GR_FAIRNESS_ARCHIVE_ROOT', str(source))
    monkeypatch.setenv('GR_FAIRNESS_ARCHIVE_OUTPUT', str(source/'analysis'))
    with pytest.raises(ValueError, match='outside the source'):
        archive_paths.output_root()
    assert list(source.iterdir()) == []


def test_paper_classic_recipe_preserves_pet_sasrec_architecture():
    recipes = json.loads((ROOT/'configs/reproduction/experiments_config_paper_classics.json').read_text())
    assert len(recipes) == 8
    config, = [r['config'] for r in recipes if r['dataset'] == 'amazon-pet-supplies' and r['config']['model'] == 'SASRec']
    assert config['embedding_size'] == 100
    assert config['history_length'] == 40 and config['padding_item_id'] == -1
    assert not config['use_wandb'] and config['metadata_text_mode'] == 'complete_metadata'
