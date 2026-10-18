"""Audit the historical comparison cohort from saved configs, reports and run logs."""
import ast
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import re
import statistics
import yaml

from scripts.paper.archive_paths import source_root, output_root
ROOT = source_root()
OUTPUT = output_root()
OUT = OUTPUT
COMPARISON = ROOT / 'docs/rq2_comparison_draft_20261011/comparison_results.csv'
with COMPARISON.open() as handle:
    rows = [r for r in csv.DictReader(handle) if r['k'] == '10']
logs = defaultdict(list)
for root in (ROOT / 'wandb', ROOT / 'fairdiverse/wandb'):
    for folder in root.glob('run-*'):
        log = folder / 'files/output.log'
        if log.is_file():
            logs[folder.name.rsplit('-', 1)[-1]].append(log)

def relative(path):
    return str(path.resolve().relative_to(ROOT))

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write_csv(name, records):
    with (OUT / name).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

fair = []
for row in rows:
    if not row['status'].startswith('recorded FairDiverse'):
        continue
    result_path = ROOT / row['source']
    config_path = result_path.with_name('config.yaml')
    config = yaml.safe_load(config_path.read_text())
    log_path = logs[row['wandb_url'].rsplit('/', 1)[-1]][0]
    text = log_path.read_text(errors='replace')
    validations = [line for line in text.splitlines() if 'eval result:' in line]
    caps = re.findall(r'Reached max_steps=(\d+)\.', text)
    cap = config['max_steps']
    assert len(validations) == 1 and caps == [str(cap)]
    assert text.index('Reached max_steps=') < text.index('eval result:')
    saved = json.loads(result_path.read_text())
    printed_results = []
    for line in text.splitlines():
        if not line.strip().startswith('{'):
            continue
        try:
            value = ast.literal_eval(line.strip())
        except (ValueError, SyntaxError):
            continue
        if isinstance(value, dict) and 'ndcg@10' in value and 'hr@5' in value:
            printed_results.append(value)
    assert printed_results
    for metric in ('hr@5', 'hr@10', 'ndcg@5', 'ndcg@10', 'gini@10'):
        assert printed_results[-1][metric] == saved[metric]
    assert config['watch_metric'] == 'hr@5'
    assert config['early_stop_metrics'] == ['hr@5']
    fair.append(dict(
        dataset=row['dataset'], tokenizer=row['tokenizer'], method=row['method'],
        optimizer_state='fresh Adam (trainer implementation)',
        max_optimizer_updates=cap, completed_optimizer_updates=int(caps[0]),
        accumulation=config['accumulate_grad_batches'], epoch_cap=config['epoch'],
        validation_interval_batches=config['val_check_interval_batches'],
        observed_validation_calls=len(validations), validation_after_step_limit=True,
        zero_step_eligible=False, selection_metric='validation Recall@5 (hr@5)',
        learning_rate=config['learning_rate'], weight_decay=config['weight_decay'],
        config=relative(config_path), config_sha256=digest(config_path),
        output_log=relative(log_path), output_log_sha256=digest(log_path),
        test_result=relative(result_path), test_log_matches_saved_result=True,
    ))
assert len(fair) == 84

fine = []
methods = {'GRID: ordinary continuation', 'GRID + IFairLRS RW (training)',
           'GRID + SPRINT (training)'}
fallback = {
    ('amazon-pet-supplies', 'rvq', 'GRID + SPRINT (training)'):
        'additional_baselines/artifacts/pet_rvq_complete_metadata_continuation_20261004/sprint/continuation.json',
    ('amazon-pet-supplies', 'rqvae', 'GRID: ordinary continuation'):
        'additional_baselines/artifacts/pet_rqvae_complete_metadata_continuation_20261004/ordinary/continuation.json',
}
for row in rows:
    if row['method'] not in methods:
        continue
    report_path = (ROOT / row['source']).with_name('continuation.json')
    if not report_path.is_file():
        report_path = ROOT / fallback[row['dataset'], row['tokenizer'], row['method']]
    report = json.loads(report_path.read_text())
    trace_path = report_path.with_name('validation_trace.json')
    trace = json.loads(trace_path.read_text())
    assert [r['update'] for r in trace] == [0, 8, 16, 24, 32]
    identity = report['identity']
    assert identity['max_updates'] == 32 and identity['validation_interval'] == 8
    assert identity['optimizer_state'] == 'require' and report['optimizer_audit']['resumed']
    assert report['selection_metric'] == 'validation/recall@5'
    winner = max(trace, key=lambda r: (r['metrics']['at_k']['5']['recall'], -r['update']))
    assert winner['update'] == report['selected_update']
    fine.append(dict(
        dataset=row['dataset'], tokenizer=row['tokenizer'], method=row['method'],
        optimizer_state='resumed source Adam', max_optimizer_updates=32,
        validation_updates='0,8,16,24,32', observed_validation_calls=len(trace),
        selected_update=report['selected_update'], zero_step_eligible=True,
        selection_metric='validation Recall@5', tie_break='earlier update',
        report=relative(report_path), report_sha256=digest(report_path),
        validation_trace=relative(trace_path), validation_trace_sha256=digest(trace_path),
    ))
assert len(fine) == 36
write_csv('fairdiverse_runs.csv', fair)
write_csv('fine_tuning_runs.csv', fine)
summary = dict(
    comparison=relative(COMPARISON), comparison_sha256=digest(COMPARISON),
    fairdiverse_runs=len(fair), fine_tuning_runs=len(fine),
    fairdiverse_validation_counts=dict(Counter(str(r['observed_validation_calls']) for r in fair)),
    fairdiverse_completed_updates=dict(Counter(str(r['completed_optimizer_updates']) for r in fair)),
    fine_tuning_validation_updates=[0, 8, 16, 24, 32],
    fine_tuning_selected_zero=sum(r['selected_update'] == 0 for r in fine),
    common_selection_criterion='Validation Recall@5; hr@5 equals Recall@5 for single-target evaluation',
    finding='All 84 FairDiverse runs validated once, after reaching their step cap; their saved checkpoints are endpoint checkpoints. The 36 resumed-Adam runs select among five validation checkpoints, including update zero.',
    limitations=['This audits the historical cohort in the comparison tables, not all repository experiments.',
                 'Fresh Adam behavior is established by the trainer implementation; FairDiverse saved artifacts do not include an optimizer-resume audit.',
                 'Protocol differences do not establish which one caused a performance difference.'],
    implementation_sources={str(p):digest(ROOT / p) for p in (
        Path('fairdiverse/recommendation/trainer.py'),
        Path('additional_baselines/continuation_grid.py'),
        Path('additional_baselines/continuation_selection.py'))},
)
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps({k:v for k,v in summary.items() if k not in ['implementation_sources','limitations']}, indent=2))
