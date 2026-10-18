"""Audit saved continuation comparisons and Beauty's separate reranker settings.

Run from any directory with python3. Uses only the standard library and saved
artifacts; does not train models or modify historical manifests/results.
"""
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from scripts.paper.archive_paths import source_root, output_root
ROOT = source_root()
OUTPUT = output_root()
OUT = OUTPUT
SOURCES = {}


def read(path):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    raw = path.read_bytes()
    SOURCES[str(path.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


def close(actual, expected):
    assert math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12), (actual, expected)


def exposure_metrics(counts):
    values = sorted(counts.values())
    total, size = sum(values), len(values)
    return (sum((2 * i - size - 1) * v for i, v in enumerate(values, 1)) / (size * total),
            -sum(v / total * math.log(v / total) for v in values if v))


table_path = ROOT / 'docs/rq2_comparison_draft_20261011/comparison_results.csv'
SOURCES[str(table_path.relative_to(ROOT))] = hashlib.sha256(table_path.read_bytes()).hexdigest()
table = list(csv.DictReader(table_path.open()))
provenance = read('docs/rq2_comparison_draft_20261011/provenance.json')
branches = {(r['dataset'], r['tokenizer'], r['method']): r
            for r in provenance['local_training_branches']}
ordinary = {(r['dataset'], r['tokenizer']): r for r in table
            if r['method'] == 'GRID: ordinary continuation' and r['k'] == '5'}
histories = read('additional_baselines/artifacts/beauty_ifairlrs_rr_popularity/original/test_histories.json')
records, deltas = [], []

for row in table:
    if row['k'] != '5' or row['method'] not in {
            'GRID: ordinary continuation', 'GRID + IFairLRS RW (training)', 'GRID + SPRINT (training)'}:
        continue
    key = row['dataset'], row['tokenizer'], row['method']
    branch = branches.get(key)
    source = ROOT / row['source']
    folder = ROOT / branch['continuation'] if branch else source.with_name('continuation.json')
    if not branch and not folder.exists():
        matched = next(b for b in branches.values()
                       if (b['dataset'], b['tokenizer']) == (row['dataset'], row['tokenizer']))
        folder = ROOT / matched['ordinary_control']
    folder = folder.parent
    config = read(folder / 'run_config.json')
    continuation = read(folder / 'continuation.json')
    trace = read(folder / 'validation_trace.json')
    result = read(source)
    selected = max(trace, key=lambda r: (r['metrics']['at_k']['5']['recall'], -r['update']))
    assert selected['update'] == continuation['selected_update']
    assert config['identity'] == continuation['identity']
    assert result['input_metadata']['checkpoint'] == continuation['selected_checkpoint']
    if branch:
        control = read(branch['ordinary_control'])
        assert continuation['identity'] == control['identity']
        assert continuation['optimizer_audit'] == control['optimizer_audit']
    stats = config.get('rw_statistics')
    if stats:
        nh, nt = sum(stats['history_counts'].values()), sum(stats['target_counts'].values())
        for g, w in stats['weights'].items():
            hs, ts = stats['history_counts'].get(g, 0) / nh, stats['target_counts'].get(g, 0) / nt
            close(w, hs / ts if ts else 0)
    record = dict(dataset=row['dataset'], tokenizer=row['tokenizer'], mode=config['mode'],
                  source=row['source'], run_config=str((folder / 'run_config.json').relative_to(ROOT)),
                  identity=config['identity'], optimizer_audit=config['optimizer_audit'],
                  native_schedule=config['native_schedule'], selected_update=selected['update'],
                  validation_recall_at_5=selected['metrics']['at_k']['5']['recall'],
                  selection_policy=continuation['selection_policy'], group=config['group'],
                  rw_statistics=stats, sprint_settings=config.get('sprint_settings'),
                  corrected_update_zero=bool(branch and branch.get('corrected_update_zero')))
    # Independently reconstruct all Beauty metrics from saved scored candidates.
    # No project evaluator or fairness helper is imported here.
    if row['dataset'] == 'amazon-beauty':
        candidates = read(folder / 'test_candidates.json')
        assert candidates['metadata'] == result['input_metadata']
        assert len(candidates['requests']) == result['metrics']['request_count'] == 22363
        assert {r['request_id'] for r in candidates['requests']} == set(histories)
        mapping = candidates['item_groups']['popularity']
        hc = Counter(mapping[str(i)] for history in histories.values() for i in history)
        gh = {g: hc[g] / sum(hc.values()) for g in set(mapping.values())}
        accum = {k: dict(recall=0, ndcg=0, counts={name: Counter(dict.fromkeys(set(m.values()), 0))
                         for name, m in candidates['item_groups'].items()}) for k in (5, 10, 20)}
        for request in candidates['requests']:
            assert len(request['targets']) == 1
            ascending = candidates['metadata']['score_convention'] == 'negative_log_probability'
            order = sorted(range(len(request['scores'])), key=request['scores'].__getitem__, reverse=not ascending)
            ranking = [request['candidate_ids'][i] for i in order[:20]]
            for k, a in accum.items():
                top = ranking[:k]
                assert len(top) == k and len(set(top)) == k
                target = request['targets'][0]
                if target in top:
                    a['recall'] += 1
                    a['ndcg'] += 1 / math.log2(top.index(target) + 2)
                for name, mapping in candidates['item_groups'].items():
                    a['counts'][name].update(mapping[str(i)] for i in top)
        record['independent_candidate_verification'] = 'Recall, NDCG, full lists, brand/popularity exposure counts, Gini and entropy at K=5/10/20'
        record['popularity_calibration'] = {}
        for k, a in accum.items():
            saved = result['metrics']['at_k'][str(k)]
            close(a['recall'] / 22363, saved['recall'])
            close(a['ndcg'] / 22363, saved['ndcg'])
            for name, counts in a['counts'].items():
                assert dict(counts) == saved['groups'][name]['exposure_counts']
                gini, entropy = exposure_metrics(counts)
                close(gini, saved['groups'][name]['gini'])
                close(entropy, saved['groups'][name]['entropy'])
            counts = a['counts']['popularity']
            gp = {g: counts[g] / sum(counts.values()) for g in gh}
            gu = {g: gp[g] - gh[g] for g in gh}
            record['popularity_calibration'][str(k)] = dict(GH=gh, GP=gp, GU=gu,
                MGU=sum(map(abs, gu.values())) / len(gu), DGU=max(gu.values()) - min(gu.values()))
        checkpoint = Path(continuation['selected_checkpoint']['path'])
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == continuation['selected_checkpoint']['sha256']
        del candidates
    records.append(record)
    print('Checked', row['dataset'], row['tokenizer'], config['mode'], flush=True)
    control_row = ordinary[row['dataset'], row['tokenizer']]
    control_result = read(control_row['source'])
    for k in (5, 10, 20):
        current = next(r for r in table if (r['dataset'], r['tokenizer'], r['method'], r['k']) == (*key, str(k)))
        m, c = result['metrics']['at_k'][str(k)], control_result['metrics']['at_k'][str(k)]
        for metric in ('recall', 'ndcg'):
            close(float(current[metric]), m[metric])
        close(float(current['brand_gini']), m['groups']['brand']['gini'])
        close(float(current['brand_entropy_bits']), m['groups']['brand']['entropy'] / math.log(2))
        for group, b in m['groups'].items():
            gini, entropy = exposure_metrics(b['exposure_counts'])
            close(gini, b['gini'])
            close(entropy, b['entropy'])
        deltas.append(dict(dataset=row['dataset'], tokenizer=row['tokenizer'], mode=config['mode'], k=k,
            selected_update=selected['update'], recall=m['recall'], ndcg=m['ndcg'],
            recall_relative_delta_percent=100 * (m['recall'] / c['recall'] - 1),
            ndcg_relative_delta_percent=100 * (m['ndcg'] / c['ndcg'] - 1),
            brand_gini=m['groups']['brand']['gini'],
            brand_gini_delta=m['groups']['brand']['gini'] - c['groups']['brand']['gini'],
            popularity_gini=m['groups'].get('popularity', {}).get('gini', ''),
            popularity_gini_delta=(m['groups']['popularity']['gini'] - c['groups']['popularity']['gini']
                                   if 'popularity' in m['groups'] and 'popularity' in c['groups'] else ''),
            most_popular_exposure_share=m['groups'].get('popularity', {}).get('exposure_shares', {}).get('4', ''),
            source=row['source']))

selection_path = 'additional_baselines/artifacts/beauty_rkmeans_validation_selected_test_356027/selection.json'
selection = read(selection_path)
assert selection['methods'] == ['rr']
rerankers = []
for method, expected in [('cpfair', 1), ('pmmf', 30)]:
    path = f'additional_baselines/artifacts/beauty_original_selected_rerank_352496/{method}_test_results.json'
    result = read(path)
    actual = {k: run['config']['backend_config']['lambda'] for k, run in result['runs'].items()}
    assert actual == {str(k): expected for k in (5, 10, 20)}
    rerankers.append(dict(method=method, rr_only_manifest=selection_path,
        rr_manifest_placeholder=selection[method + '_lambda'], actual_lambda_by_k=actual,
        actual_settings_source=path, interpretation='Inactive field in RR-only manifest; separate historical reranker run. Does not configure RW or SPRINT.'))

with (OUT / 'continuation_deltas.csv').open('w', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(deltas[0]))
    writer.writeheader()
    writer.writerows(deltas)
(OUT / 'audit.json').write_text(json.dumps(dict(
    scope='All saved ordinary/RW/SPRINT table rows: settings, selection, metric reconciliation; Beauty candidates independently recomputed',
    baseline='ordinary continuation with matching dataset/tokenizer; deltas are test results',
    verification=dict(continuation_runs=len(records), metric_rows=len(deltas), beauty_candidate_runs=9,
                      beauty_checkpoint_hashes=9, historical_manifests_modified=False),
    reranker_settings=rerankers, continuation_settings=records, sources=SOURCES,
    limitations=['One seed, no significance or training-seed uncertainty assessment',
                 'No retraining; saved checkpoints fingerprinted for Beauty only',
                 'Candidate-level reconstruction covers Beauty; other datasets checked against saved metrics and exposure counts',
                 'Corrected Pet RVQ SPRINT source has brand metrics only; its popularity fields are left missing',
                 'SPRINT weights are starting defaults, not a validated hyperparameter optimum',
                 'Checkpoint selection maximizes validation Recall@5, not fairness',
                 'Popularity calibration uses exact saved Beauty test request histories; brand and popularity are distinct fairness axes']
), indent=2) + '\n')
print(f'Verified {len(records)} run settings and {len(deltas)} metric rows; independently reconstructed all nine Beauty candidate exports.')
for r in deltas:
    if r['dataset'] == 'amazon-beauty' and r['k'] == 5:
        print(r['tokenizer'], r['mode'], 'R@5', round(r['recall'], 6),
              'delta%', round(r['recall_relative_delta_percent'], 3),
              'brand Gini', round(r['brand_gini'], 6), 'pop Gini', round(r['popularity_gini'], 6))
