"""Audit K=10 reranking coverage and recover Pet RKMeans MGU from saved lists.

No training, reranking, or setting selection is performed. Original comparison
and historical calibration exports are preserved. Uses only the Python stdlib.
"""
from collections import Counter
import csv
import hashlib
import json
import math
import runpy
from pathlib import Path

from scripts.paper.archive_paths import source_root, output_root
ROOT = source_root()
OUTPUT = output_root()
OUT = OUTPUT
COMPARISON = ROOT / 'docs/rq2_comparison_draft_20261011/comparison_results.csv'
METHODS = {'GRID + RR': 'RR', 'GRID + CPFair': 'CPFair', 'GRID + P-MMF': 'P-MMF'}
REFERENCES = {}


def reference(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    REFERENCES[str(path.relative_to(ROOT))] = digest.hexdigest()


def close(a, b):
    assert math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-11), (a, b)


def load(path):
    reference(path)
    with path.open() as stream:
        return json.load(stream)


def main():
    reference(COMPARISON)
    rows = list(csv.DictReader(COMPARISON.open()))
    controls = {(r['dataset'], r['tokenizer']): r for r in rows
                if r['k'] == '10' and r['method'] == 'GRID: saved reranking reference'}
    methods = [r for r in rows if r['k'] == '10' and r['method'] in METHODS]
    assert len(controls) == 12 and len(methods) == 36
    assert len({(r['dataset'], r['tokenizer'], r['method']) for r in methods}) == 36
    assert all(r['status'] == 'verified' for r in methods)
    method_lookup = {(r['dataset'], r['tokenizer'], METHODS[r['method']]): r for r in methods}

    calibration = {}
    historical = [
        ('docs/rkmeans_reranking_analysis_20261011/test_results.csv',
         'mgu', 'control_mgu'),
        ('docs/tokenizer_reranking_analysis_20261014/tradeoff_results.csv',
         'MGU', 'control_MGU'),
    ]
    ds_names = {'Beauty': 'amazon-beauty', 'Sports': 'amazon-sports',
                'Toys': 'amazon-toys', 'Pet Supplies': 'amazon-pet-supplies'}
    for filename, metric, baseline in historical:
        path = ROOT / filename
        reference(path)
        for row in csv.DictReader(path.open()):
            if row['k'] == '10' and row['method'] in METHODS.values():
                key = ds_names[row['dataset']], row.get('tokenizer', 'rkmeans'), row['method']
                assert key not in calibration
                calibration[key] = (float(row[baseline]), float(row[metric]))
                canonical = method_lookup[key]
                rkmeans = key[1] == 'rkmeans'
                for source_col, target_col in [
                    ('recall', 'recall'), ('ndcg', 'ndcg'),
                    ('brand_exposure_gini' if rkmeans else 'brand_gini', 'brand_gini'),
                    ('brand_exposure_entropy' if rkmeans else 'entropy_bits', 'brand_entropy_bits'),
                ]:
                    value = float(row[source_col])
                    if source_col == 'brand_exposure_entropy':
                        value /= math.log(2)
                    close(value, float(canonical[target_col]))
    assert len(calibration) == 33
    missing = [(r['dataset'], r['tokenizer'], METHODS[r['method']]) for r in methods
               if (r['dataset'], r['tokenizer'], METHODS[r['method']]) not in calibration]
    assert set(missing) == {('amazon-pet-supplies', 'rkmeans', m) for m in METHODS.values()}

    selected = ROOT / 'additional_baselines/artifacts/pet-supplies_rkmeans_validation_selected_test_357020'
    selection = load(selected / 'selection.json')
    candidate_path = Path(selection['test_candidates']['path'])
    # Only the metadata and item-group maps are needed from the 2.5 GB export.
    prefix = []
    with candidate_path.open() as stream:
        for line in stream:
            if line.lstrip().startswith('"requests":'):
                break
            prefix.append(line)
    header = json.loads(''.join(prefix).rstrip().rstrip(',') + '}')
    metadata = header['metadata']
    assert (metadata['dataset'], metadata['tokenizer'], metadata['split']['name']) == (
        'amazon-pet-supplies', 'rkmeans', 'test')
    popularity = {int(i): g for i, g in header['item_groups']['popularity'].items()}
    brands = {int(i): g for i, g in header['item_groups']['brand'].items()}
    mapping_sha = hashlib.sha256(json.dumps(sorted(popularity.items()), separators=(',', ':')).encode()).hexdigest()

    # RR stores full-population history shares and calibration at the file tail.
    rr_path = selected / 'rr_test_results.json'
    reference(rr_path)
    with rr_path.open() as stream:
        stream.seek(max(0, rr_path.stat().st_size - 30000))
        tail = stream.read()
    rr_stats = {}
    for field in ('control_ifairlrs_fairness', 'ifairlrs_fairness'):
        start = tail.index('"' + field + '"')
        start = tail.index('{', start)
        rr_stats[field] = json.JSONDecoder().raw_decode(tail[start:])[0]['10']
    correction = load(selected / 'rr_correction.json')
    assert correction['group_mapping_sha256'] == mapping_sha
    base = rr_stats['control_ifairlrs_fairness']
    for values in rr_stats.values():
        gaps = {g: values['GP'][g] - values['GH'][g] for g in values['GH']}
        close(sum(map(abs, gaps.values())) / len(gaps), values['MGU'])
        close(max(gaps.values()) - min(gaps.values()), values['DGU'])
    calibration['amazon-pet-supplies', 'rkmeans', 'RR'] = (
        base['MGU'], rr_stats['ifairlrs_fairness']['MGU'])

    history_path = candidate_path.parent / 'test_histories.json'
    histories = load(history_path)
    counts = Counter(popularity[item] for history in histories.values() for item in history)
    gh = {g: counts[g] / sum(counts.values()) for g in base['GH']}
    for group in gh:
        close(gh[group], base['GH'][group])
    request_ids = set(histories)
    del histories
    n = len(request_ids)
    assert n == 594800
    recovered = {'RR': dict(control_MGU=base['MGU'], MGU=rr_stats['ifairlrs_fairness']['MGU'])}
    print('Verified Pet RKMeans history shares and saved RR MGU', flush=True)
    for method, filename in [('CPFair', 'cpfair_test_results.json'), ('P-MMF', 'pmmf_test_results.json')]:
        report = load(selected / filename)
        assert report['input_metadata'] == metadata
        assert report['control_metrics']['request_count'] == n
        rankings = report['rankings_by_k']['10']
        assert rankings.keys() == request_ids
        pop_counts, brand_counts = Counter(), Counter()
        for ranking in rankings.values():
            assert len(ranking) == 10 and len(set(ranking)) == 10
            pop_counts.update(popularity[item] for item in ranking)
            brand_counts.update(brands[item] for item in ranking)
        gp = {g: pop_counts[g] / (n * 10) for g in gh}
        gu = {g: gp[g] - gh[g] for g in gh}
        mgu = sum(map(abs, gu.values())) / len(gu)
        calibration['amazon-pet-supplies', 'rkmeans', method] = base['MGU'], mgu
        # Confirm these lists reproduce the brand metrics in the supplied table.
        values = sorted(brand_counts[g] for g in set(brands.values()))
        total, size = sum(values), len(values)
        assert total == n * 10
        gini = sum((2*i-size-1)*v for i, v in enumerate(values, 1)) / (size*total)
        entropy = -sum(v/total * math.log2(v/total) for v in values if v)
        canonical = next(r for r in methods if r['dataset'] == 'amazon-pet-supplies'
                         and r['tokenizer'] == 'rkmeans' and METHODS[r['method']] == method)
        close(gini, float(canonical['brand_gini']))
        close(entropy, float(canonical['brand_entropy_bits']))
        recovered[method] = dict(control_MGU=base['MGU'], MGU=mgu, GH=gh, GP=gp, GU=gu,
                                 brand_gini_reproduced=gini, entropy_bits_reproduced=entropy)
        print(f'Computed {method}: MGU reduction {100*(1-mgu/base["MGU"]):.8f}%', flush=True)
        del report, rankings

    changes = []
    for row in methods:
        ds, tok, method = row['dataset'], row['tokenizer'], METHODS[row['method']]
        control = controls[ds, tok]
        baseline_mgu, mgu = calibration[ds, tok, method]
        changes.append(dict(dataset=ds, tokenizer=tok, method=method,
            recall_change_percent=100*(float(row['recall'])/float(control['recall'])-1),
            mgu_reduction_percent=100*(1-mgu/baseline_mgu),
            gini_reduction_percent=100*(1-float(row['brand_gini'])/float(control['brand_gini'])),
            entropy_increase_percent=100*(float(row['brand_entropy_bits'])/float(control['brand_entropy_bits'])-1)))
    columns = list(changes[0])[3:]
    summary = {}
    for method in METHODS.values():
        group = [r for r in changes if r['method'] == method]
        assert len(group) == 12
        summary[method] = {}
        for col in columns:
            values = [r[col] for r in group]
            summary[method][col] = dict(min=min(values), max=max(values),
                improved=sum(v > 0 for v in values), unchanged=sum(v == 0 for v in values), cases=len(values))
    with (OUT / 'case_changes.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(changes[0]))
        writer.writeheader()
        writer.writerows(changes)
    payload = dict(missing_historical_calibration=missing, recovered_pet_rkmeans=recovered,
                   summary=summary, sources=REFERENCES,
                   checks=['36 verified reranking cases and 12 matched controls at K=10',
                           'All 33 historical calibration rows match current Recall/NDCG/Gini/entropy',
                           'Pet history shares reproduced against saved RR control',
                           'Popularity mapping matches frozen RR correction',
                           'Pet CPFair/P-MMF full request IDs, full unique top-10 lists',
                           'Pet CPFair/P-MMF brand Gini and entropy reproduced against comparison CSV'],
                   limitations=['Other 11 MGU cases reuse the existing calibration exports.',
                                'Candidate membership and Recall/NDCG are not independently recomputed in this audit.',
                                'Improved means a strictly positive change using unrounded saved values; no significance claim.'])
    (OUT / 'summary.json').write_text(json.dumps(payload, indent=2) + '\n')
    runpy.run_path(str(OUT / 'make_table.py'), run_name='__main__')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
