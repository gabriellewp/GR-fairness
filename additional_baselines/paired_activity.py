"""Paired intervention effects by training activity, using saved predictions on CPU.

Commands: prepare (freeze the 12 cases), run (one case), collect (Holm and plots).
No model loading, training, generation, reranking, or test-based selection.
"""
import argparse
from array import array
import csv
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path

import ijson
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ['amazon-beauty', 'amazon-sports', 'amazon-toys', 'amazon-pet-supplies']
TOKENIZERS = ['rkmeans', 'rvq', 'rqvae']
METHODS = ['RW', 'SPRINT', 'RR', 'CPFair', 'P-MMF']
TRAINING_NAMES = {'Standard FT': 'GRID: ordinary continuation',
                  'RW': 'GRID + IFairLRS RW (training)', 'SPRINT': 'GRID + SPRINT (training)'}
METRICS = ['recall@10', 'ndcg@10']
RANK_VALUES = np.column_stack(([0.] + [1.] * 10,
                               [0.] + [1 / math.log2(i + 1) for i in range(1, 11)]))
GROUP_ORDER = ['cold_start', 'low', 'medium', 'high', 'warm']


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def reference(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=digest(path))


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def write_csv(path, rows):
    require(bool(rows), f'No rows for {path}')
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def extract(path, prefix):
    with Path(path).open('rb') as stream:
        return next(ijson.items(stream, prefix, use_float=True))


def canonical_source(row):
    """Some RKMeans rows cite the audit CSV rather than its original JSON."""
    path = ROOT / row['source']
    if path.suffix != '.csv':
        return path
    name = dict(zip(DATASETS, ['Beauty', 'Sports', 'Toys', 'Pet Supplies']))[row['dataset']]
    method = row['method'].replace('GRID + ', '')
    if row['method'] == 'GRID: saved reranking reference':
        method = 'Untreated'
    matches = [r for r in csv.DictReader(path.open())
               if r['dataset'] == name and r['method'] == method and r['k'] == row['k']]
    require(len(matches) == 1, f'Ambiguous source: {row}')
    return ROOT / matches[0]['source']


def expected(row):
    require(row['status'].split(';')[0] == 'verified', f'Unverified input: {row}')
    return {m: float(row[c]) for m, c in zip(METRICS, ['recall', 'ndcg'])}


def prepare(output, resamples=1000, seed=42):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    require(not (output / 'manifest.json').exists(), 'Manifest already exists; use a new output directory')
    comparison = ROOT / 'docs/rq2_comparison_draft_20261011/comparison_results.csv'
    rows = list(csv.DictReader(comparison.open()))
    lookup = {(r['dataset'], r['tokenizer'], r['method']): r for r in rows if r['k'] == '10'}
    cases = []
    for dataset, tokenizer in itertools.product(DATASETS, TOKENIZERS):
        rerank_row = lookup[dataset, tokenizer, 'GRID + RR']
        rr_path = canonical_source(rerank_row)
        selection_path = rr_path.parent / 'selection.json'
        selection = read_json(selection_path)
        candidate = Path(selection['test_candidates']['path'])
        require(candidate.exists(), f'Missing saved candidates: {candidate}')
        data = ROOT / 'recommendation/processed_dataset' / dataset
        case = dict(index=len(cases), dataset=dataset, tokenizer=tokenizer,
                    split=reference(data / f'{dataset}.test.ranking'),
                    activity=reference(data / 'user_activity.csv'),
                    activity_metadata=reference(data / 'user_activity_metadata.json'),
                    grid_manifest=reference(data / 'grid_data_manifest.json'),
                    selection=reference(selection_path), models={}, reports={})
        # Source digest comes from the frozen reranker selection; verified during execution.
        case['models']['Untreated'] = dict(kind='candidates', path=str(candidate),
            expected_sha256=selection['test_candidates']['sha256'],
            expected=expected(lookup[dataset, tokenizer, 'GRID: saved reranking reference']))
        for name, long_name in TRAINING_NAMES.items():
            row = lookup[dataset, tokenizer, long_name]
            report = canonical_source(row)
            path = report.parent / 'test_candidates.json'
            recovery = None
            if not path.exists():
                require((dataset, tokenizer, name) == ('amazon-pet-supplies', 'rvq', 'SPRINT'),
                        f'Missing candidate export: {path}')
                fresh = read_json(report)
                meta = extract(candidate, 'metadata')
                require(fresh.get('corrected_pet_sprint_source_result') is True,
                        'Pet RVQ SPRINT must use its corrected update-zero source')
                for key in ['checkpoint', 'sid_mapping', 'seed']:
                    require(meta[key] == fresh['input_metadata'][key], f'Corrected source differs: {key}')
                require(meta['split'] == fresh['input_metadata']['split'], 'Corrected source split differs')
                for key in ['beam_width', 'sequence_length', 'num_hierarchies']:
                    require(meta['generation'][key] == fresh['input_metadata']['generation'][key],
                            f'Corrected source generation differs: {key}')
                path = candidate
                recovery = ('Selected update zero: reuse source-checkpoint native-score candidates; '
                            'checkpoint/SID/split/generation identity checked; K=10 aggregate parity required. '
                            'The superseded historical SPRINT export is excluded.')
            require(path.exists(), f'Missing predictions: {path}')
            case['models'][name] = dict(kind='candidates', path=str(path),
                report=reference(report), expected=expected(row), recovery=recovery)
        for method in ['RR', 'CPFair', 'P-MMF']:
            row = lookup[dataset, tokenizer, 'GRID + ' + method]
            report = canonical_source(row)
            require(report.exists(), f'Missing reranked predictions: {report}')
            case['reports'][method] = dict(path=str(report), expected=expected(row))
        cases.append(case)
    manifest = dict(schema_version=1, comparison=reference(comparison), cases=cases,
        bootstrap_resamples=resamples, seed=seed, cutoff=10,
        methods=METHODS, control_rules={'RW': 'Standard FT', 'SPRINT': 'Standard FT',
            'RR': 'Untreated', 'CPFair': 'Untreated', 'P-MMF': 'Untreated'},
        inference=dict(unit='test user; exactly one held-out request per user is required',
            group_definition='Frozen training-history activity tertiles; ties kept together',
            effects='Method minus matched control; positive means effectiveness improves',
            group_intervals='Unadjusted 95% paired-user percentile bootstrap intervals',
            contrast_intervals='Difference in mean paired effects; independent resampling within the two fixed activity groups',
            bootstrap_implementation='Exact multinomial compression of paired target-rank outcomes, equivalent to resampling users for these mean statistics',
            pvalues='Two-sided paired t tests for mean effects; Welch tests on user-level changes for contrasts between disjoint activity groups',
            holm_families='Separately for fine_tuning/reranking and overall/activity/activity_contrasts; each family includes both metrics and all 12 cases',
            limitations='Evaluation-user variation conditional on saved models/predictions; no training-seed or hyperparameter-selection uncertainty; no equivalence/noninferiority claim'),
        code=reference(Path(__file__)))
    write_json(output / 'manifest.json', manifest)
    print(f'Frozen {len(cases)} cases / {len(cases)*len(METHODS)} comparisons in {output}', flush=True)
    return manifest


def verified(ref):
    require(digest(ref['path']) == ref['sha256'], f'Input changed: {ref["path"]}')


def read_population(case):
    for key in ['split', 'activity', 'activity_metadata', 'grid_manifest']:
        verified(case[key])
    meta = read_json(case['activity_metadata']['path'])
    require(meta['mapping']['sha256'] == case['activity']['sha256'], 'Activity mapping hash differs')
    grid = read_json(case['grid_manifest']['path'])
    require(meta['provenance']['source_fingerprint'] == grid['source_fingerprint'], 'Activity training source differs')
    activity = {r['user_id']: r for r in csv.DictReader(open(case['activity']['path']))}
    rows = list(csv.DictReader(open(case['split']['path']), delimiter='\t'))
    users = [r['user_id'] for r in rows]
    require(len(set(users)) == len(users) and set(users) == set(activity), 'Test/activity users differ or repeat')
    targets = np.array([int(r['target_item']) for r in rows], dtype=np.int64)
    groups = np.array([activity[u]['activity_group'] for u in users])
    counts = np.array([int(activity[u]['train_interaction_count']) for u in users], dtype=np.int64)
    require(all(g in GROUP_ORDER for g in groups), 'Unknown activity group')
    rids = [f'test:{i}:user:{u}' for i, u in enumerate(users)]
    return dict(users=users, targets=targets, groups=groups, counts=counts, rids=rids,
                rid_index={rid: i for i, rid in enumerate(rids)}, metadata=meta)


def target_rank(ranking, target):
    try:
        return ranking[:10].index(int(target)) + 1
    except ValueError:
        return 0


def validate_means(ranks, wanted, name):
    actual = RANK_VALUES[ranks].mean(axis=0)
    for j, metric in enumerate(METRICS):
        require(math.isclose(actual[j], wanted[metric], rel_tol=1e-9, abs_tol=1e-11),
                f'{name} {metric}: reconstructed {actual[j]} != table {wanted[metric]}')


def candidates(spec, population, case, keep_membership=False):
    path = Path(spec['path'])
    meta = extract(path, 'metadata')
    require(meta['split'] == {'name': 'test', 'source': case['split']}, 'Candidate test split differs')
    require(meta['seed'] == 42, 'Unexpected model seed')
    ascending = meta['score_convention'] == 'negative_log_probability'
    require(meta['score_convention'] in ['probability', 'log_probability', 'negative_log_probability', 'utility'],
            'Unknown score convention')
    n = len(population['users'])
    ranks = np.zeros(n, dtype=np.uint8)
    offsets = np.zeros(n+1, dtype=np.int64) if keep_membership else None
    items = array('i') if keep_membership else None
    size = 0
    with path.open('rb') as stream:
        for i, row in enumerate(ijson.items(stream, 'requests.item', use_float=True)):
            require(i < n and row['request_id'] == population['rids'][i], f'Candidate user/request mismatch at {i}')
            require(row['targets'] == [int(population['targets'][i])], f'Candidate target mismatch at {i}')
            ids, scores = row['candidate_ids'], row['scores']
            require(len(ids) == len(scores) and len(ids) == len(set(ids)), 'Malformed/duplicate candidates')
            require(all(math.isfinite(s) for s in scores), 'Nonfinite scores')
            order = sorted(range(len(ids)), key=scores.__getitem__, reverse=not ascending)
            ranks[i] = target_rank([ids[j] for j in order[:10]], population['targets'][i])
            if keep_membership:
                items.extend(ids)
                offsets[i+1] = len(items)
            size += 1
            if size % 100000 == 0:
                print(f'  {path.parent.name}: {size}/{n} saved candidate requests', flush=True)
    require(size == n, f'Partial candidate export: {size}/{n}')
    validate_means(ranks, spec['expected'], path)
    source = reference(path)
    if 'expected_sha256' in spec:
        require(source['sha256'] == spec['expected_sha256'], 'Candidate fingerprint differs from selection')
    return ranks, meta, (items, offsets), source


def reranked(spec, method, population, membership, meta):
    path = Path(spec['path'])
    if method == 'RR':
        frozen = extract(path, 'frozen_correction.validation_metadata')
        for key in ['checkpoint', 'sid_mapping', 'seed']:
            require(frozen[key] == meta[key], f'RR frozen {key} differs')
        for key in ['beam_width', 'sequence_length', 'num_hierarchies']:
            require(frozen['generation'][key] == meta['generation'][key], f'RR frozen generation {key} differs')
        prefix = 'rankings'
    else:
        saved_meta = extract(path, 'input_metadata')
        require(saved_meta == meta, f'{method} candidate metadata differs')
        layout = None
        with path.open('rb') as stream:
            for key, event, value in ijson.parse(stream, use_float=True):
                if event == 'start_map' and key in ['runs', 'rankings_by_k']:
                    layout = key
                    break
        require(layout is not None, f'Unknown reranking report layout: {path}')
        prefix = 'runs.10.rankings' if layout == 'runs' else 'rankings_by_k.10'
    n = len(population['users'])
    ranks, seen = np.zeros(n, dtype=np.uint8), np.zeros(n, dtype=bool)
    items, offsets = membership
    with path.open('rb') as stream:
        for rid, ranking in ijson.kvitems(stream, prefix, use_float=True):
            i = population['rid_index'].get(rid)
            require(i is not None and not seen[i], f'Unknown/duplicate ranking request: {rid}')
            require(len(ranking) >= 10 and len(ranking) == len(set(ranking)), 'Short or duplicate ranking')
            require(set(ranking[:10]).issubset(items[offsets[i]:offsets[i+1]]), 'Noncandidate top-10 item')
            ranks[i] = target_rank(ranking, population['targets'][i])
            seen[i] = True
    require(seen.all(), f'Partial {method} rankings')
    validate_means(ranks, spec['expected'], method)
    return ranks, reference(path)


def bootstrap_changes(control, method, resamples, rng):
    """Exact empirical user bootstrap for joint Recall/NDCG mean differences."""
    require(len(control) == len(method) and len(control) > 0, 'Invalid paired ranks')
    codes = control.astype(np.int64)*11 + method
    counts = np.bincount(codes, minlength=121)
    present = np.flatnonzero(counts)
    values = RANK_VALUES[present % 11] - RANK_VALUES[present // 11]
    draws = rng.multinomial(len(control), counts[present]/len(control), size=resamples)
    return draws @ values / len(control)


def paired_pvalue(values):
    if len(values) < 2:
        return None
    variance = np.var(values, ddof=1)
    if variance == 0:
        return 1. if np.mean(values) == 0 else 0.
    return float(stats.ttest_1samp(values, 0.).pvalue)


def contrast_pvalue(a, b):
    if min(len(a), len(b)) < 2:
        return None
    if np.var(a, ddof=1) == 0 and np.var(b, ddof=1) == 0:
        return 1. if np.mean(a) == np.mean(b) else 0.
    return float(stats.ttest_ind(a, b, equal_var=False).pvalue)


def effects_for_method(control, method, groups, resamples, rng):
    values_c, values_m = RANK_VALUES[control], RANK_VALUES[method]
    changes = values_m-values_c
    labels = [g for g in GROUP_ORDER if np.any(groups == g)]
    effect_rows, contrast_rows, bootstraps = [], [], {}
    for group in ['overall'] + labels:
        mask = np.ones(len(groups), dtype=bool) if group == 'overall' else groups == group
        boot = bootstrap_changes(control[mask], method[mask], resamples, rng)
        bootstraps[group] = boot
        for j, metric in enumerate(METRICS):
            low, high = np.quantile(boot[:, j], [0.025, .975])
            baseline = float(values_c[mask, j].mean())
            change = float(changes[mask, j].mean())
            effect_rows.append(dict(activity_group=group, metric=metric, users=int(mask.sum()),
                control_mean=baseline, intervention_mean=float(values_m[mask, j].mean()),
                mean_change=change, change_percentage_points=100*change,
                relative_change_percent=100*change/baseline if baseline else None,
                ci_low=float(low), ci_high=float(high), p_raw=paired_pvalue(changes[mask, j]),
                test='paired two-sided t', ci='unadjusted paired-user percentile bootstrap'))
    for a, b in itertools.combinations(labels, 2):
        for j, metric in enumerate(METRICS):
            va, vb = changes[groups == a, j], changes[groups == b, j]
            boot = bootstraps[a][:, j]-bootstraps[b][:, j]
            low, high = np.quantile(boot, [.025, .975])
            point = float(va.mean()-vb.mean())
            contrast_rows.append(dict(group_a=a, group_b=b, metric=metric, users_a=len(va), users_b=len(vb),
                mean_change_a=float(va.mean()), mean_change_b=float(vb.mean()),
                contrast=point, contrast_percentage_points=100*point, ci_low=float(low), ci_high=float(high),
                p_raw=contrast_pvalue(va, vb), test='two-sided Welch t on paired user changes',
                ci='unadjusted user bootstrap independently within each fixed activity group'))
    return effect_rows, contrast_rows


def run_case(manifest_path, index):
    manifest_path = Path(manifest_path).resolve()
    manifest = read_json(manifest_path)
    case = manifest['cases'][index]
    output = manifest_path.parent / f'case_{index:02d}_{case["dataset"]}_{case["tokenizer"]}'
    require(not (output / 'complete.json').exists(), 'Completed case exists; choose a new run directory')
    output.mkdir(parents=True, exist_ok=True)
    verified(manifest['comparison'])
    verified(manifest['code'])
    verified(case['selection'])
    population = read_population(case)
    print(f'Case {index}: {case["dataset"]}/{case["tokenizer"]}, {len(population["users"])} users', flush=True)
    vectors, metadata, sources, cache = {}, {}, {}, {}
    membership = None
    for name in ['Untreated', 'Standard FT', 'RW', 'SPRINT']:
        spec = case['models'][name]
        if 'report' in spec:
            verified(spec['report'])
        if spec['path'] in cache:
            ranks, meta = cache[spec['path']]
            validate_means(ranks, spec['expected'], name)
        else:
            ranks, meta, members, source = candidates(spec, population, case, keep_membership=name == 'Untreated')
            sources[spec['path']] = source
            cache[spec['path']] = ranks, meta
            if name == 'Untreated':
                membership = members
        vectors[name], metadata[name] = ranks, meta
    for name in ['RW', 'SPRINT']:
        for key in ['sid_mapping', 'seed', 'split']:
            require(metadata[name][key] == metadata['Standard FT'][key], f'Fine-tuning {key} differs')
        for key in ['beam_width', 'sequence_length', 'num_hierarchies']:
            require(metadata[name]['generation'][key] == metadata['Standard FT']['generation'][key],
                    f'Fine-tuning generation {key} differs')
    for method in ['RR', 'CPFair', 'P-MMF']:
        vectors[method], source = reranked(case['reports'][method], method, population, membership, metadata['Untreated'])
        sources[source['path']] = source
        print(f'  Verified {method} rankings and aggregate metrics', flush=True)
    del membership, cache
    effects, contrasts = [], []
    user_path = output / 'paired_users.csv.gz'
    with gzip.open(user_path, 'wt', compresslevel=1, newline='') as stream:
        fields = ['dataset', 'tokenizer', 'method', 'control', 'user_id', 'activity_group',
                  'train_interaction_count', 'control_target_rank_at_10', 'method_target_rank_at_10',
                  'control_recall@10', 'method_recall@10', 'delta_recall@10',
                  'control_ndcg@10', 'method_ndcg@10', 'delta_ndcg@10']
        writer = csv.writer(stream)
        writer.writerow(fields)
        for method_index, method in enumerate(METHODS):
            control_name = manifest['control_rules'][method]
            control, treated = vectors[control_name], vectors[method]
            family = 'fine_tuning' if method in ['RW', 'SPRINT'] else 'reranking'
            shared = dict(dataset=case['dataset'], tokenizer=case['tokenizer'], method=method,
                          control=control_name, comparison_family=family)
            rng = np.random.default_rng(np.random.SeedSequence([manifest['seed'], index, method_index]))
            e, c = effects_for_method(control, treated, population['groups'], manifest['bootstrap_resamples'], rng)
            effects += [dict(**shared, **row) for row in e]
            contrasts += [dict(**shared, **row) for row in c]
            a, b = RANK_VALUES[control], RANK_VALUES[treated]
            for i, user in enumerate(population['users']):
                writer.writerow([case['dataset'], case['tokenizer'], method, control_name, user,
                    population['groups'][i], int(population['counts'][i]), int(control[i]), int(treated[i]),
                    a[i, 0], b[i, 0], b[i, 0]-a[i, 0], a[i, 1], b[i, 1], b[i, 1]-a[i, 1]])
    write_csv(output / 'effects.csv', effects)
    if contrasts:
        write_csv(output / 'activity_contrasts.csv', contrasts)
    write_json(output / 'complete.json', dict(status='complete', case=index,
        dataset=case['dataset'], tokenizer=case['tokenizer'], users=len(population['users']),
        group_counts={g: int(np.sum(population['groups'] == g)) for g in np.unique(population['groups'])},
        training_activity_boundaries=population['metadata']['upper_boundaries'],
        manifest=reference(manifest_path), prediction_sources=list(sources.values()),
        effects=reference(output / 'effects.csv'),
        contrasts=reference(output / 'activity_contrasts.csv') if contrasts else None,
        paired_users=reference(user_path), bootstrap_resamples=manifest['bootstrap_resamples'],
        inference=manifest['inference'], recovery=case['models']['SPRINT'].get('recovery')))
    print(f'Complete: {output}', flush=True)


def holm(pvalues):
    pvalues = np.asarray(pvalues, dtype=float)
    require(np.isfinite(pvalues).all() and ((pvalues >= 0) & (pvalues <= 1)).all(), 'Invalid p-values')
    order = np.argsort(pvalues)
    corrected = np.minimum(1., np.maximum.accumulate((len(pvalues)-np.arange(len(pvalues)))*pvalues[order]))
    result = np.empty_like(corrected)
    result[order] = corrected
    return result


def collect(manifest_path, plots=True):
    manifest_path = Path(manifest_path).resolve()
    output, manifest = manifest_path.parent, read_json(manifest_path)
    effects, contrasts, completed = [], [], []
    manifest_ref = reference(manifest_path)
    for case in manifest['cases']:
        folder = output / f'case_{case["index"]:02d}_{case["dataset"]}_{case["tokenizer"]}'
        complete = read_json(folder / 'complete.json')
        require(complete['manifest'] == manifest_ref and complete['status'] == 'complete', 'Case manifest differs')
        verified(complete['effects'])
        effects.extend(csv.DictReader(open(complete['effects']['path'])))
        if complete['contrasts']:
            verified(complete['contrasts'])
            contrasts.extend(csv.DictReader(open(complete['contrasts']['path'])))
        completed.append(complete)
    require(len(effects) >= 12*5*2, 'Incomplete effect coverage')
    families = {}
    for row in effects:
        stratum = 'overall' if row['activity_group'] == 'overall' else 'activity'
        row['holm_family'] = row['comparison_family'] + ':' + stratum
        families.setdefault(row['holm_family'], []).append(row)
    for row in contrasts:
        row['holm_family'] = row['comparison_family'] + ':activity_contrasts'
        families.setdefault(row['holm_family'], []).append(row)
    for name, rows in families.items():
        valid = [row for row in rows if row['p_raw'] not in ['', None]]
        adjusted = holm([float(row['p_raw']) for row in valid])
        for row in rows:
            row.update(p_holm='', reject_holm_0_05=False, family_tests=len(valid))
        for row, value in zip(valid, adjusted):
            row.update(p_holm=float(value), reject_holm_0_05=bool(value < .05))
    write_csv(output / 'effects.csv', effects)
    if contrasts:
        write_csv(output / 'activity_contrasts.csv', contrasts)
    consistency = []
    for method in METHODS:
        for metric in METRICS:
            groups = sorted({r['activity_group'] for r in effects})
            for group in groups:
                selected = [r for r in effects if r['method'] == method and r['metric'] == metric and r['activity_group'] == group]
                dimensions = [('all_cases', 'all', selected)]
                dimensions += [('within_dataset', ds, [r for r in selected if r['dataset'] == ds]) for ds in DATASETS]
                dimensions += [('within_tokenizer', tok, [r for r in selected if r['tokenizer'] == tok]) for tok in TOKENIZERS]
                for dimension, value, rows in dimensions:
                    if not rows:
                        continue
                    changes = [float(r['mean_change']) for r in rows]
                    consistency.append(dict(method=method, metric=metric, activity_group=group,
                        dimension=dimension, value=value, cases=len(rows),
                        expected_cases=12 if dimension == 'all_cases' else 3 if dimension == 'within_dataset' else 4,
                        positive_cases=sum(v > 0 for v in changes), negative_cases=sum(v < 0 for v in changes),
                        unchanged_cases=sum(v == 0 for v in changes),
                        holm_positive_cases=sum(float(r['mean_change']) > 0 and r['reject_holm_0_05'] for r in rows),
                        holm_negative_cases=sum(float(r['mean_change']) < 0 and r['reject_holm_0_05'] for r in rows),
                        median_change_percentage_points=100*float(np.median(changes))))
    write_csv(output / 'consistency.csv', consistency)
    write_json(output / 'analysis_summary.json', dict(status='complete', cases=len(completed),
        comparisons=len(completed)*5, effect_rows=len(effects), activity_contrast_rows=len(contrasts),
        holm_family_sizes={k: sum(r['p_raw'] not in ['', None] for r in v) for k, v in families.items()},
        user_counts={r['dataset']+'/'+r['tokenizer']: r['group_counts'] for r in completed},
        manifest=manifest_ref, inference=manifest['inference']))
    if plots:
        plot_effects(effects, output)
    print(f'Collected {len(completed)} cases; see {output}', flush=True)


def plot_effects(rows, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors, markers = ['#2878b5', '#d95823', '#6c4c9a'], ['o', 's', '^']
    groups = ['low', 'medium', 'high']
    for metric in METRICS:
        fig, axes = plt.subplots(5, 4, figsize=(12, 11), sharex=True, sharey='row')
        for mi, method in enumerate(METHODS):
            for di, dataset in enumerate(DATASETS):
                ax = axes[mi, di]
                for ti, tokenizer in enumerate(TOKENIZERS):
                    for gi, group in enumerate(groups):
                        selected = [r for r in rows if r['method'] == method and r['dataset'] == dataset
                                    and r['tokenizer'] == tokenizer and r['metric'] == metric and r['activity_group'] == group]
                        if not selected:
                            continue
                        require(len(selected) == 1, 'Duplicate plot cell')
                        r = selected[0]
                        x = gi + (ti-1)*.16
                        ax.plot([x, x], [100*float(r['ci_low']), 100*float(r['ci_high'])], color=colors[ti], lw=1.2)
                        ax.plot(x, 100*float(r['mean_change']), markers[ti], color=colors[ti], ms=4,
                                label=tokenizer.upper() if gi == 0 else None)
                ax.axhline(0, color='0.4', lw=.7)
                ax.grid(axis='y', alpha=.15)
                ax.set_xticks(range(3), groups)
                if mi == 0:
                    ax.set_title(dataset.replace('amazon-', '').title())
                if di == 0:
                    ax.set_ylabel(method+'\nChange (percentage points)')
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=3, frameon=False)
        fig.suptitle('Paired '+metric.upper()+' changes by training activity', y=.965)
        fig.text(.5, .015, 'RW/SPRINT vs standard fine-tuning; rerankers vs untreated candidates. '
                 'Intervals: unadjusted 95% user bootstrap; fixed checkpoints, seed 42.', ha='center', fontsize=9)
        fig.tight_layout(rect=(0, .04, 1, .945))
        stem = 'activity_'+metric.replace('@', '_at_')
        fig.savefig(output / (stem+'.pdf'))
        fig.savefig(output / (stem+'.png'), dpi=170)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--resamples', type=int, default=1000)
    p.add_argument('--seed', type=int, default=42)
    p = sub.add_parser('run')
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--case', type=int, required=True)
    p = sub.add_parser('collect')
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--no-plots', action='store_true')
    args = parser.parse_args()
    if args.command == 'prepare':
        require(args.resamples >= 1000, 'Use at least 1000 bootstrap resamples')
        prepare(args.output_dir, args.resamples, args.seed)
    elif args.command == 'run':
        require(0 <= args.case < 12, 'Case must be 0..11')
        run_case(args.manifest, args.case)
    else:
        collect(args.manifest, plots=not args.no_plots)


if __name__ == '__main__':
    main()
