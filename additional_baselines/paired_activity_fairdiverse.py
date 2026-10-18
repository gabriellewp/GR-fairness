"""Extend the completed paired activity analysis with seven recorded trainers.

Stream old full-catalog COO files; decode later compact CSR files. Reconstruct
the recorded target-first, float32 score+1 ranking and verify rounded metrics.
No training, checkpoint inference, or reranking is performed.
"""
import argparse
import contextlib
import csv
import gzip
import itertools
import json
import math
from pathlib import Path
import zipfile

import numpy as np
from scipy.sparse import load_npz
import yaml

from . import paired_activity as base

ROOT = base.ROOT
FAIRDIVERSE = ['IPS', 'FOCF', 'Reg', 'FairDual', 'Minmax-SGD', 'APR', 'SDRO']
ALL_METHODS = ['RW', 'SPRINT'] + FAIRDIVERSE + ['RR', 'CPFair', 'P-MMF']
FAMILIES = {'fine_tuning': ['RW', 'SPRINT'], 'fairdiverse_training': FAIRDIVERSE,
            'reranking': ['RR', 'CPFair', 'P-MMF']}
FIELDS = ['dataset', 'tokenizer', 'method', 'control', 'user_id', 'activity_group',
          'train_interaction_count', 'control_target_rank_at_10', 'method_target_rank_at_10',
          'control_recall@10', 'method_recall@10', 'delta_recall@10',
          'control_ndcg@10', 'method_ndcg@10', 'delta_ndcg@10']


def prepare(old_manifest, output, resamples=1000, seed=42):
    old_manifest, output = Path(old_manifest).resolve(), Path(output).resolve()
    old = base.read_json(old_manifest)
    base.verified(old['code'])
    base.require(base.read_json(old_manifest.parent / 'analysis_summary.json')['comparisons'] == 60,
                 'The original 60-comparison analysis must be complete')
    base.require(not (output / 'manifest.json').exists(), 'Use a new output directory')
    rows = list(csv.DictReader(open(old['comparison']['path'])))
    lookup = {(r['dataset'], r['tokenizer'], r['method']): r for r in rows if r['k'] == '10'}
    cases = []
    for previous in old['cases']:
        ds, tok = previous['dataset'], previous['tokenizer']
        original = lookup[ds, tok, 'GRID: original (fresh evaluation)']
        original_report = base.read_json(ROOT / original['source'])
        path = Path(previous['models']['Untreated']['path'])
        # The reranking analysis uses beam 300 here, whereas source GRID uses 100.
        if (ds, tok) == ('amazon-sports', 'rvq'):
            path = ROOT / 'additional_baselines/artifacts/sports_rvq_rerank_candidates_20261011_gpu4/test_candidates.json'
        base.require(path.exists(), f'Missing source prediction export: {path}')
        meta = base.extract(path, 'metadata')
        for key in ['checkpoint', 'sid_mapping', 'seed']:
            base.require(meta[key] == original_report['input_metadata'][key], f'Source {key} differs')
        for key in ['beam_width', 'sequence_length', 'num_hierarchies']:
            base.require(meta['generation'][key] == original_report['input_metadata']['generation'][key],
                         f'Source generation {key} differs')
        case = {k: previous[k] for k in ['index', 'dataset', 'tokenizer', 'split', 'activity', 'activity_metadata', 'grid_manifest']}
        case['source_grid'] = dict(path=str(path), expected=base.expected(original),
                                   report=base.reference(ROOT / original['source']))
        case['trainers'] = {}
        for method in FAIRDIVERSE:
            row = lookup[ds, tok, f'GRID + {method} (training)']
            base.require(row['status'].startswith('recorded FairDiverse'), 'Unexpected cohort')
            report = ROOT / row['source']
            folder = report.parent
            config_path, scores = folder / 'config.yaml', folder / 'ranking_scores.npz'
            config = yaml.safe_load(config_path.read_text())
            for key, wanted in [('dataset', ds), ('grid_tokenizer', tok), ('model', 'GRID'),
                                ('seed', 42), ('grid_compact_ranking', True), ('grid_native_metrics', False)]:
                base.require(config.get(key) == wanted, f'Unexpected {key}: {folder}')
            base.require((ROOT / config['grid_checkpoint_path']).resolve() == Path(meta['checkpoint']['path']).resolve(),
                         'Trainer source checkpoint differs')
            base.require((ROOT / config['sid_path']).resolve() == Path(meta['sid_mapping']['path']).resolve(), 'Trainer SID differs')
            base.require(config['top_k_for_generation'] == meta['generation']['beam_width'], 'Trainer beam width differs')
            with zipfile.ZipFile(scores) as archive:
                fmt = np.load(archive.open('format.npy'), allow_pickle=False).item().decode()
                shape = np.load(archive.open('shape.npy'), allow_pickle=False).tolist()
                base.require(fmt in ['coo', 'csr'], 'Unsupported stored score format')
                base.require(shape == [config['user_num'], config['item_num']], 'Stored dimensions differ from config')
                expanded = archive.getinfo('data.npy').file_size
            truth = folder / 'ground_truth.npz'
            base.require(truth.exists() or fmt == 'coo', 'Compact scores require a row-aligned ground-truth artifact')
            case['trainers'][method] = dict(config=base.reference(config_path), report=base.reference(report),
                scores=str(scores), scores_size=scores.stat().st_size, layout=fmt, shape=shape,
                expanded_score_bytes=expanded, ground_truth=base.reference(truth) if truth.exists() else None,
                expected={m: float(row[c]) for m, c in zip(base.METRICS, ['recall', 'ndcg'])},
                decimals=int(config['decimals']), optimizer_updates=int(config['max_steps']),
                accumulate_grad_batches=int(config['accumulate_grad_batches']))
        cases.append(case)
    output.mkdir(parents=True, exist_ok=True)
    inference = dict(old['inference'])
    inference.update(holm_families='Separately for fine_tuning/fairdiverse_training/reranking and overall/activity/activity_contrasts; both metrics and all 12 cases within each of nine families',
        fairdiverse_interpretation='Recorded complete training-procedure effect versus original source GRID; optimizer, budget, checkpoint selection, and historical ranking protocol are not matched to standard fine-tuning.',
        fairdiverse_ranking='Historical float32 score+1, full catalog in target-first order, NumPy descending quicksort; compared with native source predictions. These are not freshly harmonized evaluation protocols.',
        fairdiverse_precision='Unrounded user statistics reconstructed from saved scores; their aggregate metrics must agree with the recorded table after rounding to the configured decimals.',
        fairdiverse_alignment='COO exports validate every stored row index and the exact target-first item order against the frozen test split. CSR exports require matching one-target-per-row ground truth.')
    manifest = dict(schema_version=1, cases=cases, methods=ALL_METHODS, fairdiverse_methods=FAIRDIVERSE,
        old_manifest=base.reference(old_manifest), old_summary=base.reference(old_manifest.parent/'analysis_summary.json'),
        comparison=old['comparison'], bootstrap_resamples=resamples, seed=seed, inference=inference,
        code=base.reference(Path(__file__)), base_code=base.reference(Path(base.__file__)))
    base.write_json(output / 'manifest.json', manifest)
    print(f'Frozen 84 FairDiverse comparisons; combined scope is 144: {output}', flush=True)


@contextlib.contextmanager
def npy_member(archive, name):
    with archive.open(name) as stream:
        version = np.lib.format.read_magic(stream)
        shape, fortran, dtype = np.lib.format._read_array_header(stream, version)
        base.require(not fortran and len(shape) == 1, f'Unexpected array layout: {name}')
        yield stream, shape[0], dtype


def read_values(stream, count, dtype):
    data = stream.read(count * dtype.itemsize)
    base.require(len(data) == count*dtype.itemsize, 'Truncated score matrix member')
    return np.frombuffer(data, dtype=dtype)


def dense_target_ranks(scores):
    """Target is column zero; use exact quicksort only for ambiguous target ties."""
    values = np.asarray(scores, dtype=np.float32)
    base.require(np.isfinite(values).all(), 'Nonfinite saved scores')
    targets = values[:, :1]
    greater = (values > targets).sum(axis=1)
    ties = (values == targets).sum(axis=1)
    ranks = np.zeros(len(values), dtype=np.uint8)
    unique = (greater < 10) & (ties == 1)
    ranks[unique] = greater[unique]+1
    # Uniform rows can be common in collapsed models. Compute their actual
    # NumPy tie ordering once, rather than sorting identical rows repeatedly.
    uniform = ties == values.shape[1]
    if uniform.any():
        order = np.argsort(np.ones(values.shape[1], dtype=np.float32))[::-1][:10]
        rank = np.flatnonzero(order == 0)
        ranks[uniform] = int(rank[0])+1 if len(rank) else 0
    ambiguous = (greater < 10) & (ties > 1) & ~uniform
    for i in np.flatnonzero(ambiguous):
        order = np.argsort(values[i])[::-1][:10]
        found = np.flatnonzero(order == 0)
        ranks[i] = int(found[0])+1 if len(found) else 0
    return ranks, int(ambiguous.sum())


def coo_ranks(path, population, shape, batch=32):
    n, catalog = shape
    targets = population['targets']
    ranks = np.zeros(n, dtype=np.uint8)
    fallbacks = 0
    with zipfile.ZipFile(path) as archive:
        with npy_member(archive, 'row.npy') as rr, npy_member(archive, 'col.npy') as cc, npy_member(archive, 'data.npy') as dd:
            rows, rn, rd = rr
            cols, cn, cd = cc
            data, dn, dt = dd
            base.require(rn == cn == dn == n*catalog, 'Old COO export must contain the complete catalog per row')
            negative_ids = np.arange(catalog-1, dtype=np.int64)
            for start in range(0, n, batch):
                stop = min(n, start+batch)
                size = (stop-start)*catalog
                row_ids = read_values(rows, size, rd).reshape(-1, catalog)
                column_ids = read_values(cols, size, cd).reshape(-1, catalog)
                values = read_values(data, size, dt).reshape(-1, catalog)
                base.require(np.all(row_ids == np.arange(start, stop)[:, None]), 'Stored COO row order differs')
                t = targets[start:stop]
                base.require(np.array_equal(column_ids[:, 0], t), 'Stored COO positives differ from test targets')
                expected_negatives = negative_ids[None, :] + (negative_ids[None, :] >= t[:, None])
                base.require(np.array_equal(column_ids[:, 1:], expected_negatives), 'Stored target-first catalog differs')
                ranks[start:stop], extra = dense_target_ranks(values)
                fallbacks += extra
                if start % (batch*500) == 0:
                    print(f'    COO {stop}/{n} rows', flush=True)
    return ranks, dict(layout='coo', user_alignment='Every row index and target-first full catalog verified',
                       ambiguous_target_tie_sorts=fallbacks)


def csr_ranks(path, population, shape, batch=4096):
    matrix = load_npz(path).tocsr()
    matrix.sort_indices()
    base.require(matrix.shape == tuple(shape) and matrix.has_canonical_format, 'Invalid compact CSR shape or duplicate entries')
    n, catalog = shape
    targets = population['targets']
    ranks = np.zeros(n, dtype=np.uint8)
    uniform_order = np.argsort(np.ones(catalog, dtype=np.float32))[::-1][:10]
    uniform_match = np.flatnonzero(uniform_order == 0)
    uniform_rank = int(uniform_match[0])+1 if len(uniform_match) else 0
    fallbacks = 0
    for start in range(0, n, batch):
        stop = min(n, start+batch)
        a, b = matrix.indptr[start], matrix.indptr[stop]
        ids, values = matrix.indices[a:b], matrix.data[a:b].astype(np.float32, copy=False)
        base.require(np.isfinite(values).all() and (values >= 1).all(), 'Stored GRID scores must be finite and shifted by +1')
        lengths = np.diff(matrix.indptr[start:stop+1])
        row_ids = np.repeat(np.arange(stop-start), lengths)
        target_scores = np.ones(stop-start, dtype=np.float32)
        target_entries = ids == targets[start:stop][row_ids]
        target_scores[row_ids[target_entries]] = values[target_entries]
        greater = np.bincount(row_ids, weights=values > target_scores[row_ids], minlength=stop-start).astype(np.int64)
        ties = np.bincount(row_ids, weights=values == target_scores[row_ids], minlength=stop-start).astype(np.int64)
        ties += (target_scores == 1)*(catalog-lengths)
        selected = np.zeros(stop-start, dtype=np.uint8)
        unique = (greater < 10) & (ties == 1)
        selected[unique] = greater[unique]+1
        uniform = ties == catalog
        selected[uniform] = uniform_rank
        ambiguous = (greater < 10) & (ties > 1) & ~uniform
        for local in np.flatnonzero(ambiguous):
            index = start+local
            lo, hi = matrix.indptr[index:index+2]
            columns = matrix.indices[lo:hi]
            target = targets[index]
            # Dataset order is [positive target, ascending remaining catalog].
            positions = columns + (columns < target)
            positions[columns == target] = 0
            dense = np.ones(catalog, dtype=np.float32)
            dense[positions] = matrix.data[lo:hi]
            order = np.argsort(dense)[::-1][:10]
            found = np.flatnonzero(order == 0)
            selected[local] = int(found[0])+1 if len(found) else 0
        ranks[start:stop] = selected
        fallbacks += int(ambiguous.sum())
        if start % (batch*25) == 0:
            print(f'    CSR {stop}/{n} rows', flush=True)
    return ranks, dict(layout='csr', user_alignment='Row-aligned ground truth required and checked',
                       ambiguous_target_tie_sorts=fallbacks)


def recorded_ranks(spec, population):
    for key in ['config', 'report']:
        base.verified(spec[key])
    config = yaml.safe_load(Path(spec['config']['path']).read_text())
    path = Path(spec['scores'])
    base.require(path.stat().st_size == spec['scores_size'], 'Stored score artifact size changed')
    base.require(spec['shape'][0] == len(population['users']), 'Recorded user count differs')
    if spec['ground_truth']:
        base.verified(spec['ground_truth'])
        truth = load_npz(spec['ground_truth']['path']).tocsr()
        base.require(truth.shape == tuple(spec['shape']), 'Ground-truth shape differs')
        base.require(np.all(np.diff(truth.indptr) == 1) and np.array_equal(truth.indices, population['targets']),
                     'Stored ground-truth row targets differ from the frozen test split')
    else:
        base.require(spec['layout'] == 'coo', 'Cannot establish target alignment')
    if spec['layout'] == 'coo':
        ranks, audit = coo_ranks(path, population, spec['shape'])
    else:
        ranks, audit = csr_ranks(path, population, spec['shape'])
    means = base.RANK_VALUES[ranks].mean(axis=0)
    for i, metric in enumerate(base.METRICS):
        actual = float(np.round(means[i], spec['decimals']))
        base.require(actual == spec['expected'][metric],
                     f'Recorded {metric} failed reproduction: {actual} vs {spec["expected"][metric]} ({path})')
    audit.update(reconstructed_unrounded_means=dict(zip(base.METRICS, means.tolist())),
                 table_rounded_means=spec['expected'], decimals=spec['decimals'], scores=base.reference(path),
                 ground_truth=spec['ground_truth'], optimizer_updates=spec['optimizer_updates'],
                 accumulate_grad_batches=spec['accumulate_grad_batches'])
    return ranks, audit


def case_folder(root, case):
    return root / f'case_{case["index"]:02d}_{case["dataset"]}_{case["tokenizer"]}'


def run_case(manifest_path, index):
    manifest_path = Path(manifest_path).resolve()
    manifest = base.read_json(manifest_path)
    for key in ['code', 'base_code', 'comparison', 'old_manifest', 'old_summary']:
        base.verified(manifest[key])
    case = manifest['cases'][index]
    output = case_folder(manifest_path.parent / 'fairdiverse_cases', case)
    base.require(not (output / 'complete.json').exists(), 'Completed case already exists')
    output.mkdir(parents=True, exist_ok=True)
    population = base.read_population(case)
    base.verified(case['source_grid']['report'])
    control, meta, _, source_ref = base.candidates(case['source_grid'], population, case)
    print(f'FairDiverse case {index}: {case["dataset"]}/{case["tokenizer"]}; {len(control)} users', flush=True)
    effects, contrasts, audits = [], [], {}
    with gzip.open(output/'paired_users.csv.gz', 'wt', newline='', compresslevel=1) as stream:
        writer = csv.writer(stream)
        writer.writerow(FIELDS)
        for mi, method in enumerate(FAIRDIVERSE):
            print(f'  Reconstructing {method}', flush=True)
            ranks, audits[method] = recorded_ranks(case['trainers'][method], population)
            rng = np.random.default_rng(np.random.SeedSequence([manifest['seed'], index, mi+100]))
            e, c = base.effects_for_method(control, ranks, population['groups'], manifest['bootstrap_resamples'], rng)
            shared = dict(dataset=case['dataset'], tokenizer=case['tokenizer'], method=method,
                          control='Source GRID', comparison_family='fairdiverse_training')
            effects += [dict(**shared, **r) for r in e]
            contrasts += [dict(**shared, **r) for r in c]
            a, b = base.RANK_VALUES[control], base.RANK_VALUES[ranks]
            for i, user in enumerate(population['users']):
                writer.writerow([case['dataset'], case['tokenizer'], method, 'Source GRID', user,
                    population['groups'][i], int(population['counts'][i]), int(control[i]), int(ranks[i]),
                    a[i, 0], b[i, 0], b[i, 0]-a[i, 0], a[i, 1], b[i, 1], b[i, 1]-a[i, 1]])
            print(f'  Verified and analysed {method}', flush=True)
    base.write_csv(output/'effects.csv', effects)
    base.write_csv(output/'activity_contrasts.csv', contrasts)
    base.write_json(output/'complete.json', dict(status='complete', case=index,
        dataset=case['dataset'], tokenizer=case['tokenizer'], users=len(control),
        group_counts={g: int(np.sum(population['groups'] == g)) for g in np.unique(population['groups'])},
        manifest=base.reference(manifest_path), effects=base.reference(output/'effects.csv'),
        contrasts=base.reference(output/'activity_contrasts.csv'), paired_users=base.reference(output/'paired_users.csv.gz'),
        source_grid_predictions=source_ref, score_reproduction=audits, inference=manifest['inference']))
    print(f'Complete: {output}', flush=True)


def merge(manifest_path, plots=True):
    manifest_path = Path(manifest_path).resolve()
    root, manifest = manifest_path.parent, base.read_json(manifest_path)
    for key in ['code', 'base_code', 'old_manifest', 'old_summary']:
        base.verified(manifest[key])
    old_path = Path(manifest['old_manifest']['path'])
    for case in manifest['cases']:
        previous = base.read_json(case_folder(old_path.parent, case)/'complete.json')
        new = base.read_json(case_folder(root/'fairdiverse_cases', case)/'complete.json')
        base.require(previous['manifest'] == manifest['old_manifest'] and new['manifest'] == base.reference(manifest_path),
                     'A completed case belongs to a different frozen analysis')
        base.require(previous['group_counts'] == new['group_counts'], 'Activity populations differ')
        folder = case_folder(root, case)
        folder.mkdir(exist_ok=True)
        for kind in ['effects', 'contrasts']:
            rows = []
            for record in [previous, new]:
                base.verified(record[kind])
                rows.extend(csv.DictReader(open(record[kind]['path'])))
            base.require({r['method'] for r in rows} == set(ALL_METHODS), 'Missing intervention family')
            base.require(len({(r['method'],r['metric'],r['activity_group']) if kind == 'effects'
                              else (r['method'],r['metric'],r['group_a'],r['group_b']) for r in rows}) == len(rows),
                         'Duplicated effect/contrast hypotheses')
            filename = 'effects.csv' if kind == 'effects' else 'activity_contrasts.csv'
            base.write_csv(folder/filename, rows)
        base.write_json(folder/'complete.json', dict(status='complete', case=case['index'],
            dataset=case['dataset'], tokenizer=case['tokenizer'], group_counts=new['group_counts'],
            manifest=base.reference(manifest_path), effects=base.reference(folder/'effects.csv'),
            contrasts=base.reference(folder/'activity_contrasts.csv'),
            paired_users_files=[previous['paired_users'],new['paired_users']],
            source_results=[str(case_folder(old_path.parent,case)/'complete.json'),
                            str(case_folder(root/'fairdiverse_cases',case)/'complete.json')]))
    # Reuse the identical correction and descriptive-consistency implementation.
    # No existing analysis outputs or manifests are modified.
    original_methods = base.METHODS
    try:
        base.METHODS = ALL_METHODS
        base.collect(manifest_path, plots=False)
    finally:
        base.METHODS = original_methods
    summary = base.read_json(root/'analysis_summary.json')
    summary.update(comparisons=144, methods=ALL_METHODS, comparison_families=FAMILIES,
        source_analysis=manifest['old_manifest'], recorded_score_comparisons=84)
    base.write_json(root/'analysis_summary.json', summary)
    if plots:
        plot_families(list(csv.DictReader(open(root/'effects.csv'))), root)
    print(f'Combined all 144 comparisons in {root}', flush=True)


def plot_families(rows, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors, markers = ['#2878b5', '#d95823', '#6c4c9a'], ['o', 's', '^']
    groups = ['low', 'medium', 'high']
    controls = {'fine_tuning': 'RW/SPRINT vs standard fine-tuning',
                'fairdiverse_training': 'Recorded FairDiverse training vs source GRID; full procedure and evaluation-protocol effects',
                'reranking': 'Rerankers vs matched untreated candidates'}
    for family, methods in FAMILIES.items():
        for metric in base.METRICS:
            height = 2*len(methods)+1.5
            fig, axes = plt.subplots(len(methods), 4, figsize=(12, height), sharex=True, sharey='row', squeeze=False)
            for mi, method in enumerate(methods):
                for di, dataset in enumerate(base.DATASETS):
                    ax = axes[mi,di]
                    for ti, tokenizer in enumerate(base.TOKENIZERS):
                        for gi, group in enumerate(groups):
                            selected = [r for r in rows if (r['method'],r['dataset'],r['tokenizer'],r['metric'],r['activity_group'])
                                        == (method,dataset,tokenizer,metric,group)]
                            if not selected:
                                continue
                            base.require(len(selected) == 1, 'Duplicated plot point')
                            r = selected[0]
                            x = gi + (ti-1)*.16
                            ax.plot([x,x], [100*float(r['ci_low']),100*float(r['ci_high'])], color=colors[ti],lw=1.2)
                            ax.plot(x,100*float(r['mean_change']),markers[ti],color=colors[ti],ms=4,
                                    label=['RKMeans','RVQ','RQ-VAE'][ti] if gi == 0 else None)
                    ax.axhline(0,color='.4',lw=.7)
                    ax.grid(axis='y',alpha=.15)
                    ax.set_xticks(range(3),groups)
                    if mi == 0:
                        ax.set_title(dataset.replace('amazon-','').title())
                    if di == 0:
                        ax.set_ylabel(method+'\nChange (percentage points)')
            handles, labels = axes[0,0].get_legend_handles_labels()
            fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,1-.03/height),ncol=3,frameon=False)
            fig.suptitle(controls[family]+'\nPaired '+metric.upper()+' changes by training activity',y=1-.40/height,fontsize=11)
            fig.text(.5,.12/height,'Pointwise 95% user-bootstrap intervals; conditional on saved checkpoints and predictions. '
                     'No training-seed uncertainty.',ha='center',fontsize=8)
            fig.tight_layout(rect=(0,.40/height,1,1-.95/height))
            stem='activity_'+family+'_'+metric.replace('@','_at_')
            fig.savefig(output/(stem+'.pdf'))
            fig.savefig(output/(stem+'.png'),dpi=160)
            plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    p=commands.add_parser('prepare')
    p.add_argument('--old-manifest',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--resamples',type=int,default=1000)
    p.add_argument('--seed',type=int,default=42)
    p=commands.add_parser('run')
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--case',type=int,required=True)
    p=commands.add_parser('merge')
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--no-plots',action='store_true')
    args=parser.parse_args()
    if args.command=='prepare':
        base.require(args.resamples>=1000,'Use at least 1000 bootstrap replicates')
        prepare(args.old_manifest,args.output_dir,args.resamples,args.seed)
    elif args.command=='run':
        base.require(0<=args.case<12,'Case must be 0..11')
        run_case(args.manifest,args.case)
    else:
        merge(args.manifest,plots=not args.no_plots)


if __name__=='__main__':
    main()
