"""Recover K=10 popularity exposure/calibration from frozen experiment artifacts.

CPU only; no model inference, training, or new reranking. Cases and historical
score conventions come from the completed paired-activity audits. Each method
is cached separately. Run with numpy, scipy, and ijson installed::

    python -m additional_baselines.popularity_fairness_analysis --case 0
    python docs/make_popularity_fairness.py
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import zipfile

import numpy as np
from scipy.sparse import load_npz
try:
    import ijson
except ModuleNotFoundError:
    # Reuse an existing local ijson installation without changing the active
    # environment or giving its other packages priority over this environment.
    if not os.environ.get('POPULARITY_FAIRNESS_IJSON_PATH'):
        raise
    sys.path.append(os.environ['POPULARITY_FAIRNESS_IJSON_PATH'])
    import ijson

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/popularity_fairness_20261017'
OLD = ROOT / 'docs/paired_activity_20261016/manifest.json'
FD = ROOT / 'docs/paired_activity_all_methods_20261016/manifest.json'
METHODS = ['Standard FT', 'RW', 'SPRINT', 'IPS', 'FOCF', 'Reg', 'FairDual',
           'Minmax-SGD', 'APR', 'SDRO', 'CPFair', 'P-MMF', 'RR']


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def reference(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b''):
            digest.update(chunk)
    return dict(path=str(path), sha256=digest.hexdigest())


def verified(spec):
    require(reference(spec['path']) == spec, f'Changed frozen input: {spec["path"]}')


def extract(path, prefix):
    with Path(path).open('rb') as stream:
        return next(ijson.items(stream, prefix, use_float=True))


def mapping_digest(mapping):
    return hashlib.sha256(json.dumps(sorted(mapping.items()), separators=(',', ':')).encode()).hexdigest()


def gini(counts):
    counts = np.sort(np.asarray(counts, dtype=float))
    require(len(counts) > 0 and counts.sum() > 0, 'Empty exposure')
    return float(np.dot(2*np.arange(1, len(counts)+1)-len(counts)-1, counts)
                 / (len(counts)*counts.sum()))


def metrics(shares, gh):
    gp = np.asarray(shares, dtype=float)
    require(gp.shape == (5,) and np.isclose(gp.sum(), 1, atol=1e-12), 'Invalid five-bin shares')
    gu = gp-np.asarray(gh)
    return dict(popularity_gini_at_10=gini(gp), mgu_at_10=float(np.abs(gu).mean()),
                dgu_at_10=float(gu.max()-gu.min()), popularity_shares=gp.tolist(),
                history_shares=list(gh), group_gaps=gu.tolist())


def dataset_context(old_case):
    """Validate training-derived bins and pool each test request's last 40 items."""
    folder = OUT / old_case['dataset']
    if (folder/'context.json').exists():
        context = read(folder/'context.json')
        for key in ['split', 'train', 'candidate_groups']:
            spec = context[key]
            # Candidate group evidence is a header, linked to the older audit's
            # full-file fingerprint; its mapping is rechecked in every case.
            if key != 'candidate_groups':
                verified(spec)
        return context, np.load(folder/'population.npz')['targets']
    verified(old_case['split'])
    candidate = Path(old_case['models']['Untreated']['path'])
    groups = extract(candidate, 'item_groups')
    popularity = {int(i): int(g) for i, g in groups['popularity'].items()}
    brand = {int(i): str(g) for i, g in groups['brand'].items()}
    catalog = len(popularity)
    require(set(popularity) == set(range(catalog)) == set(brand), 'Catalog IDs must be contiguous')
    split = Path(old_case['split']['path'])
    train = split.with_name(old_case['dataset']+'.train')
    counts, seen = Counter(), set()
    with train.open(newline='') as stream:
        for r in csv.DictReader(stream, delimiter='\t'):
            user, target = int(r['user_id:token']), int(r['item_id:token'])
            if user not in seen:
                counts.update(json.loads(r['history_behaviors'])[-40:])
                seen.add(user)
            counts[target] += 1
    ordered = sorted(range(catalog), key=lambda i: (counts[i], i))
    expected = {item: min(index*5//catalog, 4) for index, item in enumerate(ordered)}
    require(popularity == expected, 'Frozen bins differ from training-frequency bins')
    pop_array = np.array([popularity[i] for i in range(catalog)], dtype=np.int8)
    labels = sorted(set(brand.values()))
    brand_index = {g: n for n, g in enumerate(labels)}
    brand_array = np.array([brand_index[brand[i]] for i in range(catalog)], dtype=np.int32)
    history_counts = np.zeros(catalog, dtype=np.int64)
    users, targets = [], []
    with split.open(newline='') as stream:
        for r in csv.DictReader(stream, delimiter='\t'):
            users.append(int(r['user_id']))
            targets.append(int(r['target_item']))
            history = json.loads(r['history_behaviors'])[-40:]
            # np.add.at counts duplicate items within a history correctly.
            np.add.at(history_counts, history, 1)
    require(len(set(users)) == len(users), 'Expected one test request per user')
    gh_counts = np.bincount(pop_array, weights=history_counts, minlength=5)
    context = dict(dataset=old_case['dataset'], requests=len(users), catalog=catalog,
        split=old_case['split'], train=reference(train),
        candidate_groups=dict(path=str(candidate),
            sha256=old_case['models']['Untreated']['expected_sha256']),
        popularity_mapping_sha256=mapping_digest({i: str(g) for i,g in popularity.items()}),
        popularity_mapping={str(i):str(g) for i,g in popularity.items()},
        brand_labels=labels, brand_mapping=groups['brand'],
        history_length_cap=40, history_counts=gh_counts.astype(int).tolist(),
        history_shares=(gh_counts/gh_counts.sum()).tolist())
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder/'population.npz', targets=np.array(targets, dtype=np.int64),
                        users=np.array(users, dtype=np.int64), popularity=pop_array, brand=brand_array)
    write(folder/'context.json', context)
    return context, np.asarray(targets)


def native_result(report, context):
    verified(report)
    saved = read(report['path'])['metrics']
    at = saved['at_k']['10']
    require(saved['request_count'] == context['requests'], 'Native request count differs')
    exposure = at['groups'].get('popularity')
    if exposure is None:
        return None
    counts = [exposure['exposure_counts'].get(str(g), 0) for g in range(5)]
    result = metrics(np.array(counts)/sum(counts), context['history_shares'])
    require(abs(result['popularity_gini_at_10']-exposure['gini']) < 1e-12, 'Native Gini mismatch')
    result.update(popularity_counts=counts, exposure_positions=sum(counts),
        recall_at_10=at['recall'], ndcg_at_10=at['ndcg'],
        brand_gini_at_10=at['groups']['brand']['gini'], sources=[report],
        reconstruction='Saved native unweighted top-10 exposure counts')
    return result


class Aggregate:
    def __init__(self, context):
        self.context = context
        self.pop = np.array([int(context['popularity_mapping'][str(i)]) for i in range(context['catalog'])])
        lookup = {g:i for i,g in enumerate(context['brand_labels'])}
        self.brand = np.array([lookup[context['brand_mapping'][str(i)]] for i in range(context['catalog'])])
        self.pc = np.zeros(5, dtype=np.int64)
        self.bc = np.zeros(len(lookup), dtype=np.int64)
        self.n = self.hits = 0
        self.ndcg = 0.
        self.digest = hashlib.sha256()

    def add(self, items, targets):
        items = np.asarray(items, dtype=np.int64)
        if items.ndim == 1:
            items = items[None, :]
        require(items.shape[1] <= 10, 'Unexpected cutoff')
        self.pc += np.bincount(self.pop[items].ravel(), minlength=5)
        self.bc += np.bincount(self.brand[items].ravel(), minlength=len(self.bc))
        hits = items == np.asarray(targets)[:, None]
        require(np.all(hits.sum(axis=1) <= 1), 'Duplicate target')
        self.hits += int(hits.any(axis=1).sum())
        self.ndcg += float((hits/np.log2(np.arange(items.shape[1])+2)).sum())
        self.n += len(items)
        self.digest.update(np.sort(items, axis=1).astype('<i4').tobytes())

    def finish(self):
        require(self.n == self.context['requests'], 'Not the complete test split')
        result = metrics(self.pc/self.pc.sum(), self.context['history_shares'])
        result.update(popularity_counts=self.pc.tolist(), brand_counts=self.bc.tolist(),
            exposure_positions=int(self.pc.sum()), brand_gini_at_10=gini(self.bc),
            recall_at_10=self.hits/self.n, ndcg_at_10=self.ndcg/self.n,
            top10_item_sets_sha256=self.digest.hexdigest())
        return result


def positions_to_items(positions, targets):
    q = np.maximum(positions-1, 0)
    return np.where(positions == 0, targets[:,None], q+(q >= targets[:,None]))


def dense_top10(values):
    """Fast unique-score top ten; ties use the recorded full NumPy quicksort."""
    values = np.asarray(values, dtype=np.float32)
    candidate = np.argpartition(values, -10, axis=1)[:, -10:]
    selected = np.take_along_axis(values, candidate, axis=1)
    order = np.argsort(selected, axis=1)[:, ::-1]
    top = np.take_along_axis(candidate, order, axis=1)
    ranked = np.take_along_axis(values, top, axis=1)
    threshold = ranked[:, -1]
    ties = (np.diff(ranked, axis=1) == 0).any(axis=1)
    ties |= (values == threshold[:, None]).sum(axis=1) > 1
    for row in np.flatnonzero(ties):
        top[row] = np.argsort(values[row])[::-1][:10]
    return top


def sparse_top10(matrix, targets, aggregate, batch=4096):
    """Reproduce full-catalog ranking without materializing the Pet catalog."""
    matrix = matrix.tocsr()
    matrix.sort_indices()
    require(matrix.has_canonical_format, 'Duplicate sparse entries')
    n, catalog = matrix.shape
    uniform = np.argsort(np.ones(catalog, dtype=np.float32))[::-1][:10]
    audit = dict(uniform_rows=0, full_catalog_tie_fallback_rows=0)
    for start in range(0, n, batch):
        stop = min(n, start+batch)
        lengths = np.diff(matrix.indptr[start:stop+1])
        width = max(11, int(lengths.max(initial=0)))
        values = np.full((stop-start, width), -np.inf, dtype=np.float32)
        ids = np.zeros((stop-start, width), dtype=np.int64)
        a,b = matrix.indptr[start], matrix.indptr[stop]
        rows = np.repeat(np.arange(stop-start), lengths)
        offsets = np.arange(b-a)-np.repeat(matrix.indptr[start:stop]-a, lengths)
        values[rows, offsets] = matrix.data[a:b]
        ids[rows, offsets] = matrix.indices[a:b]
        require(np.isfinite(matrix.data[a:b]).all() and (matrix.data[a:b] >= 1).all(),
                'Scores must preserve historical +1 shift')
        order = np.argsort(values, axis=1)[:, ::-1][:, :10]
        top = np.take_along_axis(ids, order, axis=1)
        ranked = np.take_along_axis(values, order, axis=1)
        # Internal ties among non-target items cannot change exposure or target
        # rank. Full catalog quicksort is needed only for ties crossing the
        # cutoff, a tied selected target, or implicit baseline-score fillers.
        cutoff = ranked[:, -1, None]
        ambiguous = ranked[:, -1] <= 1
        ambiguous |= ((values == cutoff).sum(axis=1)
                      > (ranked == cutoff).sum(axis=1))
        positive = top == targets[start:stop, None]
        target_score = np.where(positive, ranked, -np.inf).max(axis=1)
        ambiguous |= positive.any(axis=1) & ((ranked == target_score[:, None]).sum(axis=1) > 1)
        def resolve_tie(local):
            row = start+local
            lo,hi = matrix.indptr[row:row+2]
            if np.all(matrix.data[lo:hi].astype(np.float32) == 1):
                positions = uniform
                is_uniform = True
            else:
                columns = matrix.indices[lo:hi]
                target = targets[row]
                positions = columns+(columns < target)
                positions[columns == target] = 0
                dense = np.ones(catalog, dtype=np.float32)
                dense[positions] = matrix.data[lo:hi]
                positions = np.argsort(dense)[::-1][:10]
                is_uniform = False
            items = positions_to_items(positions[None,:], targets[row:row+1])[0]
            return local, items, is_uniform

        # NumPy releases the GIL while sorting. Bounded parallel full-catalog
        # fallbacks retain the exact tie result and avoid one-core bottlenecks.
        with ThreadPoolExecutor(max_workers=4) as pool:
            for local,items,is_uniform in pool.map(resolve_tie, np.flatnonzero(ambiguous)):
                top[local] = items
                audit['uniform_rows' if is_uniform else 'full_catalog_tie_fallback_rows'] += 1
        aggregate.add(top, targets[start:stop])
        if start % (batch*25) == 0:
            print(f'    recovered {stop}/{n} requests', flush=True)
    return audit


def recorded_result(spec, old_audit, context, targets):
    for key in ['config', 'report']:
        verified(spec[key])
    scores = reference(spec['scores'])
    require(scores == old_audit['scores'], 'Scores changed since full row/column alignment audit')
    require(spec['shape'] == [len(targets), context['catalog']], 'Score dimensions differ')
    if spec['ground_truth']:
        verified(spec['ground_truth'])
        truth = load_npz(spec['ground_truth']['path']).tocsr()
        require(np.all(np.diff(truth.indptr) == 1) and np.array_equal(truth.indices, targets),
                'Saved ground truth differs from frozen targets')
    agg = Aggregate(context)
    with zipfile.ZipFile(spec['scores']) as archive:
        with archive.open('data.npy') as stream:
            version = np.lib.format.read_magic(stream)
            shape, fortran, dtype = np.lib.format._read_array_header(stream, version)
            full = shape == (len(targets)*context['catalog'],)
            if full:
                require(not fortran, 'Unexpected score order')
                for start in range(0, len(targets), 128):
                    stop = min(len(targets), start+128)
                    count = (stop-start)*context['catalog']
                    raw = stream.read(count*dtype.itemsize)
                    require(len(raw) == count*dtype.itemsize, 'Truncated scores')
                    values = np.frombuffer(raw, dtype=dtype).reshape(-1, context['catalog'])
                    require(np.isfinite(values).all(), 'Nonfinite scores')
                    top = dense_top10(values)
                    agg.add(positions_to_items(top, targets[start:stop]), targets[start:stop])
                audit = dict(layout='full COO', alignment='Reused SHA256-matched prior every-row/item alignment audit')
    if not full:
        audit = sparse_top10(load_npz(spec['scores']), targets, agg)
        audit.update(layout=old_audit['layout'], alignment=old_audit['user_alignment'])
    result = agg.finish()
    saved = read(spec['report']['path'])
    for key, saved_key in [('recall_at_10','hr@10'), ('ndcg_at_10','ndcg@10'), ('brand_gini_at_10','gini@10')]:
        require(round(result[key], spec['decimals']) == saved[saved_key],
                f'Historical {key} failed reproduction: {result[key]} vs {saved[saved_key]}')
    for key, metric in [('recall_at_10','recall@10'), ('ndcg_at_10','ndcg@10')]:
        require(abs(result[key]-old_audit['reconstructed_unrounded_means'][metric]) < 1e-11,
                f'Unrounded activity audit differs: {key}')
    result.update(sources=[scores, spec['report'], spec['config']], reconstruction=audit,
        prior_score_reproduction=old_audit,
        runtime=dict(python=sys.version, numpy=np.__version__, ijson=ijson.__version__),
        verified_metrics=['Recall@10', 'NDCG@10', 'brand Gini@10'])
    return result


def calibration_result(values, context, sources):
    gh = [values['GH'][str(g)] for g in range(5)]
    require(np.allclose(gh, context['history_shares'], rtol=0, atol=1e-12), 'History reference differs')
    result = metrics([values['GP'][str(g)] for g in range(5)], gh)
    for name, key in [('MGU','mgu_at_10'), ('DGU','dgu_at_10')]:
        require(abs(result[key]-values[name]) < 1e-12, 'Saved calibration mismatch')
    result.update(sources=sources, reconstruction='Previously verified saved-ranking popularity shares; Gini recomputed over all five bins')
    return result


def reranking_results(old_case, context, targets):
    ds,tok = old_case['dataset'],old_case['tokenizer']
    if tok != 'rkmeans':
        path = ROOT / f'docs/tokenizer_reranking_analysis_20261014/calibration/{ds}_{tok}.json'
        calibration = read(path)
        require(calibration['dataset'] == ds and calibration['tokenizer'] == tok,
                'Calibration case differs')
        candidate = old_case['models']['Untreated']
        require(dict(path=candidate['path'], sha256=candidate['expected_sha256'])
                in calibration['sources'], 'Calibrated candidate fingerprint differs')
        for method in ['RR', 'CPFair', 'P-MMF']:
            require(any(s['path'] == old_case['reports'][method]['path']
                        for s in calibration['sources']), 'Calibrated reranking source differs')
        for source in calibration['sources']:
            # Large candidate/ranking files have already been verified by the
            # frozen paired-activity study. The reused calibration is hashed.
            require(Path(source['path']).exists(), 'Missing calibrated source')
        return {name:calibration_result(calibration['results'][key]['10'], context,
                [reference(path)] + calibration['sources'])
                for name,key in [('Saved reference','control'),('RR','rr'),('CPFair','cpfair'),('P-MMF','pmmf')]}
    rr_path = old_case['reports']['RR']['path']
    frozen = extract(rr_path, 'frozen_correction')
    require(frozen['group_mapping_sha256'] == context['popularity_mapping_sha256'], 'RR bins differ')
    rr_reference = reference(rr_path)
    results = {}
    for name,prefix in [('Saved reference','control_ifairlrs_fairness'),('RR','ifairlrs_fairness')]:
        values = extract(rr_path, prefix)['10']
        results[name] = calibration_result(values, context, [rr_reference])
    for method,key in [('CPFair','cpfair'),('P-MMF','pmmf')]:
        if ds != 'amazon-pet-supplies':
            path = ROOT / f'additional_baselines/artifacts/rkmeans_calibration_357290/{ds.removeprefix("amazon-")}_{key}_calibration.json'
            calibration = read(path)
            require(calibration['group_mapping_sha256'] == context['popularity_mapping_sha256'], 'Calibration bins differ')
            require(calibration['saved_rankings']['path'] == old_case['reports'][method]['path'], 'Reranking report differs')
            results[method] = calibration_result(calibration['reranked']['10'], context,
                                                 [reference(path), calibration['saved_rankings']])
        else:
            # No precomputed RKMeans/Pet calibration: stream only saved top ten.
            path = Path(old_case['reports'][method]['path'])
            meta = extract(path, 'input_metadata')
            native_meta = extract(old_case['models']['Untreated']['path'], 'metadata')
            require(meta == native_meta, 'Reranking metadata differs from fixed pool')
            prefix = None
            with path.open('rb') as stream:
                for location,event,key in ijson.parse(stream):
                    if location == '' and event == 'map_key' and key in ['runs', 'rankings_by_k']:
                        prefix = 'runs.10.rankings' if key == 'runs' else 'rankings_by_k.10'
                        break
            require(prefix is not None, 'Unknown saved reranking layout')
            population = np.load(OUT/ds/'population.npz')
            rid_index = {f'test:{i}:user:{u}':i for i,u in enumerate(population['users'])}
            seen = set()
            aggregate = Aggregate(context)
            with path.open('rb') as stream:
                for rid,items in ijson.kvitems(stream, prefix, use_float=True):
                    require(rid in rid_index and rid not in seen, 'Reranking request ID differs')
                    seen.add(rid)
                    index = rid_index[rid]
                    require(len(items[:10]) == len(set(items[:10])), 'Duplicate reranked item')
                    aggregate.add(items[:10], targets[index:index+1])
            result = aggregate.finish()
            expected = old_case['reports'][method]['expected']
            require(abs(result['recall_at_10']-expected['recall@10']) < 1e-12, 'Reranked recall differs')
            require(abs(result['ndcg_at_10']-expected['ndcg@10']) < 1e-11, 'Reranked NDCG differs')
            result.update(sources=[reference(path)], reconstruction='Streamed saved K=10 rerankings')
            results[method] = result
    return results


def run_case(index):
    old_case = read(OLD)['cases'][index]
    case = read(FD)['cases'][index]
    context,targets = dataset_context(old_case)
    folder = OUT / f'case_{index:02d}_{case["dataset"]}_{case["tokenizer"]}'
    folder.mkdir(parents=True, exist_ok=True)
    print(f'Case {index}: {case["dataset"]}/{case["tokenizer"]}', flush=True)
    groups = extract(case['source_grid']['path'], 'item_groups')
    require(groups['popularity'] == context['popularity_mapping'], 'Source bins differ between tokenizers')
    require(groups['brand'] == context['brand_mapping'], 'Source brand mapping differs')
    rerank_cache = folder/'reranking.json'
    if not rerank_cache.exists():
        write(rerank_cache, reranking_results(old_case, context, targets))
    results = read(rerank_cache)
    for method in ['Standard FT','RW','SPRINT']:
        path = folder/(method.replace(' ','_')+'.json')
        if not path.exists():
            spec = old_case['models'][method]
            groups = extract(spec['path'], 'item_groups')
            require(groups['popularity'] == context['popularity_mapping'], 'Fine-tuning bins changed')
            result = native_result(spec['report'], context)
            if result is None:
                require(method == 'SPRINT' and index == 10, 'Unexpected missing native exposure counts')
                require(spec['path'] == case['source_grid']['path'], 'Update-zero recovery differs from source')
                result = dict(results['Saved reference'])
                result['reconstruction'] = 'Update-zero checkpoint; identical source predictions confirmed by frozen activity recovery'
            write(path, result)
        results[method] = read(path)
    source_path = folder/'Original.json'
    if not source_path.exists():
        result = native_result(case['source_grid']['report'], context)
        if result is None and index != 4:
            require(case['source_grid']['path'] == old_case['models']['Untreated']['path'], 'Source differs from reranking control')
            result = dict(results['Saved reference'])
            result['reconstruction'] += '; source and saved reference use identical candidate artifact'
        elif result is None:
            # Sports RVQ source uses beam 100; saved reranking control uses 300.
            path = case['source_grid']['path']
            meta = extract(path, 'metadata')
            aggregate = Aggregate(context)
            users = np.load(OUT/case['dataset']/'population.npz')['users']
            with Path(path).open('rb') as stream:
                for i,r in enumerate(ijson.items(stream,'requests.item',use_float=True)):
                    require(r['request_id'] == f'test:{i}:user:{users[i]}' and r['targets'] == [int(targets[i])], 'Source rows differ')
                    order = sorted(range(len(r['scores'])), key=lambda j:r['scores'][j],
                                   reverse=meta['score_convention'] != 'negative_log_probability')[:10]
                    aggregate.add([r['candidate_ids'][j] for j in order], targets[i:i+1])
            result = aggregate.finish()
            result.update(sources=[reference(path)], reconstruction='Native stable-score top ten; source beam 100 (reranking reference beam 300)')
            require(abs(result['recall_at_10']-case['source_grid']['expected']['recall@10']) < 1e-12, 'Source Recall differs')
        write(source_path, result)
    results['Original'] = read(source_path)
    prior_path = FD.parent/'fairdiverse_cases'/folder.name/'complete.json'
    prior = read(prior_path)
    for method,spec in case['trainers'].items():
        path = folder/(method+'.json')
        if not path.exists():
            print(f'  recovering {method}', flush=True)
            result = recorded_result(spec, prior['score_reproduction'][method], context, targets)
            result['prior_audit'] = reference(prior_path)
            write(path, result)
            print(f'  {method}: pop Gini {result["popularity_gini_at_10"]:.6f}; all three reproduction checks passed', flush=True)
        results[method] = read(path)
    require(set(results) == set(METHODS + ['Original','Saved reference']), 'Incomplete case')
    write(folder/'complete.json', dict(case=index, dataset=case['dataset'], tokenizer=case['tokenizer'],
          requests=context['requests'], results=results, frozen_activity_manifests=[reference(OLD),reference(FD)],
          context=reference(OUT/case['dataset']/'context.json'), code=reference(__file__)))
    print(f'Completed case {index}', flush=True)


def main():
    global OLD, FD, OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=int, required=True, choices=range(12))
    parser.add_argument('--source-manifest', type=Path, default=OLD)
    parser.add_argument('--fairdiverse-manifest', type=Path, default=FD)
    parser.add_argument('--output-dir', type=Path, default=OUT)
    args = parser.parse_args()
    OLD, FD, OUT = args.source_manifest.resolve(), args.fairdiverse_manifest.resolve(), args.output_dir.resolve()
    if OUT == OLD.parent or OUT == FD.parent:
        raise ValueError('Use a separate output directory for recovered popularity metrics')
    run_case(args.case)


if __name__ == '__main__':
    main()
