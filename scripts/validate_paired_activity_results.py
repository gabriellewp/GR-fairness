"""Independent checks of the combined activity analysis and one full user case.

Uses the Python standard library, without importing the analysis implementation.
Run after the all-method merger finishes; writes validation.json in its output.
"""
import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
from statistics import median


def check(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, message):
    check(math.isclose(float(actual), float(expected), rel_tol=1e-9, abs_tol=1e-11), message)


def read_csv(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


def verify(ref):
    value = hashlib.sha256()
    with Path(ref['path']).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    check(value.hexdigest() == ref['sha256'], f"Changed artifact: {ref['path']}")


def validate(root):
    summary = json.loads((root / 'analysis_summary.json').read_text())
    manifest = json.loads((root / 'manifest.json').read_text())
    verify(summary['manifest'])
    check(summary['status'] == 'complete' and summary['comparisons'] == 144, 'Incomplete analysis')
    effects = read_csv(root / 'effects.csv')
    contrasts = read_csv(root / 'activity_contrasts.csv')
    check(len(effects) == 1152 and len(contrasts) == 864, 'Unexpected hypothesis coverage')
    controls = {m: 'Source GRID' for m in manifest['fairdiverse_methods']}
    controls.update(RW='Standard FT', SPRINT='Standard FT', RR='Untreated', CPFair='Untreated')
    controls['P-MMF'] = 'Untreated'
    lookup = {}
    families = defaultdict(list)
    for row in effects:
        check(row['control'] == controls[row['method']], 'Incorrect reference')
        key = tuple(row[k] for k in ['dataset', 'tokenizer', 'method', 'metric', 'activity_group'])
        check(key not in lookup, 'Duplicate effect')
        lookup[key] = row
        close(float(row['intervention_mean']) - float(row['control_mean']), row['mean_change'], 'Effect sign or subtraction')
        close(100 * float(row['mean_change']), row['change_percentage_points'], 'Effect units')
        check(float(row['ci_low']) <= float(row['ci_high']), 'Reversed interval')
        families[row['holm_family']].append(row)
    expected_keys = {(case['dataset'], case['tokenizer'], method, metric, group)
                     for case in manifest['cases'] for method in controls
                     for metric in ['recall@10', 'ndcg@10']
                     for group in ['overall', 'low', 'medium', 'high']}
    check(set(lookup) == expected_keys, 'Missing dataset/tokenizer/method/group/metric')
    for case in manifest['cases']:
        folder = root / f"case_{case['index']:02d}_{case['dataset']}_{case['tokenizer']}"
        record = json.loads((folder / 'complete.json').read_text())
        check(record['manifest'] == summary['manifest'], 'Completed case belongs to another analysis')
        for field in ['effects', 'contrasts']:
            verify(record[field])
    for key, row in lookup.items():
        if key[-1] != 'overall':
            continue
        group_rows = [lookup[key[:-1] + (g,)] for g in ['low', 'medium', 'high']]
        n = sum(int(r['users']) for r in group_rows)
        check(n == int(row['users']), 'Missing users in activity groups')
        for field in ['control_mean', 'intervention_mean', 'mean_change']:
            close(sum(float(r[field]) * int(r['users']) for r in group_rows) / n, row[field], 'Group/overall reconciliation')
    contrast_keys = set()
    for row in contrasts:
        key = tuple(row[k] for k in ['dataset', 'tokenizer', 'method', 'metric'])
        contrast_key = key + (row['group_a'], row['group_b'])
        check(contrast_key not in contrast_keys, 'Duplicate group contrast')
        contrast_keys.add(contrast_key)
        a, b = (lookup[key + (row[g],)] for g in ['group_a', 'group_b'])
        close(float(a['mean_change']) - float(b['mean_change']), row['contrast'], 'Direct group contrast')
        close(100 * float(row['contrast']), row['contrast_percentage_points'], 'Contrast units')
        families[row['holm_family']].append(row)
    check(len(families) == 9, 'Incorrect multiple-testing families')
    for name, rows in families.items():
        ordered = sorted(rows, key=lambda r: float(r['p_raw']))
        previous = 0.0
        for i, row in enumerate(ordered):
            previous = max(previous, min(1.0, (len(rows) - i) * float(row['p_raw'])))
            close(previous, row['p_holm'], f'Holm correction: {name}')
            check((row['reject_holm_0_05'] == 'True') == (previous < .05), 'Incorrect significance flag')
    consistency = read_csv(root / 'consistency.csv')
    check(len(consistency) == 768, 'Incomplete consistency summary')
    for row in consistency:
        selected = [r for r in effects if all(r[k] == row[k] for k in ['method', 'metric', 'activity_group'])]
        if row['dimension'] != 'all_cases':
            column = 'dataset' if row['dimension'] == 'within_dataset' else 'tokenizer'
            selected = [r for r in selected if r[column] == row['value']]
        check(len(selected) == int(row['cases']) == int(row['expected_cases']), 'Incomplete consistency coverage')
        check(sum(float(r['mean_change']) > 0 for r in selected) == int(row['positive_cases']), 'Consistency signs')
        check(sum(float(r['mean_change']) < 0 for r in selected) == int(row['negative_cases']), 'Negative consistency signs')
        check(sum(float(r['mean_change']) == 0 for r in selected) == int(row['unchanged_cases']), 'Unchanged consistency signs')
        for sign, predicate in [('positive', lambda v: v > 0), ('negative', lambda v: v < 0)]:
            count = sum(predicate(float(r['mean_change'])) and r['reject_holm_0_05'] == 'True' for r in selected)
            check(count == int(row[f'holm_{sign}_cases']), 'Corrected-significance consistency counts')
        close(100 * median(float(r['mean_change']) for r in selected), row['median_change_percentage_points'], 'Consistency median')
    # Independently reconstruct every exported user metric in Toys / RQ-VAE.
    # This checks 12 interventions, their references, and all three activity groups.
    folder = root / 'case_08_amazon-toys_rqvae'
    complete = json.loads((folder / 'complete.json').read_text())
    case = manifest['cases'][8]
    verify(case['split'])
    verify(case['activity'])
    users = {r['user_id'] for r in read_csv(case['activity']['path'])}
    activity = {r['user_id']: r['activity_group'] for r in read_csv(case['activity']['path'])}
    totals, counts, seen = defaultdict(float), Counter(), defaultdict(set)
    for ref in complete['paired_users_files']:
        verify(ref)
        with gzip.open(ref['path'], 'rt') as stream:
            for row in csv.DictReader(stream):
                method, user, group = row['method'], row['user_id'], row['activity_group']
                check(user in users and user not in seen[method], 'Duplicate or unexpected test user')
                check(activity[user] == group, 'Activity assignment changed')
                check(row['control'] == controls[method], 'User reference differs')
                seen[method].add(user)
                counts[method, group] += 1
                for metric in ['recall@10', 'ndcg@10']:
                    values = {}
                    for variant in ['control', 'method']:
                        rank = int(row[f'{variant}_target_rank_at_10'])
                        check(0 <= rank <= 10, 'Invalid target rank')
                        value = float(rank > 0) if metric == 'recall@10' else 1 / math.log2(rank + 1) if rank else 0.0
                        close(value, row[f'{variant}_{metric}'], 'User metric differs from target rank')
                        values[variant] = value
                    delta = values['method'] - values['control']
                    close(delta, row[f'delta_{metric}'], 'Incorrect user delta')
                    totals[method, group, metric] += delta
    check(set(seen) == set(controls) and all(s == users for s in seen.values()), 'Incomplete user comparisons')
    for (method, group, metric), total in totals.items():
        row = lookup['amazon-toys', 'rqvae', method, metric, group]
        close(total / counts[method, group], row['mean_change'], 'User/group effect reconciliation')
    result = dict(status='passed', comparisons=144, effect_rows=len(effects), contrast_rows=len(contrasts),
                  holm_family_sizes={k: len(v) for k, v in families.items()},
                  checked=['Reference rules', 'Group/overall weighted means', 'Direct group contrasts',
                           'Nine Holm families and significance flags', 'Cross-case sign counts and coverage'],
                  independent_user_case='amazon-toys/rqvae', user_case_methods=len(seen),
                  user_case_users_per_method=len(users), user_case_rows=sum(counts.values()),
                  limitations='User-row recomputation covers one of 12 cases; raw score reproduction is recorded by the workers, not repeated here.')
    result['input_sha256'] = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                              for name in ['manifest.json', 'effects.csv', 'activity_contrasts.csv', 'consistency.csv']}
    result['validator'] = dict(path=str(Path(__file__).resolve()),
                               sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (root / 'validation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output_dir', type=Path)
    validate(parser.parse_args().output_dir.resolve())
