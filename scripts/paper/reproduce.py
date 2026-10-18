"""Regenerate the frozen historical paper's tables and plots from aggregate inputs."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

from scripts.experiment_labels import method_label

ROOT = Path(__file__).resolve().parents[2]
DATASETS = {'amazon-beauty': 'Beauty', 'amazon-sports': 'Sports',
            'amazon-toys': 'Toys', 'amazon-pet-supplies': 'Pet Supplies'}
TOKENS = {'rkmeans': 'RKMeans', 'rvq': 'RVQ', 'rqvae': 'RQ-VAE'}


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def verify_inputs(inputs):
    manifest = json.loads((ROOT/'paper/source_manifest.json').read_text())
    for entry in manifest['inputs']:
        relative = Path(entry['path']).relative_to('paper/inputs')
        if hashlib.sha256((inputs/relative).read_bytes()).hexdigest() != entry['sha256']:
            raise ValueError(f'Frozen paper input changed: {relative}')
    rows = read_csv(inputs/'comparison_results.csv')
    keys = {(r['dataset'], r['tokenizer'], r['method'], r['k']) for r in rows}
    if len(rows) != len(keys) or len(rows) != 564:
        raise ValueError('Expected 564 distinct complete comparison rows')
    if any(r['recall'] == '' or r['ndcg'] == '' for r in rows):
        raise ValueError('Incomplete comparison metrics')
    return rows


def render_comparison(rows, output):
    tex = [r'% Historical cohort, seed 42. Brand entropy is in bits.']
    for dataset, title in DATASETS.items():
        tex.extend([r'\begin{table*}[p]', r'\centering\small',
            r'\caption{'+title+r': Recall, NDCG, brand Gini and entropy at $K=5/10/20$. '
            r'Training rows retain their recorded evaluation protocols; rerankers use matched saved candidates.}',
            r'\label{tab:comparison-'+dataset.removeprefix('amazon-')+'}',
            r'\begin{tabular}{llcccc}', r'\toprule',
            r'Model / variant & Tokenizer & Recall & NDCG & Brand Gini & Brand entropy \\', r'\midrule'])
        selected = [r for r in rows if r['dataset'] == dataset]
        cases = list(dict.fromkeys((r['tokenizer'], r['method']) for r in selected))
        for tokenizer, method in cases:
            matched = sorted((r for r in selected if (r['tokenizer'], r['method']) == (tokenizer, method)), key=lambda r: int(r['k']))
            assert [int(r['k']) for r in matched] == [5, 10, 20]
            values = [' / '.join(f'{float(r[key]):.4f}' for r in matched)
                      for key in ['recall', 'ndcg', 'brand_gini', 'brand_entropy_bits']]
            tex.append(' & '.join([method_label(method), TOKENS.get(tokenizer, '--'), *values]) + r' \\')
        tex.extend([r'\bottomrule', r'\end{tabular}', r'\end{table*}'])
    output.write_text('\n'.join(tex)+'\n')


def render_classics(rows, output):
    lookup = {(r['dataset'], r['tokenizer'], r['method']): r for r in rows if r['k'] == '10'}
    tex = [r'\begin{table*}[t]', r'\centering\small',
           r'\caption{Classic and original GRID results at $K=10$. GRID shows the best Recall/NDCG and brand Gini range across tokenizers.}',
           r'\label{tab:classics}', r'\begin{tabular}{lccccccccc}', r'\toprule',
           r'Dataset & SASRec R & N & Gini & GRU4Rec R & N & Gini & GRID R & N & Gini range \\', r'\midrule']
    for dataset, label in DATASETS.items():
        values = []
        for method in ['SASRec', 'GRU4Rec']:
            row = lookup[dataset, '', method]
            values += [f'{float(row[key]):.4f}' for key in ['recall', 'ndcg', 'brand_gini']]
        originals = [lookup[dataset, tok, 'GRID: original (fresh evaluation)'] for tok in TOKENS]
        values += [f'{max(float(r[key]) for r in originals):.4f}' for key in ['recall', 'ndcg']]
        ginis = [float(r['brand_gini']) for r in originals]
        values.append(f'{min(ginis):.4f}--{max(ginis):.4f}')
        tex.append(' & '.join([label, *values])+r' \\')
    tex += [r'\bottomrule', r'\end{tabular}', r'\end{table*}']
    output.write_text('\n'.join(tex)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, default=ROOT/'paper/inputs')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'artifacts/paper')
    args = parser.parse_args()
    inputs, output = args.inputs.resolve(), args.output_dir.resolve()
    rows = verify_inputs(inputs)
    if output.is_relative_to(inputs) or (output.exists() and any(output.iterdir())):
        raise FileExistsError('Choose a new output directory outside the frozen inputs')
    tables, figures = output/'tables', output/'figures'
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    comparison = tables/'comparison_tables.tex'
    render_comparison(rows, comparison)
    # Preserve the original rounded-input figure protocol, verifying that the
    # freshly rendered tables contain exactly the frozen scientific values.
    from scripts.paper.make_training_interventions import parse_tables
    assert parse_tables(comparison) == parse_tables(inputs/'comparison_tables.tex')
    render_classics(rows, tables/'classics.tex')
    from scripts import build_grid_baseline_comparison as baseline
    baseline.OUT, baseline.PAPER_CSV = tables, inputs/'published_grid_reference.csv'
    shutil.copyfile(inputs/'pet_published_reference.json', tables/'pet_published_reference.json')
    baseline.write_baseline_comparison([dict(r, source='') for r in rows], dataset_statistics_path=inputs/'dataset_statistics.json')

    commands = [
        ['scripts.paper.make_training_interventions', '--source', comparison, '--table-dir', tables, '--figure-dir', figures],
        ['scripts.paper.make_popularity_fairness', '--source-dir', inputs/'popularity', '--comparison-results', inputs/'comparison_results.csv', '--table-dir', tables, '--figure-dir', figures],
        ['scripts.paper.make_training_interventions_combined', '--training-changes', tables/'training_interventions_changes.csv', '--popularity-changes', tables/'popularity_fairness_changes.csv', '--comparison-tables', comparison, '--table-dir', tables, '--output', figures/'fig_training_interventions_combined'],
        ['scripts.paper.make_reranking_table', '--comparison-results', inputs/'comparison_results.csv', '--case-changes', inputs/'reranking/case_changes.csv', '--output-dir', tables],
        ['scripts.plot_activity_baseline_losses', '--source', inputs/'activity', '--output', figures/'fig_activity_baseline_losses'],
    ]
    for command in commands:
        subprocess.run([sys.executable, '-m', *map(str, command)], cwd=ROOT, check=True)
    lookup = {(r['dataset'], r['tokenizer'], r['method']): r for r in rows if r['k'] == '10'}
    summaries = {}
    for name, method in [('RW', 'GRID + IFairLRS RW (training)'), ('SPRINT', 'GRID + SPRINT (training)')]:
        changes = [100*(float(lookup[ds, tok, method]['recall'])/float(lookup[ds, tok, 'GRID: ordinary continuation']['recall'])-1)
                   for ds in DATASETS for tok in TOKENS]
        summaries[name] = dict(recall_change_percent_min=min(changes), recall_change_percent_max=max(changes))
    rr = [r for r in read_csv(inputs/'reranking/case_changes.csv') if r['method'] == 'RR' and r['tokenizer'] == 'rkmeans']
    summaries['RKMeans_RR_at_10'] = dict(mgu_reduction_percent_min=min(float(r['mgu_reduction_percent']) for r in rr),
                                      mgu_reduction_percent_max=max(float(r['mgu_reduction_percent']) for r in rr))
    (tables/'unrounded_manuscript_summaries.json').write_text(json.dumps(summaries, indent=2)+'\n')
    products = sorted(p for p in output.rglob('*') if p.is_file())
    (output/'SHA256SUMS').write_text(''.join(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(output)}\n' for p in products))
    print(f'Paper reproduction complete: {len(products)} outputs in {output}')


if __name__ == '__main__':
    main()
