"""Render signed relative changes from the completed reranking audit.

All five columns use 100 * (method / matched control - 1). The original audit
CSV retains its reduction convention for MGU/Gini; signs are converted here.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/paper/tables'
COMPARISON = ROOT / 'paper/inputs/comparison_results.csv'


def main():
    global OUT, COMPARISON
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison-results', type=Path, default=COMPARISON)
    parser.add_argument('--case-changes', type=Path, default=ROOT/'paper/inputs/reranking/case_changes.csv')
    parser.add_argument('--output-dir', type=Path, default=OUT)
    args = parser.parse_args()
    OUT, COMPARISON = args.output_dir, args.comparison_results
    OUT.mkdir(parents=True, exist_ok=True)
    source = list(csv.DictReader(COMPARISON.open()))
    lookup = {(r['dataset'], r['tokenizer'], r['method']): r for r in source if r['k'] == '10'}
    changes = []
    for r in csv.DictReader(args.case_changes.open()):
        ds, tok, method = r['dataset'], r['tokenizer'], r['method']
        control = lookup[ds, tok, 'GRID: saved reranking reference']
        row = lookup[ds, tok, 'GRID + ' + method]
        assert row['status'] == control['status'] == 'verified'
        assert abs(float(r['recall_change_percent']) - 100*(float(row['recall'])/float(control['recall'])-1)) < 1e-9
        assert abs(float(r['gini_reduction_percent']) - 100*(1-float(row['brand_gini'])/float(control['brand_gini']))) < 1e-9
        assert abs(float(r['entropy_increase_percent']) - 100*(float(row['brand_entropy_bits'])/float(control['brand_entropy_bits'])-1)) < 1e-9
        changes.append(dict(dataset=ds, tokenizer=tok, method=method,
            recall_change_percent=float(r['recall_change_percent']),
            ndcg_change_percent=100*(float(row['ndcg'])/float(control['ndcg'])-1),
            mgu_change_percent=-float(r['mgu_reduction_percent']),
            gini_change_percent=-float(r['gini_reduction_percent']),
            entropy_change_percent=float(r['entropy_increase_percent'])))
    assert len(changes) == len({(r['dataset'], r['tokenizer'], r['method']) for r in changes}) == 36
    columns = list(changes[0])[3:]
    summary = {}
    for method in ('RR', 'CPFair', 'P-MMF'):
        group = [r for r in changes if r['method'] == method]
        assert len(group) == 12
        summary[method] = {col: dict(min=min(r[col] for r in group), max=max(r[col] for r in group)) for col in columns}
    with (OUT / 'signed_case_changes.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(changes[0]))
        writer.writeheader()
        writer.writerows(changes)
    (OUT / 'signed_summary.json').write_text(json.dumps(summary, indent=2) + '\n')

    def number(value):
        # Avoid signed zero. The caption states that entries are rounded.
        return f'{0.0 if round(value, 1) == 0 else value:.1f}'

    tex = [r'\begin{table}[!tb]', r'\centering\small',
           r'\caption{Reranking at $K=10$: minimum--maximum relative changes (\%) against matched untreated candidates across 12 dataset--tokenizer cases, computed as $100(M_{\mathrm{reranked}}/M_{\mathrm{control}}-1)$. Arrows indicate the preferred direction: positive changes improve Recall, NDCG, and brand entropy; negative changes improve popularity MGU and brand Gini. Values are rounded to one decimal place.}',
           r'\label{tab:rerank-summary}', r'\setlength{\tabcolsep}{3pt}',
           r'\begin{tabular}{@{}lccccc@{}}', r'\toprule',
           r'Method & Recall@10 $\uparrow$ & NDCG@10 $\uparrow$ & MGU $\downarrow$ & Gini $\downarrow$ & Entropy $\uparrow$\\',
           r'\midrule']
    for method, values in summary.items():
        cells = [f'${number(values[col]["min"])}$ to ${number(values[col]["max"])}$' for col in columns]
        tex.append(' & '.join([method] + cells) + r' \\')
    tex += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    (OUT / 'rerank_summary.tex').write_text('\n'.join(tex) + '\n')
    print('\n'.join(tex))


if __name__ == '__main__':
    main()
