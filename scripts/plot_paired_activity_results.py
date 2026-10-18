"""Render publication figures from the verified all-method activity tables.

Keeps frozen analysis scripts and original figures intact. Writes *_readable
PDF/PNG files, with a shared units label instead of overlapping row labels.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


FAMILIES = {
    'fine_tuning': (['RW', 'SPRINT'], 'RW and SPRINT vs standard fine-tuning', ''),
    'fairdiverse_training': (['IPS', 'FOCF', 'Reg', 'FairDual', 'Minmax-SGD', 'APR', 'SDRO'],
                            'FairDiverse training vs original GRID',
                            'Recorded training procedures and evaluation protocols'),
    'reranking': (['RR', 'CPFair', 'P-MMF'], 'Rerankers vs their untreated candidate ordering', ''),
}
DATASETS = ['amazon-beauty', 'amazon-sports', 'amazon-toys', 'amazon-pet-supplies']
TOKENS = [('rkmeans', 'RKMeans', '#2878b5', 'o'),
          ('rvq', 'RVQ', '#d95823', 's'), ('rqvae', 'RQ-VAE', '#6c4c9a', '^')]


def render(root):
    source = root / 'effects.csv'
    with source.open() as stream:
        rows = list(csv.DictReader(stream))
    lookup = {(r['method'], r['dataset'], r['tokenizer'], r['metric'], r['activity_group']): r
              for r in rows}
    if len(lookup) != len(rows) or len(rows) != 1152:
        raise ValueError('Expected 1,152 distinct effect rows')
    outputs = []
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.edgecolor': '#555555', 'axes.labelcolor': '#222222',
                         'pdf.fonttype': 42, 'ps.fonttype': 42}):
        for family, (methods, title, qualifier) in FAMILIES.items():
            for metric in ['recall@10', 'ndcg@10']:
                height = 2.0 * len(methods) + 1.8
                fig, axes = plt.subplots(len(methods), 4, figsize=(13.5, height),
                                         sharex=True, sharey='row', squeeze=False)
                fig.subplots_adjust(left=.09, right=.99, bottom=.95/height,
                                    top=1-1.15/height, hspace=.20, wspace=.10)
                label = 'Recall@10' if metric == 'recall@10' else 'NDCG@10'
                fig.suptitle(title + (f'\n{qualifier}' if qualifier else ''),
                             y=1-.40/height, fontsize=12)
                fig.supylabel(f'Change in {label} (percentage points)', x=.012, fontsize=11)
                for mi, method in enumerate(methods):
                    for di, dataset in enumerate(DATASETS):
                        ax = axes[mi, di]
                        for ti, (tokenizer, _, color, marker) in enumerate(TOKENS):
                            for gi, group in enumerate(['low', 'medium', 'high']):
                                row = lookup[method, dataset, tokenizer, metric, group]
                                x = gi + (ti-1)*.16
                                low, high = 100*float(row['ci_low']), 100*float(row['ci_high'])
                                ax.vlines(x, low, high, color=color, linewidth=1.2)
                                ax.plot(x, 100*float(row['mean_change']), marker,
                                        color=color, markersize=4)
                        ax.axhline(0, color='#555555', linewidth=.8)
                        ax.grid(axis='y', color='#dddddd', linewidth=.6)
                        ax.set_axisbelow(True)
                        ax.set_xticks([0, 1, 2], ['low', 'medium', 'high'])
                        ax.set_xlim(-.4, 2.4)
                        if di == 0:
                            ax.set_ylabel(method, fontsize=11, labelpad=10)
                        if mi == 0:
                            ax.set_title(['Beauty', 'Sports', 'Toys', 'Pet Supplies'][di], fontsize=11)
                handles = [Line2D([], [], color=c, marker=m, linestyle='', label=t)
                           for _, t, c, m in TOKENS]
                fig.legend(handles=handles, loc='upper center', ncol=3, frameon=False,
                           bbox_to_anchor=(.55, 1.0), borderaxespad=.15)
                fig.text(.54, .37/height,
                         'Pointwise 95% paired-user bootstrap intervals; fixed checkpoints and recommendations.\n'
                         'No training-seed uncertainty. Users per case: Beauty 22,363; Sports 35,598; '
                         'Toys 19,412; Pet Supplies 594,800.',
                         ha='center', va='center', fontsize=8)
                stem = root / f"activity_{family}_{metric.replace('@', '_at_')}_readable"
                for extension in ['pdf', 'png']:
                    path = stem.with_suffix('.' + extension)
                    fig.savefig(path, dpi=160)
                    outputs.append(path.name)
                plt.close(fig)
    provenance = dict(source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                      script=str(Path(__file__).resolve()),
                      script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      outputs=outputs, estimate='Intervention minus its declared reference, in percentage points',
                      uncertainty='Pointwise 95% paired-user bootstrap intervals; not multiplicity-adjusted')
    (root / 'readable_figures.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(f'Created {len(outputs)} readable figure files in {root}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output_dir', type=Path)
    render(parser.parse_args().output_dir.resolve())
