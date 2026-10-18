#!/usr/bin/env python3
"""Build the K=10 companion from recovered frozen popularity evaluations.

Run after all twelve CPU recovery cases:
    python docs/make_popularity_fairness.py

Writes figures/fig_popularity_fairness.pdf, PNG, caption, a LaTeX table,
case-level metrics, changes, summaries, extrema, and provenance.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys

from scripts.paper import make_training_interventions as style
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT/'paper/inputs/popularity'
FIGURES = ROOT/'artifacts/paper/figures'
OUTPUT = ROOT/'artifacts/paper/tables'
COMPARISON = ROOT/'paper/inputs/comparison_results.csv'
FIELDS = ['popularity_gini_at_10', 'mgu_at_10', 'dgu_at_10']
LABELS = ['Popularity Gini@10', 'MGU@10', 'DGU@10']
TOKENIZERS = dict(rkmeans='RKMeans', rvq='RVQ', rqvae='RQ-VAE')


def read(path):
    return json.loads(path.read_text())


def reference(path):
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def collect():
    changes, absolute, sources = [], [], []
    comparison_path = COMPARISON
    with comparison_path.open() as stream:
        lookup = {(r['dataset'],r['tokenizer'],r['display_method']):r
                  for r in csv.DictReader(stream) if r['k'] == '10'}
    sources.append(reference(comparison_path))
    for index in range(12):
        paths = list(SOURCE.glob(f'case_{index:02d}_*/complete.json'))
        if len(paths) != 1:
            raise ValueError(f'Case {index} is incomplete; run CPU recovery first')
        path = paths[0]
        complete = read(path)
        sources.append(reference(path))
        results = complete['results']
        for name,result in results.items():
            label = 'IFairLRS RW' if name == 'RW' else name
            variant = (style.ORIGINAL if name == 'Original' else style.RERANKING_REFERENCE
                       if name == 'Saved reference' else style.METHODS[label])
            reported = lookup[complete['dataset'],complete['tokenizer'],variant]
            for field,column in [('recall_at_10','recall'),('ndcg_at_10','ndcg'),
                                 ('brand_gini_at_10','brand_gini')]:
                if field in result:
                    actual = result[field]
                    expected = float(reported[column])
                    if label in style.FAIRDIVERSE:
                        actual = round(actual,4)
                    if abs(actual-expected) > 1e-11:
                        raise ValueError(f'Current comparison differs: {complete["case"]}/{label}/{field}')
            bounds = {'popularity_gini_at_10':.8,'mgu_at_10':.4,'dgu_at_10':2}
            for field,bound in bounds.items():
                if not np.isfinite(result[field]) or not -1e-12 <= result[field] <= bound+1e-12:
                    raise ValueError(f'Invalid metric: {label}/{field}')
            row = dict(dataset=complete['dataset'].removeprefix('amazon-'),
                       tokenizer=TOKENIZERS[complete['tokenizer']],
                       method=label,
                       requests=complete['requests'])
            for field in FIELDS:
                row[field] = result[field]
            row.update({f'popularity_share_bin_{g}':result['popularity_shares'][g] for g in range(5)})
            row.update({f'history_share_bin_{g}':result['history_shares'][g] for g in range(5)})
            row['source_cache'] = str(path.relative_to(SOURCE))
            absolute.append(row)
            if name in ['Original','Saved reference']:
                continue
            reference_name = 'Saved reference' if name in style.RERANKERS else 'Original'
            ref = results[reference_name]
            row = dict(row, reference=reference_name)
            for field in FIELDS:
                row[f'reference_{field}'] = ref[field]
                row[f'delta_{field}'] = result[field]-ref[field]
            changes.append(row)
    # Same case order and jitter assignment as the effectiveness/brand figure.
    order = {method:i for i,method in enumerate(style.METHODS)}
    dataset_order = {ds:i for i,ds in enumerate(style.DATASETS)}
    changes.sort(key=lambda r:(order[r['method']], dataset_order[r['dataset']], style.TOKENIZERS.index(r['tokenizer'])))
    if len(changes) != 156 or len(absolute) != 180:
        raise ValueError('Expected 156 changes and 180 absolute evaluations')
    summaries = []
    for method in style.METHODS:
        rows = [r for r in changes if r['method'] == method]
        if len(rows) != 12:
            raise ValueError(f'Incomplete method: {method}')
        summary = dict(method=method, cases=12)
        for field in FIELDS:
            values = [r['delta_'+field] for r in rows]
            summary.update({f'{field}_median':statistics.median(values),
                            f'{field}_min':min(values), f'{field}_max':max(values),
                            f'{field}_improved_cases':sum(v < -1e-12 for v in values),
                            f'{field}_worsened_cases':sum(v > 1e-12 for v in values)})
        summaries.append(summary)
    return changes,absolute,summaries,sources


def draw(changes):
    plt.rcParams.update({
        'font.family':'serif', 'font.serif':['Nimbus Roman','Times New Roman','DejaVu Serif'],
        'mathtext.fontset':'stix', 'font.size':7.5, 'axes.labelsize':7,
        'xtick.labelsize':6.5, 'ytick.labelsize':7, 'legend.fontsize':6.5,
        'axes.spines.top':False, 'axes.spines.right':False, 'axes.linewidth':.6,
        'pdf.fonttype':42,
    })
    fig = plt.figure(figsize=(4.8, 7.4))
    layout = fig.add_gridspec(3,3,width_ratios=(1.9,1,1),left=.17,right=.99,
                             bottom=.07,top=.875,wspace=.08,hspace=.43)
    offsets = np.random.default_rng(42).uniform(-.12,.12,12)
    labels, titles = [], []
    for panel,(field,title) in enumerate(zip(FIELDS,LABELS)):
        ax = fig.add_subplot(layout[panel,0])
        case_axes = [fig.add_subplot(layout[panel,col]) for col in (1,2)]
        delta = 'delta_'+field
        for i,method in enumerate(style.METHODS):
            values = [r[delta] for r in changes if r['method'] == method]
            color = style.TEAL if method in style.RERANKERS else style.BLUE if method in style.CONTINUATION else style.ORANGE
            ax.scatter(values,i+offsets,s=12,color=color,alpha=.85,edgecolor='white',linewidth=.3,zorder=3)
            ax.plot([np.median(values)]*2,[i-.32,i+.32],color='#0b0b0b',linewidth=1.2,zorder=4)
        ax.axvline(0,color='#52514e',linewidth=.6)
        for boundary in (2.5,9.5):
            ax.axhline(boundary,color='#d5d4d1',linewidth=.5)
        ax.set_ylim(12.5,-.5)
        ax.set_yticks(range(13))
        ax.set_yticklabels([m+(r'$^{\dagger}$' if m in style.FAIRDIVERSE else '') for m in style.METHODS])
        titles.append(ax.set_title(title+' change',loc='left',fontsize=8,fontweight='bold',pad=7))
        ax.set_xlabel('Absolute change (negative = better)')
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.grid(axis='x',color='#ecebe8',linewidth=.5)
        ax.set_axisbelow(True)
        ax.margins(x=.10)
        labels.extend(style.draw_case_columns(ax,case_axes,changes,delta,offsets))
    handles = [
        Line2D([],[],marker='o',ls='',color=style.BLUE,ms=3.5,label='Fine-tuning (32-update cap)\nResumed Adam state'),
        Line2D([],[],marker='o',ls='',color=style.ORANGE,ms=3.5,label='FairDiverse trainer$^{\\dagger}$\nFresh Adam optimizer'),
        Line2D([],[],marker='o',ls='',color=style.TEAL,ms=3.5,label='Reranking'),
        Line2D([],[],marker='|',ls='',color='#0b0b0b',ms=7,markeredgewidth=1.2,label='Median of 12 cases'),
        Line2D([],[],marker='D',ls='',color='#262626',ms=4,markerfacecolor='none',markeredgewidth=.55,label='Extreme cases'),
        Line2D([],[],color='#52514e',lw=.6,label="No change (each case's reference)"),
    ]
    legend = fig.legend(handles=handles,loc='upper center',ncol=2,frameon=False,bbox_to_anchor=(.5,1),columnspacing=1,handletextpad=.3)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = []
    for ax,text in labels:
        box = text.get_window_extent(renderer)
        column = ax.get_window_extent(renderer)
        if box.x0 < column.x0-1 or box.x1 > column.x1+1:
            raise ValueError(f'Case label exceeds its column: {text.get_text()}')
        boxes.append(box)
    if any(a.overlaps(b) for i,a in enumerate(boxes) for b in boxes[i+1:]):
        raise ValueError('Extreme-case labels overlap')
    if any(legend.get_window_extent(renderer).overlaps(title.get_window_extent(renderer)) for title in titles):
        raise ValueError('Legend overlaps a metric title')
    fig.savefig(FIGURES/'fig_popularity_fairness.pdf',metadata={'Title':'GRID popularity exposure equality and calibration'})
    fig.savefig(FIGURES/'fig_popularity_fairness.png',dpi=240)
    plt.close(fig)


def write_table(summaries):
    lines = [r'% Generated by docs/make_popularity_fairness.py; requires booktabs.',
        r'\begin{table}[!tb]',r'\centering\scriptsize',
        r'\caption{Popularity fairness at $K=10$. Entries are the median [minimum, maximum]',
        r'absolute change over 12 dataset--tokenizer cases, each relative to its own',
        r'original GRID checkpoint (training) or untreated saved-candidate ordering (reranking).',
        r'Negative changes mean lower popularity exposure inequality ($G_{\rm pop}$)',
        r'or lower mismatch to pooled test-history popularity shares (MGU, DGU).',
        r'Ranges describe cases, not uncertainty. Five training-frequency bins are frozen;',
        r'histories contain at most the last 40 items per request. Entries are rounded',
        r'to four decimals; seed 42.',
        r'$\dagger$: recorded FairDiverse protocols, with fresh Adam optimizers and',
        r'different schedules, are retained; native fine-tuning resumes source Adam',
        r'with a 32-update cap and update zero eligible. These descriptive comparisons',
        r'do not isolate causal method effects.}',r'\label{tab:popularity-fairness}',
        r'\setlength{\tabcolsep}{1.15pt}',r'\begin{tabular}{@{}lccc@{}}',r'\toprule',
        r'Method & $\Delta G_{\rm pop}@10$ & $\Delta\mathrm{MGU}@10$ & $\Delta\mathrm{DGU}@10$ \\',r'\midrule']
    for index,row in enumerate(summaries):
        if index in (3,10):
            lines.append(r'\midrule')
        name = row['method'] + (r'$^{\dagger}$' if row['method'] in style.FAIRDIVERSE else '')
        cells = []
        for field in FIELDS:
            cells.append('{} [{}, {}]'.format(*(style.signed(row[field+'_'+stat],4) for stat in ['median','min','max'])))
        lines.append(' & '.join([name]+cells)+r' \\')
    lines.extend([r'\bottomrule',r'\end{tabular}',r'\end{table}',''])
    (OUTPUT/'popularity_fairness_table.tex').write_text('\n'.join(lines))


def draw_compact(changes):
    """122-mm paper version; the taller main figure retains endpoint names."""
    fig,axes = plt.subplots(1,3,figsize=(4.8,3.4),sharey=True)
    fig.subplots_adjust(left=.17,right=.985,bottom=.19,top=.76,wspace=.16)
    offsets = np.random.default_rng(42).uniform(-.12,.12,12)
    for ax,field,title in zip(axes,FIELDS,LABELS):
        delta = 'delta_'+field
        for i,method in enumerate(style.METHODS):
            rows = [r for r in changes if r['method'] == method]
            values = np.array([r[delta] for r in rows])
            color = style.TEAL if method in style.RERANKERS else style.BLUE if method in style.CONTINUATION else style.ORANGE
            ax.scatter(values,i+offsets,s=10,color=color,alpha=.85,edgecolor='white',linewidth=.25,zorder=3)
            ax.plot([np.median(values)]*2,[i-.32,i+.32],color='#0b0b0b',linewidth=1.1,zorder=4)
            for _,value,ties in style.extreme_cases(rows,delta):
                indices = [rows.index(row) for row in ties]
                ax.scatter([value]*len(indices),i+offsets[indices],marker='D',s=20,
                           facecolors='none',edgecolors='#262626',linewidths=.5,zorder=5)
        ax.axvline(0,color='#52514e',linewidth=.6)
        for boundary in (2.5,9.5):
            ax.axhline(boundary,color='#d5d4d1',linewidth=.5)
        extent = max(abs(r[delta]) for r in changes)
        unit = .1 if field != 'mgu_at_10' else .05
        bound = np.ceil(extent/unit)*unit
        ax.set_xlim(-bound*1.08,bound*1.08)
        ax.set_xticks([-bound,0,bound])
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.set_ylim(12.5,-.5)
        ax.set_yticks(range(13))
        ax.set_title(title,fontsize=7.5,fontweight='bold',pad=7)
        ax.set_xlabel('Absolute change',fontsize=7)
        ax.grid(axis='x',color='#ecebe8',linewidth=.5)
        ax.set_axisbelow(True)
    axes[0].set_yticklabels([m+(r'$^{\dagger}$' if m in style.FAIRDIVERSE else '') for m in style.METHODS])
    handles = [
        Line2D([],[],marker='o',ls='',color=style.BLUE,ms=3.5,label='Fine-tuning (32-update cap)\nResumed Adam state'),
        Line2D([],[],marker='o',ls='',color=style.ORANGE,ms=3.5,label='FairDiverse trainer$^{\\dagger}$\nFresh Adam optimizer'),
        Line2D([],[],marker='o',ls='',color=style.TEAL,ms=3.5,label='Reranking'),
        Line2D([],[],marker='|',ls='',color='#0b0b0b',ms=7,markeredgewidth=1.1,label='Median of 12 cases'),
        Line2D([],[],marker='D',ls='',color='#262626',ms=4,markerfacecolor='none',markeredgewidth=.5,label='Extreme cases'),
        Line2D([],[],color='#52514e',lw=.6,label="No change (each case's reference)"),
    ]
    legend = fig.legend(handles=handles,loc='upper center',ncol=2,frameon=False,
                        bbox_to_anchor=(.5,1.01),columnspacing=1,handletextpad=.3)
    fig.text(.57,.025,'Negative = more equal exposure / better history alignment',
             ha='center',fontsize=6.5)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    if any(legend.get_window_extent(renderer).overlaps(ax.title.get_window_extent(renderer)) for ax in axes):
        raise ValueError('Compact legend overlaps a panel title')
    fig.savefig(FIGURES/'fig_popularity_fairness_compact.pdf',metadata={'Title':'GRID popularity fairness at K=10 (compact)'})
    fig.savefig(FIGURES/'fig_popularity_fairness_compact.png',dpi=240)
    plt.close(fig)


def main():
    global SOURCE, FIGURES, OUTPUT, COMPARISON
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, default=SOURCE)
    parser.add_argument('--comparison-results', type=Path, default=COMPARISON)
    parser.add_argument('--figure-dir', type=Path, default=FIGURES)
    parser.add_argument('--table-dir', type=Path, default=OUTPUT)
    parser.add_argument('--latex-preview', action='store_true')
    args = parser.parse_args()
    SOURCE, FIGURES, OUTPUT, COMPARISON = args.source_dir, args.figure_dir, args.table_dir, args.comparison_results
    FIGURES.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    changes,absolute,summaries,sources = collect()
    for filename,rows in [('popularity_fairness_changes.csv',changes),('popularity_fairness_metrics.csv',absolute),
                          ('popularity_fairness_summary.csv',summaries)]:
        style.write_csv(OUTPUT/filename,rows)
    extrema = []
    for method in style.METHODS:
        rows = [r for r in changes if r['method'] == method]
        for field in FIELDS:
            for side,value,ties in style.extreme_cases(rows,'delta_'+field):
                for row in ties:
                    extrema.append(dict(method=method,metric=field,extremum=side,value=value,
                                        dataset=row['dataset'],tokenizer=row['tokenizer'],tied_cases=len(ties)))
    style.write_csv(OUTPUT/'popularity_fairness_extrema.csv',extrema)
    draw(changes)
    draw_compact(changes)
    write_table(summaries)
    caption = r'''\caption{Changes in GRID popularity fairness at $K=10$.
Top: popularity-group Gini measures unequal exposure across five catalog bins
defined by training interaction frequency. Middle and bottom: MGU and DGU
measure mismatch between recommendation shares and pooled test-history shares,
using at most the last 40 history items per request.
For $GU_g=GP_g-GH_g$, $\mathrm{MGU}=\frac{1}{5}\sum_g|GU_g|$ and
$\mathrm{DGU}=\max_g GU_g-\min_g GU_g$. Exposure counts each recommendation once,
without rank discounting; zero-exposure bins remain included.
Every dot is one of 12 cases (four datasets $\times$ three tokenizers),
and every change is $M_m-M_0$: negative means less inequality or better calibration.
Each training case uses its own original GRID reference; rerankers use the
untreated ordering of their exact saved candidate pool. The vertical zero line
means no change. Short black bars show medians; outlined diamonds mark the
minimum and maximum changes, identified in the adjacent columns (all ties listed).
Pet denotes Pet Supplies. Vertical jitter separates dots only.
Blue: fine-tuning resumes source Adam, with a 32-update cap and update zero eligible.
Orange ($\dagger$): FairDiverse training uses fresh Adam and different schedules;
metrics are recovered from saved scores under the recorded historical full-catalog
ranking rules, rather than a newly harmonized evaluation.
Teal: reranking changes saved orderings without model updates.
Sports--RVQ source evaluation uses beam 100; its reranking pool and reference use 300.
Metrics use unrounded saved exposures or reconstructed rankings; seed 42.
Ranges are descriptive, not confidence intervals; these comparisons do not isolate
causal method effects. Popularity exposure equality and history calibration are
distinct objectives, and neither implies brand exposure equality.}
\label{fig:popularity-fairness}
'''
    (FIGURES/'fig_popularity_fairness_caption.tex').write_text(caption)
    (FIGURES/'fig_popularity_fairness_compact_caption.tex').write_text(r'''\caption{Changes in popularity fairness at $K=10$.
Each row contains 12 cases (four datasets $\times$ three tokenizers).
Each dot is an absolute change relative to that case's original GRID checkpoint
(training) or untreated saved-candidate ordering (reranking); negative is better.
Popularity Gini measures unequal exposure across five frozen training-frequency
bins. MGU and DGU measure mismatch between pooled recommendation and test-history
shares, using each request's last at most 40 history items:
for $GU_g=GP_g-GH_g$, $\mathrm{MGU}=\frac{1}{5}\sum_g|GU_g|$ and
$\mathrm{DGU}=\max_gGU_g-\min_gGU_g$.
Black bars show medians; diamonds mark extrema; jitter only separates points.
Panels use different numerical scales. Blue methods resume source Adam with a
32-update cap; orange methods ($\dagger$) use fresh Adam and different schedules,
retaining recorded historical ranking protocols; teal methods rerank without
model updates. Sports--RVQ uses source beam 100 and reranking beam 300.
Results use unrounded recovered metrics, seed 42; comparisons are descriptive.}
\label{fig:popularity-fairness-compact}
''')
    provenance = dict(cases=12,methods=13,paired_changes=156,cutoff=10,seed=42,
        source_cases=sources,code=[reference(Path(__file__)),reference(ROOT/'additional_baselines/popularity_fairness_analysis.py')],
        reference_rules='Original fresh GRID for all training methods; untreated exact saved pool for rerankers; paired within dataset/tokenizer',
        grouping='Five approximately equal catalog-size bins sorted by training frequency then item ID; verified against frozen candidate maps',
        calibration='Pooled last-40-item test-request histories; GP-GH, MGU=mean(abs(GU)), DGU=max(GU)-min(GU)',
        weighting='Unweighted top-10 item exposure; method summaries weight each of the 12 cases equally',
        precision='Unrounded saved shares/counts and reconstructed rankings; differs from rounded TeX inputs used in the effectiveness/brand figure',
        inference='Single seed; descriptive ranges, no seed confidence intervals or causal isolation; historical FairDiverse ranking conventions retained',
        outputs=[str(p) for p in [FIGURES/'fig_popularity_fairness.pdf',FIGURES/'fig_popularity_fairness.png',FIGURES/'fig_popularity_fairness_caption.tex',FIGURES/'fig_popularity_fairness_compact.pdf',FIGURES/'fig_popularity_fairness_compact.png',FIGURES/'fig_popularity_fairness_compact_caption.tex']])
    (OUTPUT/'popularity_fairness_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    if args.latex_preview:
        style.render_table(OUTPUT/'popularity_fairness_table.tex',FIGURES/'popularity_fairness_table_preview.pdf')
    print('Wrote',FIGURES/'fig_popularity_fairness.pdf')
    for row in summaries:
        print(row['method'],*[round(row[f+'_median'],5) for f in FIELDS])


if __name__ == '__main__':
    main()
