#!/usr/bin/env python3
"""Combine the existing effectiveness/brand and popularity intervention figures.

Run: python docs/make_training_interventions_combined.py

Preserves the original plotted values: Recall/brand use rounded table inputs;
popularity metrics use unrounded recovered evaluations. No evaluations rerun.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from scripts.paper import make_training_interventions as style
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, FuncFormatter
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TRAINING = ROOT/'artifacts/paper/tables/training_interventions_changes.csv'
POPULARITY = ROOT/'artifacts/paper/tables/popularity_fairness_changes.csv'
OUT = ROOT/'artifacts/paper/figures/fig_training_interventions_combined'
DATA = ROOT/'artifacts/paper/tables'
KEY = ('dataset','tokenizer','method')
PANELS = [
    ('recall_at_10_relative_change_percent', 'Recall@10', 'Relative change (%)\nPositive = better', [-80,-40,0,20]),
    ('brand_gini_at_10_absolute_change', 'Brand Gini@10', 'Absolute change\nNegative = better', [-.05,0,.15]),
    ('delta_popularity_gini_at_10', 'Popularity Gini@10', 'Absolute change\nNegative = better', [-.15,0,.15]),
    ('delta_mgu_at_10', 'MGU@10', 'Absolute change\nNegative = better', [-.06,0,.06]),
    ('delta_dgu_at_10', 'DGU@10', 'Absolute change\nNegative = better', [-.2,0,.2]),
]
DATASET_CODES = {'beauty':'B','sports':'S','toys':'T','pet-supplies':'P'}
TOKENIZER_CODES = {'RKMeans':'1','RVQ':'2','RQ-VAE':'3'}


def case_code(row):
    return DATASET_CODES[row['dataset']]+TOKENIZER_CODES[row['tokenizer']]


def tied_codes(cases):
    """Compress tied tokenizers of a dataset: P1/P3 becomes P1/3."""
    grouped = {}
    for row in cases:
        grouped.setdefault(DATASET_CODES[row['dataset']],[]).append(TOKENIZER_CODES[row['tokenizer']])
    return ','.join(ds+'/'.join(tokens) for ds,tokens in grouped.items())


def reference(path):
    return dict(path=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def keyed(path):
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    result = {tuple(row[k] for k in KEY):row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f'Duplicate dataset/tokenizer/method in {path}')
    return result


def collect():
    training,popularity = keyed(TRAINING),keyed(POPULARITY)
    expected = {(ds,tok,m) for ds in style.DATASETS for tok in style.TOKENIZERS for m in style.METHODS}
    if set(training) != expected or set(popularity) != expected:
        raise ValueError('Both sources must contain the same 156 method/case combinations')
    current,_ = style.calculate_changes(style.parse_tables(style.SOURCE))
    for row in current:
        key = tuple(row[k] for k in KEY)
        for field,_,_,_ in PANELS[:2]:
            if not math.isclose(row[field],float(training[key][field]),rel_tol=0,abs_tol=1e-12):
                raise ValueError('Training figure CSV differs from the current comparison tables')
    records = []
    for method in style.METHODS:
        for ds in style.DATASETS:
            for tok in style.TOKENIZERS:
                key = ds,tok,method
                old,pop = training[key],popularity[key]
                wanted = style.RERANKING_REFERENCE if method in style.RERANKERS else style.ORIGINAL
                pop_reference = 'Saved reference' if method in style.RERANKERS else 'Original'
                if old['reference_variant'] != wanted or pop['reference'] != pop_reference:
                    raise ValueError(f'Reference rules differ for {key}')
                record = dict(dataset=ds,tokenizer=tok,method=method,reference_variant=wanted)
                record.update({field:float((old if index<2 else pop)[field])
                               for index,(field,_,_,_) in enumerate(PANELS)})
                if not all(math.isfinite(record[f]) for f,_,_,_ in PANELS):
                    raise ValueError('Nonfinite plotted change')
                records.append(record)
    return records


def draw(records):
    plt.rcParams.update({
        'font.family':'serif','font.serif':['Nimbus Roman','Times New Roman','DejaVu Serif'],
        'mathtext.fontset':'stix','font.size':7,'axes.labelsize':6.5,
        'xtick.labelsize':6.5,'ytick.labelsize':7,'legend.fontsize':6.3,
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,
        'axes.edgecolor':'#52514e','xtick.color':'#52514e','ytick.color':'#52514e',
        'pdf.fonttype':42,
    })
    fig = plt.figure(figsize=(4.8,5.15))
    layout = fig.add_gridspec(2,3,left=.17,right=.99,bottom=.135,top=.915,wspace=.19,hspace=.48)
    offsets = np.random.default_rng(42).uniform(-.12,.12,12)
    mark_count = 0
    axes,case_labels,headers = [],[],[]
    for index,(field,title,xlabel,ticks) in enumerate(PANELS):
        cell = layout[index//3,index%3].subgridspec(1,2,width_ratios=(2,1.1),wspace=.06)
        ax = fig.add_subplot(cell[0,0])
        code_ax = fig.add_subplot(cell[0,1])
        axes.extend([ax,code_ax])
        code_ax.set_xlim(0,1)
        code_ax.set_ylim(12.5,-.5)
        code_ax.axis('off')
        headers.append(code_ax.text(.5,1.012,'Min → Max',ha='center',va='bottom',fontsize=6,
                                    transform=code_ax.transAxes,color='#353534'))
        for i,method in enumerate(style.METHODS):
            cases = [r for r in records if r['method'] == method]
            values = np.array([r[field] for r in cases])
            if len(values) != 12:
                raise ValueError('Every row must contain twelve cases')
            color = style.TEAL if method in style.RERANKERS else style.BLUE if method in style.CONTINUATION else style.ORANGE
            ax.scatter(values,i+offsets,s=10,color=color,alpha=.85,edgecolor='white',linewidth=.25,zorder=3)
            mark_count += len(values)
            ax.plot([np.median(values)]*2,[i-.32,i+.32],color='#0b0b0b',linewidth=1.1,zorder=4)
            for _,value,ties in style.extreme_cases(cases,field):
                indices = [cases.index(row) for row in ties]
                ax.scatter([value]*len(indices),i+offsets[indices],marker='D',s=20,
                           facecolors='none',edgecolors='#262626',linewidths=.5,zorder=5)
            endpoints = style.extreme_cases(cases,field)
            label = tied_codes(endpoints[0][2])+' → '+tied_codes(endpoints[1][2])
            case_labels.append((code_ax,code_ax.text(.5,i,label,ha='center',va='center',
                                                    fontsize=6,color='#353534')))
        ax.axvline(0,color='#52514e',linewidth=.6)
        for boundary in (2.5,9.5):
            ax.axhline(boundary,color='#d5d4d1',linewidth=.5)
            code_ax.axhline(boundary,color='#d5d4d1',linewidth=.5)
        ax.set_ylim(12.5,-.5)
        ax.set_yticks(range(13))
        ax.set_yticklabels(list(style.METHODS)
                           if index in (0,3) else [])
        ax.tick_params(axis='y',length=3 if index in (0,3) else 0)
        # Center the panel title over its plot and its case-code column.
        title_ax = fig.add_subplot(layout[index//3,index%3],frame_on=False)
        title_ax.axis('off')
        title_ax.set_title(f'({chr(97+index)}) {title}',fontsize=7.3,fontweight='bold',pad=18)
        axes.append(title_ax)
        ax.set_xlabel(xlabel,labelpad=4)
        ax.margins(x=.10)
        limits = ax.get_xlim()
        ax.set_xticks([v for v in ticks if limits[0] <= v <= limits[1]])
        ax.xaxis.set_major_formatter(FormatStrFormatter('%d') if index==0 else
            FuncFormatter(lambda value,_:'0' if abs(value)<1e-12 else
                          f'{value:.2f}'.replace('-0.','−.').replace('0.','.')))
        ax.grid(axis='x',color='#ecebe8',linewidth=.5)
        ax.set_axisbelow(True)
    legend_ax = fig.add_subplot(layout[1,2])
    axes.append(legend_ax)
    legend_ax.axis('off')
    legend_ax.set_title('12 cases per method',loc='left',fontsize=7.3,fontweight='bold',pad=7)
    handles = [
        Line2D([],[],marker='o',ls='',color=style.BLUE,ms=3.5,
               label='Fine-tuning\nResumed Adam\n32-update cap'),
        Line2D([],[],marker='o',ls='',color=style.ORANGE,ms=3.5,
               label='FairDiverse trainer\nFresh Adam\nDifferent schedules'),
        Line2D([],[],marker='o',ls='',color=style.TEAL,ms=3.5,
               label='Reranking\nNo model updates'),
        Line2D([],[],marker='|',ls='',color='#0b0b0b',ms=7,markeredgewidth=1.1,
               label='Median of 12 cases'),
        Line2D([],[],marker='D',ls='',color='#262626',ms=4,markerfacecolor='none',markeredgewidth=.5,
               label='Min / max cases'),
        Line2D([],[],color='#52514e',lw=.6,label='No change from\ncase-specific reference'),
    ]
    legend = legend_ax.legend(handles=handles,loc='upper left',frameon=False,
                              borderaxespad=0,handletextpad=.45,handlelength=1.2,labelspacing=.75)
    keys = [
        fig.text(.56,.038,'Cases: B = Beauty; S = Sports; T = Toys; P = Pet Supplies',
                 ha='center',fontsize=6.2),
        fig.text(.56,.014,'Tokenizers: 1 = RKMeans; 2 = RVQ; 3 = RQ-VAE; / = tied cases',
                 ha='center',fontsize=6.2),
    ]
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    canvas = fig.bbox
    text_boxes = []
    for ax in axes:
        visible_text = [ax.title]
        if ax.axison:
            visible_text.extend([ax.xaxis.label,*ax.get_xticklabels(),*ax.get_yticklabels()])
        for text in visible_text:
            if not text.get_text():
                continue
            box = text.get_window_extent(renderer)
            if box.x0 < -1 or box.x1 > canvas.width+1 or box.y0 < -1 or box.y1 > canvas.height+1:
                raise ValueError(f'Text is outside the figure: {text.get_text()}')
            text_boxes.append((text.get_text(),box))
    for ax,text in case_labels:
        box = text.get_window_extent(renderer)
        column = ax.get_window_extent(renderer)
        if box.x0 < column.x0-1 or box.x1 > column.x1+1:
            raise ValueError(f'Case code exceeds its column: {text.get_text()}')
        text_boxes.append((text.get_text(),box))
    for text in [*keys,*headers]:
        text_boxes.append((text.get_text(),text.get_window_extent(renderer)))
    for i,(label,a) in enumerate(text_boxes):
        if a.x0 < -1 or a.x1 > canvas.width+1 or a.y0 < -1 or a.y1 > canvas.height+1:
            raise ValueError(f'Text is outside the figure: {label}')
        for other,b in text_boxes[i+1:]:
            if a.overlaps(b):
                raise ValueError(f'Axis text overlaps: {label!r} / {other!r}')
    box = legend.get_window_extent(renderer)
    panel = legend_ax.get_window_extent(renderer)
    if box.x1 > panel.x1+1 or box.y0 < panel.y0-1 or box.overlaps(legend_ax.title.get_window_extent(renderer)):
        raise ValueError('Legend does not fit its panel')
    if mark_count != 780:
        raise ValueError('Expected 156 points in each of five metric panels')
    fig.savefig(OUT.with_suffix('.pdf'),metadata={'Title':'GRID interventions: effectiveness, exposure equality, and popularity calibration'})
    fig.savefig(OUT.with_suffix('.png'),dpi=260)
    plt.close(fig)


def main():
    global TRAINING, POPULARITY, OUT, DATA
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-changes', type=Path, default=TRAINING)
    parser.add_argument('--popularity-changes', type=Path, default=POPULARITY)
    parser.add_argument('--comparison-tables', type=Path, default=style.SOURCE)
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--table-dir', type=Path, default=DATA)
    args = parser.parse_args()
    TRAINING, POPULARITY, OUT, DATA = args.training_changes, args.popularity_changes, args.output, args.table_dir
    style.SOURCE = args.comparison_tables
    OUT.parent.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    records = collect()
    draw(records)
    style.write_csv(DATA/'training_interventions_combined_changes.csv',records)
    extrema = []
    summaries = []
    for method in style.METHODS:
        cases = [r for r in records if r['method'] == method]
        summary = dict(method=method,cases=12)
        for field,_,_,_ in PANELS:
            values = [r[field] for r in cases]
            summary.update({field+'_median':float(np.median(values)),field+'_min':min(values),field+'_max':max(values)})
            for side,value,ties in style.extreme_cases(cases,field):
                for row in ties:
                    extrema.append(dict(method=method,metric=field,extremum=side,value=value,
                                        dataset=row['dataset'],tokenizer=row['tokenizer'],tied_cases=len(ties),
                                        case_code=case_code(row),plotted_endpoint_label=tied_codes(ties)))
        summaries.append(summary)
    style.write_csv(DATA/'training_interventions_combined_summary.csv',summaries)
    style.write_csv(DATA/'training_interventions_combined_extrema.csv',extrema)
    OUT.with_name(OUT.name+'_caption.tex').write_text(r'''\caption{GRID intervention effects at $K=10$ over 12 cases per row
(four datasets $\times$ three tokenizers). References are case-matched original
GRID for training and untreated saved-candidate orderings for reranking.
(a) Relative Recall change, $100(R_m/R_0-1)$; positive is better.
(b--e) Absolute metric changes; negative is better.
Popularity Gini uses five frozen training-frequency bins.
MGU and DGU measure mismatch between pooled recommendation and test-history
popularity shares (40-item history cap).
Black bars: medians; diamonds: extrema; zero lines: no change.
Min$\to$Max columns identify endpoint cases; slashes indicate ties.
Blue resumes Adam (32-update cap, update zero eligible); orange uses fresh Adam
with different schedules and historical ranking protocols; teal reranks.
Sports--RVQ source/reranking beams are 100/300.
Recall/brand use rounded table inputs; popularity uses unrounded recovered metrics.
Seed 42; descriptive comparisons; independent panel scales.}
\label{fig:training-interventions-combined}
''')
    provenance = dict(cases=12,methods=13,panels=5,points_per_panel=156,total_case_metric_points=780,
        source_files=[reference(TRAINING),reference(POPULARITY),reference(style.SOURCE)],
        code=reference(Path(__file__)),
        join='Exact one-to-one dataset/tokenizer/method; reference rules agree for all 156 rows',
        data_validation='Recall and brand changes match the current comparison_tables.tex and original figure CSV; popularity changes copied without recomputation',
        precision='Recall/brand use rounded table values, popularity metrics use unrounded recovered values; original plotted changes preserved',
        presentation='Two rows, three columns; five metric panels with aligned Min-to-Max case-code columns and one shared legend; same case jitter in every panel',
        case_key=dict(datasets=DATASET_CODES,tokenizers=TOKENIZER_CODES,ties='Slash-separated tokenizer codes; e.g. P1/3 means Pet RKMeans and RQ-VAE'),
        endpoint_rows=len(PANELS)*len(style.METHODS),endpoint_labels=2*len(PANELS)*len(style.METHODS),
        endpoint_case_identities=len(extrema),
        tied_endpoint_labels=len({(r['method'],r['metric'],r['extremum']) for r in extrema if r['tied_cases']>1}),
        verification='780 case/metric points; all minimum and maximum case identities shown including ties; labels bounded and nonoverlapping; legend contained in sixth panel')
    (DATA/'training_interventions_combined_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print('Wrote',OUT.with_suffix('.pdf'))


if __name__ == '__main__':
    main()
