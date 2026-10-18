"""Create the paper figure linking original GRID Recall to APR/SDRO losses.

Panel A uses original-GRID control means from effects.csv. Panel B preserves
the saved absolute low-minus-high contrasts, pointwise bootstrap intervals,
and Holm decisions from the complete 504-test activity-contrast family.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator


ROOT = Path(__file__).resolve().parents[1]
DATASETS = [('amazon-beauty', 'Beauty'), ('amazon-sports', 'Sports'),
            ('amazon-toys', 'Toys'), ('amazon-pet-supplies', 'Pet Supplies')]
TOKENS = [('rkmeans', 'RKMeans', '#2878b5', 'o', '-'),
          ('rvq', 'RVQ', '#d95823', 's', '--'),
          ('rqvae', 'RQ-VAE', '#6c4c9a', '^', ':')]
GROUPS = ['low', 'medium', 'high']
TRAINERS = ['IPS', 'FOCF', 'Reg', 'FairDual', 'Minmax-SGD', 'APR', 'SDRO']


def require(condition, message):
    if not condition:
        raise ValueError(message)


def same(a, b, message):
    require(math.isclose(float(a), float(b), rel_tol=1e-10, abs_tol=1e-12), message)


def reference(path):
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prepare_data(source):
    effects = read_csv(source / 'effects.csv')
    contrasts = read_csv(source / 'activity_contrasts.csv')
    lookup = {(r['dataset'], r['tokenizer'], r['method'], r['activity_group']): r
              for r in effects if r['metric'] == 'recall@10'}
    require(len(lookup) == len([r for r in effects if r['metric'] == 'recall@10']), 'Duplicate effects')
    panel_a, panel_b = [], []
    for dataset, _ in DATASETS:
        for tokenizer, _, _, _, _ in TOKENS:
            for group in GROUPS:
                rows = [lookup[dataset, tokenizer, method, group] for method in TRAINERS]
                for row in rows:
                    require(row['control'] == 'Source GRID', 'Panel A needs original source GRID')
                    same(row['control_mean'], rows[0]['control_mean'], 'Original GRID means differ across trainers')
                    require(row['users'] == rows[0]['users'], 'Activity populations differ')
                baseline = float(rows[0]['control_mean'])
                require(0 < baseline <= 1, 'Invalid baseline Recall')
                panel_a.append(dict(dataset=dataset, tokenizer=tokenizer, activity_group=group,
                                    users=int(rows[0]['users']), recall_at_10=baseline,
                                    recall_at_10_percent=100*baseline))
            for method in ['APR', 'SDRO']:
                selected = [r for r in contrasts if (r['dataset'], r['tokenizer'], r['method'],
                            r['metric'], r['group_a'], r['group_b']) ==
                            (dataset, tokenizer, method, 'recall@10', 'low', 'high')]
                require(len(selected) == 1, 'Missing or duplicate direct low-high contrast')
                row = selected[0]
                low, high = [lookup[dataset, tokenizer, method, group] for group in ['low', 'high']]
                same(float(low['mean_change']) - float(high['mean_change']), row['contrast'], 'Contrast differs from effects')
                same(row['mean_change_a'], low['mean_change'], 'Low effect differs')
                same(row['mean_change_b'], high['mean_change'], 'High effect differs')
                require(float(low['mean_change']) < 0 and float(high['mean_change']) < 0,
                        'Loss-direction labels require losses in both groups')
                require(row['holm_family'] == 'fairdiverse_training:activity_contrasts' and
                        int(row['family_tests']) == 504, 'Preserve the full original Holm family')
                significant = float(row['p_holm']) < .05
                require(significant == (row['reject_holm_0_05'] == 'True'), 'Holm decision differs')
                panel_b.append(dict(dataset=dataset, tokenizer=tokenizer, method=method,
                    low_mean_change=float(low['mean_change']), high_mean_change=float(high['mean_change']),
                    contrast=float(row['contrast']), contrast_percentage_points=float(row['contrast_percentage_points']),
                    ci_low_percentage_points=100*float(row['ci_low']), ci_high_percentage_points=100*float(row['ci_high']),
                    p_raw=float(row['p_raw']), p_holm=float(row['p_holm']), holm_significant=significant,
                    holm_family=row['holm_family'], family_tests=int(row['family_tests']),
                    low_relative_change_percent=float(low['relative_change_percent']),
                    high_relative_change_percent=float(high['relative_change_percent'])))
    pet = [lookup['amazon-pet-supplies', tokenizer, 'SDRO', group]
           for tokenizer, _, _, _, _ in TOKENS for group in GROUPS]
    relative_range = [min(float(r['relative_change_percent']) for r in pet),
                      max(float(r['relative_change_percent']) for r in pet)]
    require(all(-77 < float(r['relative_change_percent']) < -72 for r in pet), 'Check caption relative-loss range')
    return panel_a, panel_b, relative_range


def render(source, output):
    panel_a, panel_b, relative_range = prepare_data(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    alookup = {(r['dataset'], r['tokenizer'], r['activity_group']): r for r in panel_a}
    blookup = {(r['dataset'], r['tokenizer'], r['method']): r for r in panel_b}
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 8,
                         'axes.titlesize': 9, 'axes.labelsize': 8, 'legend.fontsize': 8,
                         'axes.edgecolor': '#555555', 'axes.linewidth': .6,
                         'xtick.labelsize': 8, 'ytick.labelsize': 8,
                         'pdf.fonttype': 42, 'ps.fonttype': 42}):
        fig = plt.figure(figsize=(7.2, 4.9))
        grid = fig.add_gridspec(2, 4, left=.10, right=.985, bottom=.135, top=.84,
                               hspace=.70, wspace=.18)
        top_axes, bottom_axes = [], []
        for di, (dataset, label) in enumerate(DATASETS):
            top = fig.add_subplot(grid[0, di], sharey=top_axes[0] if di else None)
            bottom = fig.add_subplot(grid[1, di], sharey=bottom_axes[0] if di else None)
            top_axes.append(top)
            bottom_axes.append(bottom)
            for ti, (tokenizer, _, color, marker, style) in enumerate(TOKENS):
                baseline = [alookup[dataset, tokenizer, group]['recall_at_10_percent'] for group in GROUPS]
                top.plot(range(3), baseline, color=color, marker=marker, markersize=3.4,
                         linestyle=style, linewidth=.9, markeredgewidth=.5)
                for mi, method in enumerate(['APR', 'SDRO']):
                    row = blookup[dataset, tokenizer, method]
                    x = mi + (ti-1)*.17
                    lo, hi = row['ci_low_percentage_points'], row['ci_high_percentage_points']
                    bottom.vlines(x, lo, hi, color=color, linewidth=1)
                    bottom.plot([x-.025, x+.025], [lo, lo], color=color, linewidth=.7)
                    bottom.plot([x-.025, x+.025], [hi, hi], color=color, linewidth=.7)
                    bottom.plot(x, row['contrast_percentage_points'], marker, color=color,
                                markersize=3.8, markeredgewidth=.5)
                    if row['holm_significant']:
                        bottom.text(x, hi+.14, '*', ha='center', va='bottom', fontsize=9,
                                    color='#222222')
            top.set_title(label, pad=6)
            top.set_xticks([0, 1, 2], ['Low', 'Med.', 'High'])
            top.set_xlim(-.25, 2.25)
            top.set_ylim(0, 8.6)
            top.yaxis.set_major_locator(MultipleLocator(2))
            bottom.set_xticks([0, 1], ['APR', 'SDRO'])
            bottom.set_xlim(-.45, 1.45)
            bottom.set_ylim(-1.35, 4.65)
            bottom.yaxis.set_major_locator(MultipleLocator(1))
            bottom.axhline(0, color='#555555', linewidth=.75)
            for ax in [top, bottom]:
                ax.grid(axis='y', color='#e5e5e5', linewidth=.5)
                ax.set_axisbelow(True)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.tick_params(axis='both', length=2.5, width=.6)
                if di:
                    ax.tick_params(labelleft=False)
        top_axes[0].set_ylabel('Original Recall@10 (%)', labelpad=5)
        bottom_axes[0].set_ylabel('Low − high change\n(percentage points)', labelpad=5)
        handles = [Line2D([], [], color=c, marker=m, linestyle=s, linewidth=.9,
                          markersize=4, label=label) for _, label, c, m, s in TOKENS]
        fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.54, .995),
                   frameon=False, ncol=3, handlelength=2, columnspacing=1.8)
        fig.text(.10, .905, 'A  Original GRID effectiveness by activity group', fontsize=9, weight='bold')
        fig.text(.10, .465, 'B  Differential losses under APR and SDRO', fontsize=9, weight='bold')
        fig.text(.10, .060, 'Positive: high-activity users lose more. Negative: low-activity users lose more.', fontsize=7)
        fig.text(.10, .025, 'Intervals: pointwise 95% bootstrap CIs.  * Holm-adjusted p < 0.05 (full 504-test family).', fontsize=7)
        fig.savefig(output.with_suffix('.pdf'), metadata={'Title': 'Original GRID effectiveness and differential APR/SDRO losses',
                                                        'Creator': 'plot_activity_baseline_losses.py'})
        fig.savefig(output.with_suffix('.png'), dpi=300)
        plt.close(fig)
    caption = r'''\caption{Original effectiveness and differential absolute losses across user-activity groups.
(A) Original GRID Recall@10 for frozen low-, medium-, and high-activity groups.
(B) Low-minus-high contrasts in mean per-user Recall@10 changes under APR and SDRO,
where each change is intervention minus original GRID. Positive contrasts indicate
larger absolute losses for high-activity users; negative contrasts indicate larger
losses for low-activity users. Intervals are pointwise 95\% user-bootstrap intervals
(1,000 replicates); asterisks denote Holm-adjusted $p<0.05$ from Welch tests on
user-level changes, retaining the full 504-test activity-contrast family.
Baseline Recall differs across activity groups. On Pet Supplies, SDRO's relative
Recall losses are descriptively similar across groups within each tokenizer
(approximately 72--76\% across the dataset); significance refers to absolute
contrasts, not relative differences. Comparisons include the recorded training
and evaluation protocols; uncertainty is conditional on the saved models.}
\label{fig:activity-baseline-losses}
'''
    caption_path = output.with_name(output.name + '_caption.tex')
    caption_path.write_text(caption)
    a_path = output.with_name(output.name + '_panel_a.csv')
    b_path = output.with_name(output.name + '_panel_b.csv')
    write_csv(a_path, panel_a)
    write_csv(b_path, panel_b)
    outputs = [output.with_suffix('.pdf'), output.with_suffix('.png'), caption_path, a_path, b_path]
    provenance = dict(sources={name: reference(source/name) for name in ['effects.csv', 'activity_contrasts.csv']},
        script=reference(Path(__file__)), outputs=[reference(p) for p in outputs],
        panel_a_points=len(panel_a), panel_b_contrasts=len(panel_b),
        holm_significant_contrasts=sum(r['holm_significant'] for r in panel_b),
        baseline_checks='Original GRID control means and group counts agree across all seven FairDiverse methods',
        contrast_checks='All 24 contrasts agree with differences of group effects; all groups have negative APR/SDRO effects',
        correction='Original 504-test family retained; no correction recomputed on the displayed subset',
        interval='Pointwise 95% user-bootstrap interval; not Holm-adjusted',
        pet_sdro_relative_change_range_percent=relative_range,
        relative_changes='Descriptive caption context only; no relative-scale test or equivalence claim',
        units={'panel_a': 'Recall@10 percent', 'panel_b': 'Absolute Recall@10 change contrast in percentage points'})
    output.with_name(output.name + '_provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    print(f'Created {output.with_suffix(".pdf")} and PNG, caption, plot data, provenance')
    print(f'Checked {len(panel_a)} baselines, {len(panel_b)} contrasts; '
          f'{provenance["holm_significant_contrasts"]} Holm-significant contrasts')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT/'docs/paired_activity_all_methods_20261016')
    parser.add_argument('--output', type=Path, default=ROOT/'figures/fig_activity_baseline_losses')
    args = parser.parse_args()
    render(args.source.resolve(), args.output.resolve())
