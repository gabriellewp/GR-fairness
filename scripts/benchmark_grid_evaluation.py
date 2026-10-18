"""Compare legacy and batched evaluation on real held-out Pet Supplies rows.

Run on a GPU node before enabling batching for the pending array tasks.
The report checks native beam identity, raw/shifted scores, metrics and sparse
exports; it records timing and rejects changes to top-K catalog rankings.
"""

import argparse
import ast
import csv
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
print('[benchmark] Loading GRID evaluation dependencies', flush=True)

import numpy as np
import torch

from additional_baselines.ifairlrs_grid import original_module
from fairdiverse.recommendation.evaluator import Ranking_Evaluator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/reproduction/experiments_config_amazon-pet-supplies_grid_tokenizers.json')
    parser.add_argument('--exp-id', type=int, nargs='+', default=[57])
    parser.add_argument('--requests', type=int, default=32)
    parser.add_argument('--start-row', type=int, default=0)
    parser.add_argument('--production-loader', action='store_true',
                        help='Include the actual ranking Dataset/DataLoader in evaluator timing')
    parser.add_argument('--beam-check-only', action='store_true',
                        help='Check generated SID parity and catalog scores without timing evaluators')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.requests < 1 or args.batch_size < 1 or args.start_row < 0:
        parser.error('requests/batch-size must be positive and start-row non-negative')
    experiment_ids, output = args.exp_id, args.output
    for experiment_id in experiment_ids:
        args.exp_id = experiment_id
        args.output = (output.with_name(f'{output.stem}_{experiment_id}{output.suffix}')
                       if len(experiment_ids) > 1 else output)
        run(args)


def run(args):
    torch.set_num_threads(4)
    torch.manual_seed(42)
    experiment = next(e for e in json.loads((ROOT / args.config).read_text()) if e['exp_id'] == args.exp_id)
    config = dict(experiment['config'], dataset=experiment['dataset'], device=args.device)
    data = ROOT / 'recommendation/processed_dataset' / experiment['dataset']
    config.update(json.loads((data / 'dataset_stats.json').read_text()))
    config.update(group_num=config['group_num'], decimals=8, user_fairness=False)
    print(f'[benchmark] Loading {config["grid_tokenizer"]} checkpoint', flush=True)
    model = original_module('grid_model').GRIDBaseModel(config).to(args.device).eval()
    rows = []
    with (data / (experiment['dataset'] + '.valid.ranking')).open() as stream:
        for row_index, row in enumerate(csv.DictReader(stream, delimiter='\t')):
            if row_index < args.start_row:
                continue
            rows.append(row)
            if len(rows) >= args.requests:
                break
    history_length = config['history_length']
    histories, items = [], []
    catalog = torch.arange(config['item_num'])
    for row in rows:
        history = ast.literal_eval(row['history_behaviors'])[-history_length:]
        histories.append(history + [-1] * (history_length - len(history)))
        target = int(row['target_item'])
        items.append(torch.cat((torch.tensor([target]), catalog[catalog != target])))
    users = torch.tensor([int(row['user_id']) for row in rows])
    histories, items = torch.tensor(histories), torch.stack(items)
    batch = (users, histories, items, torch.ones(len(rows), dtype=torch.long))
    loader = [batch]
    if args.production_loader:
        import pandas as pd
        from torch.utils.data import DataLoader
        from fairdiverse.recommendation.sampler import RankingTestDataset
        frame = pd.DataFrame(rows)
        frame['user_id'] = frame['user_id'].astype(int)
        frame['target_item'] = frame['target_item'].astype(int)
        frame['history_behaviors'] = frame['history_behaviors'].apply(ast.literal_eval)
        loader = DataLoader(RankingTestDataset(frame, config), batch_size=128, shuffle=False)
    iid2pid = {int(k): int(v) for k, v in json.loads((data / 'iid2pid.json').read_text()).items()}

    def sync():
        if args.device.startswith('cuda'):
            torch.cuda.synchronize()

    def timed(call):
        sync()
        start = time.perf_counter()
        result = call()
        sync()
        return result, time.perf_counter() - start

    # Warm CUDA kernels outside timing. Capture one full set of legacy beams
    # to check that batching never mixes users or changes selected candidates.
    feed = {'user_ids': users.to(args.device), 'history_ids': histories.to(args.device)}
    with torch.no_grad():
        print('[benchmark] Checking individual versus batched beams and scores', flush=True)
        model.generate(feed['history_ids'][:1], feed['user_ids'][:1])
        old_beams, old_scores = [], []
        for i in range(len(rows)):
            old_scores.append(model.full_predict({k: v[i:i+1] for k, v in feed.items()}, items[i:i+1].to(args.device)).cpu())
            old_beams.append(model._last_generated_sids.cpu())
        old_scores = torch.cat(old_scores)
        fast_scores = model.full_predict_batch(feed, items, args.batch_size)
        old_beams = torch.cat(old_beams)
        fast_beams = model._last_generated_sids
        same_beams = torch.equal(old_beams, fast_beams)
        beam_differences = (old_beams != fast_beams).any(dim=-1)
        differing_positions = beam_differences.nonzero().tolist()
        diagnostic_k = max(config['topk'])
        same_native_topk = torch.equal(old_beams[:, :diagnostic_k], fast_beams[:, :diagnostic_k])
        # A beam permutation below the largest reported cutoff leaves native
        # ranking metrics unchanged. Require the full multiset as well so
        # target/prefix coverage across all generated beams is preserved.
        same_beam_multisets = all(
            sorted(map(tuple, old.tolist())) == sorted(map(tuple, fast.tolist()))
            for old, fast in zip(old_beams, fast_beams)
        )
        torch.testing.assert_close(old_scores, fast_scores, rtol=1e-5, atol=1e-8)
        old_order = np.argsort((old_scores + 1).numpy(), axis=1)[:, ::-1]
        fast_order = np.argsort((fast_scores + 1).numpy(), axis=1)[:, ::-1]
        same_topk = np.array_equal(old_order[:, :max(config['topk'])], fast_order[:, :max(config['topk'])])

    beam_check = {
        'experiment_id': args.exp_id, 'tokenizer': config['grid_tokenizer'],
        'requests': len(rows), 'start_row': args.start_row,
        'native_beams_identical': same_beams,
        'native_topk_beams_identical': same_native_topk,
        'native_beam_multisets_identical': same_beam_multisets,
        'catalog_topk_identical': same_topk,
        'score_max_difference': float((old_scores - fast_scores).abs().max()),
        'beam_differences': [
            {'dataset_row': args.start_row + row, 'beam_rank': beam + 1,
             'legacy_sid': old_beams[row, beam].tolist(),
             'batched_sid': fast_beams[row, beam].tolist()}
            for row, beam in differing_positions
        ],
        'passed': same_native_topk and same_beam_multisets and same_topk,
    }
    if args.device.startswith('cuda'):
        beam_check['gpu_device'] = torch.cuda.get_device_name()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    beam_path = (args.output if args.beam_check_only else args.output.with_name(
        f'{args.output.stem}_beam_check{args.output.suffix}'))
    beam_path.write_text(json.dumps(beam_check, indent=2) + '\n')
    print(json.dumps(beam_check, indent=2), flush=True)
    if args.beam_check_only:
        if not beam_check['passed']:
            raise SystemExit('Batched evaluation changed generated SID membership or reported rankings')
        return

    def evaluate(batch_size):
        evaluator = Ranking_Evaluator.__new__(Ranking_Evaluator)
        evaluator.config = dict(config, grid_eval_generation_batch_size=batch_size,
                                grid_native_diagnostics=True, grid_native_metrics=False)
        evaluator.iid2pid = iid2pid
        result, seconds = timed(lambda: evaluator.eval(loader, model, store_scores=True))
        return result, seconds, evaluator

    print('[benchmark] Timing complete legacy evaluator', flush=True)
    (legacy_metrics, legacy_export), legacy_seconds, legacy_evaluator = evaluate(0)
    print('[benchmark] Timing complete batched evaluator', flush=True)
    (fast_metrics, fast_export), fast_seconds, fast_evaluator = evaluate(args.batch_size)
    native_evaluator = Ranking_Evaluator.__new__(Ranking_Evaluator)
    native_evaluator.config = dict(config, grid_native_metrics=True)
    native = native_evaluator.eval(loader, model)
    same_export_structure = (np.array_equal(legacy_export.indices, fast_export.indices)
                             and np.array_equal(legacy_export.indptr, fast_export.indptr))
    export_max_difference = (float(np.max(np.abs(legacy_export.data - fast_export.data)))
                             if same_export_structure and legacy_export.nnz else None)
    report = {
        'experiment_id': args.exp_id, 'tokenizer': config['grid_tokenizer'],
        'checkpoint': model.loaded_checkpoint_path, 'requests': len(rows),
        'start_row': args.start_row, 'production_loader': args.production_loader,
        'generation_batch_size': args.batch_size, 'device': args.device,
        'legacy_seconds': legacy_seconds, 'batched_seconds': fast_seconds,
        'speedup': legacy_seconds / fast_seconds,
        'native_beams_identical': same_beams, 'catalog_topk_identical': same_topk,
        'native_topk_beams_identical': same_native_topk,
        'native_beam_multisets_identical': same_beam_multisets,
        'beam_difference_positions': [
            {'dataset_row': args.start_row + row, 'beam_rank': beam + 1}
            for row, beam in differing_positions
        ],
        'metrics_identical': legacy_metrics == fast_metrics,
        'native_diagnostic_identical': native == fast_evaluator.last_grid_native_result,
        'export_structure_identical': same_export_structure,
        'export_max_difference': export_max_difference,
        'score_max_difference': float((old_scores - fast_scores).abs().max()),
        'ground_truth_identical': (legacy_evaluator.last_ground_truth != fast_evaluator.last_ground_truth).nnz == 0,
    }
    if args.device.startswith('cuda'):
        report['gpu_device'] = torch.cuda.get_device_name()
        report['peak_allocated_gpu_bytes'] = torch.cuda.max_memory_allocated()
    report['passed'] = all(report[k] for k in (
        'native_topk_beams_identical', 'native_beam_multisets_identical',
        'catalog_topk_identical', 'metrics_identical',
        'native_diagnostic_identical', 'export_structure_identical', 'ground_truth_identical',
    )) and export_max_difference is not None and export_max_difference <= 1e-6
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)
    if not report['passed']:
        raise SystemExit('Batched evaluation failed parity; do not enable it for new tasks')


if __name__ == '__main__':
    main()
