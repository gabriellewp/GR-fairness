"""Repair the three compact-COO FairDual cases without changing frozen workers.

The original array and manifest remain intact. This extension validates the old
compact row/item order, then uses the already tested compact ranking algorithm.
"""
import argparse
from pathlib import Path
import zipfile

import numpy as np
from scipy.sparse import load_npz

from . import paired_activity as base
from . import paired_activity_fairdiverse as fd


def compact_coo_ranks(path, population, shape):
    matrix = load_npz(path)
    base.require(matrix.format == 'coo' and matrix.shape == tuple(shape), 'Expected compact COO scores')
    rows, cols = matrix.row, matrix.col
    base.require(len(rows) > 0 and rows.min() >= 0 and rows.max() < shape[0], 'Invalid compact row IDs')
    base.require(cols.min() >= 0 and cols.max() < shape[1], 'Invalid compact catalog IDs')
    base.require(np.all(np.diff(rows) >= 0), 'Compact COO must preserve sequential test-row order')
    targets = population['targets'][rows]
    positions = cols + (cols < targets)
    positions[cols == targets] = 0
    same_row = rows[1:] == rows[:-1]
    base.require(np.all(np.diff(positions)[same_row] > 0),
                 'Compact items do not preserve the positive-first ascending catalog order')
    covered = np.unique(rows[cols == targets])
    del matrix, rows, cols, targets, positions, same_row
    ranks, audit = fd.csr_ranks(path, population, shape)
    audit.update(layout='coo_compact',
        user_alignment=('Explicit sequential COO row indices and strictly ordered target-first catalog positions checked; '
                        'missing positive targets recovered from the frozen test split under the deterministic '
                        'shuffle=False test-loader row-order assumption. Existing ground truth is also checked by the caller.'),
        target_present_rows=int(len(covered)), reader_extension=base.reference(Path(__file__)))
    return ranks, audit


def run(repair_path, index):
    repair_path = Path(repair_path).resolve()
    repair = base.read_json(repair_path)
    for key in ['manifest', 'code']:
        base.verified(repair[key])
    base.require(index in repair['cases'], 'Case is outside the declared repair set')
    original = fd.coo_ranks

    def reader(path, population, shape, batch=32):
        with zipfile.ZipFile(path) as archive:
            with fd.npy_member(archive, 'data.npy') as (_, count, _):
                compact = count != shape[0]*shape[1]
        return compact_coo_ranks(path, population, shape) if compact else original(path, population, shape, batch=batch)

    try:
        fd.coo_ranks = reader
        fd.run_case(repair['manifest']['path'], index)
    finally:
        fd.coo_ranks = original
    manifest = base.read_json(repair['manifest']['path'])
    folder = fd.case_folder(Path(repair['manifest']['path']).parent/'fairdiverse_cases', manifest['cases'][index])
    complete = base.read_json(folder/'complete.json')
    complete['repair'] = base.reference(repair_path)
    base.write_json(folder/'complete.json', complete)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repair', type=Path, required=True)
    parser.add_argument('--case', type=int, required=True)
    args = parser.parse_args()
    run(args.repair, args.case)


if __name__ == '__main__':
    main()
