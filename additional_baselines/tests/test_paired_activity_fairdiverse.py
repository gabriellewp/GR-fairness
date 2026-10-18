"""Historical score reconstruction must match evaluator quicksort and user rows."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, save_npz

from additional_baselines import paired_activity_fairdiverse as fd
from additional_baselines import paired_activity_fairdiverse_retry as repair


def brute(scores):
    ranks=[]
    for row in np.asarray(scores,dtype=np.float32):
        found=np.flatnonzero(np.argsort(row)[::-1][:10]==0)
        ranks.append(int(found[0])+1 if len(found) else 0)
    return ranks


class RecordedScoreTests(unittest.TestCase):
    def test_dense_fast_paths_match_full_quicksort(self):
        rng=np.random.default_rng(7)
        # Many ties, unique targets, all-equal rows, and targets below cutoff.
        scores=rng.choice([1.,1.000001,1.1,1.2,1.3],size=(120,37)).astype(np.float32)
        scores[0]=1.
        scores[1]=np.arange(37)+1
        scores[2]=np.arange(37,0,-1)
        ranks,_=fd.dense_target_ranks(scores)
        self.assertEqual(ranks.tolist(),brute(scores))

    def test_stream_coo_positive_alignment_and_exact_ranks(self):
        with tempfile.TemporaryDirectory() as tmp:
            n,catalog=4,17
            targets=np.array([4,1,12,0])
            ids=np.array([[t]+[i for i in range(catalog) if i!=t] for t in targets])
            scores=np.random.default_rng(8).choice([1.,1.1,1.2],size=(n,catalog)).astype(np.float32)
            path=Path(tmp)/'scores.npz'
            matrix=coo_matrix((scores.ravel(),(np.repeat(np.arange(n),catalog),ids.ravel())),shape=(n,catalog))
            save_npz(path,matrix)
            ranks,_=fd.coo_ranks(path,{'targets':targets},[n,catalog],batch=2)
            self.assertEqual(ranks.tolist(),brute(scores))
            with self.assertRaisesRegex(ValueError,'positives differ'):
                fd.coo_ranks(path,{'targets':targets[::-1]},[n,catalog],batch=2)

    def test_compact_csr_preserves_implicit_baseline_and_target_first_ties(self):
        with tempfile.TemporaryDirectory() as tmp:
            n,catalog=5,31
            targets=np.array([8,2,25,3,14])
            ids=np.array([[t]+[i for i in range(catalog) if i!=t] for t in targets])
            scores=np.ones((n,catalog),dtype=np.float32)
            scores[0,[0,2,3,4]]=[1.1,1.2,1.1,1.05]
            scores[1,:15]=1.2
            scores[2,0]=1.3
            scores[3,10:]=1.4
            stored=scores>1
            rows=np.repeat(np.arange(n),catalog)[stored.ravel()]
            cols=ids.ravel()[stored.ravel()]
            data=scores.ravel()[stored.ravel()]
            matrix=coo_matrix((data,(rows,cols)),shape=(n,catalog)).tocsr()
            path=Path(tmp)/'scores.npz'
            save_npz(path,matrix)
            ranks,_=fd.csr_ranks(path,{'targets':targets},[n,catalog],batch=2)
            self.assertEqual(ranks.tolist(),brute(scores))

    def test_invalid_compact_shift_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'scores.npz'
            save_npz(path,csr_matrix(np.array([[.5,0,0]],dtype=np.float32)))
            with self.assertRaisesRegex(ValueError,'shifted by'):
                fd.csr_ranks(path,{'targets':np.array([0])},[1,3])

    def test_compact_coo_order_and_missing_positive(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'compact.npz'
            # Positive target 4 appears first in row 0; positive target 2 is
            # absent from row 1. The frozen split supplies that missed target.
            matrix=coo_matrix((np.array([1.1,1.2,1.05,1.1,1.2],dtype=np.float32),
                               ([0,0,0,1,1],[4,1,7,0,6])),shape=(2,12))
            save_npz(path,matrix)
            population={'targets':np.array([4,2])}
            ranks,audit=repair.compact_coo_ranks(path,population,[2,12])
            dense=np.ones((2,12),dtype=np.float32)
            dense[0,[0,2,7]]=[1.1,1.2,1.05]
            dense[1,[1,6]]=[1.1,1.2]
            self.assertEqual(ranks.tolist(),brute(dense))
            self.assertEqual(audit['target_present_rows'],1)
            bad=coo_matrix((np.array([1.2,1.1]),([0,0],[7,4])),shape=(2,12))
            save_npz(path,bad)
            with self.assertRaisesRegex(ValueError,'positive-first'):
                repair.compact_coo_ranks(path,population,[2,12])


if __name__=='__main__':
    unittest.main()
