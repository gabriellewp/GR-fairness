import sys
import unittest
import numpy as np

from additional_baselines.rerank.fast_pmmf import project_sorted, next_dual, FastPMMFReranker
from additional_baselines.rerank.legacy import load_class
from additional_baselines.tests.test_rerank_lookup import LookupParityTests
from additional_baselines.rerank import FairDiverseReranker


class ExactProjectionTests(unittest.TestCase):
    def test_projection_matches_legacy_qp_with_nonuniform_weights(self):
        module = sys.modules[load_class("PMMF").__module__]
        rng = np.random.default_rng(42)
        for strength in (0.,.5,5.,50.):
            for size in (3,16,32):
                rho = rng.uniform(.2,2.,size)
                z = np.sort(rng.normal(-1.,2.,size))
                dual = z/rho
                exact = project_sorted(dual,rho,strength)
                legacy = module.compute_projection_maxmin_fairness_with_order(dual,rho,strength)
                np.testing.assert_allclose(exact,legacy,atol=3e-5,rtol=1e-5)
                self.assertGreaterEqual(np.cumsum(exact*rho).min(),-strength-1e-10)
                self.assertLessEqual(np.square(rho*(exact-dual)).sum(),np.square(rho*(legacy-dual)).sum()+1e-4)

    def test_zero_strength_and_inactive_projection(self):
        np.testing.assert_array_equal(project_sorted([-3.,-2.,1.],np.ones(3),0.),[0.,0.,1.])
        np.testing.assert_array_equal(project_sorted([-3.,-2.,1.],np.ones(3),10.),[-3.,-2.,1.])
        np.testing.assert_allclose(project_sorted([-3.,-2.,1.],np.ones(3),1.),[-1.,0.,1.])

    def test_complete_rankings_match_legacy_non_tied_fixtures(self):
        fixture = LookupParityTests()
        fixture.setUp()
        try:
            for strength in (0.,1.,30.):
                inputs = fixture.fixture(42)
                kwargs = dict(model="PMMF",group="brand",fairness_lambda=strength,
                              weights={"0":1.,"1":1.,"2":1.})
                for k in (1,5):
                    legacy = FairDiverseReranker(**kwargs).rerank(inputs,k)
                    exact = FastPMMFReranker(**kwargs).rerank(inputs,k)
                    self.assertEqual(legacy.rankings,exact.rankings)
                    self.assertEqual(legacy.diagnostics,exact.diagnostics)
        finally:
            fixture.doCleanups()


if __name__ == "__main__":
    unittest.main()
