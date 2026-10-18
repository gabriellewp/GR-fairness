import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("rq2_classic_test", Path(__file__).resolve().parents[2] / "scripts/rq2_classic_evaluation.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class CommonExposureTest(unittest.TestCase):
    def test_equal_exposure_has_half_mmf_and_bits_entropy(self):
        result = module.count_fairness([1, 1, 1, 1])
        self.assertEqual(result["mmf"], .5)
        self.assertEqual(result["gini"], 0)
        self.assertEqual(result["entropy_bits"], 2)

    def test_unexposed_catalog_groups_are_included(self):
        result = module.count_fairness([0, 0, 4, 4])
        self.assertEqual(result["group_count"], 4)
        self.assertEqual(result["mmf"], 0)
        self.assertEqual(result["gini"], .5)
        self.assertEqual(result["entropy_bits"], 1)

    def test_mmf_floors_half_of_odd_group_count(self):
        self.assertEqual(module.count_fairness([1, 2, 3, 4, 5])["mmf"], 3/15)


if __name__ == "__main__":
    unittest.main()
