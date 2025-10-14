from __future__ import annotations
import unittest
from fractions import Fraction
import numpy as np
import numeric


class TestNumeric(unittest.TestCase):
    """Unit tests specific to the numeric.py module"""

    def test_get_and_registry(self):
        """Verify that all numeric modules are registered"""
        names = numeric.get_name_of_modules()
        self.assertIn("tropical_min_plus", names)
        self.assertIn("tropical_max_plus", names)
        self.assertIn("ocaml_float", names)
        for name in names:
            obj = numeric.get(name)
            self.assertTrue(hasattr(obj, "add"))
            self.assertTrue(hasattr(obj, "compare"))

    # === INT ===
    def test_numeric_int_operations(self):
        N = numeric.NumericInt()
        self.assertEqual(N.add(2, 3), 5)
        self.assertEqual(N.neg(4), -4)
        self.assertEqual(N.compare(5, 5), 0)
        self.assertEqual(N.compare(5, 2), 1)
        self.assertEqual(N.compare(2, 5), -1)
        self.assertEqual(N.div(10, 2), 5)
        self.assertEqual(N.pow(2, 3), 8)
        self.assertEqual(N.to_string(7), "7")

    # === FLOAT ===
    def test_numeric_float_operations(self):
        N = numeric.NumericFloat()
        self.assertAlmostEqual(N.add(1.5, 2.5), 4.0)
        self.assertAlmostEqual(N.mul(2.0, 3.0), 6.0)
        self.assertEqual(N.compare(1.0, 2.0), -1)
        self.assertEqual(N.compare(2.0, 1.0), 1)
        self.assertEqual(N.compare(1.0, 1.0), 0)
        self.assertEqual(N.pow(2.0, 3), 8.0)
        self.assertEqual(N.of_int(3), 3.0)

    # === BIG INT ===
    def test_numeric_bigint_operations(self):
        N = numeric.NumericBigInt()
        self.assertEqual(N.add(1000000000, 2), 1000000002)
        self.assertEqual(N.mul(10**6, 2), 2_000_000)
        self.assertEqual(N.compare(5, 6), -1)
        self.assertEqual(N.compare(7, 6), 1)
        self.assertEqual(N.to_string(42), "42")

    # === BIG RATIONAL ===
    def test_numeric_bigrat_operations(self):
        N = numeric.NumericBigRat()
        a, b = Fraction(1, 2), Fraction(3, 4)
        self.assertEqual(N.add(a, b), Fraction(5, 4))
        self.assertEqual(N.mul(a, b), Fraction(3, 8))
        self.assertEqual(N.div(b, a), Fraction(3, 2))
        self.assertEqual(N.pow(Fraction(2, 3), 2), Fraction(4, 9))
        self.assertEqual(N.compare(Fraction(3, 4), Fraction(3, 4)), 0)
        self.assertEqual(N.compare(Fraction(1, 4), Fraction(1, 3)), -1)
        self.assertEqual(N.to_string(Fraction(2, 3)), "2/3")

    # === TROPICAL MIN-PLUS ===
    def test_tropical_minplus(self):
        N = numeric.TropicalNumericMinPlus()
        self.assertEqual(N.add(3, 5), 3)        # min(3,5)
        self.assertEqual(N.mul(3, 5), 8)        # 3+5
        self.assertEqual(N.div(7, 2), 5)        # 7-2
        self.assertEqual(N.pow(2, 3), 6)
        self.assertEqual(N.compare(5, 5), 0)
        self.assertEqual(N.compare(3, 5), -1)
        self.assertEqual(N.compare(5, 3), 1)
        self.assertEqual(N.min(3, 5), 3)
        self.assertEqual(N.max(3, 5), 5)
        self.assertEqual(N.one, 0.0)
        self.assertTrue(np.isinf(N.zero))

    # === TROPICAL MAX-PLUS ===
    def test_tropical_maxplus(self):
        N = numeric.TropicalNumericMaxPlus()
        self.assertEqual(N.add(3, 5), 5)        # max(3,5)
        self.assertEqual(N.mul(3, 5), 8)        # 3+5
        self.assertEqual(N.div(7, 2), 5)
        self.assertEqual(N.pow(2, 3), 6)
        self.assertEqual(N.compare(5, 5), 0)
        self.assertEqual(N.compare(3, 5), -1)
        self.assertEqual(N.compare(5, 3), 1)
        self.assertEqual(N.min(3, 5), 3)
        self.assertEqual(N.max(3, 5), 5)
        self.assertEqual(N.one, 0.0)
        self.assertTrue(np.isneginf(N.zero))

    def test_tropical_array_compare(self):
        """Verify compare() on numpy arrays"""
        N = numeric.TropicalNumericMinPlus()
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 2.0, 3.0])
        self.assertEqual(N.compare(x, y), 0)
        self.assertEqual(N.compare(np.array([1, 2]), np.array([1, 3])), -1)
        self.assertEqual(N.compare(np.array([2, 3]), np.array([1, 3])), 1)


if __name__ == "__main__":
    unittest.main()
