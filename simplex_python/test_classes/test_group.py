"""Regression tests covering the ordered-group helpers in group.py."""

from __future__ import annotations
import unittest
import numeric
import group


class TestGroup(unittest.TestCase):
    """Unit tests specific to the group.py module"""

    @classmethod
    def setUpClass(cls):
        """Initialize numeric modules and groups for tests"""
        cls.NumFloat = numeric.NumericFloat()
        cls.G_float = group.GroupFromNumeric(cls.NumFloat)
        cls.G_int = group.IntGroup()
        cls.G_rev = group.ReverseOrder(cls.G_int)
        cls.G_cart = group.CartesianProduct(cls.G_int, cls.G_int)
        cls.G_sparse = group.CartesianPowerSparse(cls.G_int)

    # === Basic tests ===
    def test_groupfromnumeric_operations(self):
        """Verify addition, negation and comparison"""
        self.assertEqual(self.G_float.add(2.0, 3.0), 5.0)
        self.assertEqual(self.G_float.neg(4.0), -4.0)
        self.assertEqual(self.G_float.compare(3.0, 3.0), 0)
        self.assertEqual(self.G_float.compare(5.0, 3.0), 1)
        self.assertEqual(self.G_float.compare(1.0, 4.0), -1)
        self.assertEqual(self.G_float.max(1.0, 4.0), 4.0)
        self.assertEqual(self.G_float.zero(), 0.0)
        self.assertEqual(self.G_float.to_string(3.0), "3.0")

    def test_intgroup_basic(self):
        """Test basic operations of IntGroup"""
        self.assertEqual(self.G_int.add(2, 3), 5)
        self.assertEqual(self.G_int.neg(2), -2)
        self.assertEqual(self.G_int.compare(2, 5), -1)
        self.assertEqual(self.G_int.compare(5, 2), 1)
        self.assertEqual(self.G_int.compare(3, 3), 0)
        self.assertEqual(self.G_int.max(2, 5), 5)
        self.assertEqual(self.G_int.zero(), 0)

    def test_reverse_order(self):
        """Verify order inversion"""
        self.assertEqual(self.G_rev.compare(5, 2), -1)  
        self.assertEqual(self.G_rev.compare(2, 5), 1)
        self.assertEqual(self.G_rev.add(2, 3), 5)
        self.assertEqual(self.G_rev.neg(3), -3)
        self.assertEqual(self.G_rev.zero(), 0)
        self.assertEqual(self.G_rev.max(3, 5), 3)  

    # === Tests on Cartesian products ===
    def test_cartesian_product_operations(self):
        """Basic operations of Cartesian product"""
        a = (1, 2)
        b = (3, 4)
        res_add = self.G_cart.add(a, b)
        self.assertEqual(res_add, (4, 6))
        res_neg = self.G_cart.neg(a)
        self.assertEqual(res_neg, (-1, -2))
        self.assertEqual(self.G_cart.compare((1, 2), (1, 3)), -1)
        self.assertEqual(self.G_cart.compare((2, 1), (1, 3)), 1)
        self.assertEqual(self.G_cart.max((1, 2), (1, 1)), (1, 2))
        self.assertEqual(self.G_cart.to_string((1, 2)), "[|1; 2|]")

    def test_cartesian_power_sparse_add_and_compare(self):
        """Verify addition and comparison of sparse vectors"""
        x = [(0, 2), (2, 5)]
        y = [(1, 3)]
        added = self.G_sparse.add(x, y)
        self.assertEqual(sorted(added), [(0, 2), (1, 3), (2, 5)])
        negated = self.G_sparse.neg(x)
        self.assertEqual(negated, [(0, -2), (2, -5)])

        self.assertEqual(self.G_sparse.compare([(0, 1)], [(0, 2)]), -1)
        self.assertEqual(self.G_sparse.compare([(0, 2)], [(0, 1)]), 1)
        self.assertEqual(self.G_sparse.compare([(0, 1)], [(0, 1)]), 0)
        # zero and to_string
        self.assertEqual(self.G_sparse.zero(), [])
        self.assertTrue(self.G_sparse.to_string([(0, 2)]).startswith("[("))

    def test_cartesian_power_sparse_from_list(self):
        """Verify sorting and cleaning of from_list"""
        lst = [(2, 5), (0, 1), (1, 0)]
        res = self.G_sparse.from_list(lst)

        self.assertEqual(res, [(0, 1), (2, 5)])

    # === Consistency tests ===
    def test_zero_elements(self):
        """Check neutral elements"""
        self.assertEqual(self.G_int.add(self.G_int.zero(), 5), 5)
        self.assertEqual(self.G_float.add(self.G_float.zero(), 5.0), 5.0)
        self.assertEqual(self.G_cart.zero(), (0, 0))

    def test_sum_method_in_additive_group(self):
        """Verify iterated sum in AdditiveGroup"""
        nums = [1, 2, 3, 4]
        res = self.G_int.sum(nums)
        self.assertEqual(res, 10)

    def test_to_string_methods(self):
        """Verify to_string in various groups"""
        self.assertIsInstance(self.G_int.to_string(5), str)
        self.assertIn("5", self.G_cart.to_string((5, 6)))
        self.assertIn("3", self.G_float.to_string(3.0))


if __name__ == "__main__":
    unittest.main()
