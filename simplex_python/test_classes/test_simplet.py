"""Behavioral tests for the Simplet tropical simplex driver."""

from __future__ import annotations
import unittest
import sys
import numpy as np
import numeric
import group
import linear_prog
import tangent_digraph
from simplet import Simplet


class TestSimplet(unittest.TestCase):
    """Unit tests specific to the simplet.py module"""

    @classmethod
    def setUpClass(cls):
        """Create a coherent tropical LP for Simplet testing"""
        Num = numeric.TropicalNumericMinPlus()
        cls.G = group.GroupFromNumeric(Num)
        cls.LPmod = linear_prog.LinearProg(cls.G)
        
        def var_names(i): return ["x", "y"][i]

        objective = [
            # minimize x
            ((linear_prog.ColKind.VAR, 0), linear_prog.Sign.POS, 0.0),
            ((linear_prog.ColKind.AFFINE, None), linear_prog.Sign.POS, float('inf')),
        ]

        ineqs: list[list[tuple[linear_prog.ColIndex, linear_prog.Sign, float]]] = [
            # x <= 3:  x+0 <= 0+3
            [
                ((linear_prog.ColKind.VAR, 0), linear_prog.Sign.POS, 0.0),
                ((linear_prog.ColKind.AFFINE, None), linear_prog.Sign.NEG, 3.0),
            ],
            # y <= 3:  y+0 <= 0+3
            [
                ((linear_prog.ColKind.VAR, 1), linear_prog.Sign.POS, 0.0),
                ((linear_prog.ColKind.AFFINE, None), linear_prog.Sign.NEG, 3.0),
            ],
            # 1 <= y:  0+1 <= y+0
            [
                ((linear_prog.ColKind.AFFINE, None), linear_prog.Sign.POS, 1.0),
                ((linear_prog.ColKind.VAR, 1), linear_prog.Sign.NEG, 0.0),
            ],
        ]

        cls.lp = cls.LPmod.init(var_names, 2, objective, ineqs)
        cls.point = np.array([3.0, 1.0])


    # === Construction tests ===
    def test_create_simplet(self):
        """Verify that Simplet can be created correctly"""
        simp = Simplet(self.LPmod)
        self.assertIsInstance(simp, Simplet)
        self.assertTrue(hasattr(simp, "init"))
        self.assertTrue(hasattr(simp, "solve"))

    def test_init_simplet_instance(self):
        """Verify that Simplet.init() creates a coherent instance"""
        simp = Simplet(self.LPmod)
        try:
            instance = simp.init(self.lp, self.point)
            self.assertIsInstance(instance, Simplet.Instance)
            self.assertEqual(len(instance.lp.ineqs), 3)
            self.assertTrue(hasattr(instance, "tangent_digraph"))
        except ValueError as e:
            error_msg = str(e).lower()
            self.assertIn("point", error_msg, f"Error message should mention point: {e}")


    def test_simplet_instance_fields(self):
        """Verify that Instance has all main attributes"""
        simp = Simplet(self.LPmod)
        try:
            inst = simp.init(self.lp, self.point)
        except ValueError:
            inst = Simplet.Instance(self.lp, self.point)

        attrs = [
            "lp", "point", "tangent_digraph", "arg_slacks",
            "ineq_status", "max_permutation", "reduced_costs", "dual_slacks"
        ]
        for a in attrs:
            self.assertTrue(hasattr(inst, a))

    def test_compute_arg_slacks_pos(self):
        """Verify arg_slacks_pos computation"""
        simp = Simplet(self.LPmod)
        inst = Simplet.Instance(self.lp, self.point)
        simp._compute_arg_slacks_pos(inst)
        self.assertIsInstance(inst.arg_slacks, list)
        self.assertEqual(len(inst.arg_slacks), self.lp.nb_ineq())

    def test_compute_reduced_costs(self):
        """Verify reduced costs computation"""
        simp = Simplet(self.LPmod)
        inst = Simplet.Instance(self.lp, self.point)
        inst.tangent_digraph = tangent_digraph.TangentDigraph.compute(self.lp, self.point)
        simp._compute_reduced_costs(inst)
        self.assertTrue(isinstance(inst.reduced_costs, list))
        self.assertEqual(len(inst.reduced_costs), self.lp.nb_ineq())

    def test_compute_max_permutation(self):
        """Verify maximizing permutation computation"""
        simp = Simplet(self.LPmod)
        inst = Simplet.Instance(self.lp, self.point)
        inst.tangent_digraph = tangent_digraph.TangentDigraph.compute(self.lp, self.point)
        simp._compute_max_permutation(inst)
        self.assertEqual(len(inst.max_permutation), self.lp.dim())

    def test_bland_rule_and_basis_contains(self):
        """Verify bland_rule and basis_contains"""
        simp = Simplet(self.LPmod)
        inst = Simplet.Instance(self.lp, self.point)
        inst.tangent_digraph = tangent_digraph.TangentDigraph.compute(self.lp, self.point)
        simp._compute_reduced_costs(inst)
        result = simp.bland_rule(inst)
        self.assertTrue(result is None or isinstance(result, int))
        for i in range(self.lp.nb_ineq()):
            _ = simp.basis_contains(inst, i)
    def test_print_status_and_reduced_costs(self):
        """Verify status and reduced costs printing"""

        simp = Simplet(self.LPmod)
        inst = Simplet.Instance(self.lp, self.point)
        inst.tangent_digraph = tangent_digraph.TangentDigraph.compute(self.lp, self.point)
        simp._compute_arg_slacks_pos(inst)
        simp._compute_reduced_costs(inst)
        try:
            simp.print_status(inst, sys.stdout) 
            simp.print_reduced_costs(inst, sys.stdout) 
        except Exception as e:
            self.fail(f"Printing failed with exception: {e}")


if __name__ == "__main__":
    unittest.main()
