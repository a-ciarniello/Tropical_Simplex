# === test_simplet.py ===
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

        # === LP Definition ===
        # minimize x + 2
        # s.t. x + 1 <= y + 1
        #      y + 1 <= z + 1
        #      z + 1 <= x + 1
        def var_names(i): return ["x", "y", "z"][i]

        objective = [
            ((linear_prog.ColKind.VAR, 0), linear_prog.Sign.POS, 0.0),
            ((linear_prog.ColKind.AFFINE, None), linear_prog.Sign.POS, 2.0),
        ]

        ineqs: list[list[tuple[linear_prog.ColIndex, linear_prog.Sign, float]]] =[
            [
                ((linear_prog.ColKind.VAR, 0), linear_prog.Sign.POS, 1.0),
                ((linear_prog.ColKind.VAR, 1), linear_prog.Sign.NEG, 1.0),
            ],
            [
                ((linear_prog.ColKind.VAR, 1), linear_prog.Sign.POS, 1.0),
                ((linear_prog.ColKind.VAR, 2), linear_prog.Sign.NEG, 1.0),
            ],
            [
                ((linear_prog.ColKind.VAR, 2), linear_prog.Sign.POS, 1.0),
                ((linear_prog.ColKind.VAR, 0), linear_prog.Sign.NEG, 1.0),
            ],
        ]

        cls.lp = cls.LPmod.init(var_names, 3, objective, ineqs)
        cls.point = np.array([0.0, 0.0, 0.0])  # symmetric point

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
            self.assertIsInstance(instance, Simplet.SimpletInstance)
            self.assertEqual(len(instance.lp.ineqs), 3)
            self.assertTrue(hasattr(instance, "tangent_digraph"))
        except ValueError as e:
            # can fail if the point is not basic, but the module must respond in a managed way
            self.assertIn("punto", str(e).lower())

    def test_simplet_instance_fields(self):
        """Verify that SimpletInstance has all main attributes"""
        simp = Simplet(self.LPmod)
        try:
            inst = simp.init(self.lp, self.point)
        except ValueError:
            inst = Simplet.SimpletInstance(self.lp, self.point)

        attrs = [
            "lp", "point", "tangent_digraph", "arg_slacks",
            "ineq_status", "max_permutation", "reduced_costs", "dual_slacks"
        ]
        for a in attrs:
            self.assertTrue(hasattr(inst, a))

    def test_compute_arg_slacks_pos(self):
        """Verify arg_slacks_pos computation"""
        simp = Simplet(self.LPmod)
        inst = Simplet.SimpletInstance(self.lp, self.point)
        simp._compute_arg_slacks_pos(inst)
        self.assertIsInstance(inst.arg_slacks, list)
        self.assertEqual(len(inst.arg_slacks), self.lp.nb_ineq())

    def test_compute_reduced_costs(self):
        """Verify reduced costs computation"""
        simp = Simplet(self.LPmod)
        inst = Simplet.SimpletInstance(self.lp, self.point)
        inst.tangent_digraph = tangent_digraph.TangentDigraph.compute(self.lp, self.point)
        simp._compute_reduced_costs(inst)
        self.assertTrue(isinstance(inst.reduced_costs, list))
        self.assertEqual(len(inst.reduced_costs), self.lp.nb_ineq())

    def test_compute_max_permutation(self):
        """Verify maximizing permutation computation"""
        simp = Simplet(self.LPmod)
        inst = Simplet.SimpletInstance(self.lp, self.point)
        inst.tangent_digraph = tangent_digraph.TangentDigraph.compute(self.lp, self.point)
        simp._compute_max_permutation(inst)
        self.assertEqual(len(inst.max_permutation), self.lp.dim())

    def test_bland_rule_and_basis_contains(self):
        """Verify bland_rule and basis_contains"""
        simp = Simplet(self.LPmod)
        inst = Simplet.SimpletInstance(self.lp, self.point)
        inst.tangent_digraph = tangent_digraph.TangentDigraph.compute(self.lp, self.point)
        simp._compute_reduced_costs(inst)
        result = simp.bland_rule(inst)
        self.assertIn(result, [None, 0, 1, 2])
        # basis_contains
        for i in range(self.lp.nb_ineq()):
            _ = simp.basis_contains(inst, i)
    def test_print_status_and_reduced_costs(self):
        """Verify status and reduced costs printing"""

        simp = Simplet(self.LPmod)
        inst = Simplet.SimpletInstance(self.lp, self.point)
        simp._compute_arg_slacks_pos(inst)
        simp._compute_reduced_costs(inst)
        try:
            simp.print_status(inst, sys.stdout)  # print_status can handle None, but we use stdout to see output
            simp.print_reduced_costs(inst, sys.stdout)  # print_reduced_costs requires a valid TextIO
        except Exception as e:
            self.fail(f"Printing failed with exception: {e}")


if __name__ == "__main__":
    unittest.main()
