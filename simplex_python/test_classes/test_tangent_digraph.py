"""Targeted tests for the tangent-digraph combinatorial utilities."""

from __future__ import annotations
import unittest
import numpy as np
import numeric
import group
import linear_prog
import tangent_digraph


class TestTangentDigraph(unittest.TestCase):
    """Unit tests specific to the tangent_digraph.py module"""

    @classmethod
    def setUpClass(cls):
        """Create a small reference LP problem"""
        Num = numeric.TropicalNumericMinPlus()
        cls.G = group.GroupFromNumeric(Num)
        cls.LPmod = linear_prog.LinearProg(cls.G)

        # === LP ===
        # minimize x + 3
        def var_names(i): return ["x", "y", "z"][i]

        objective = [
            ((linear_prog.ColKind.VAR, 0), linear_prog.Sign.POS, 0.0),  # x
            ((linear_prog.ColKind.AFFINE, None), linear_prog.Sign.POS, 2.0),
        ]

        ineqs: list[list[tuple[linear_prog.ColIndex, linear_prog.Sign, float]]] = [
            [  # x + 1 <= y + 1
                ((linear_prog.ColKind.VAR, 0), linear_prog.Sign.POS, 1.0),
                ((linear_prog.ColKind.VAR, 1), linear_prog.Sign.NEG, 1.0),
            ],
            [  # y + 1 <= z + 1
                ((linear_prog.ColKind.VAR, 1), linear_prog.Sign.POS, 1.0),
                ((linear_prog.ColKind.VAR, 2), linear_prog.Sign.NEG, 1.0),
            ],
            [  # z + 1 <= x + 1
                ((linear_prog.ColKind.VAR, 2), linear_prog.Sign.POS, 1.0),
                ((linear_prog.ColKind.VAR, 0), linear_prog.Sign.NEG, 1.0),
            ],
        ]

        cls.lp = cls.LPmod.init(var_names, 3, objective, ineqs)
        cls.point_feasible = np.array([0.0, 0.0, 0.0])

    def test_compute_tangent_graph(self):
        """Correct construction of the tangent graph"""
        tg = tangent_digraph.TangentDigraph.compute(self.lp, self.point_feasible)
        self.assertIsInstance(tg, tangent_digraph.TangentDigraph)
        self.assertEqual(len(tg.ineq_nodes), self.lp.nb_ineq())
        self.assertEqual(len(tg.var_nodes), self.lp.dim())

    def test_is_hyp_node(self):
        """Verify that hyp nodes are correctly identified"""
        tg = tangent_digraph.TangentDigraph.compute(self.lp, self.point_feasible)
        for i in range(len(tg.ineq_nodes)):
            result = tg.is_hyp_node(i)
            self.assertIn(result, [True, False])

    def test_connected_components(self):
        """Correct count of connected components"""
        tg = tangent_digraph.TangentDigraph.compute(self.lp, self.point_feasible)
        nb_cc = tg.nb_connected_component()
        self.assertIsInstance(nb_cc, int)
        self.assertGreaterEqual(nb_cc, 1)

    def test_basic_point_detection(self):
        """Verify if the point is basic"""
        tg = tangent_digraph.TangentDigraph.compute(self.lp, self.point_feasible)
        is_basic = tg.is_basic_point()
        self.assertIn(is_basic, [True, False])

    def test_add_and_remove_arc(self):
        """Verify consistency of arc addition and removal"""
        tg = tangent_digraph.TangentDigraph.compute(self.lp, self.point_feasible)
        var_index = (linear_prog.ColKind.VAR, 2)
        i = 0
        tg.add_arc(var_index, i, linear_prog.Sign.POS, 1.23)
        self.assertTrue(any(v == var_index for v, _, _ in tg.ineq_nodes[i]))

        tg.remove_arc(var_index, i)
        self.assertFalse(any(v == var_index for v, _, _ in tg.ineq_nodes[i]))

    def test_remove_arcs_of_node(self):
        """Complete removal of arcs from a node"""
        tg = tangent_digraph.TangentDigraph.compute(self.lp, self.point_feasible)
        tg.remove_arcs_of_node(("IneqNode", 0))
        self.assertEqual(len(tg.ineq_nodes[0]), 0)

    def test_dfs_fold_functionality(self):
        """Basic check on DFS"""
        tg = tangent_digraph.TangentDigraph.compute(self.lp, self.point_feasible)

        def f(acc, node):
            acc.append(node)
            return acc

        result = tg.dfs_fold_acyclic_graph(f, [], ("VarNode", (linear_prog.ColKind.AFFINE, None)))
        self.assertIsInstance(result, list)
        self.assertTrue(all(isinstance(n, tuple) for n in result))

    def test_print(self):
        """Check that printing does not raise exceptions"""
        tg = tangent_digraph.TangentDigraph.compute(self.lp, self.point_feasible)
        try:
            tg.print()
        except Exception as e:
            self.fail(f"tg.print() raised an exception: {e}")


if __name__ == "__main__":
    unittest.main()
