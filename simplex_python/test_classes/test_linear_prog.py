"""Unit tests validating the canonical LP container implementation."""

import unittest
import numpy as np
import numeric
import group
import linear_prog

class TestLinearProg(unittest.TestCase):
    """Validate dimension bookkeeping, slack computations, and feasibility checks."""

    def setUp(self):
        """Build a minimal tropical LP used across every test case."""
        # 1. Define the numeric and group structure (Min-Plus algebra)
        Num = numeric.TropicalNumericMinPlus()
        G = group.GroupFromNumeric(Num)

        # 2. Define variable names
        def var_names(i):
            return ["x", "y"][i]

        # 3. Define the LP problem components
        # Objective: minimize x + 3 (tropical) -> min(x, 3)
        objective = [
            ((linear_prog.ColKind.VAR, 0), linear_prog.Sign.POS, 0.0),
            ((linear_prog.ColKind.AFFINE, None), linear_prog.Sign.POS, 3.0),
        ]

        # Constraint: x + 2 <= y + 4 (tropical)
        ineqs: list[list[tuple[linear_prog.ColIndex, linear_prog.Sign, float]]] = [[
            ((linear_prog.ColKind.VAR, 0), linear_prog.Sign.POS, 2.0),
            ((linear_prog.ColKind.VAR, 1), linear_prog.Sign.NEG, 4.0),
        ]]
        
        # 4. Create the LP instance and the test point
        LP_mod = linear_prog.LinearProg(G)
        self.lp = LP_mod.init(var_names, nb_var=2, objective=objective, ineqs=ineqs)
        self.point = np.array([0.0, 1.0])  # Corresponds to x=0, y=1

    def test_dimensions(self):
        """LP should report both dimension and inequality count accurately."""
        self.assertEqual(self.lp.dim(), 2, "LP dimension mismatch")
        self.assertEqual(self.lp.nb_ineq(), 1, "Unexpected number of inequalities")

    def test_compute_slack_args(self):
        """Slack arguments should capture the minimum-attaining row entries."""
        slack_args = self.lp.compute_slack_args((linear_prog.RowKind.INEQ, 0), self.point)
        
        # The expected result is the information of the term that achieved the minimum
        expected_args = [
            ((linear_prog.ColKind.VAR, 0), linear_prog.Sign.POS, 2.0)
        ]
        
        self.assertEqual(slack_args, expected_args, "Unexpected slack-arg selection")

    def test_point_feasibility(self):
        """Point should satisfy the inequality when the minimum is positive-sided."""
        is_feasible = self.lp.is_point_feasible(self.point)
        self.assertTrue(is_feasible, "Reference point expected to be feasible")

    def test_point_infeasibility(self):
        """Point should be rejected when only negative terms realize the minimum."""
        infeasible_point = np.array([4.0, 1.0])
        is_feasible = self.lp.is_point_feasible(infeasible_point)
        self.assertFalse(is_feasible, "Point incorrectly marked as feasible")


if __name__ == "__main__":
    # This makes the test script runnable
    unittest.main()