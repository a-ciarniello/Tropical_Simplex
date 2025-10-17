import unittest
import numpy as np
import numeric
import group
import linear_prog

class TestLinearProg(unittest.TestCase):
    """
    Test suite per il modulo linear_prog.
    
    Questa classe verifica le funzionalità principali della classe LP,
    inclusa la sua creazione, il calcolo delle dimensioni, il calcolo dello slack
    e i controlli di fattibilità.
    I commenti nel codice sono in inglese per coerenza.
    """

    def setUp(self):
        """
        Set up a common testing environment before each test method is run.
        This creates a small, consistent LP instance for all tests.
        """
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
        """
        Verify that the LP reports its dimensions correctly.
        """
        self.assertEqual(self.lp.dim(), 2, "La dimensione dell'LP dovrebbe essere 2")
        self.assertEqual(self.lp.nb_ineq(), 1, "L'LP dovrebbe avere 1 disequazione")

    def test_compute_slack_args(self):
        """
        Test the computation of slack arguments for a given inequality and point.
        
        For the inequality x+2 <= y+4 and point x=0, y=1:
        - The positive term's value is x + 2 = 0 + 2 = 2.0
        - The negative term's value is y + 4 = 1 + 4 = 5.0
        The minimum is 2.0, achieved by the positive term.
        """
        slack_args = self.lp.compute_slack_args((linear_prog.RowKind.INEQ, 0), self.point)
        
        # The expected result is the information of the term that achieved the minimum
        expected_args = [
            ((linear_prog.ColKind.VAR, 0), linear_prog.Sign.POS, 2.0)
        ]
        
        self.assertEqual(slack_args, expected_args, "Gli argomenti dello slack non sono calcolati correttamente")

    def test_point_feasibility(self):
        """
        Check if a given point is correctly identified as feasible.
        
        A point is feasible if the minimum slack is not achieved exclusively
        by negative terms. In our case, the minimum is achieved by a positive
        term, so the point is feasible.
        """
        is_feasible = self.lp.is_point_feasible(self.point)
        self.assertTrue(is_feasible, "Il punto [0.0, 1.0] dovrebbe essere fattibile")

    def test_point_infeasibility(self):
        """
        Check if an infeasible point is correctly identified.
        
        Let's test with point x=4, y=1.
        - Positive term: x + 2 = 4 + 2 = 6.0
        - Negative term: y + 4 = 1 + 4 = 5.0
        The minimum is 5.0, achieved by the negative term. Since only the
        negative term achieves the minimum, the point is infeasible.
        """
        infeasible_point = np.array([4.0, 1.0])
        is_feasible = self.lp.is_point_feasible(infeasible_point)
        self.assertFalse(is_feasible, "Il punto [4.0, 1.0] dovrebbe essere non fattibile")


if __name__ == "__main__":
    # This makes the test script runnable
    unittest.main()