import numpy as np
import numeric
import group
import linear_prog

def test_linear_prog():

    # === 1. Numeric group definition ===
    Num = numeric.TropicalNumericMinPlus()
    G = group.GroupFromNumeric(Num)

    # === 2. Variable names definition ===
    def var_names(i):
        return ["x", "y"][i]

    # === 3. Construction of a small LP problem ===

    # minimize x + 3
    objective = [
        ((linear_prog.ColKind.VAR, 0), linear_prog.Sign.POS, 0.0),  # x
        ((linear_prog.ColKind.AFFINE, None), linear_prog.Sign.POS, 3.0),
    ]

    # Constraint: x + 2 <= y + 4
    ineqs: list[list[tuple[linear_prog.ColIndex, linear_prog.Sign, float]]] = [[
        ((linear_prog.ColKind.VAR, 0), linear_prog.Sign.POS, 2.0),     # x + 2
        ((linear_prog.ColKind.VAR, 1), linear_prog.Sign.NEG, 4.0),     # y + 4
    ]]

    # === 4. LP instance creation ===
    LP_mod = linear_prog.LinearProg(G)
    lp = LP_mod.init(var_names, nb_var=2, objective=objective, ineqs=ineqs)

    # === 5. Candidate point ===
    point = np.array([0.0, 1.0])  # x=0, y=1

    # === 6. Main methods test ===
    print("=== LP structure ===")
    lp.pretty_print()

    print("\n=== Slack args for the inequality ===")
    slack_args = lp.compute_slack_args((linear_prog.RowKind.INEQ, 0), point)
    print(slack_args)

    print("\n=== Feasibility check ===")
    feasible = lp.is_point_feasible(point)
    print("Feasible point?", feasible)

    print("\n=== Dimensions ===")
    print(f"dim = {lp.dim()}, nb_ineq = {lp.nb_ineq()}")

if __name__ == "__main__":
    test_linear_prog()
