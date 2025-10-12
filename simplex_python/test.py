import numeric, group, linear_prog
import numpy as np
from typing import List, Tuple, Any

Num = numeric.get("tropical_min_plus")
G = group.GroupFromNumeric(Num)
LPmod = linear_prog.LinearProg(G)

# Use explicit type annotations to help Pylance
objective: List[Tuple[linear_prog.ColIndex, linear_prog.Sign, Any]] = [
    ((linear_prog.ColKind.VAR, 0), linear_prog.Sign.POS, 3.0),
    ((linear_prog.ColKind.AFFINE, None), linear_prog.Sign.POS, 4.0)
]
ineqs: List[List[Tuple[linear_prog.ColIndex, linear_prog.Sign, Any]]] = [
    [((linear_prog.ColKind.VAR, 0), linear_prog.Sign.POS, 3.0),
     ((linear_prog.ColKind.VAR, 1), linear_prog.Sign.NEG, 5.0)]
]

lp = LPmod.init(lambda j: f"x{j}", 2, objective, ineqs)
lp.pretty_print()

point = np.array([0.0, 5.0])
print("Feasible:", lp.is_point_feasible(point))
