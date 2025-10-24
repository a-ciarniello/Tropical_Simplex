from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import sys
from enum import Enum
import numeric, group, linear_prog, parser
from simplet import Simplet
from typing import Optional, List, Tuple, TextIO
from perturbed_lp import PerturbedLP


class LinearProg:
    def __init__(self, group: group.OrderedGroup):
        self._impl = linear_prog.LinearProg(group)
    def init(self, var_names, nb_var, obj, ineq):
        return self._impl.init(var_names, nb_var, obj, ineq)

# Use the LP class from linear_prog module
LP = linear_prog.LP

# ---------- Esiti ----------
class Solution(Enum):
    INFEASIBLE = "Infeasible"
    UNBOUNDED = "Unbounded"
    OPTIMUM = "Optimum"

# ---------- main porting ----------
def run_main(input_filename: str,
             verbose: bool = False,
             log_file_name: str = "log") -> Tuple[Solution, np.ndarray]:


    lexbuf = parser.lexer_from_file(input_filename)


    try:
        numeric_name, var_names = parser.lexer_header(lexbuf)
    except Exception as e:
        raise RuntimeError(f"lexer error while reading header: {e}") from e

 
    Num = numeric.get(numeric_name)
    Parse = parser.Parser(Num)

    try:
        obj, ineq, basic_point_list, is_maximize = Parse.main(lexbuf) 
    except Exception as e:
        raise RuntimeError(f"parse error: {e}") from e

    print("input file parsed")
    print(f"Semiring: {numeric_name}")
    print(f"Objective: {'maximize' if is_maximize else 'minimize'}")

  
    G = group.GroupFromNumeric(Num)

    LPmod = LinearProg(G)

    nb_var = len(var_names)

    var_names_array: List[str] = [""] * nb_var
    for name, idx in var_names.items():
        var_names_array[idx] = name
    var_names_fun = lambda j: var_names_array[j]

 
    lp: LP = LPmod.init(var_names_fun, nb_var, obj, ineq)


    log: Optional[TextIO] = open(log_file_name, "w", encoding="utf-8") if verbose else None
    try:
        if log:
            print("\n parsed successfully\n", file=log)
        lp.pretty_print()

        basic_point_given = basic_point_list.size > 0 if hasattr(basic_point_list, 'size') else len(basic_point_list) > 0

        if basic_point_given:
            # ----- Caso: base fornita -> Fase II diretta -----
            basic_point = np.array(basic_point_list)
            Simp = Simplet(LPmod._impl)
            phaseII = Simp.init(lp, basic_point)

            print("applying tropical simplex method with given input basic point")
            if log: print("\n------------------\napplying tropical simplex method with given input basic point\n", file=log)

            # Select the appropriate pivot rule based on the objective direction
            pivot_rule = Simp.get_pivot_rule_for_objective(maximize=is_maximize)
            Simp.solve(phaseII, pivot_rule, log)

            opt = Simp.basic_point(phaseII)
            opt_projected = opt.copy()  # Copia l'array NumPy
            return (Solution.OPTIMUM, opt_projected)

        else:
            # ----- Caso: nessuna base -> Fase I + Fase II -----
            PertLP = PerturbedLP(LPmod._impl)
            phaseI_lp, basic_point = PertLP.phaseI(lp)

            if log:
                print("\n------------------\nphaseI lp:\n", file=log)
            phaseI_lp.pretty_print()

            SimpletI = Simplet(LPmod._impl) 
            phaseI = SimpletI.init(phaseI_lp, basic_point)

            print("solving phaseI")
            if log: print("\n------------------\ncall simplex method on phaseI lp\n", file=log)
            # Phase I is always a minimization problem (finding feasibility)
            pivot_rule_phaseI = SimpletI.get_pivot_rule_for_objective(maximize=False)
            SimpletI.solve(phaseI, pivot_rule_phaseI, log)

            phaseI_opt_basic_point = SimpletI.basic_point(phaseI)
            feasible = SimpletI.basis_contains(phaseI, PertLP.phaseI_infeasibility_var_lower_bound_row(lp))

            if not feasible:
                return (Solution.INFEASIBLE, np.array([]))

            # ---- Fase II ----
            print("solving phaseII")
            phaseII_lp = PertLP.phaseII(lp)
            if log:
                print("\n---------\nphaseII lp:\n", file=log)
            phaseII_lp.pretty_print()

            phaseII_basic_point = phaseI_opt_basic_point[:lp.dim()] 
            SimpletII = Simplet(LPmod._impl)
            phaseII = SimpletII.init(phaseII_lp, phaseII_basic_point)

            if log: print("\n------------------\ncall simplex method on phaseII lp\n", file=log)
            # Phase II uses the original objective direction
            pivot_rule_phaseII = SimpletII.get_pivot_rule_for_objective(maximize=is_maximize)
            SimpletII.solve(phaseII, pivot_rule_phaseII, log)

            ub_row = PertLP.phaseII_upperbound_row(lp)
            basis_has_inf_plane = SimpletII.basis_contains(phaseII, ub_row)
            infinity_plane_red_cost = SimpletII.red_cost(phaseII, ub_row)

            if not basis_has_inf_plane:
                opt = SimpletII.basic_point(phaseII)
                opt_projected = np.array([PertLP.project(x) for x in opt])
                return (Solution.OPTIMUM, opt_projected)
            else:
                # red_cost: Some(Pos, _) -> unbounded; None -> bounded; Some(Neg, _) -> assert false
                unbounded = (infinity_plane_red_cost is not None and infinity_plane_red_cost[0] == "Pos")
                if unbounded:
                    return (Solution.UNBOUNDED, np.array([]))
                else:
                    if log:
                        print(f" ===\n last pivot on {ub_row} to obtain a point with finite entries\n ===\n", file=log)
                    SimpletII.pivot(phaseII, ub_row)
                    SimpletII.print_status(phaseII, log)
                    opt = SimpletII.basic_point(phaseII)
                    opt_projected = np.array([PertLP.project(x) for x in opt])
                    return (Solution.OPTIMUM, opt_projected)
    finally:
        if log: log.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Test di base se non vengono forniti argomenti
        print("Uso: python main.py <file.lp> [--verbose]")
        print("\nTest di base del gruppo tropicale:")
        Num = numeric.get("tropical_min_plus")
        G = group.GroupFromNumeric(Num)
        print("Zero:", G.zero())
        print("Trop Add(2,5):", G.add(2, 5))
        print("Compare(2,5):", G.compare(2, 5))
        print("Trop Mux(2,5):", G.mul(2, 5))
    else:
        # Risolve il problema LP fornito
        input_file = sys.argv[1]
        verbose = "--verbose" in sys.argv or "-v" in sys.argv
        
        try:
            solution, point = run_main(input_file, verbose)

            print("\n--- Final Result ---")
            print(f"\nResult: {solution.value}")
            if solution == Solution.OPTIMUM:
                print("Optimal point:")
                for i, val in enumerate(point):
                    print(f"  x{i}: {val}")
            elif solution == Solution.INFEASIBLE:
                print("The problem is infeasible")
            elif solution == Solution.UNBOUNDED:
                print("The problem is unbounded")
                
        except Exception as e:
            print(f"Error during solving: {e}")
            import traceback
            traceback.print_exc()
