from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import sys
import os
from datetime import datetime
from enum import Enum
from simplet import Simplet
import numeric, group, linear_prog, parser
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
    print(f"\nVariables: {var_names}")
    print(f"\nSemiring: {numeric_name}")
    print(f"\nObjective: {'maximize' if is_maximize else 'minimize'}\n")

  
    G = group.GroupFromNumeric(Num)

    LPmod = LinearProg(G)

    nb_var = len(var_names)

    var_names_array: List[str] = [""] * nb_var
    for name, idx in var_names.items():
        var_names_array[idx] = name
    var_names_fun = lambda j: var_names_array[j]

 
    lp: LP = LPmod.init(var_names_fun, nb_var, obj, ineq)

    # Create log file name with current date and time
    timestamp = datetime.now().strftime("%Y_%m_%d--%H.%M.%S")
    log_file_name = f"{log_file_name}_{timestamp}.txt"

    os.makedirs(os.path.dirname("Logs/"), exist_ok=True)
    save_path = os.path.join("Logs/", log_file_name)

    log: Optional[TextIO] = open(save_path, "w", encoding="utf-8")
    try:
        if log:
            print(f"Input file: {input_filename}\n\nparsed successfully\n", file=log)
        lp.pretty_print()

        basic_point_given = basic_point_list.size > 0 if hasattr(basic_point_list, 'size') else len(basic_point_list) > 0


        if basic_point_given:
            # ----- Caso: base fornita -> tentativo Fase II diretta, con fallback a Fase I -----
            try:
                basic_point = np.array(basic_point_list)
                Simp = Simplet(LPmod._impl)
                phaseII = Simp.init(lp, basic_point)

                print("applying tropical simplex method with given input basic point")
                if log:
                    print("\n------------------\napplying tropical simplex method with given input basic point\n", file=log)

                pivot_rule = Simp.get_pivot_rule_for_objective(maximize=is_maximize)
                Simp.solve(phaseII, pivot_rule, log, max_iterations=1000)

                opt = Simp.basic_point(phaseII)
                opt_projected = opt.copy()
                return (Solution.OPTIMUM, opt_projected)

            except ValueError as e:
                print(f"Given basic point rejected ({e}), falling back to Phase I.")
                if log:
                    print(f"\n------------------\nGiven basic point rejected ({e}), falling back to Phase I.\n", file=log)
                basic_point_given = False



        if not basic_point_given:
            # ----- Caso: nessuna base -> Fase I + Fase II -----
            PertLP = PerturbedLP(LPmod._impl)
            phaseI_lp, basic_point = PertLP.phaseI(lp)

            print("\n------------------\nInitial basic point for phaseI: \n", basic_point)
        
            if log:
                print("\n------------------\nphaseI lp:", file=log)

            print("\n------------------\n \nphaseI lp constructed:")
            phaseI_lp.pretty_print()

            SimpletI = Simplet(PertLP.LP_pert_mod) 
            phaseI = SimpletI.init(phaseI_lp, basic_point)

            print("solving phaseI")
            if log: print("\n------------------\ncall simplex method on phaseI lp\n", file=log)


            pivot_rule_phaseI = SimpletI.get_pivot_rule_for_objective(maximize=False)
            
            try:
                SimpletI.solve(phaseI, pivot_rule_phaseI, log, max_iterations=50)
            except RuntimeError as e:
                print(f"\n*** RuntimeError caught: {e}")
                raise
            except Exception as e:
                print(f"\n*** Unexpected error: {type(e).__name__}: {e}")
                raise

            print("phaseI solved")

            phaseI_opt_basic_point = SimpletI.basic_point(phaseI)
            feasible = SimpletI.basis_contains(phaseI, PertLP.phaseI_infeasibility_var_lower_bound_row(lp))

            print(f"phaseI optimal basic point: \n {phaseI_opt_basic_point}")
            print(f"feasibility from phaseI: {feasible}")


            if not feasible:
                return (Solution.INFEASIBLE, np.array([]))

            # ---- Fase II ----
            print("solving phaseII")
            phaseII_lp = PertLP.phaseII(lp)
            if log:
                print("\n---------\nphaseII lp:\n", file=log)
            phaseII_lp.pretty_print()

            phaseII_basic_point = PertLP.phaseII_initial_point_from_phaseI_opt(lp, phaseI_opt_basic_point)
            SimpletII = Simplet(LPmod._impl)
            phaseII = SimpletII.init(phaseII_lp, phaseII_basic_point)

            if log: print("\n------------------\ncall simplex method on phaseII lp\n", file=log)
            # Phase II uses the original objective direction
            pivot_rule_phaseII = SimpletII.get_pivot_rule_for_objective(maximize=is_maximize)
            SimpletII.solve(phaseII, pivot_rule_phaseII, log, max_iterations=1000)


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

    # Safeguard: all code paths above should return; if not, raise.
    raise RuntimeError("run main terminated without producing a result")

if __name__ == "__main__":
    if len(sys.argv) < 2:

        ## Execute a predefined test problem
        print("\nNo input file provided, running predefined test problem 'non_generic.lp'\n")

        input_file = "simplex_python/problems/non_generic.lp"
        verbose = False

        try:
            solution, point = run_main(input_file)

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


    else:
        # Risolve il problema LP fornito
        input_file = sys.argv[1]
        
        try:
            solution, point = run_main(input_file)

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
