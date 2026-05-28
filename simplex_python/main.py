from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import sys
import os
import numeric, group, linear_prog, parser, tangent_digraph
from datetime import datetime
from enum import Enum
from simplet import Simplet

from typing import Optional, List, Tuple, TextIO
from perturbed_lp import PerturbedLP, PinftyError


class LinearProg:
    def __init__(self, group: group.OrderedGroup):
        self._impl = linear_prog.LinearProg(group)
    def init(self, var_names, nb_var, obj, ineq):
        return self._impl.init(var_names, nb_var, obj, ineq)


LP = linear_prog.LP

# ---------- Results ----------
class Solution(Enum):
    INFEASIBLE = "Infeasible"
    UNBOUNDED = "Unbounded"
    OPTIMUM = "Optimum"


def Feasibility_check_for_Phase1(phase1_inst: Simplet.Instance, pert_lp: PerturbedLP, original_lp: linear_prog.LP):
    """Validate the Phase I optimum against the original LP (feasible + basic)."""

    projected_coords = []

    def _to_scalar(val):
        if isinstance(val, np.ndarray):
            if val.size == 1:
                return val.item()
            raise ValueError("Phase I projection produced vector-valued coordinate")
        return val

    for j in range(original_lp.dim()):
        try:
            projected = pert_lp.project(phase1_inst.point[j])
        except PinftyError:
            projected = float("inf")
        projected_coords.append(_to_scalar(projected))

    candidate_point = np.array(projected_coords, dtype=object)


    try:
        tg = tangent_digraph.TangentDigraph.compute(original_lp, candidate_point)
    except Exception as e:
        return False, f"Phase I projected point cannot build tangent digraph: {e}"

    if not original_lp.is_point_feasible(candidate_point):
        feasibility = False
        msg = "Phase I optimal point is not feasible for original LP."
    else:
        is_basic = tg.is_basic_point()
        feasibility = True
        msg = "Phase I optimal point is feasible for original LP."
        if not is_basic:
            msg += " (Warning: point is not basic; proceeding anyway.)"

    return feasibility, msg




# ---------- Main Script ----------
def run_main(input_filename: str,
             log_file_name: str = "log") -> Tuple[Solution, np.ndarray]:

    lexbuf = parser.lexer_from_file(input_filename)

    try:
        numeric_name, var_names, semiring = parser.lexer_header(lexbuf)
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
    print(f"\nSemiring: {'min-plus' if semiring == 'minplus' else 'max-plus'}")
    print(f"\nObjective: {'maximize' if is_maximize else 'minimize'}\n\n")

    G = group.GroupFromNumeric(Num)

    LPmod = LinearProg(G)

    nb_var = len(var_names)

    var_names_array: List[str] = [""] * nb_var

    for name, idx in var_names.items():
        var_names_array[idx] = name
    var_names_fun = lambda j: var_names_array[j]
 
    lp: LP = LPmod.init(var_names_fun, nb_var, obj, ineq)

    timestamp = datetime.now().strftime("%Y_%m_%d--%H.%M.%S")
    log_file_name = f"{log_file_name}_{timestamp}.txt"

    os.makedirs(os.path.dirname("D:\\Logs/"), exist_ok=True)
    save_path = os.path.join("D:\\","Logs/", log_file_name)

    log: Optional[TextIO] = open(save_path, "w", encoding="utf-8")
    try:
        if log:
            print(f"Input file: {input_filename}\n\nparsed successfully\n", file=log)
        lp.pretty_print()

        basic_point_given = basic_point_list.size > 0 if hasattr(basic_point_list, 'size') else len(basic_point_list) > 0

        if basic_point_given:
            try:
                basic_point = np.array(basic_point_list)
                Simp = Simplet(LPmod._impl)
                phaseII = Simp.init(lp, basic_point)

                print("applying tropical simplex method with given input basic point")

                print("\n------------------------\nsolving phaseII\n------------------------\n")
                print(f".\n.\n.\n.\n.\n.\n.\n.\n.")
                if log:
                    print("\n------------------\napplying tropical simplex method with given input basic point\n", file=log)

                pivot_rule = Simp.get_pivot_rule_for_objective(maximize=is_maximize)
                Simp.solve(phaseII, pivot_rule, log)

                opt = Simp.basic_point(phaseII)
                opt_projected = opt.copy()
                feasible_phaseII = lp.is_point_feasible(opt_projected)

                print("\n------------------------\nphaseII solved\n------------------------\n")
                print("\n==============================================\n")

                if feasible_phaseII:
                    return (Solution.OPTIMUM, opt_projected)

                print("Phase II point violates at least one constraint; retrying via Phase I.")
                if log:
                    print("Phase II point violates at least one constraint; retrying via Phase I.", file=log)
                basic_point_given = False

            except ValueError as e:
                print(f"Given basic point rejected ({e}), falling back to Phase I.")
                if log:
                    print(f"\n------------------\nGiven basic point rejected ({e}), falling back to Phase I.\n", file=log)
                basic_point_given = False
                
        if not basic_point_given:
            PertLP = PerturbedLP(LPmod._impl)
            phaseI_lp, basic_point = PertLP.phaseI(lp)

            if log:
                print(f"\n------------------\nphaseI lp:\n\n{phaseI_lp.to_string()}", file=log)

            SimpletI = Simplet(PertLP.LP_pert_mod) 
            phaseI = SimpletI.init(phaseI_lp, basic_point)

            print("\n==============================================\n\nPhaseI\n\n")

            print("\n------------------------\nsolving phaseI\n------------------------\n")
            print(f".\n.\n.\n.\n.\n.\n.\n.\n.")

            if log: print("\n------------------\ncall simplex method on phaseI lp\n------------------\n", file=log)

            pivot_rule_phaseI = SimpletI.get_pivot_rule_for_objective(maximize=False)

            try:
                SimpletI.solve(phaseI, pivot_rule_phaseI, log)
            except RuntimeError as e:
                print(f"\n*** RuntimeError caught: {e}")
                raise
            except Exception as e:
                print(f"\n*** Unexpected error: {type(e).__name__}: {e}")
                raise

            print("\n------------------------\nphaseI solved\n------------------------\n")

            phaseI_opt_basic_point = SimpletI.basic_point(phaseI)
            feasible, msg = Feasibility_check_for_Phase1(phaseI, PertLP, lp)


            print(f"\nFeasibility from phaseI: {feasible}\n{msg}\n")

            if not feasible:
                return (Solution.INFEASIBLE, np.array([]))        

            if feasible:
                print("\n==============================================\n\nPhaseII\n\n")

                phaseII_lp = PertLP.phaseII(lp)
                phaseII_basic_point = PertLP.phaseII_initial_point_from_phaseI_opt(lp, phaseI_opt_basic_point)    

                if log:
                    print(f"\n---------\nphaseII lp:\n{phaseII_lp.to_string()}\n", file=log)
                phaseII_lp.pretty_print()

                if not phaseII_lp.is_point_feasible(phaseII_basic_point):
                    print("\nPhase II initial point is not feasible - returning INFEASIBLE\n")
                    if log:
                        print("\nPhase II initial point is not feasible - returning INFEASIBLE\n", file=log)
                    return (Solution.INFEASIBLE, np.array([]))

                SimpletII = Simplet(PertLP.LP_pert_mod)
                phaseII = SimpletII.init(phaseII_lp, phaseII_basic_point, require_basic=False)

                print("\n------------------------\nsolving phaseII\n------------------------\n")
                print(f".\n.\n.\n.\n.\n.\n.\n.\n.")

                if log: print("\n------------------\ncall simplex method on phaseII lp\n------------------\n", file=log)

                pivot_rule_phaseII = SimpletII.get_pivot_rule_for_objective(maximize=is_maximize)
                SimpletII.solve(phaseII, pivot_rule_phaseII, log)

                ub_row = PertLP.phaseII_upperbound_row(lp)
                basis_has_inf_plane = SimpletII.basis_contains(phaseII, ub_row)
                infinity_plane_red_cost = SimpletII.red_cost(phaseII, ub_row)
                
                print("\n------------------------\nphaseII solved\n------------------------\n")
                print("\n==============================================\n")

                if not basis_has_inf_plane:
                    opt = SimpletII.basic_point(phaseII)
                    opt_projected = np.array([PertLP.project(x) for x in opt])
                    return (Solution.OPTIMUM, opt_projected)
                else:
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


    raise RuntimeError("run main terminated without producing a result")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Error: missing LP input file. usage: .\\main.py <problem.lp>")
        sys.exit(1)

    else:

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
