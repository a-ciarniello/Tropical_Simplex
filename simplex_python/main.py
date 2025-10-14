from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import sys
from enum import Enum
import numeric, group, linear_prog, parser
from simplet import Simplet
from typing import Optional, Callable, List, Tuple, Dict, Any, Protocol, Iterable, TextIO




class SimpletProto(Protocol):
    @dataclass
    class Instance:
        pass
    def init(self, lp: "LP", basic_point: np.ndarray) -> "SimpletProto.Instance": ...
    def solve(self, inst: "SimpletProto.Instance",
              pivot_rule: Callable, log: Optional[TextIO]) -> None: ...
    def bland_rule(self, *args, **kwargs): ...
    def basic_point(self, inst: "SimpletProto.Instance") -> np.ndarray: ...
    def basis_contains(self, inst: "SimpletProto.Instance", row: int) -> bool: ...
    def red_cost(self, inst: "SimpletProto.Instance", row: int) -> Optional[Tuple[str, Any]]: ...
    def pivot(self, inst: "SimpletProto.Instance", row: int) -> None: ...
    def print(self, inst: "SimpletProto.Instance",
              log: Optional[TextIO]) -> None: ...
    

class LinearProg:
    def __init__(self, group: group.OrderedGroup):
        self._impl = linear_prog.LinearProg(group)
    def init(self, var_names, nb_var, obj, ineq):
        return self._impl.init(var_names, nb_var, obj, ineq)

# Use the LP class from linear_prog module
LP = linear_prog.LP

class PerturbedLP:
    def __init__(self, lp_mod: LinearProg):
        self.LPmod = lp_mod

    # Gli identificatori di riga usati nel main OCaml:
    def phaseI_infeasibility_var_lower_bound_row(self, lp: LP) -> int:
        # TODO
        raise NotImplementedError

    def phaseII_upperbound_row(self, lp: LP) -> int:
        # TODO
        raise NotImplementedError

    def phaseI(self, lp: LP) -> Tuple[LP, np.ndarray]:
        # TODO
        raise NotImplementedError

    def phaseII(self, lp: LP) -> LP:
        # TODO
        raise NotImplementedError

    def print(self, lp: LP, log: Optional[TextIO]) -> None:
        lp.pretty_print()

    def project(self, x: Any) -> Optional[Any]:
        # In OCaml: Some (proiezione) / None. 
        # In Python: Optional
        # TODO
        return x 



# ---------- Lexer/Parser placeholders ----------

def lexer_from_file(fp: str):
    """Legge il file LP e restituisce il contenuto"""
    with open(fp, "r", encoding="utf-8") as f:
        return f.read()

def lexer_header(lexbuf: str) -> Tuple[str, Dict[str, int]]:
    """Parsing dell'header per ottenere numeric_name e var_names"""
    return parser.lexer_header(lexbuf)

class Parser:
    def __init__(self, Num: numeric.NumericBase):
        self.Num = Num
        self._parser = parser.Parser(Num)

    def main(self, content: str) -> Tuple[Any, Any, np.ndarray]:
        """Parsing completo del file LP"""
        return self._parser.main(content)

# ---------- Esiti ----------

class Solution(Enum):
    INFEASIBLE = "Infeasible"
    UNBOUNDED = "Unbounded"
    OPTIMUM = "Optimum"

# ---------- main porting ----------

def run_main(input_filename: str,
             verbose: bool = False,
             log_file_name: str = "log") -> Tuple[Solution, np.ndarray]:


    lexbuf = lexer_from_file(input_filename)


    try:
        numeric_name, var_names = lexer_header(lexbuf)
    except Exception as e:
        raise RuntimeError(f"lexer error while reading header: {e}") from e

 
    Num = numeric.get(numeric_name)
    Parse = Parser(Num)

    try:
        obj, ineq, basic_point_list = Parse.main(lexbuf) 
    except Exception as e:
        raise RuntimeError(f"parse error: {e}") from e

    print("input file parsed")

  
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
            print("\n parsed lp\n", file=log)
        lp.pretty_print()

        basic_point_given = basic_point_list.size > 0 if hasattr(basic_point_list, 'size') else len(basic_point_list) > 0

        if basic_point_given:
            # ----- Caso: base fornita -> Fase II diretta -----
            basic_point = np.array(basic_point_list)
            Simp = Simplet(LPmod._impl)
            phaseII = Simp.init(lp, basic_point)

            print("applying tropical simplex method with given input basic point")
            if log: print("\n------------------\napplying tropical simplex method with given input basic point\n", file=log)

            Simp.solve(phaseII, lambda inst: Simp.bland_rule(inst), log)

            opt = Simp.basic_point(phaseII)
            opt_projected = opt.copy()  # Copia l'array NumPy
            return (Solution.OPTIMUM, opt_projected)

        else:
            # ----- Caso: nessuna base -> Fase I + Fase II -----
            PertLP = PerturbedLP(LPmod)
            phaseI_lp, basic_point = PertLP.phaseI(lp)

            if log:
                print("\n------------------\nphaseI lp:\n", file=log)
            PertLP.print(phaseI_lp, log)

            SimpletI = Simplet(LPmod._impl) 
            phaseI = SimpletI.init(phaseI_lp, basic_point)

            print("solving phaseI")
            if log: print("\n------------------\ncall simplex method on phaseI lp\n", file=log)
            SimpletI.solve(phaseI, lambda inst: SimpletI.bland_rule(inst), log)

            phaseI_opt_basic_point = SimpletI.basic_point(phaseI)
            feasible = SimpletI.basis_contains(phaseI, PertLP.phaseI_infeasibility_var_lower_bound_row(lp))

            if not feasible:
                return (Solution.INFEASIBLE, np.array([]))

            # ---- Fase II ----
            print("solving phaseII")
            phaseII_lp = PertLP.phaseII(lp)
            if log:
                print("\n---------\nphaseII lp:\n", file=log)
            PertLP.print(phaseII_lp, log)

            phaseII_basic_point = phaseI_opt_basic_point[:lp.dim()] 
            SimpletII = Simplet(LPmod._impl)
            phaseII = SimpletII.init(phaseII_lp, phaseII_basic_point)

            if log: print("\n------------------\ncall simplex method on phaseII lp\n", file=log)
            SimpletII.solve(phaseII, lambda inst: SimpletII.bland_rule(inst), log)

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
        print("Add(2,5):", G.add(2, 5))
        print("Compare(2,5):", G.compare(2, 5))
        print("Max(2,5):", G.max(2, 5))
    else:
        # Risolve il problema LP fornito
        input_file = sys.argv[1]
        verbose = "--verbose" in sys.argv or "-v" in sys.argv
        
        try:
            solution, point = run_main(input_file, verbose)
            print(f"\nRisultato: {solution.value}")
            if solution == Solution.OPTIMUM:
                print("Punto ottimo:")
                for i, val in enumerate(point):
                    print(f"  x{i}: {val}")
            elif solution == Solution.INFEASIBLE:
                print("Il problema è infattibile")
            elif solution == Solution.UNBOUNDED:
                print("Il problema è illimitato")
                
        except Exception as e:
            print(f"Errore durante la risoluzione: {e}")
            import traceback
            traceback.print_exc()
