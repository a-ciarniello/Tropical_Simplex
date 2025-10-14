from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, List, Tuple, Optional, Union
from enum import Enum
import numpy as np
import group


# Type safety for Enums
class Sign(Enum):
    POS = "Pos"
    NEG = "Neg"

class ColKind(Enum):
    AFFINE = "Affine"
    VAR = "Var"

class RowKind(Enum):
    OBJECTIVE = "Objective"
    INEQ = "Ineq"

# Union types using Type aliases  
ColIndex = Union[
    Tuple[ColKind, None],  # (ColKind.AFFINE, None)
    Tuple[ColKind, int]    # (ColKind.VAR, j)
]

RowIndex = Union[
    Tuple[RowKind, None],  # (RowKind.OBJECTIVE, None)
    Tuple[RowKind, int]    # (RowKind.INEQ, i)
]


@dataclass
class LP:
    """Rappresenta un tropical linear program."""
    G: group.OrderedGroup
    vars: List[List[Tuple[RowIndex, Sign, Any]]]
    affine_col: List[Tuple[RowIndex, Sign, Any]]
    objective: List[Tuple[ColIndex, Sign, Any]]
    ineqs: List[List[Tuple[ColIndex, Sign, Any]]]
    var_names: Callable[[int], str]

    # --- Metodi diretti ---
    def dim(self) -> int:
        return len(self.vars)

    def nb_ineq(self) -> int:
        return len(self.ineqs)

    def get_row(self, row_index: RowIndex):
        kind, idx = row_index
        if kind == RowKind.OBJECTIVE:
            return self.objective
        elif kind == RowKind.INEQ:
            if idx is None:
                raise ValueError("Inequality index cannot be None")
            return self.ineqs[idx]
        else:
            raise ValueError(f"Invalid row_index {row_index}")

    def get_col(self, col_index: ColIndex):
        kind, idx = col_index
        if kind == ColKind.AFFINE:
            return self.affine_col
        elif kind == ColKind.VAR:
            if idx is None:
                raise ValueError("Variable index cannot be None")
            return self.vars[idx]
        else:
            raise ValueError(f"Invalid col_index {col_index}")

    def compute_entry_plus_var(self, entry: Any, col_index: ColIndex, point: np.ndarray):
        kind, idx = col_index
        if kind == ColKind.AFFINE:
            if entry is None:
                raise ValueError("Entry cannot be None for AFFINE column")
            return entry
        elif kind == ColKind.VAR:
            if idx is None:
                raise ValueError("Variable index cannot be None")
            if entry is None:
                raise ValueError("Entry cannot be None for VAR column")
            return entry + point[idx]  # Regular addition for coefficient computation

    def compute_slack_args(self, row_index: RowIndex, point: np.ndarray):
        row = self.get_row(row_index)
        result: List[Tuple[ColIndex, Sign, Any]] = []

        for col_index, sign, entry in row:
            slack = self.compute_entry_plus_var(entry, col_index, point)
            if not result:
                result = [(col_index, sign, entry)]
            else:
                old_col_index, old_sign, old_entry = result[0]
                old_slack = self.compute_entry_plus_var(old_entry, old_col_index, point)
                cmp_val = self.G.compare(slack, old_slack)
                if cmp_val == 0:
                    result.append((col_index, sign, entry))
                elif cmp_val == -1:  # slack < old_slack → new minimum found
                    result = [(col_index, sign, entry)]
        return result

    def is_point_feasible(self, point: np.ndarray) -> bool:
        if self.dim() != len(point):
            raise ValueError("Dimension mismatch between LP and point")

        for i in range(self.nb_ineq()):
            arg = self.compute_slack_args((RowKind.INEQ, i), point)
            pos, neg = False, False
            for _, sign, _ in arg:
                if sign == Sign.POS:
                    pos = True
                elif sign == Sign.NEG:
                    neg = True
            if not pos and not neg:
                raise AssertionError("Invalid inequality row (no signs found)")
            # In algebra tropicale, se solo il termine negativo (costante) raggiunge il minimo,
            # significa che il vincolo NON è soddisfatto (il punto è fuori dalla regione feasible)
            # Se solo il termine positivo raggiunge il minimo, il vincolo È soddisfatto
            # Se entrambi raggiungono il minimo, siamo sul bordo (feasible)
            if neg and not pos:
                return False
        return True

    def pretty_print(self):
        print(f"dim = {self.dim()}, nb_ineq = {self.nb_ineq()}")
        print("Objective:")
        for cidx, sign, val in self.objective:
            print(f"  {sign} {cidx}: {self.G.to_string(val)}")
        print("Inequalities:")
        for i, ineq in enumerate(self.ineqs):
            terms = ", ".join(
                f"{sign} {col}: {self.G.to_string(val)}" for col, sign, val in ineq
            )
            print(f"  {i}: {terms}")


class LinearProg:
    """Equivalente a LinearProg.Make(G) in OCaml."""

    def __init__(self, G: group.OrderedGroup):
        self.G = G

    def init(
        self,
        var_names: Callable[[int], str],
        nb_var: int,
        objective: List[Tuple[ColIndex, Sign, Any]],
        ineqs: List[List[Tuple[ColIndex, Sign, Any]]],
    ) -> LP:
        
        vars_cols: List[List[Tuple[RowIndex, Sign, Any]]] = [[] for _ in range(nb_var)]
        affine_col: List[Tuple[RowIndex, Sign, Any]] = []

        
        def process_input_row(row_index: RowIndex, w_i):
            for col_index, sign, entry in w_i:
                if col_index[0] == ColKind.AFFINE:
                    affine_col.append((row_index, sign, entry))
                elif col_index[0] == ColKind.VAR:
                    j = col_index[1]
                    if j is None:
                        raise ValueError("Variable index cannot be None")
                    vars_cols[j].append((row_index, sign, entry))

        for i, w_i in enumerate(ineqs):
            process_input_row((RowKind.INEQ, i), w_i)
        process_input_row((RowKind.OBJECTIVE, None), objective)

        return LP(
            G=self.G,
            vars=vars_cols,
            affine_col=affine_col,
            objective=objective,
            ineqs=ineqs,
            var_names=var_names,
        )

