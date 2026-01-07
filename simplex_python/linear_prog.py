from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, List, Tuple, Union, Optional
from enum import Enum
import numpy as np
import group


class Sign(Enum):
    POS = "Pos"
    NEG = "Neg"


class ColKind(Enum):
    AFFINE = "Affine"
    VAR = "Var"


class RowKind(Enum):
    OBJECTIVE = "Objective"
    INEQ = "Ineq"


ColIndex = Union[Tuple[ColKind, None], Tuple[ColKind, int]]
RowIndex = Union[Tuple[RowKind, None], Tuple[RowKind, int]]


def _default_var_name(j: int) -> str:
    return f"x{j}"


@dataclass
class LP:
    G: group.OrderedGroup
    vars: List[List[Tuple[RowIndex, Sign, Any]]]
    affine_col: List[Tuple[RowIndex, Sign, Any]]
    objective: List[Tuple[ColIndex, Sign, Any]]
    ineqs: List[List[Tuple[ColIndex, Sign, Any]]]
    var_names: Callable[[int], str]

    def dim(self) -> int:
        return len(self.vars)

    def nb_ineq(self) -> int:
        return len(self.ineqs)

    def get_row(self, row_index: RowIndex):
        kind, idx = row_index
        if kind == RowKind.OBJECTIVE:
            return self.objective
        if kind == RowKind.INEQ and idx is not None:
            return self.ineqs[idx]
        raise ValueError(f"Invalid row_index {row_index}")

    def get_col(self, col_index: ColIndex):
        kind, idx = col_index
        if kind == ColKind.AFFINE:
            return self.affine_col
        if kind == ColKind.VAR and idx is not None:
            return self.vars[idx]
        raise ValueError(f"Invalid col_index {col_index}")

    def compute_entry_plus_var(self, entry: Any, col_index: ColIndex, point: np.ndarray):
        kind, idx = col_index
        if kind == ColKind.AFFINE:
            return entry
        if kind == ColKind.VAR and idx is not None:
            # Tropical linear terms use classical addition on coefficients (group.mul)
            return self.G.mul(entry, point[idx])
        raise ValueError(f"Invalid col_index {col_index}")

    def compute_slack_args(self, row_index: RowIndex, point: np.ndarray):
        row = self.get_row(row_index)

        def step(current: List[Tuple[ColIndex, Sign, Any]], col_sign_entry):
            col_index, sign, entry = col_sign_entry
            slack = self.compute_entry_plus_var(entry, col_index, point)
            if not current:
                return [(col_index, sign, entry)]
            old_col_index, _, old_entry = current[0]
            old_slack = self.compute_entry_plus_var(old_entry, old_col_index, point)
            cmp_val = self.G.compare(slack, old_slack)
            if cmp_val == 0:
                return current + [(col_index, sign, entry)]
            # Match OCaml: keep values when slack > old_slack (cmp == 1)
            if cmp_val == 1:
                return [(col_index, sign, entry)]
            return current

        minima = []
        for col_sign_entry in row:
            minima = step(minima, col_sign_entry)

        # Preserve original order of minima (OCaml reverses at end)
        minima.reverse()
        return minima

    def is_point_feasible(self, point: np.ndarray, allow_all_neg: bool = False) -> bool:
        if self.dim() != len(point):
            raise ValueError("dimension mismatch between LP and point")

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
            if neg and not pos and not allow_all_neg:
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

    def _pretty_print_objective(self, myprintf):
        row = self.objective
        row_pos, row_neg = self._split_row(row)
        if not row_pos and not row_neg:
            myprintf("no objective")
        elif row_pos and not row_neg:
            myprintf("minimize ")
            self._pretty_print_scalar_product(row_pos, myprintf)
        elif row_neg and not row_pos:
            myprintf("maximize ")
            self._pretty_print_scalar_product(row_neg, myprintf)
        else:
            raise ValueError("Objective row must not mix POS and NEG")

    def _pretty_print_ineq(self, ineq_index: int, myprintf):
        row = self.ineqs[ineq_index]
        row_pos, row_neg = self._split_row(row)
        self._pretty_print_scalar_product(row_pos, myprintf)
        myprintf(" >= ")
        self._pretty_print_scalar_product(row_neg, myprintf)

    def _split_row(self, row):
        row_pos, row_neg = [], []
        for col_index, sign, entry in row:
            if sign == Sign.POS:
                row_pos.append((col_index, entry))
            else:
                row_neg.append((col_index, entry))
        return row_pos, row_neg

    def _pretty_print_scalar_product(self, entries, myprintf):
        if not entries:
            myprintf("-oo")
            return
        def fmt(col_index, entry):
            col_kind, j = col_index
            if col_kind == ColKind.AFFINE:
                return self.G.to_string(entry)
            name = self.var_names(j) if j is not None else f"x?"
            op = self.G.operation_string
            if self.G.compare(entry, self.G.zero()) == 0:
                return name
            return f"{self.G.to_string(entry)}{op}{name}"

        first, *rest = entries
        col_index, entry = first
        myprintf(f"{fmt(col_index, entry)}")
        for col_index, entry in rest:
            myprintf(f", {fmt(col_index, entry)}")

    def to_string(self) -> str:
        buf = []
        buf.append(f"dim = {self.dim()}, nb_ineq = {self.nb_ineq()}\n")
        buf.append("Objective:\n")
        buf.append(f"  {self._row_to_string(self.get_row((RowKind.OBJECTIVE, None)))}\n")
        buf.append("Inequalities:\n")
        for i in range(self.nb_ineq()):
            buf.append(f"  {i}: {self._row_to_string(self.get_row((RowKind.INEQ, i)))}\n")
        return "".join(buf)

    def _row_to_string(self, row):
        parts = []
        for col_index, sign, entry in row:
            col_kind, j = col_index
            col_str = f"(<ColKind.{col_kind.name}: '{col_kind.value}'>, {j})" if col_kind == ColKind.VAR else "(<ColKind.AFFINE: 'Affine'>, None)"
            sign_str = f"Sign.{sign.name}"
            parts.append(f"{sign_str} {col_str}: {self.G.to_string(entry)}")
        return ", ".join(parts)


class LinearProg:
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

        def process_input_entry(row_index: RowIndex, col_index: ColIndex, sign: Sign, entry: Any):
            kind, j = col_index
            if kind == ColKind.AFFINE:
                old_col = affine_col
            elif kind == ColKind.VAR:
                if j is None or j < 0 or j >= nb_var:
                    raise ValueError(f"invalid variable index {j}")
                old_col = vars_cols[j]
            else:
                raise ValueError(f"Invalid col_index {col_index}")

            def select(new_entry):
                if kind == ColKind.AFFINE:
                    affine_col.clear(); affine_col.extend(new_entry)
                else:
                    vars_cols[j] = new_entry

            if old_col and old_col[0][0] == row_index:
                _, old_sign, old_entry = old_col[0]
                if sign == old_sign:
                    kept = (row_index, sign, self.G.max(entry, old_entry))
                else:
                    cmp = self.G.compare(entry, old_entry)
                    if cmp == 0:
                        kept = (row_index, Sign.POS, entry)
                    elif cmp == 1:
                        kept = (row_index, sign, entry)
                    else:
                        kept = (row_index, old_sign, old_entry)
                select([kept] + old_col[1:])
            else:
                select([(row_index, sign, entry)] + old_col)

        def process_input_row(row_index: RowIndex, w_i):
            for col_index, sign, entry in w_i:
                process_input_entry(row_index, col_index, sign, entry)

        for i, w_i in enumerate(ineqs):
            process_input_row((RowKind.INEQ, i), w_i)
        process_input_row((RowKind.OBJECTIVE, None), objective)

        nb_ineq = len(ineqs)
        rows: List[List[Tuple[ColIndex, Sign, Any]]] = [[] for _ in range(nb_ineq)]
        objective_row: List[Tuple[ColIndex, Sign, Any]] = []

        def add_entry_to_row(col_index: ColIndex, row_index: RowIndex, sign: Sign, entry: Any):
            kind, i = row_index
            if kind == RowKind.OBJECTIVE:
                objective_row.append((col_index, sign, entry))
            else:
                rows[i].append((col_index, sign, entry))

        for row_index, sign, entry in affine_col:
            add_entry_to_row((ColKind.AFFINE, None), row_index, sign, entry)
        for j, col in enumerate(vars_cols):
            for row_index, sign, entry in col:
                add_entry_to_row((ColKind.VAR, j), row_index, sign, entry)

        return LP(
            G=self.G,
            vars=vars_cols,
            affine_col=affine_col,
            objective=objective_row,
            ineqs=rows,
            var_names=var_names or _default_var_name,
        )

