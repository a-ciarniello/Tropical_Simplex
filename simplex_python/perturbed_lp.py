from __future__ import annotations

import numpy as np
from typing import Any, List, Optional, Tuple
import group
import linear_prog

Sign = linear_prog.Sign
ColKind = linear_prog.ColKind
RowKind = linear_prog.RowKind
ColIndex = linear_prog.ColIndex
RowIndex = linear_prog.RowIndex


class PinftyError(Exception):
    pass


class PerturbedLP:
    """Perturbed Phase I/II LP builder over (F,G,H) = Int * G * sparse Int."""

    def __init__(self, lp_module: linear_prog.LinearProg):
        self.LP = lp_module
        self.G = lp_module.G

        self.F = group.IntGroup()
        self.H = group.CartesianPowerSparse(group.IntGroup())
        self.PertG = group.CartesianTriple(self.F, self.G, self.H)

        self.LP_pert_mod = linear_prog.LinearProg(self.PertG)

    # --- Low-level accessors ---
    def _from_entries(self, f: int, g: Any, h: List[Tuple[int, int]]) -> Any:
        return self.PertG.from_entries(f, g, h)

    def _first(self, fgh: Any) -> int:
        return self.PertG.first(fgh)

    def _second(self, fgh: Any) -> Any:
        return self.PertG.second(fgh)

    def _third(self, fgh: Any) -> Any:
        return self.PertG.third(fgh)

    # --- Projection ---
    def project(self, fgh: Any) -> Optional[Any]:
        f = self.PertG.first(fgh)
        if f > 0:
            raise PinftyError()
        if f == 0:
            return self.PertG.second(fgh)
        else:  # f < 0
            return float("-inf")

    # --- Special row indices ---
    def phaseII_upperbound_row(self, lp: linear_prog.LP) -> int:
        return lp.dim() + lp.nb_ineq()

    def phaseI_infeasibility_var_lower_bound_row(self, lp: linear_prog.LP) -> int:
        return lp.dim() + lp.nb_ineq()

    # --- Phase I ---
    def phaseI(self, lp: linear_prog.LP) -> Tuple[linear_prog.LP, np.ndarray]:
        dim, nb_ineq = lp.dim(), lp.nb_ineq()
        infeas_var = dim

        processed_rows = self._process_input_rows(lp)
        processed_rows = [row + [((ColKind.VAR, infeas_var), Sign.POS, self.PertG.from_entries(0, self.G.zero(), self.H.zero()))] for row in processed_rows]

        lower_bounds = self._lower_bounds_builder(lp)
        infeas_lb_row = [
            ((ColKind.AFFINE, None), Sign.NEG, self._phaseI_lower_bound()),
            ((ColKind.VAR, infeas_var), Sign.POS, self.PertG.from_entries(0, self.G.zero(), self.H.zero())),
        ]
        inf_plane = [((ColKind.VAR, infeas_var), Sign.NEG, self.PertG.from_entries(0, self.G.zero(), self.H.zero()))] + self._infinity_plane_row(lp)

        matrix = lower_bounds + processed_rows
        matrix.insert(0, infeas_lb_row)
        matrix.insert(0, inf_plane)

        nb_columns = dim + 2
        matrix_rev = list(reversed(matrix))
        perturbed = self._epsilon_perturbation(matrix_rev, 1, nb_columns)
        perturbed_matrix = list(reversed(perturbed))

        objective_row = self._epsilon_perturb_row(
            [((ColKind.VAR, infeas_var), Sign.POS, self.PertG.from_entries(0, self.G.zero(), self.H.zero()))], 0, nb_columns
        )

        phaseI_lp = self.LP_pert_mod.init(
            var_names=lambda j: f"phaseI_var_{j}",
            nb_var=dim + 1,
            objective=objective_row,
            ineqs=perturbed_matrix,
        )

        initial_basic_point = np.empty(dim + 1, dtype=object)

        for j in range(dim):
            i = 1 + nb_ineq + j
            l = self._lower_bound(j)
            h = self.H.add(
                self.H.neg(self._epsilon_perturbation_coeff(i, j, nb_columns)),
                self.H.neg(self._epsilon_perturbation_coeff(i, nb_columns - 1, nb_columns)),
            )
            initial_basic_point[j] = self.PertG.add(l, self.PertG.from_entries(0, self.G.zero(), h))

        j = dim
        i = 1 + nb_ineq + dim + 1
        h = self.H.add(
            self._epsilon_perturbation_coeff(i, j, nb_columns),
            self._epsilon_perturbation_coeff(i, nb_columns - 1, nb_columns),
        )
        initial_basic_point[j] = self.PertG.add(self._upper_bound(), self.PertG.from_entries(0, self.G.zero(), h))

        return phaseI_lp, initial_basic_point

    # --- Phase II ---
    def phaseII(self, lp: linear_prog.LP) -> linear_prog.LP:
        dim, nb_ineq = lp.dim(), lp.nb_ineq()

        processed_rows = self._process_input_rows(lp)
        lower_bounds = self._lower_bounds_builder(lp)
        infinity_plane = self._infinity_plane_row(lp)

        nb_columns = dim + 2

        matrix = lower_bounds + processed_rows
        m_rev = list(reversed(matrix))
        perturbed_m = self._epsilon_perturbation(m_rev, 1, nb_columns)

        inf_row_index = dim + nb_ineq + 2
        pert_inf = self._epsilon_perturbation([infinity_plane], inf_row_index, nb_columns)

        pert_matrix = list(reversed(pert_inf + perturbed_m))

        objective_row = self._epsilon_perturb_row(
            self._process_row(lp.get_row((RowKind.OBJECTIVE, None))), 0, nb_columns
        )

        return self.LP_pert_mod.init(
            var_names=lp.var_names,
            nb_var=dim,
            objective=objective_row,
            ineqs=pert_matrix,
        )

    # --- Phase II basic point reconstruction ---
    def phaseII_initial_point_from_phaseI_opt(self, lp: linear_prog.LP, phaseI_opt: np.ndarray) -> np.ndarray:
        dim = lp.dim()
        initial_basic_point = np.empty(dim, dtype=object)

        for j in range(dim):
            initial_basic_point[j] = phaseI_opt[j]

        return initial_basic_point

    # --- Helper constructors ---
    def _affine_perturbation(self) -> Any:
        return self.PertG.from_entries(-1, self.G.zero(), self.H.zero())

    def _lower_bound(self, col_index: int) -> Any:
        return self.PertG.from_entries(-2, self.G.zero(), self.H.zero())

    def _phaseI_lower_bound(self) -> Any:
        return self.PertG.from_entries(-3, self.G.zero(), self.H.zero())

    def _upper_bound(self) -> Any:
        return self.PertG.from_entries(1, self.G.zero(), self.H.zero())

    def _epsilon_perturbation_coeff(self, i: int, j: int, nb_columns: int) -> List[Tuple[int, int]]:
        return self.H.from_list([(i * nb_columns + j, 1)])

    def _epsilon_perturb_row(self, row: List[Tuple[ColIndex, Sign, Any]], row_index: int, nb_columns: int) -> List[Tuple[ColIndex, Sign, Any]]:
        new_row: List[Tuple[ColIndex, Sign, Any]] = []
        for col_index, sign, fgh in row:
            j = col_index[1] if col_index[0] == ColKind.VAR else nb_columns - 1
            f = self.PertG.first(fgh)
            g = self.PertG.second(fgh)
            h = self._epsilon_perturbation_coeff(row_index, j, nb_columns)
            if sign == Sign.NEG:
                h = self.H.neg(h)
            pert_fgh = self.PertG.from_entries(f, g, h)
            new_row.insert(0, (col_index, sign, pert_fgh))
        return new_row

    def _epsilon_perturbation(self, rows: List[List[Tuple[ColIndex, Sign, Any]]], first_row_index: int, nb_columns: int) -> List[List[Tuple[ColIndex, Sign, Any]]]:
        new_rows: List[List[Tuple[ColIndex, Sign, Any]]] = []
        for offset, row in enumerate(rows):
            row_index = first_row_index + offset
            new_rows.insert(0, self._epsilon_perturb_row(row, row_index, nb_columns))
        return new_rows

    def _process_row(self, row: List[Tuple[ColIndex, Sign, Any]]) -> List[Tuple[ColIndex, Sign, Any]]:
        return [
            (col_index, sign, self.PertG.from_entries(0, entry, self.H.zero()))
            for col_index, sign, entry in row
        ]

    def _process_input_rows(self, lp: linear_prog.LP) -> List[List[Tuple[ColIndex, Sign, Any]]]:
        # Match the OCaml construction order: prepend each processed row so the
        # resulting list is reversed (last input inequality first). Row indices
        # used for epsilon perturbations rely on this ordering.
        processed: List[List[Tuple[ColIndex, Sign, Any]]] = []
        for i in range(lp.nb_ineq()):
            row = lp.get_row((RowKind.INEQ, i))
            new_row = self._process_row(row)
            has_affine = any(col_index[0] == ColKind.AFFINE for col_index, _, _ in row)
            if not has_affine:
                new_row.append(((ColKind.AFFINE, None), Sign.POS, self._affine_perturbation()))
            processed.insert(0, new_row)
        return processed

    def _infinity_plane_row(self, lp: linear_prog.LP) -> List[Tuple[ColIndex, Sign, Any]]:
        row: List[Tuple[ColIndex, Sign, Any]] = [((ColKind.AFFINE, None), Sign.POS, self._upper_bound())]
        for j in range(lp.dim()):
            row.append(((ColKind.VAR, j), Sign.NEG, self.PertG.from_entries(0, self.G.zero(), self.H.zero())))
        return row

    def _lower_bounds_builder(self, lp: linear_prog.LP) -> List[List[Tuple[ColIndex, Sign, Any]]]:
        rows: List[List[Tuple[ColIndex, Sign, Any]]] = []
        for j in range(lp.dim()):
            row = [
                ((ColKind.AFFINE, None), Sign.NEG, self._lower_bound(j)),
                ((ColKind.VAR, j), Sign.POS, self.PertG.from_entries(0, self.G.zero(), self.H.zero())),
            ]
            rows.insert(0, row)
        return rows
