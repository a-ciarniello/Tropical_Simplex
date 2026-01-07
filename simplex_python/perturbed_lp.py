from __future__ import annotations
import numpy as np
from typing import Any, List, Tuple, Optional
import group, linear_prog
from linear_prog import LP, Sign, ColKind, RowKind, ColIndex, RowIndex


class PinftyError(Exception):
    """Exception raised when projection results in positive infinity."""
    pass


class PerturbedLP:
    """Factory for Phase I/II perturbed LPs (F×G×H) built from an LP over G."""

    def __init__(self, lp_module: linear_prog.LinearProg):
        # Base group G
        self.G = lp_module.G

        # Coordinates use classical float arithmetic
        import numeric
        self.CoordNumeric = numeric.NumericFloat()

        # Perturbation groups (integer, classical order)
        self.F = group.IntGroup()
        self.H = group.CartesianPowerSparse(group.IntGroup())

        # Product group (F, G, H)
        self.PertG = group.CartesianTriple(self.F, self.G, self.H)

        # LP factory over (F, G, H)
        self.LP_pert_mod = linear_prog.LinearProg(self.PertG)

    # === Helpers for (F,G,H) ===
    def _from_entries(self, f: int, g: Any, h: List[Tuple[int, int]]) -> Any:
        return self.PertG.from_entries(f, g, h)

    def _first(self, fgh: Any) -> int:
        return self.PertG.first(fgh)

    def _second(self, fgh: Any) -> Any:
        return self.PertG.second(fgh)

    def _third(self, fgh: Any) -> List[Tuple[int, int]]:
        return self.PertG.third(fgh)

    # === Debug formatting ===
    def _format_row(self, row: List[Tuple[ColIndex, Sign, Any]]) -> str:
        parts = []
        for col_index, sign, fgh in row:
            sign_str = "POS" if sign == Sign.POS else "NEG"
            col_str = f"VAR {col_index[1]}" if col_index[0] == ColKind.VAR else "AFFINE"
            parts.append(f"{sign_str}, {col_str}: {self.PertG.to_string(fgh)}")
        return "; ".join(parts)

    def _format_matrix(self, matrix: List[List[Tuple[ColIndex, Sign, Any]]]) -> str:
        return "\n".join(f"Row {i}: {self._format_row(row)}" for i, row in enumerate(matrix))

    # === Core public methods ===
    def project(self, fgh: Any) -> Optional[Any]:
        f = self._first(fgh)
        if f > 0:
            raise PinftyError("Projection would result in positive infinity")
        if f == 0:
            return self._second(fgh)
        return None

    def phaseII_upperbound_row(self, lp: LP) -> int:
        return lp.dim() + lp.nb_ineq()

    def phaseI_infeasibility_var_lower_bound_row(self, lp: LP) -> int:
        return lp.dim() + lp.nb_ineq()

    def phaseI(self, lp: LP) -> Tuple[LP, np.ndarray]:
        phaseI_lower_bound = self._from_entries(-3, self.CoordNumeric.zero, self.H.zero())
        upper_bound = self._from_entries(1, self.CoordNumeric.zero, self.H.zero())
        dim, nb_ineq = lp.dim(), lp.nb_ineq()
        infeasibility_var_idx = dim

        # Process input inequalities
        processed_rows = self._process_input_rows(lp)
        identity_coeff = self._from_entries(0, self.G.one(), self.H.zero())
        for row in processed_rows:
            row.append(((ColKind.VAR, infeasibility_var_idx), Sign.POS, identity_coeff))

        # Other structural rows
        lower_bounds_rows = self._lower_bounds_builder(lp)
        infeasibility_var_lb_row = [
            ((ColKind.AFFINE, None), Sign.NEG, phaseI_lower_bound),
            ((ColKind.VAR, infeasibility_var_idx), Sign.POS, identity_coeff),
        ]
        inf_plane_row = [((ColKind.VAR, infeasibility_var_idx), Sign.NEG, identity_coeff)] + self._infinity_plane_row(dim, upper_bound)

        # Assemble and perturb matrix
        matrix = lower_bounds_rows + processed_rows
        matrix.insert(0, infeasibility_var_lb_row)
        matrix.insert(0, inf_plane_row)

        print(f"\n Matrix before perturbation: \n{self._format_matrix(matrix)}\n")

        nb_columns = dim + 2
        matrix.reverse()
        perturbed_m = self._epsilon_perturbation(matrix, 1, nb_columns)
        perturbed_m.reverse()

        print(f"\n Perturbed matrix created: \n{self._format_matrix(perturbed_m)}\n")

        phaseI_lp = self.LP_pert_mod.init(
            var_names=lambda j: f"phaseI_var_{j}",
            nb_var=dim + 1,
            objective=self._epsilon_perturb_row([((ColKind.VAR, infeasibility_var_idx), Sign.POS, identity_coeff)], 0, nb_columns),
            ineqs=perturbed_m,
        )

        # Initial basic point
        initial_basic_point = np.empty(dim + 1, dtype=object)
        for j in range(dim):
            i = 1 + nb_ineq + j
            l = self._lower_bound(j, lp)
            h_pert = self.H.add(
                self.H.neg(self._epsilon_perturbation_coeff(i, j, nb_columns)),
                self.H.neg(self._epsilon_perturbation_coeff(i, nb_columns - 1, nb_columns)),
            )
            initial_basic_point[j] = self.PertG.add(l, self._from_entries(0, self.CoordNumeric.zero, h_pert))

        # Infeasibility variable: saturate its lower-bound row (index dim + nb_ineq in perturbed matrix)
        j = dim
        h_terms = []
        w_row_index = dim + nb_ineq + 1  # row_index used during perturbation for w lower bound
        h_terms.extend(self.H.neg(self._epsilon_perturbation_coeff(w_row_index, j, nb_columns)))
        h_terms.extend(self.H.neg(self._epsilon_perturbation_coeff(w_row_index, nb_columns - 1, nb_columns)))
        h_pert = self.H.from_list(h_terms)
        initial_basic_point[j] = self.PertG.add(phaseI_lower_bound, self._from_entries(0, self.CoordNumeric.zero, h_pert))

        return (phaseI_lp, initial_basic_point)

    def phaseII(self, lp: LP) -> LP:
        dim, nb_ineq = lp.dim(), lp.nb_ineq()
        upper_bound = self._from_entries(1, self.CoordNumeric.zero, self.H.zero())

        processed_input_rows = self._process_input_rows(lp)
        lower_bounds_rows = self._lower_bounds_builder(lp)

        nb_columns = dim + 1

        # Perturb original inequalities first: row indices 1..nb_ineq
        perturbed_input = self._epsilon_perturbation(processed_input_rows, 1, nb_columns)

        # Perturb lower bounds after inequalities: row indices (1 + nb_ineq)..(nb_ineq + dim)
        perturbed_lbs = self._epsilon_perturbation(lower_bounds_rows, 1 + nb_ineq, nb_columns)

        # Infinity plane gets the last index (nb_ineq + dim)
        infinity_plane = self._infinity_plane_row(dim, upper_bound)
        pert_inf_plane = self._epsilon_perturbation(
            [infinity_plane], 1 + nb_ineq + dim, nb_columns
        )

        # Final matrix order: lower bounds, original inequalities, infinity plane
        final_matrix = perturbed_lbs + perturbed_input + pert_inf_plane

        obj_row_g = lp.get_row((RowKind.OBJECTIVE, None))
        objective_row = self._g_row_to_fgh_row(obj_row_g)
        perturbed_objective = self._epsilon_perturb_row(objective_row, 0, nb_columns)

        return self.LP_pert_mod.init(
            var_names=lp.var_names,
            nb_var=dim,
            objective=perturbed_objective,
            ineqs=final_matrix,
        )

    def phaseII_initial_point_from_phaseI_opt(
        self, lp: LP, phaseI_opt: np.ndarray
    ) -> np.ndarray:
        """Build a basic Phase II starting point aligned with Phase II ε indices.

        We reuse the F and G components from the Phase I optimum, but rebuild the H
        component so that each variable saturates its lower-bound row in Phase II.
        """

        dim, nb_ineq = lp.dim(), lp.nb_ineq()
        nb_columns = dim + 1
        initial_basic_point = np.empty(dim, dtype=object)

        # Lower-bound rows are perturbed starting at row_index = 1 + nb_ineq.
        # _lower_bounds_builder inserts rows in reverse (j = dim-1 .. 0),
        # so variable j sits at row_index = 1 + nb_ineq + (dim-1-j).
        for j in range(dim):
            row_index = 1 + nb_ineq + (dim - 1 - j)

            h_terms = self.H.add(
                self.H.neg(self._epsilon_perturbation_coeff(row_index, j, nb_columns)),
                self.H.neg(
                    self._epsilon_perturbation_coeff(row_index, nb_columns - 1, nb_columns)
                ),
            )

            # Tie with all original inequalities (row_index 1..nb_ineq in Phase II perturbation)
            # so that POS and NEG reach the same slack on each row.
            for tie_row_index in range(1, 1 + nb_ineq):
                h_terms = self.H.add(
                    h_terms,
                    self.H.neg(
                        self._epsilon_perturbation_coeff(tie_row_index, j, nb_columns)
                    ),
                )
                h_terms = self.H.add(
                    h_terms,
                    self.H.neg(
                        self._epsilon_perturbation_coeff(
                            tie_row_index, nb_columns - 1, nb_columns
                        )
                    ),
                )

            # Force symmetric ties on each inequality row: if row i has a POS var j,
            # add eps to var j and the affine so that POS/NEG tie; if a row has only NEG,
            # force a POS tie on the first variable encountered.
            for i in range(nb_ineq):
                row = lp.get_row((RowKind.INEQ, i))
                row_idx = 1 + i  # perturbation index for inequality i in Phase II

                pos_vars = [col for col, sign, _ in row if sign == Sign.POS and col[0] == ColKind.VAR]

                # If no POS vars, pick the first var (if any) to create a POS tie
                if not pos_vars:
                    fallback_var = next((col for col, sign, _ in row if col[0] == ColKind.VAR), None)
                    if fallback_var is not None:
                        pos_vars = [fallback_var]

                for col_index in pos_vars:
                    if col_index[0] != ColKind.VAR or col_index[1] is None:
                        continue
                    j_var = col_index[1]
                    # tie var j_var
                    h_terms = self.H.add(
                        h_terms,
                        self.H.neg(
                            self._epsilon_perturbation_coeff(row_idx, j_var, nb_columns)
                        ),
                    )
                    h_terms = self.H.add(
                        h_terms,
                        self.H.neg(
                            self._epsilon_perturbation_coeff(
                                row_idx, nb_columns - 1, nb_columns
                            )
                        ),
                    )
                    # symmetric tie on affine (opposite sign to keep the same magnitude)
                    h_terms = self.H.add(
                        h_terms,
                        self.H.neg(
                            self.H.neg(
                                self._epsilon_perturbation_coeff(
                                    row_idx, nb_columns - 1, nb_columns
                                )
                            )
                        ),
                    )

            base = self._from_entries(
                self._first(phaseI_opt[j]),
                self._second(phaseI_opt[j]),
                self._third(phaseI_opt[j]),
            )

            initial_basic_point[j] = self.PertG.add(
                base, self._from_entries(0, self.CoordNumeric.zero, h_terms)
            )

        return initial_basic_point

    # === Helper methods ===
    def _affine_perturbation(self, row_index: int, lp: LP) -> Any:
        return self._from_entries(-1, self.CoordNumeric.zero, self.H.zero())

    def _lower_bound(self, col_index: int, lp: LP) -> Any:
        return self._from_entries(-2, self.CoordNumeric.zero, self.H.zero())

    def _epsilon_perturbation_coeff(self, i: int, j: int, nb_columns: int) -> List[Tuple[int, int]]:
        return self.H.from_list([(i * nb_columns + j, 1)])

    def _epsilon_perturb_row(self, row: List, row_index: int, nb_columns: int) -> List:
        new_row = []
        for col_index, sign, fgh in row:
            j = col_index[1] if col_index[0] == ColKind.VAR else nb_columns - 1
            f = self._first(fgh)
            g = self._second(fgh)
            h = self._epsilon_perturbation_coeff(row_index, j, nb_columns)
            if sign == Sign.NEG:
                h = self.H.neg(h)
            pert_fgh = self._from_entries(f, g, h)
            new_row.insert(0, (col_index, sign, pert_fgh))
        return new_row

    def _epsilon_perturbation(self, rows: List, first_row_index: int, nb_columns: int) -> List:
        new_rows = []
        for i, row in enumerate(rows):
            row_index = i + first_row_index
            new_row = self._epsilon_perturb_row(row, row_index, nb_columns)
            new_rows.insert(0, new_row)
        return new_rows

    def _g_row_to_fgh_row(self, old_row: List) -> List:
        return [
            (col_index, sign, self._from_entries(0, g, self.H.zero()))
            for col_index, sign, g in old_row
        ]

    def _process_input_rows(self, lp: LP) -> List:
        new_rows = []
        for i in range(lp.nb_ineq()):
            old_row = lp.get_row((RowKind.INEQ, i))
            processed_row = self._g_row_to_fgh_row(old_row)
            has_affine = any(c[0][0] == ColKind.AFFINE for c in old_row)
            if not has_affine:
                fgh = self._affine_perturbation(i, lp)
                processed_row.append(((ColKind.AFFINE, None), Sign.POS, fgh))
            new_rows.append(processed_row)
        return new_rows

    def _infinity_plane_row(self, dim: int, upper_bound: Any) -> List:
        row: List[Tuple[ColIndex, Sign, Any]] = [((ColKind.AFFINE, None), Sign.POS, upper_bound)]
        identity_coeff = self._from_entries(0, self.G.one(), self.H.zero())
        for j in range(dim):
            row.append(((ColKind.VAR, j), Sign.NEG, identity_coeff))
        return row

    def _lower_bounds_builder(self, lp: LP) -> List:
        rows = []
        identity_coeff = self._from_entries(0, self.G.one(), self.H.zero())
        for j in range(lp.dim()):
            fgh = self._lower_bound(j, lp)
            row: List[Tuple[ColIndex, Sign, Any]] = [
                ((ColKind.AFFINE, None), Sign.NEG, fgh),
                ((ColKind.VAR, j), Sign.POS, identity_coeff),
            ]
            rows.insert(0, row)
        return rows
