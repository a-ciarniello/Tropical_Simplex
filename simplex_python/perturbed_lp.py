from __future__ import annotations
import numpy as np
from typing import Any, Callable, List, Tuple, Dict, Optional, TextIO
import group, linear_prog
from linear_prog import LP, Sign, ColKind, RowKind, ColIndex, RowIndex

class PinftyError(Exception):
    """Exception raised when projection results in positive infinity."""
    pass

class PerturbedLP:
    """
    Python translation of the PertLP OCaml module.
    
    This class takes a linear program (LP) defined over an ordered group G
    and constructs perturbed versions of it (Phase I and Phase II LPs)
    over a new group F * G * H, where F and H are integer groups used for perturbation.
    This process is detailed in sections 4.4.1 and 4.4.2 of "Tropical aspects of
    linear programming" by Pascal Benchimol, 2014.
    """

    def __init__(self, lp_module: linear_prog.LinearProg):
        """
        Initializes the perturbed LP factory.

        Args:
            lp_module: An instance of linear_prog.LinearProg for the original group.
        """
        # Original group G (for tropical operations on coefficients)
        self.G = lp_module.G
        
        # IMPORTANT: Coordinates of basic points use standard algebra (0.0), not tropical zero (±inf)
        # This numeric module is used only for coordinate values in the G component of (F, G, H)
        import numeric
        self.CoordNumeric = numeric.NumericFloat()
        
        # Perturbation groups F and H use standard integer group (NOT tropical!)
        # F and H are used for perturbation coefficients, not for tropical operations
        self.F = group.IntGroup()
        self.H = group.CartesianPowerSparse(group.IntGroup())
        
        # We model the cartesian triple (F, G, H) using CartesianTriple (flat tuple structure)
        # This matches OCaml's MakeCartesianTriple implementation
        self.PertG = group.CartesianTriple(self.F, self.G, self.H)
        
        # Create a LinearProg factory for the new perturbed group
        self.LP_pert_mod = linear_prog.LinearProg(self.PertG)

    # === Group Helper Methods for (F, G, H) tuples ===
    def _from_entries(self, f: int, g: Any, h: List[Tuple[int, int]]) -> Any:
        """Creates a perturbed group element from f, g, and h components."""
        return self.PertG.from_entries(f, g, h)

    def _first(self, fgh: Any) -> int:
        """Extracts the F component from a perturbed element."""
        return self.PertG.first(fgh)

    def _second(self, fgh: Any) -> Any:
        """Extracts the G component from a perturbed element."""
        return self.PertG.second(fgh)

    def _third(self, fgh: Any) -> List[Tuple[int, int]]:
        """Extracts the H component from a perturbed element."""
        return self.PertG.third(fgh)

    # === Core Public Methods ===

    def project(self, fgh: Any) -> Optional[Any]:
        """
        Projects a perturbed value back to the original group G.
        - Returns g if f == 0.
        - Returns None (representing -inf) if f < 0.
        - Raises PinftyError if f > 0.
        """
        f = self._first(fgh)
        if f > 0:
            raise PinftyError("Projection would result in positive infinity")
        elif f == 0:
            return self._second(fgh)
        else:  # f < 0
            return None

    def phaseII_upperbound_row(self, lp: LP) -> int:
        """Index of the row that enforces upper bounds in phaseII."""
        return lp.dim() + lp.nb_ineq()

    def phaseI_infeasibility_var_lower_bound_row(self, lp: LP) -> int:
        """Index of the row that enforces a lower bound on the phaseI extra variable."""
        # Note: In the original logic, this row ends up at the same index as the phaseII upper bound row
        # due to the construction order and list reversals.
        return lp.dim() + lp.nb_ineq()

    def phaseI(self, lp: LP) -> Tuple[LP, np.ndarray]:
        """
        Constructs the Phase I LP to find a feasible basic point.
        
        Returns:
            A tuple containing the Phase I LP and an initial basic point for it.
        """
        # --- 1. Define constant perturbed values ---
        # Use standard zero (0.0) for coordinate values, not tropical zero (±inf)
        phaseI_lower_bound = self._from_entries(-3, self.CoordNumeric.zero, self.H.zero())
        upper_bound = self._from_entries(1, self.CoordNumeric.zero, self.H.zero())
        dim, nb_ineq = lp.dim(), lp.nb_ineq()
        infeasibility_var_idx = dim

        # --- 2. Process input inequalities ---
        # Corresponds to `new_rows_builder` and `add_infeasibility_var`
        processed_rows = self._process_input_rows(lp)
        # Use multiplicative identity coefficient for variable
        identity_coeff = self._from_entries(0, self.G.one(), self.H.zero())
        for row in processed_rows:
            # Add the new infeasibility variable to each original inequality
            new_coeff = ( (ColKind.VAR, infeasibility_var_idx), Sign.POS, identity_coeff )
            row.insert(0, new_coeff)

        # --- 3. Build other rows ---
        lower_bounds_rows = self._lower_bounds_builder(lp)
        identity_coeff = self._from_entries(0, self.G.one(), self.H.zero())
        infeasibility_var_lb_row = [
            ( (ColKind.AFFINE, None), Sign.NEG, phaseI_lower_bound ),
            ( (ColKind.VAR, infeasibility_var_idx), Sign.POS, identity_coeff )
        ]
        
        phaseII_inf_plane = self._infinity_plane_row(dim, upper_bound)
        # Add infeasibility var with NEGATIVE sign to the infinity plane (OCaml uses Neg)
        inf_plane_row = [( (ColKind.VAR, infeasibility_var_idx), Sign.NEG, identity_coeff )] + phaseII_inf_plane
        
        # --- 4. Assemble and perturb the matrix ---
        # IMPORTANT: OCaml order is processed_rows + lower_bounds_rows (not lower_bounds first!)
        # See perturbedLP.ml line 231: lower_bounds_builder appends to processed_rows
        matrix = processed_rows + lower_bounds_rows
        matrix.insert(0, infeasibility_var_lb_row)
        matrix.insert(0, inf_plane_row)

        matrix.reverse()

        nb_columns = dim + 2  # Original vars + infeasibility var + affine
        perturbed_m = self._epsilon_perturbation(matrix, 1, nb_columns)

        perturbed_m.reverse()
        perturbed_matrix = perturbed_m

        # --- 5. Assemble and perturb the objective function ---
        identity_coeff = self._from_entries(0, self.G.one(), self.H.zero())
        objective_row = [( (ColKind.VAR, infeasibility_var_idx), Sign.POS, identity_coeff )]
        perturbed_objective = self._epsilon_perturb_row(objective_row, 0, nb_columns)
        
        # --- 6. Create the Phase I LP ---
        phaseI_lp = self.LP_pert_mod.init(
            var_names=lambda j: f"phaseI_var_{j}",
            nb_var=dim + 1,
            objective=perturbed_objective,
            ineqs=perturbed_matrix
        )

        # --- 7. Compute the initial basic point ---
        initial_basic_point = np.empty(dim + 1, dtype=object)
        
        # For original variables
        # OCaml uses i = 1 + nb_ineq + j (see perturbedLP.ml line 252)
        # This is NOT the perturbation index of the inequality, but a separate index for point calculation
        
        for j in range(dim):
            # Use OCaml formula directly
            i = 1 + nb_ineq + j
            l = self._lower_bound(j, lp)
            
            print(f"  Variable {j}: using i={i}")

            h_pert = self.H.add(
                self.H.neg(self._epsilon_perturbation_coeff(i, j, nb_columns)),
                self.H.neg(self._epsilon_perturbation_coeff(i, nb_columns - 1, nb_columns))
            )
            # Use standard zero for coordinate perturbations
            pertl = self.PertG.add(l, self._from_entries(0, self.CoordNumeric.zero, h_pert))
            initial_basic_point[j] = pertl
            
        # For the infeasibility variable
        j = dim
        # OCaml uses i = 1 + nb_ineq + dim + 1 (see perturbedLP.ml line 267)
        i = 1 + nb_ineq + dim + 1

        print(f"Infeasibility variable (j={j}): using i={i} (OCaml formula: 1 + nb_ineq + dim + 1)")

        l = upper_bound
        h_pert = self.H.add(
            self._epsilon_perturbation_coeff(i, j, nb_columns),
            self._epsilon_perturbation_coeff(i, nb_columns - 1, nb_columns)
        )
        # Use standard zero for coordinate perturbations
        pertl = self.PertG.add(l, self._from_entries(0, self.CoordNumeric.zero, h_pert))
        initial_basic_point[j] = pertl
        
        return (phaseI_lp, initial_basic_point)


    def phaseII(self, lp: LP) -> LP:
        """
        Constructs the perturbed Phase II LP from the original LP.
        """
        dim, nb_ineq = lp.dim(), lp.nb_ineq()
        # Use standard zero for coordinate values
        upper_bound = self._from_entries(1, self.CoordNumeric.zero, self.H.zero())
        
        # --- 1. Process input rows and add lower bounds ---
        processed_input_rows = self._process_input_rows(lp)
        matrix = self._lower_bounds_builder(lp) + processed_input_rows
        
        # --- 2. Perturb the main matrix ---
        nb_columns = dim + 1 # Original vars + affine
        perturbed_matrix = self._epsilon_perturbation(matrix, 1, nb_columns)
        
        # --- 3. Create and perturb the infinity plane row separately ---
        infinity_plane = self._infinity_plane_row(dim, upper_bound)
        # The row index for perturbation must be unique
        pert_inf_plane = self._epsilon_perturbation([infinity_plane], dim + nb_ineq + 2, nb_columns)
        
        final_matrix = perturbed_matrix + pert_inf_plane

        # --- 4. Process and perturb the objective function ---
        obj_row_g = lp.get_row((RowKind.OBJECTIVE, None))
        objective_row = self._g_row_to_fgh_row(obj_row_g)
        perturbed_objective = self._epsilon_perturb_row(objective_row, 0, nb_columns)
        
        # --- 5. Create the Phase II LP ---
        phaseII_lp = self.LP_pert_mod.init(
            var_names=lp.var_names,
            nb_var=dim,
            objective=perturbed_objective,
            ineqs=final_matrix
        )
        
        return phaseII_lp

    # === Internal Helper Methods ===
    
    def _affine_perturbation(self, row_index: int, lp: LP) -> Any:
        """Perturbation for rows without an affine term.
        Uses standard zero for coordinate value."""
        return self._from_entries(-1, self.CoordNumeric.zero, self.H.zero())

    def _lower_bound(self, col_index: int, lp: LP) -> Any:
        """Perturbation for lower bounds on variables.
        Uses standard zero for coordinate value."""
        return self._from_entries(-2, self.CoordNumeric.zero, self.H.zero())

    def _epsilon_perturbation_coeff(self, i: int, j: int, nb_columns: int) -> List[Tuple[int, int]]:
        """Creates the H component (sparse vector) for perturbation."""
        return self.H.from_list([(i * nb_columns + j, 1)])

    def _epsilon_perturb_row(self, row: List, row_index: int, nb_columns: int) -> List:
        """Applies epsilon perturbation to a single row."""
        new_row = []
        for col_index, sign, fgh in row:
            if col_index[0] == ColKind.VAR:
                j = col_index[1]
            else: # Affine
                j = nb_columns - 1
            
            f = self._first(fgh)
            g = self._second(fgh)
            h = self._epsilon_perturbation_coeff(row_index, j, nb_columns)

            if sign == Sign.NEG:
                h = self.H.neg(h)
            
            pert_fgh = self._from_entries(f, g, h)
            new_row.append((col_index, sign, pert_fgh))
        return new_row

    def _epsilon_perturbation(self, rows: List, first_row_index: int, nb_columns: int) -> List:
        """Applies epsilon perturbation to a list of rows."""
        new_rows = []
        for i, row in enumerate(rows):
            row_index = i + first_row_index
            new_row = self._epsilon_perturb_row(row, row_index, nb_columns)
            new_rows.append(new_row)
        return new_rows

    def _g_row_to_fgh_row(self, old_row: List) -> List:
        """Converts a row with entries in G to a row with entries in PertG (F=0, H=0)."""
        new_row = []
        for col_index, sign, g in old_row:
            fgh = self._from_entries(0, g, self.H.zero())
            new_row.append((col_index, sign, fgh))
        return new_row

    def _process_input_rows(self, lp: LP) -> List:
        """Processes original inequalities for perturbation."""
        new_rows = []
        for i in range(lp.nb_ineq()):
            old_row = lp.get_row((RowKind.INEQ, i))
            processed_row = self._g_row_to_fgh_row(old_row)
            
            # If the original row has no affine term, add a perturbed one.
            has_affine = any(c[0] == ColKind.AFFINE for c in old_row)
            if not has_affine:
                fgh = self._affine_perturbation(i, lp)
                processed_row.append(((ColKind.AFFINE, None), Sign.POS, fgh))
            new_rows.append(processed_row)
        return new_rows

    def _infinity_plane_row(self, dim: int, upper_bound: Any) -> List:
        """Builds the row representing the 'infinity plane' (upper bounds).
        Uses multiplicative identity (G.one) for variable coefficients."""
        row: List[Tuple[ColIndex, Sign, Any]] = [( (ColKind.AFFINE, None), Sign.POS, upper_bound )]
        identity_coeff = self._from_entries(0, self.G.one(), self.H.zero())
        for j in range(dim):
            row.append(( (ColKind.VAR, j), Sign.NEG, identity_coeff ))
        return row

    def _lower_bounds_builder(self, lp: LP) -> List:
        """Builds the rows representing lower bounds on original variables.
        Uses multiplicative identity (G.one) for variable coefficients."""
        rows = []
        identity_coeff = self._from_entries(0, self.G.one(), self.H.zero())
        for j in range(lp.dim()):
            fgh = self._lower_bound(j, lp)
            row: List[Tuple[ColIndex, Sign, Any]] = [
                ( (ColKind.AFFINE, None), Sign.NEG, fgh ),
                ( (ColKind.VAR, j), Sign.POS, identity_coeff )
            ]
            rows.append(row)
        return rows