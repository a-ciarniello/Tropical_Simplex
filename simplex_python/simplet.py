from typing import Any, Callable, Optional, Tuple, TextIO, List
from enum import Enum
import numpy as np
import linear_prog
import tangent_digraph


class IneqStatus(Enum):
    """
    Status of inequalities during the tropical pivoting process.
    
    Attributes:
        BREAK_HYP: Inequality that was in the basis and might be violated
                   during movement along the pivoting direction.
        BASIS: Inequality currently saturated by the basic point (in the basis).
        ENT_HYP: Inequality that might enter the basis during pivoting.
        INACTIVE: Inequality leaving the basis and no longer considered.
    """
    BREAK_HYP = "BreakHyp"
    BASIS = "Basis" 
    ENT_HYP = "EntHyp"
    INACTIVE = "Inactive"


class Simplet:
    """
    Python translation of Simplet.Make(LP).
    
    Implements the tropical simplex algorithm described in Chapter 7 of
    "Tropical aspects of linear programming", Pascal Benchimol, PhD Thesis, 2014.
    
    This module provides the core functionality for solving tropical linear programs
    using a simplex-like algorithm adapted to the tropical semiring.
    """

    def __init__(self, lp_module_or_instance):
        """
        Initialize the Simplet module with reference to the LinearProg module or LP instance.
        
        Args:
            lp_module_or_instance: Either an LP instance or the LinearProg module.
                                   If it's an LP instance, it should have 'G' and 'nb_ineq' attributes.
                                   If it's a module, it should have a 'G' attribute for the group.
        """
        if hasattr(lp_module_or_instance, 'G') and hasattr(lp_module_or_instance, 'nb_ineq'):
            # It's an LP instance
            self.LP = lp_module_or_instance
            self.G = lp_module_or_instance.G
        else:
            # It's a module
            self.LP = lp_module_or_instance  
            self.G = lp_module_or_instance.G

    class SimpletInstance:
        """
        An instance of the tropical simplet containing all necessary data structures.
        Corresponds to the type 't' in the original OCaml module.
        
        Attributes:
            lp: The tropical linear program.
            point: The current basic point.
            tangent_digraph: The tangent digraph at the current basic point.
            arg_slacks: Graph between inequality i and argmax_j (|W^+_ij| + x_j).
            var_in_arg_slack: Inverse indexing for arg_slacks by variable.
            affine_var_in_arg_slack: Inequalities where affine variable is in arg_slack.
            direction: List of variable indices defining the pivoting direction.
            arg_lambdas: For each inequality, argmax_{j in direction} (|W_ij| + x_j).
            ineq_status: Status of each inequality (BASIS, BREAK_HYP, ENT_HYP, INACTIVE).
            max_permutation: Maximizing permutation sigma : [n] -> I in tropical permanent of A_I.
            reduced_costs: Reduced costs for each inequality.
            dual_slacks: Dual slacks for each variable.
            var_seen: Boolean array for Dijkstra algorithm.
            iteration: Iteration counter.
        """
        
        def __init__(self, lp: linear_prog.LP, basic_point: np.ndarray):
            self.lp = lp
            self.point = basic_point.copy()  # Current basic point
            
            # Main data structures
            self.tangent_digraph: Optional[tangent_digraph.TangentDigraph] = None
            
            # Graph between Ineq i and argmax_j (|W^+_ij| + x_j)
            self.arg_slacks: List[List[Tuple[Any, Any]]] = [[] for _ in range(lp.nb_ineq())]
            self.var_in_arg_slack: List[List[int]] = [[] for _ in range(lp.dim())]
            self.affine_var_in_arg_slack: List[int] = []
            
            # For pivoting
            self.direction: List[Any] = []  # List of var_index
            # argmax_{j in direction} (|W_ij| + x_j)
            self.arg_lambdas: List[List[Tuple[Any, str, Any]]] = [[] for _ in range(lp.nb_ineq())]
            self.ineq_status: List[IneqStatus] = [IneqStatus.INACTIVE for _ in range(lp.nb_ineq())]
            
            # For reduced costs
            self.max_permutation: List[Optional[Tuple[int, str, Any]]] = [None for _ in range(lp.dim())]
            self.reduced_costs: List[Optional[Tuple[str, Any]]] = [None for _ in range(lp.nb_ineq())]
            self.dual_slacks: List[Optional[Tuple[str, Any]]] = [None for _ in range(lp.dim())]
            self.var_seen: List[bool] = [False for _ in range(lp.dim())]  # For Dijkstra
            
            # Iteration counter
            self.iteration = 0

    def init(self, lp: linear_prog.LP, basic_point: np.ndarray) -> "Simplet.SimpletInstance":
        """
        Create a new simplet instance for a given LP and basic point.
        
        Args:
            lp: The tropical linear program (must be non-degenerate).
            basic_point: Basic point (must be feasible and saturate exactly dim inequalities).
            
        Returns:
            A fully initialized SimpletInstance.
            
        Raises:
            ValueError: If basic_point is not feasible or not a basic point.
        """
        nb_ineq = lp.nb_ineq()
        dim = lp.dim()
        
        # Verify that each row in lp has at least one non-zero coefficient
        for i in range(nb_ineq):
            try:
                w_i = lp.get_row((linear_prog.RowKind.INEQ, i))
                if not w_i:  # Empty list
                    raise ValueError(f"Simplet.init: all coefficients are zero in inequality {i}. "
                                   "At least one non-zero coefficient is required.")
            except AttributeError:
                # Fallback if get_row is not implemented
                pass
        
        # Build the tangent digraph
        tg = tangent_digraph.TangentDigraph.compute(lp, basic_point)
        
        # Verify that the given point is a basic point
        if not tg.is_basic_point():
            raise ValueError("Simplet.init: the input point is not a basic point.")
        
        # Check if the basic point is feasible
        if hasattr(lp, 'is_point_feasible') and not lp.is_point_feasible(basic_point):
            raise ValueError("Simplet.init: the input point is not feasible.")
        
        # Create the instance
        simplet_inst = Simplet.SimpletInstance(lp, basic_point)
        simplet_inst.tangent_digraph = tg
        
        # Initialize ineq_status based on the tangent digraph
        for i in range(nb_ineq):
            if tg.is_hyp_node(i):
                simplet_inst.ineq_status[i] = IneqStatus.BASIS
        
        # Compute arg_slacks and reduced_costs
        self._compute_arg_slacks_pos(simplet_inst)
        self._compute_reduced_costs(simplet_inst)
        
        return simplet_inst

    def _compute_arg_slacks_pos(self, simplet_inst: "Simplet.SimpletInstance") -> None:
        """
        Compute the graph between var j and argmax_j (W^+_ij + x_j) and store it in simplet_inst.
        Corresponds to compute_arg_slacks_pos in the original OCaml.
        
        Uses compute_slack_args from LinearProg to find the argmin of coefficients
        for each inequality, then filters only those with positive sign.
        """
        nb_ineq = simplet_inst.lp.nb_ineq()
        dim = simplet_inst.lp.dim()
        
        # Reset data structures
        simplet_inst.arg_slacks = [[] for _ in range(nb_ineq)]
        simplet_inst.var_in_arg_slack = [[] for _ in range(dim)]
        simplet_inst.affine_var_in_arg_slack = []
        
        for i in range(nb_ineq):
            # Compute argmin for inequality i
            arg_slack = simplet_inst.lp.compute_slack_args((linear_prog.RowKind.INEQ, i), simplet_inst.point)
            
            # Process each element in the argmin
            for col_index, sign, entry in arg_slack:
                # In OCaml, check the sign and process only positive ones
                if sign == linear_prog.Sign.NEG:
                    continue
                elif sign == linear_prog.Sign.POS:
                    # Add to arg_slacks (store only var_index and entry, not the sign)
                    simplet_inst.arg_slacks[i].append((col_index, entry))
                    
                    # Update var_in_arg_slack for reverse indexing
                    if col_index[0] == linear_prog.ColKind.AFFINE:
                        # If it's the affine variable, add it to the special list
                        simplet_inst.affine_var_in_arg_slack.append(i)
                    elif col_index[0] == linear_prog.ColKind.VAR:
                        # If it's an ordinary variable j, add it to var_in_arg_slack[j]
                        j = col_index[1]
                        if j is not None:
                            simplet_inst.var_in_arg_slack[j].append(i)
    
    def _compute_reduced_costs(self, simplet_inst: "Simplet.SimpletInstance") -> None:
        """
        Compute reduced costs using Dijkstra's algorithm for longest paths.
        Corresponds to compute_reduced_costs in the original OCaml.
        
        Reduced costs are calculated as distances in the bipartite graph between variables
        and inequalities, using the tangent digraph to determine the optimal permutation.
        """
        nb_ineq = simplet_inst.lp.nb_ineq()
        dim = simplet_inst.lp.dim()
        
        # Compute the maximizing permutation
        self._compute_max_permutation(simplet_inst)
        
        # Reset data structures
        simplet_inst.reduced_costs = [None for _ in range(nb_ineq)]
        simplet_inst.dual_slacks = [None for _ in range(dim)]
        simplet_inst.var_seen = [False for _ in range(dim)]
        
        # Compute mu = max(0, c_1+x_1, ..., c_n+x_n) from the objective function
        mu = self._compute_objective_value(simplet_inst)
        neg_mu = self.G.neg(mu)
        
        # Initialize distances from objective to variable nodes
        self._initialize_dual_slacks_from_objective(simplet_inst, neg_mu)
        
        # Dijkstra's algorithm for longest paths
        # Consider only VarNodes; when considering a VarNode j:
        # -mark VarNode j as seen (in the tree of longest paths)
        # -go to IneqNode sigma(j) with arc cost mu
        # -explore unseen neighbor VarNodes of IneqNode sigma(j)
        
        finished = False
        while not finished:
            # Find the VarNode with the longest distance not yet in the tree
            max_j = None
            max_val = None
            
            for current_j in range(dim):
                if simplet_inst.var_seen[current_j]:
                    continue
                
                current_val = simplet_inst.dual_slacks[current_j]
                if current_val is None:
                    continue
                    
                if max_val is None:
                    max_j = current_j
                    max_val = current_val
                else:
                    current_sign, current_entry = current_val
                    max_sign, max_entry = max_val
                    cmp = self.G.compare(current_entry, max_entry)
                    if cmp == 1:  # current > max
                        max_j = current_j
                        max_val = current_val
            
            if max_j is None:
                finished = True
            else:
                # Mark variable as seen
                simplet_inst.var_seen[max_j] = True
                
                # Get sigma(j) - the inequality corresponding to this variable
                sigma_j_data = simplet_inst.max_permutation[max_j]
                if sigma_j_data is None:
                    continue
                    
                sigma_j, sign, entry = sigma_j_data
                
                # Get the dual slack (distance) at this variable
                dual_slack_j = simplet_inst.dual_slacks[max_j]
                if dual_slack_j is None:
                    continue
                j_sign, j_entry = dual_slack_j
                
                # Get slack of inequality sigma_j
                arg_slack_sigma_j = simplet_inst.arg_slacks[sigma_j]
                if not arg_slack_sigma_j:
                    continue
                    
                var_slack, entry_slack = arg_slack_sigma_j[0]
                slack_sigma_j = simplet_inst.lp.compute_entry_plus_var(entry_slack, var_slack, simplet_inst.point)
                
                # Compute reduced cost for inequality sigma_j
                entry_sigma_j = self.G.add(j_entry, mu)
                sign_sigma_j = linear_prog.Sign.POS.value if j_sign == sign else linear_prog.Sign.NEG.value
                simplet_inst.reduced_costs[sigma_j] = (sign_sigma_j, entry_sigma_j)
                
                # Visit neighbors of inequality sigma_j
                ineq_row = simplet_inst.lp.get_row((linear_prog.RowKind.INEQ, sigma_j))
                for arc_index, arc_sign, arc_entry in ineq_row:
                    if arc_index[0] != linear_prog.ColKind.VAR:
                        continue
                    
                    arc_var_index = arc_index[1]
                    if arc_var_index is None or simplet_inst.var_seen[arc_var_index]:
                        continue
                    
                    # Compute new distance
                    x_l = simplet_inst.point[arc_var_index]
                    neg_slack_sigma_j = self.G.neg(slack_sigma_j)
                    
                    # new_dist = -mu + x_l + arc_entry - slack_sigma_j + entry_sigma_j
                    terms = [neg_mu, x_l, arc_entry, neg_slack_sigma_j, entry_sigma_j]
                    new_dist = self.G.sum(terms)
                    
                    # Determine sign of the new path
                    new_sign = linear_prog.Sign.POS.value if sign_sigma_j == arc_sign.value else linear_prog.Sign.NEG.value
                    
                    # Check if this is a better path
                    current_dual_slack = simplet_inst.dual_slacks[arc_var_index]
                    better = False
                    if current_dual_slack is None:
                        better = True
                    else:
                        current_sign, current_entry = current_dual_slack
                        cmp = self.G.compare(current_entry, new_dist)
                        if cmp == -1:  # current < new_dist
                            better = True
                    
                    if better:
                        simplet_inst.dual_slacks[arc_var_index] = (new_sign, new_dist)
        
        # Rescale reduced costs
        for i in range(nb_ineq):
            arg_slack_i = simplet_inst.arg_slacks[i]
            if not arg_slack_i:
                continue
                
            var_slack, entry_slack = arg_slack_i[0]
            slack_i = simplet_inst.lp.compute_entry_plus_var(entry_slack, var_slack, simplet_inst.point)
            
            z_i = simplet_inst.reduced_costs[i]
            if z_i is not None:
                sign, value = z_i
                neg_slack_i = self.G.neg(slack_i)
                rescaled_red_cost = self.G.add(value, neg_slack_i)
                simplet_inst.reduced_costs[i] = (sign, rescaled_red_cost)
    
    def _compute_objective_value(self, simplet_inst: "Simplet.SimpletInstance") -> Any:
        """
        Compute the value of the objective function at the current point.
        mu = max(0, c_1+x_1, ..., c_n+x_n)
        """
        # Initialize with zero (constant term)
        mu = self.G.zero()
        
        # Compute the maximum over the objective function terms
        objective_terms = simplet_inst.lp.get_row((linear_prog.RowKind.OBJECTIVE, None))
        for col_index, sign, entry in objective_terms:
            if sign == linear_prog.Sign.POS:
                term_value = simplet_inst.lp.compute_entry_plus_var(entry, col_index, simplet_inst.point)
                mu = self.G.max(mu, term_value)
        
        return mu
    
    def _initialize_dual_slacks_from_objective(self, simplet_inst: "Simplet.SimpletInstance", neg_mu: Any) -> None:
        """
        Initialize distances from the objective to variable nodes.
        """
        objective_terms = simplet_inst.lp.get_row((linear_prog.RowKind.OBJECTIVE, None))
        
        for col_index, sign, entry in objective_terms:
            if col_index[0] == linear_prog.ColKind.VAR:
                j = col_index[1]
                if j is not None:
                    # Compute the scaled distance
                    scaled_entry = self.G.add(neg_mu, entry)
                    simplet_inst.dual_slacks[j] = (sign.value, scaled_entry)
    
    def _compute_max_permutation(self, simplet_inst: "Simplet.SimpletInstance") -> None:
        """
        Compute the maximizing permutation sigma : [n] -> I in the tropical permanent of A_I.
        Corresponds to compute_max_permutation in the original OCaml.
        
        Uses DFS in the tangent digraph to determine the optimal permutation.
        """
        dim = simplet_inst.lp.dim()
        simplet_inst.max_permutation = [None for _ in range(dim)]
        
        if simplet_inst.tangent_digraph is not None:
            # Implement DFS in the tangent digraph
            def compute_sigma(acc, node):
                node_type, node_value = node
                if node_type == "IneqNode":
                    ineq_index = int(node_value) if isinstance(node_value, int) else None
                    if ineq_index is not None:
                        # Get all variables connected to this inequality node
                        if simplet_inst.tangent_digraph is not None:
                            ineq_node = simplet_inst.tangent_digraph.get_ineq_node(ineq_index)
                            # In a hyperplan node, there should be at least one VAR with each sign
                            # We assign this inequality to ALL connected non-affine variables
                            # that don't have a permutation yet
                            for var_index, sign, entry in ineq_node:
                                if var_index[0] == linear_prog.ColKind.VAR:
                                    j = var_index[1]
                                    if j is not None and simplet_inst.max_permutation[j] is None:
                                        simplet_inst.max_permutation[j] = (ineq_index, sign.value, entry)
                return acc
            
            # Start DFS from the affine node
            start_node = ("VarNode", (linear_prog.ColKind.AFFINE, None))
            simplet_inst.tangent_digraph.dfs_fold_acyclic_graph(compute_sigma, None, start_node)
        
        # Verify that all variables have a permutation
        for j in range(dim):
            if simplet_inst.max_permutation[j] is None:
                # Fallback if DFS didn't find a complete permutation
                print(f"Warning: permutation for variable {j} not found")

    def _bound_on_length_by_ineq(self, simplet_inst: "Simplet.SimpletInstance", ineq_index: int) -> Optional[Any]:
        """
        During pivoting, compute the bound on the length of the current ordinary segment
        given by inequality ineq_index.
        
        Args:
            simplet_inst: The simplet instance.
            ineq_index: The index of the inequality.
            
        Returns:
            The bound value if applicable, None otherwise.
            
        Raises:
            ValueError: If the slack of the inequality is +infinity.
        """
        arg_slack = simplet_inst.arg_slacks[ineq_index]
        arg_lambda = simplet_inst.arg_lambdas[ineq_index]
        ineq_status = simplet_inst.ineq_status[ineq_index]
        
        if not arg_slack:  # Empty list
            raise ValueError(f"Error, simplet._bound_on_length_by_ineq: slack of ineq_{ineq_index} is +oo")
        
        if not arg_lambda:
            return None
            
        var_index_slack, entry_slack = arg_slack[0]
        var_index_lambda, sign_lambda, entry_lambda = arg_lambda[0]
        
        if ineq_status in [IneqStatus.BASIS, IneqStatus.INACTIVE]:
            return None
        elif ineq_status == IneqStatus.ENT_HYP and sign_lambda == linear_prog.Sign.POS.value:
            return None
        elif ineq_status == IneqStatus.BREAK_HYP or (ineq_status == IneqStatus.ENT_HYP and sign_lambda == linear_prog.Sign.NEG.value):
            # Compute the actual bound: entry_slack - entry_lambda
            # This represents the maximum segment length before the inequality changes status
            bound = self.G.substract(entry_slack, entry_lambda)
            return bound
        
        return None

    def bland_rule(self, inst: "Simplet.SimpletInstance") -> Optional[int]:
        """
        Find the first inequality with negative reduced cost for minimization problems.
        Implements Bland's rule to avoid cycling.
        
        This rule is appropriate when solving tropical LP minimization problems
        (e.g., min-plus semiring with minimize objective, or max-plus semiring with maximize objective).
        
        Args:
            inst: The simplet instance.
            
        Returns:
            Index of the first inequality with negative reduced cost, or None if optimal.
        """
        nb_ineq = inst.lp.nb_ineq()
        
        for i in range(nb_ineq):
            # Verify if the ineq is a hyp_node (in the basis) using TangentDigraph
            if inst.tangent_digraph is None or not inst.tangent_digraph.is_hyp_node(i):
                continue
                
            red_cost = inst.reduced_costs[i]
            if red_cost is not None and red_cost[0] == linear_prog.Sign.NEG.value:
                return i
        
        return None
    
    def bland_rule_maximize(self, inst: "Simplet.SimpletInstance") -> Optional[int]:
        """
        Find the first inequality with positive reduced cost for maximization problems.
        Implements Bland's rule to avoid cycling.
        
        This rule is appropriate when solving tropical LP maximization problems
        (e.g., min-plus semiring with maximize objective, or max-plus semiring with minimize objective).
        The sign convention is reversed compared to minimization.
        
        Args:
            inst: The simplet instance.
            
        Returns:
            Index of the first inequality with positive reduced cost, or None if optimal.
        """
        nb_ineq = inst.lp.nb_ineq()
        
        for i in range(nb_ineq):
            # Verify if the ineq is a hyp_node (in the basis) using TangentDigraph
            if inst.tangent_digraph is None or not inst.tangent_digraph.is_hyp_node(i):
                continue
                
            red_cost = inst.reduced_costs[i]
            # For maximization, positive reduced cost indicates improvement potential
            if red_cost is not None and red_cost[0] == linear_prog.Sign.POS.value:
                return i
        
        return None

    def get_pivot_rule_for_objective(self, maximize: bool = False) -> Callable[["Simplet.SimpletInstance"], Optional[int]]:
        """
        Return the appropriate pivot rule based on the optimization direction.
        
        This method allows solving hybrid problems where the semiring and objective direction
        may not match the standard convention (e.g., min-plus with maximize, or max-plus with minimize).
        
        Args:
            maximize: If True, return the pivot rule for maximization objectives.
                     If False (default), return the pivot rule for minimization objectives.
                     
        Returns:
            The appropriate pivot rule function (bland_rule or bland_rule_maximize).
            
        Examples:
            Standard cases:
            - min-plus semiring + minimize objective: get_pivot_rule_for_objective(maximize=False)
            - max-plus semiring + maximize objective: get_pivot_rule_for_objective(maximize=False)
            
            Hybrid cases:
            - min-plus semiring + maximize objective: get_pivot_rule_for_objective(maximize=True)
            - max-plus semiring + minimize objective: get_pivot_rule_for_objective(maximize=True)
        """
        if maximize:
            return self.bland_rule_maximize
        else:
            return self.bland_rule

    def basis_contains(self, inst: "Simplet.SimpletInstance", ineq_index: int) -> bool:
        """
        Check if inequality ineq_index is saturated by the current basic point.
        
        Args:
            inst: The simplet instance.
            ineq_index: The index of the inequality to check.
            
        Returns:
            True if the inequality is in the basis, False otherwise.
        """
        if inst.tangent_digraph is not None:
            return inst.tangent_digraph.is_hyp_node(ineq_index)
        else:
            # Fallback to ineq_status if TangentDigraph is not available
            return inst.ineq_status[ineq_index] == IneqStatus.BASIS

    def pivot(self, inst: "Simplet.SimpletInstance", i_out: int) -> None:
        """
        Perform tropical pivoting from the current basic point along the edge defined by I \\ {i_out}.
        This is the implementation of the core of the tropical simplex algorithm.
        
        The complete algorithm includes:
        1. Initialize inequality status
        2. Remove hyp node from tangent digraph  
        3. Compute direction and break hyp
        4. Traverse the tropical edge through ordinary segments
        5. Update data structures
        
        Args:
            inst: The simplet instance.
            i_out: Index of the inequality leaving the basis.
            
        Raises:
            ValueError: If i_out is not in the basis or tangent digraph is not initialized.
        """
        nb_ineq = inst.lp.nb_ineq()
        
        # Verify that i_out belongs to the basis
        if not self.basis_contains(inst, i_out):
            raise ValueError("Simplet.pivot: the input index does not belong to the basis")
        
        if inst.tangent_digraph is None:
            raise ValueError("Simplet.pivot: tangent digraph not initialized")
        
        # 1. Initialize ineq_status based on the tangent digraph
        for i in range(nb_ineq):
            if inst.tangent_digraph.is_hyp_node(i):
                inst.ineq_status[i] = IneqStatus.BREAK_HYP
            else:
                inst.ineq_status[i] = IneqStatus.ENT_HYP
        
        # 2. Mark hyp node as leaving the basis
        inst.ineq_status[i_out] = IneqStatus.INACTIVE
        
        # 3. Find the unique incoming arc to hyp node i_out
        incoming_arc = self._find_incoming_arc(inst, i_out)
        if incoming_arc is None:
            raise ValueError(f"Simplet.pivot: no incoming arc found for hyp node {i_out}")
        
        incoming_var_index, incoming_sign, incoming_entry = incoming_arc
        
        # 4. Remove hyp node i_out from tangent digraph
        inst.tangent_digraph.remove_arcs_of_node(("IneqNode", i_out))
        
        # 5. Compute break hyp and direction
        self._compute_direction_and_break_hyp(inst, incoming_var_index)
        
        # 6. Compute ent hyp and arg_lambda  
        self._compute_arg_lambdas(inst)
        
        # 7. Traverse the tropical edge
        new_point = self._traverse_tropical_edge(inst, incoming_var_index, incoming_entry)
        
        # 8. Update data structures
        inst.point = new_point
        inst.iteration += 1
        
        # Rebuild tangent digraph from the new point
        inst.tangent_digraph = tangent_digraph.TangentDigraph.compute(inst.lp, new_point)
        
        # Recompute data structures after pivot
        self._compute_arg_slacks_pos(inst)
        self._compute_reduced_costs(inst)

    def _find_incoming_arc(self, inst: "Simplet.SimpletInstance", i_out: int) -> Optional[Tuple[linear_prog.ColIndex, linear_prog.Sign, Any]]:
        """
        Find the unique incoming arc to hyp node i_out.
        A hyp node has exactly one incoming arc (POS) and one outgoing arc (NEG).
        
        Args:
            inst: The simplet instance.
            i_out: Index of the hyp node.
            
        Returns:
            Tuple of (var_index, sign, entry) for the incoming arc, or None if not found.
        """
        if inst.tangent_digraph is None:
            return None
            
        ineq_node_arcs = inst.tangent_digraph.get_ineq_node(i_out)
        
        # Search for the incoming arc (positive sign)
        for var_index, sign, entry in ineq_node_arcs:
            if sign == linear_prog.Sign.POS:
                return (var_index, sign, entry)
        
        return None

    def _compute_direction_and_break_hyp(self, inst: "Simplet.SimpletInstance", incoming_var_index: linear_prog.ColIndex) -> None:
        """
        Compute the direction of movement and identify break_hyp inequalities.
        The direction is determined by the incoming arc of the node leaving the basis.
        
        Args:
            inst: The simplet instance.
            incoming_var_index: Index of the variable of the incoming arc.
        """
        # The direction is given by the variable of the incoming arc
        if incoming_var_index[0] == linear_prog.ColKind.VAR:
            j = incoming_var_index[1]
            if j is not None:
                inst.direction = [j]  # Direction along variable j
        else:
            # If the incoming arc is from the affine variable, empty direction
            inst.direction = []

    def _compute_arg_lambdas(self, inst: "Simplet.SimpletInstance") -> None:
        """
        Compute arg_lambda for each inequality during pivoting.
        arg_lambda[i] = argmax_{j in direction} (|W_ij| + x_j)
        
        Args:
            inst: The simplet instance.
        """
        nb_ineq = inst.lp.nb_ineq()
        
        # Reset arg_lambdas
        inst.arg_lambdas = [[] for _ in range(nb_ineq)]
        
        for i in range(nb_ineq):
            # For each inequality, compute the argmax over the direction
            if inst.direction:
                # If there's a defined direction
                max_val = None
                best_args = []
                
                for j in inst.direction:
                    # Find the coefficient W_ij for variable j in inequality i
                    row_i = inst.lp.get_row((linear_prog.RowKind.INEQ, i))
                    for col_index, sign, entry in row_i:
                        if col_index[0] == linear_prog.ColKind.VAR and col_index[1] == j:
                            # Compute |W_ij| + x_j
                            val = inst.lp.compute_entry_plus_var(entry, col_index, inst.point)
                            
                            if max_val is None or self.G.compare(val, max_val) > 0:
                                max_val = val
                                best_args = [(col_index, sign.value, entry)]
                            elif max_val is not None and self.G.compare(val, max_val) == 0:
                                best_args.append((col_index, sign.value, entry))
                
                inst.arg_lambdas[i] = best_args
            else:
                # No direction, empty arg_lambda
                inst.arg_lambdas[i] = []

    def _traverse_tropical_edge(self, inst: "Simplet.SimpletInstance", incoming_var_index: linear_prog.ColIndex, incoming_entry: Any) -> np.ndarray:
        """
        Traverse the tropical edge to the new basic point.
        This is the core of the algorithm: move along the direction until a new inequality becomes active.
        
        Args:
            inst: The simplet instance.
            incoming_var_index: Index of the incoming variable.
            incoming_entry: Entry value of the incoming arc.
            
        Returns:
            The new basic point after traversing the edge.
        """
        current_point = inst.point.copy()
        
        # If the direction is empty (movement along affine variable), the point doesn't change
        if not inst.direction:
            return current_point
        
        # Compute the maximum lambda for which we remain feasible
        lambda_max = None
        
        for i in range(inst.lp.nb_ineq()):
            if inst.ineq_status[i] in [IneqStatus.ENT_HYP, IneqStatus.BREAK_HYP]:
                bound = self._bound_on_length_by_ineq(inst, i)
                if bound is not None:
                    if lambda_max is None or self.G.compare(bound, lambda_max) < 0:
                        lambda_max = bound
        
        # If there's no bound, use a small default value
        if lambda_max is None:
            lambda_max = self.G.one  # Unit movement
        
        # Move the point in the computed direction
        for j in inst.direction:
            if j < len(current_point):
                # x_j := x_j + lambda_max
                current_point[j] = self.G.add(current_point[j], lambda_max)
        
        return current_point


    def solve(
        self,
        inst: "Simplet.SimpletInstance",
        pivot_rule: Callable[["Simplet.SimpletInstance"], Optional[int]],
        log: Optional[TextIO] = None,
        max_iterations: Optional[int] = None,
    ) -> None:
        """
        Solve the tropical LP by iterating pivots until optimality is reached.
        
        Args:
            inst: The simplet instance.
            pivot_rule: Function that selects which inequality should leave the basis.
                        Returns None when optimal.
            log: Optional output stream for logging the solving process.
            max_iterations: Optional cap on the number of pivot iterations (for debugging).
        """
        print("\nSTARTING SIMPLEX\n")

        nb_iter = 1
        is_optimal = False
        
        def myprintf(outchannel, format_str, *args):
            if outchannel is not None:
                print(format_str % args, file=outchannel)

        while not is_optimal:
            # --- debug cap per evitare loop infiniti ---
            if max_iterations is not None and nb_iter > max_iterations:
                myprintf(log, "\n[DEBUG] Reached max_iterations=%d, stopping simplex.", max_iterations)
                print(f"[DEBUG] Reached max_iterations={max_iterations}, stopping simplex.")
                break
            # --------------------------------------------

            myprintf(log, "\niteration %d", nb_iter)
            self.print_status(inst, log)
            
            leaving_ineq = pivot_rule(inst)
            
            if leaving_ineq is None:
                is_optimal = True
                myprintf(log, "\nOptimal reached!")
            else:
                myprintf(log, "\npivoting on %d", leaving_ineq)
                self.pivot(inst, leaving_ineq)
                nb_iter += 1


    # --- Access methods ---
    def basic_point(self, inst: "Simplet.SimpletInstance") -> np.ndarray:
        """
        Return the current basic point.
        
        Args:
            inst: The simplet instance.
            
        Returns:
            The current basic point as a numpy array.
        """
        return inst.point

    def red_cost(self, inst: "Simplet.SimpletInstance", i: int) -> Optional[Tuple[str, Any]]:
        """
        Return the reduced cost for inequality i.
        
        Args:
            inst: The simplet instance.
            i: Index of the inequality.
            
        Returns:
            Tuple of (sign, value) for the reduced cost, or None if not available.
        """
        if 0 <= i < len(inst.reduced_costs):
            return inst.reduced_costs[i]
        return None

    def lp(self, inst: "Simplet.SimpletInstance") -> linear_prog.LP:
        """
        Return the associated linear program.
        
        Args:
            inst: The simplet instance.
            
        Returns:
            The LP instance.
        """
        return inst.lp

    def tangent_digraph(self, inst: "Simplet.SimpletInstance") -> Any:
        """
        Return the tangent digraph at the current basic point.
        
        Args:
            inst: The simplet instance.
            
        Returns:
            The TangentDigraph instance.
        """
        return inst.tangent_digraph

    def print_status(self, inst: "Simplet.SimpletInstance", log: Optional[TextIO] = None) -> None:
        """
        Print the current status of the simplet.
        
        Args:
            inst: The simplet instance.
            log: Optional output stream for printing. If None, nothing is printed.
        """
        if log is None:
            return
            
        print(f"Simplet iteration {inst.iteration}", file=log)
        
        # Print the basis
        print("Basis:", file=log)
        basis_indices = [i for i in range(len(inst.ineq_status)) 
                        if inst.ineq_status[i] == IneqStatus.BASIS]
        print(f"  {basis_indices}", file=log)
        
        # Print the basic point
        print("Basic point:", file=log)
        for j, x_val in enumerate(inst.point):
            print(f"  x{j}: {x_val}", file=log)
        
        # Print reduced costs
        print("Reduced costs:", file=log)
        for i, red_cost in enumerate(inst.reduced_costs):
            if red_cost is None:
                red_cost_str = "null"
            else:
                sign, entry = red_cost
                red_cost_str = f"{sign} {entry}"
            print(f"  y{i}: {red_cost_str}", file=log)
        
        print(file=log)  # Empty line
        
        if log:
            log.flush()

    def print_reduced_costs(self, inst: "Simplet.SimpletInstance", log: TextIO) -> None:
        """
        Print only the reduced costs.
        
        Args:
            inst: The simplet instance.
            log: Output stream for printing.
        """
        print("Reduced costs:", file=log)
        for i, red_cost in enumerate(inst.reduced_costs):
            if red_cost is None:
                red_cost_str = "null"
            else:
                sign, entry = red_cost
                red_cost_str = f"{sign} {entry}"
            print(f"y{i}: {red_cost_str}", file=log)
        
        if log:
            log.flush()