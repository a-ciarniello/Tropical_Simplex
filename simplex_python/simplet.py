"""Pivot routines implementing the tropical simplex algorithm."""

from __future__ import annotations
from typing import Any, Callable, List, Optional, Tuple
import sys
import numpy as np
import linear_prog
import tangent_digraph
import group


IneqStatus = linear_prog.Sign 


class Simplet:
    """Stateful driver that runs the tropical simplex method on an LP model."""
    def __init__(self, lp_module_or_instance):
        if hasattr(lp_module_or_instance, "G") and hasattr(lp_module_or_instance, "nb_ineq"):
            self.LP = lp_module_or_instance  # LP instance
            self.G: group.OrderedGroup = lp_module_or_instance.G
        else:
            self.LP = lp_module_or_instance  # LP module
            self.G: group.OrderedGroup = lp_module_or_instance.G

    class Instance:
        """Mutable solver state tied to a specific LP and basic point."""

        def __init__(self, lp: linear_prog.LP, point: np.ndarray):
            self.lp = lp
            self.point = point.copy()
            self.tangent_digraph: tangent_digraph.TangentDigraph = tangent_digraph.TangentDigraph.compute(lp, point)
            nb_ineq = lp.nb_ineq()
            dim = lp.dim()
            self.arg_slacks: List[List[Tuple[linear_prog.ColIndex, Any]]] = [[] for _ in range(nb_ineq)]
            self.var_in_arg_slack: List[List[int]] = [[] for _ in range(dim)]
            self.affine_var_in_arg_slack: List[int] = []
            self.direction: List[linear_prog.ColIndex] = []
            self.arg_lambdas: List[List[Tuple[linear_prog.ColIndex, str, Any]]] = [[] for _ in range(nb_ineq)]
            self.ineq_status: List[str] = ["Inactive" for _ in range(nb_ineq)]
            self.max_permutation: List[Optional[Tuple[int, str, Any]]] = [None for _ in range(dim)]
            self.reduced_costs: List[Optional[Tuple[str, Any]]] = [None for _ in range(nb_ineq)]
            self.dual_slacks: List[Optional[Tuple[str, Any]]] = [None for _ in range(dim)]
            self.var_seen: List[bool] = [False for _ in range(dim)]
            self.iteration = 0

    # ---- Helpers ----
    def _compute_arg_slacks_pos(self, inst: "Simplet.Instance") -> None:
        nb_ineq = inst.lp.nb_ineq()
        dim = inst.lp.dim()
        inst.arg_slacks = [[] for _ in range(nb_ineq)]
        inst.var_in_arg_slack = [[] for _ in range(dim)]
        inst.affine_var_in_arg_slack = []
        for i in range(nb_ineq):
            arg_slack = inst.lp.compute_slack_args((linear_prog.RowKind.INEQ, i), inst.point)
            pos_entries = [(col_index, entry) for col_index, sign, entry in arg_slack if sign == linear_prog.Sign.POS]


            if not pos_entries and arg_slack:
                pos_entries = [(arg_slack[0][0], arg_slack[0][2])]

            for col_index, entry in pos_entries:
                inst.arg_slacks[i].append((col_index, entry))
                if col_index[0] == linear_prog.ColKind.AFFINE:
                    inst.affine_var_in_arg_slack.append(i)
                else:
                    j = col_index[1]
                    if j is not None:
                        inst.var_in_arg_slack[j].append(i)

    def _bound_on_length_by_ineq(self, inst: "Simplet.Instance", ineq_index: int) -> Optional[Any]:
        arg_slack = inst.arg_slacks[ineq_index]
        arg_lambda = inst.arg_lambdas[ineq_index]
        ineq_status = inst.ineq_status[ineq_index]
        if not arg_slack:
            raise ValueError("Error, slack is +oo")
        if not arg_lambda:
            return None
        (var_slack, entry_slack) = arg_slack[0]
        (var_lambda, sign_lambda, entry_lambda) = arg_lambda[0]
        if ineq_status in ("Basis", "Inactive"):
            return None
        if ineq_status == "EntHyp" and sign_lambda == linear_prog.Sign.POS.value:
            return None
        slack_val = inst.lp.compute_entry_plus_var(entry_slack, var_slack, inst.point)
        lambda_val = inst.lp.compute_entry_plus_var(entry_lambda, var_lambda, inst.point)
        return self.G.substract(slack_val, lambda_val)

    def _compute_direction_and_break_hyp(self, inst: "Simplet.Instance", incoming_var_index: linear_prog.ColIndex) -> None:
        def visit(acc: List[linear_prog.ColIndex], node):
            node_type, node_val = node
            if node_type == "IneqNode":
                i = node_val
                if inst.ineq_status[i] != "BreakHyp":
                    raise ValueError("expected BreakHyp while initializing direction")
                inst.ineq_status[i] = "Basis"
                return acc
            else:  
                acc.append(node_val)
                return acc
        direction = inst.tangent_digraph.dfs_fold_acyclic_graph(visit, [], ("VarNode", incoming_var_index))
        inst.direction = direction

    def _deactivate_ent_hyp_touching_direction(self, inst: "Simplet.Instance") -> None:
        for var_index in inst.direction:
            kind, j = var_index
            if kind == linear_prog.ColKind.AFFINE:
                candidates = inst.affine_var_in_arg_slack
            else:
                candidates = inst.var_in_arg_slack[j]
            for i in candidates:
                if inst.ineq_status[i] == "EntHyp":
                    inst.ineq_status[i] = "Inactive"

    def _compute_arg_lambdas(self, inst: "Simplet.Instance") -> None:
        nb_ineq = inst.lp.nb_ineq()
        inst.arg_lambdas = [[] for _ in range(nb_ineq)]
        for var_index in inst.direction:
            for row_index, sign, entry in inst.lp.get_col(var_index):
                if row_index[0] != linear_prog.RowKind.INEQ:
                    continue
                ineq_idx = row_index[1]
                old_arg = inst.arg_lambdas[ineq_idx]
                val = inst.lp.compute_entry_plus_var(entry, var_index, inst.point)
                if not old_arg:
                    inst.arg_lambdas[ineq_idx] = [(var_index, sign.value, entry)]
                else:
                    old_var, old_sign, old_entry = old_arg[0]
                    old_val = inst.lp.compute_entry_plus_var(old_entry, old_var, inst.point)
                    cmp = self.G.compare(val, old_val)
                    if cmp == 0:
                        inst.arg_lambdas[ineq_idx] = [(var_index, sign.value, entry)] + old_arg
                    elif cmp == 1:
                        inst.arg_lambdas[ineq_idx] = [(var_index, sign.value, entry)]

    def _traverse_break_hyp(self, inst: "Simplet.Instance", break_index: int) -> None:
        arg_lambda = inst.arg_lambdas[break_index]
        if len(arg_lambda) != 1:
            raise ValueError("arg_lambda at breakHyp must be singleton")
        new_var, new_sign, new_entry = arg_lambda[0]
        ineq_node = inst.tangent_digraph.get_ineq_node(break_index)
        same_oriented = [arc for arc in ineq_node if arc[1].value == new_sign]
        if len(same_oriented) != 1:
            raise ValueError("breakHyp node degree error")
        old_var = same_oriented[0][0]
        inst.tangent_digraph.remove_arc(old_var, break_index)

        
        def upd(acc, node):
            ntype, nval = node
            if ntype == "IneqNode":
                i = nval
                if inst.ineq_status[i] != "BreakHyp":
                    raise ValueError("expected BreakHyp while traversing")
                inst.ineq_status[i] = "Basis"
                return acc
            else:
                acc.append(nval)
                return acc
        new_dir_vars = inst.tangent_digraph.dfs_fold_acyclic_graph(upd, [], ("IneqNode", break_index))
        inst.direction = new_dir_vars + inst.direction

        
        for var_index in new_dir_vars:
            kind, j = var_index
            if kind == linear_prog.ColKind.AFFINE:
                candidates = inst.affine_var_in_arg_slack
            else:
                candidates = inst.var_in_arg_slack[j]
            for i in candidates:
                if inst.ineq_status[i] == "EntHyp":
                    inst.ineq_status[i] = "Inactive"

        inst.tangent_digraph.add_arc(new_var, break_index, linear_prog.Sign(new_sign), new_entry)

       
        for var_index in new_dir_vars:
            for row_index, sign, entry in inst.lp.get_col(var_index):
                if row_index[0] != linear_prog.RowKind.INEQ:
                    continue
                ineq_idx = row_index[1]
                old_arg = inst.arg_lambdas[ineq_idx]
                val = inst.lp.compute_entry_plus_var(entry, var_index, inst.point)
                if not old_arg:
                    inst.arg_lambdas[ineq_idx] = [(var_index, sign.value, entry)]
                else:
                    old_var, old_sign, old_entry = old_arg[0]
                    old_val = inst.lp.compute_entry_plus_var(old_entry, old_var, inst.point)
                    cmp = self.G.compare(val, old_val)
                    if cmp == 0:
                        inst.arg_lambdas[ineq_idx] = [(var_index, sign.value, entry)] + old_arg
                    elif cmp == 1:
                        inst.arg_lambdas[ineq_idx] = [(var_index, sign.value, entry)]

    def _traverse_ordinary_segment(self, inst: "Simplet.Instance") -> Optional[int]:
        nb_ineq = inst.lp.nb_ineq()
        dim = inst.lp.dim()

        length = None
        arg_len: List[int] = []
        for i in range(nb_ineq):
            bound_i = self._bound_on_length_by_ineq(inst, i)
            if bound_i is None:
                continue
            if length is None:
                length = bound_i
                arg_len = [i]
            else:
                cmp = self.G.compare(bound_i, length)
                if cmp == 1:
                    continue
                if cmp == 0:
                    arg_len.append(i)
                else:
                    length = bound_i
                    arg_len = [i]
        if length is None or len(arg_len) != 1:
            raise ValueError("unbounded or ambiguous ordinary segment")
        length_ineq = arg_len[0]

        # update point
        has_affine = any(k == linear_prog.ColKind.AFFINE for k, _ in inst.direction)
        if has_affine:
            for j in range(dim):
                inst.point[j] = self.G.substract(inst.point[j], length)
        for var_index in inst.direction:
            kind, j = var_index
            if kind == linear_prog.ColKind.VAR and j is not None:
                inst.point[j] = self.G.add(inst.point[j], length)

        status = inst.ineq_status[length_ineq]
        if status == "BreakHyp":
            self._traverse_break_hyp(inst, length_ineq)
            return None
        if status == "EntHyp":
            return length_ineq
        raise ValueError("unexpected status on length-defining inequality")

    def _compute_max_permutation(self, inst: "Simplet.Instance") -> None:
        dim = inst.lp.dim()
        inst.max_permutation = [None for _ in range(dim)]

        def visit(acc, node):
            ntype, nval = node
            if ntype == "IneqNode":
                i = nval
                ineq_node = inst.tangent_digraph.get_ineq_node(i)
                vars_here = [(var, sign, entry) for var, sign, entry in ineq_node if var[0] == linear_prog.ColKind.VAR]
                if len(vars_here) != 1 and len(vars_here) != 2:
                    raise ValueError("bad degree for hyp node")
                chosen = None
                for var_index, sign, entry in vars_here:
                    j = var_index[1]
                    if j is not None and inst.max_permutation[j] is None:
                        chosen = (j, sign, entry)
                        break
                if chosen is None:
                    return acc
                j, sign, entry = chosen
                inst.max_permutation[j] = (i, sign.value, entry)
            return acc

        inst.tangent_digraph.dfs_fold_acyclic_graph(visit, None, ("VarNode", (linear_prog.ColKind.AFFINE, None)))
        for j, sigma_j in enumerate(inst.max_permutation):
            if sigma_j is None:
                raise ValueError(f"sigma({j}) not defined")

    def _compute_reduced_costs(self, inst: "Simplet.Instance") -> None:
        nb_ineq = inst.lp.nb_ineq()
        dim = inst.lp.dim()
        if not any(inst.arg_slacks):
            self._compute_arg_slacks_pos(inst)
        self._compute_max_permutation(inst)
        inst.reduced_costs = [None for _ in range(nb_ineq)]
        inst.dual_slacks = [None for _ in range(dim)]
        inst.var_seen = [False for _ in range(dim)]

        mu = self.G.one()
        for var_index, sign, entry in inst.lp.get_row((linear_prog.RowKind.OBJECTIVE, None)):
            if var_index[0] == linear_prog.ColKind.AFFINE:
                continue
            val = inst.lp.compute_entry_plus_var(entry, var_index, inst.point)
            mu = self.G.max(mu, val)
        neg_mu = self.G.neg(mu)

        for var_index, sign, entry in inst.lp.get_row((linear_prog.RowKind.OBJECTIVE, None)):
            if var_index[0] != linear_prog.ColKind.VAR:
                continue
            j = var_index[1]
            x_j = inst.point[j]
            scaled = self.G.sum([neg_mu, x_j, entry])
            inst.dual_slacks[j] = (sign.value, scaled)

        def find_max_unseen():
            best_j = None
            best_val = None
            for j in range(dim):
                if inst.var_seen[j]:
                    continue
                val = inst.dual_slacks[j]
                if val is None:
                    continue
                if best_val is None or self.G.compare(val[1], best_val[1]) == 1:
                    best_j = j
                    best_val = val
            return best_j

        finished = False
        while not finished:
            j = find_max_unseen()
            if j is None:
                finished = True
                break
            inst.var_seen[j] = True
            sigma_j, sigma_sign, sigma_entry = inst.max_permutation[j]
            j_sign, j_entry = inst.dual_slacks[j]
            slack_sigma = inst.lp.compute_entry_plus_var(inst.arg_slacks[sigma_j][0][1], inst.arg_slacks[sigma_j][0][0], inst.point)
            entry_sigma = self.G.add(j_entry, mu)
            sign_sigma = "Pos" if (j_sign == sigma_sign) else "Neg"
            inst.reduced_costs[sigma_j] = (sign_sigma, entry_sigma)
            for arc_index, arc_sign, arc_entry in inst.lp.get_row((linear_prog.RowKind.INEQ, sigma_j)):
                if arc_index[0] != linear_prog.ColKind.VAR:
                    continue
                l = arc_index[1]
                if inst.var_seen[l]:
                    continue
                x_l = inst.point[l]
                new_dist = self.G.sum([neg_mu, x_l, arc_entry, self.G.neg(slack_sigma), entry_sigma])
                new_sign = "Pos" if (sign_sigma != arc_sign.value) else "Neg"
                better = False
                current = inst.dual_slacks[l]
                if current is None:
                    better = True
                else:
                    _, cur_val = current
                    if self.G.compare(new_dist, cur_val) == -1:
                        better = True
                if better:
                    inst.dual_slacks[l] = (new_sign, new_dist)


        for i in range(nb_ineq):
            slack_i = inst.lp.compute_entry_plus_var(inst.arg_slacks[i][0][1], inst.arg_slacks[i][0][0], inst.point)
            z_i = inst.reduced_costs[i]
            if z_i is None:
                continue
            sign, value = z_i
            inst.reduced_costs[i] = (sign, self.G.add(value, self.G.neg(slack_i)))


    def init(self, lp: linear_prog.LP, basic_point: np.ndarray) -> "Simplet.Instance":
        
        self.G = lp.G
        tg = tangent_digraph.TangentDigraph.compute(lp, basic_point)

        if not tg.is_basic_point():
            raise ValueError("input point is not a basic point")
        if not lp.is_point_feasible(basic_point, allow_all_neg=True):
            raise ValueError("input point is not feasible")
        
        inst = Simplet.Instance(lp, basic_point)
        inst.tangent_digraph = tg
        nb_ineq = lp.nb_ineq()

        for i in range(nb_ineq):
            if tg.is_hyp_node(i):
                inst.ineq_status[i] = "Basis"
        self._compute_arg_slacks_pos(inst)
        self._compute_reduced_costs(inst)

        return inst

    def basis_contains(self, inst: "Simplet.Instance", ineq_index: int) -> bool:
        return inst.tangent_digraph.is_hyp_node(ineq_index)

    def red_cost(self, inst: "Simplet.Instance", i: int):
        return inst.reduced_costs[i]

    def basic_point(self, inst: "Simplet.Instance") -> np.ndarray:
        return inst.point

    def print_status(self, inst: "Simplet.Instance", out=None):
        if out:
            print("basis:", file=out)
            for i in range(inst.lp.nb_ineq()):
                if inst.tangent_digraph.is_hyp_node(i):
                    print(f"{i}, ", end="", file=out)
            print("\npoint:", file=out)
            for j, x in enumerate(inst.point):
                print(f"x{j}: {self.G.to_string(x)}", file=out)
            print("reduced costs:", file=out)
            for i, rc in enumerate(inst.reduced_costs):
                if rc is None:
                    s = "null"
                else:
                    sgn, v = rc
                    s = f"{sgn} {self.G.to_string(v)}"
                print(f"y{i}: {s}", file=out)

    def print_reduced_costs(self, inst: "Simplet.Instance", out=None):
        target = out or sys.stdout
        for i, rc in enumerate(inst.reduced_costs):
            if rc is None:
                s = "null"
            else:
                sgn, v = rc
                s = f"{sgn} {self.G.to_string(v)}"
            print(f"y{i}: {s}", file=target)

    def bland_rule(self, inst: "Simplet.Instance") -> Optional[int]:
        for i in range(inst.lp.nb_ineq()):
            if not inst.tangent_digraph.is_hyp_node(i):
                continue
            rc = inst.reduced_costs[i]
            if rc is not None and rc[0] == linear_prog.Sign.NEG.value:
                return i
        return None

    def get_pivot_rule_for_objective(self, maximize: bool = False) -> Callable[["Simplet.Instance"], Optional[int]]:
        return self.bland_rule

    def _find_incoming_arc(self, inst: "Simplet.Instance", i_out: int):
        arcs = inst.tangent_digraph.get_ineq_node(i_out)
        for var_index, sign, entry in arcs:
            if sign == linear_prog.Sign.POS:
                return (var_index, sign, entry)
        return None

    def pivot(self, inst: "Simplet.Instance", i_out: int) -> None:
        """Execute one pivot that removes inequality ``i_out`` from the basis."""
        if not self.basis_contains(inst, i_out):
            raise ValueError("input index does not belong to the basis")
        nb_ineq = inst.lp.nb_ineq()
        for i in range(nb_ineq):
            if inst.tangent_digraph.is_hyp_node(i):
                inst.ineq_status[i] = "BreakHyp"
            else:
                inst.ineq_status[i] = "EntHyp"
        inst.ineq_status[i_out] = "Inactive"

        incoming = self._find_incoming_arc(inst, i_out)
        if incoming is None:
            raise ValueError("no incoming arc found")
        incoming_var_index, _, incoming_entry = incoming

        inst.tangent_digraph.remove_arcs_of_node(("IneqNode", i_out))

        self._compute_direction_and_break_hyp(inst, incoming_var_index)
        self._deactivate_ent_hyp_touching_direction(inst)
        self._compute_arg_lambdas(inst)


        visited = 0
        while True:
            if visited > inst.lp.dim():
                raise ValueError("visited more ordinary segments than dimension")
            visited += 1
            i_ent = self._traverse_ordinary_segment(inst)
            if i_ent is None:
                continue
            arg_lambda = inst.arg_lambdas[i_ent]
            if len(arg_lambda) != 1 or arg_lambda[0][1] != linear_prog.Sign.NEG.value:
                raise ValueError("arg_lambda at new basic point must be singleton with Neg")
            out_var, out_sign, out_entry = arg_lambda[0]

            inst.tangent_digraph.add_arc(out_var, i_ent, linear_prog.Sign.NEG, out_entry)

            arg_slack = inst.arg_slacks[i_ent]
            if len(arg_slack) != 1:
                raise ValueError("arg_slack at new basic point must be singleton")
            in_var, in_entry = arg_slack[0]

            inst.tangent_digraph.add_arc(in_var, i_ent, linear_prog.Sign.POS, in_entry)

            break

        inst.arg_slacks = [[] for _ in range(nb_ineq)]
        self._compute_arg_slacks_pos(inst)
        self._compute_reduced_costs(inst)

    def solve(self, inst: "Simplet.Instance", pivot_rule: Callable[["Simplet.Instance"], Optional[int]], out=None, max_iterations: int = 10000) -> None:
        """Iteratively pivot until optimality or ``max_iterations`` is reached."""
        it = 1
        while True:
            if it > max_iterations:
                raise RuntimeError("maximum iterations reached")
            if out:
                print(f"\niteration {it}", file=out)
                self.print_status(inst, out)
            leaving = pivot_rule(inst)
            if leaving is None:
                break
            self.pivot(inst, leaving)
            it += 1
