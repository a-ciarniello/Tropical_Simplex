from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Tuple, Union
import sys
import linear_prog


Sign = linear_prog.Sign
ColKind = linear_prog.ColKind
RowKind = linear_prog.RowKind
ColIndex = linear_prog.ColIndex
RowIndex = linear_prog.RowIndex
VarIndex = ColIndex
IneqIndex = int

VarNodeIndex = Tuple[str, VarIndex]  # ("VarNode", var_index)
IneqNodeIndex = Tuple[str, int]      # ("IneqNode", i)
NodeIndex = Union[VarNodeIndex, IneqNodeIndex]


@dataclass
class TangentDigraph:
    var_nodes: List[List[Tuple[IneqIndex, Sign, Any]]]
    affine_var_node: List[Tuple[IneqIndex, Sign, Any]]
    ineq_nodes: List[List[Tuple[VarIndex, Sign, Any]]]

    # --- Accessors ---
    def get_ineq_node(self, ineq_index: int) -> List[Tuple[VarIndex, Sign, Any]]:
        try:
            return self.ineq_nodes[ineq_index]
        except IndexError as exc:
            raise ValueError("tangent_digraph.get_ineq_node: index out of bounds") from exc

    def get_var_node(self, var_index: VarIndex) -> List[Tuple[IneqIndex, Sign, Any]]:
        kind, j = var_index
        try:
            if kind == ColKind.AFFINE:
                return self.affine_var_node
            if kind == ColKind.VAR and j is not None:
                return self.var_nodes[j]
        except IndexError as exc:
            raise ValueError("tangent_digraph.get_var_node: index out of bounds") from exc
        raise ValueError(f"tangent_digraph.get_var_node: invalid var_index {var_index}")

    def is_hyp_node(self, ineq_index: int) -> bool:
        arcs = self.get_ineq_node(ineq_index)
        nb_pos, nb_neg = 0, 0
        for _, sign, _ in arcs:
            if sign == Sign.POS:
                nb_pos += 1
            else:
                nb_neg += 1
            if nb_pos > 1 or nb_neg > 1:
                return False
        return nb_pos == 1 and nb_neg == 1

    # --- Construction ---
    @classmethod
    def compute(cls, lp: linear_prog.LP, point: Any) -> "TangentDigraph":
        dim = lp.dim()
        nb_ineq = lp.nb_ineq()

        if len(point) != dim:
            raise ValueError(
                f"Error while computing tangent digraph: linear program has {dim} variables and input point has     dimension {len(point)}"
            )

        var_nodes: List[List[Tuple[IneqIndex, Sign, Any]]] = [[] for _ in range(dim)]
        affine_var_node: List[Tuple[IneqIndex, Sign, Any]] = []
        ineq_nodes: List[List[Tuple[VarIndex, Sign, Any]]] = [[] for _ in range(nb_ineq)]

        def compute_hyp_nodes(ineq_index: int, hyp_nodes: List[int]) -> List[int]:
            if ineq_index >= nb_ineq:
                return hyp_nodes
            arg = lp.compute_slack_args((RowKind.INEQ, ineq_index), point)
            pos, neg = False, False
            for _, sign, _ in arg:
                if sign == Sign.POS:
                    pos = True
                else:
                    neg = True
            if not pos:
                raise ValueError(
                    f"Error while initializing tangent digraph. Input point does not satisfy inequality indexed by {ineq_index}"
                )
            if pos and neg:
                ineq_nodes[ineq_index] = arg
                return compute_hyp_nodes(ineq_index + 1, hyp_nodes + [ineq_index])
            return compute_hyp_nodes(ineq_index + 1, hyp_nodes)

        hyp_nodes = compute_hyp_nodes(0, [])

        def add_arcs_from_ineq(ineq_index: int) -> None:
            for var_index, sign, entry in ineq_nodes[ineq_index]:
                if var_index[0] == ColKind.AFFINE:
                    affine_var_node.append((ineq_index, sign, entry))
                else:
                    j = var_index[1]
                    var_nodes[j].append((ineq_index, sign, entry))

        for i in hyp_nodes:
            add_arcs_from_ineq(i)

        return cls(var_nodes=var_nodes, affine_var_node=affine_var_node, ineq_nodes=ineq_nodes)

    # --- Mutations ---
    def contains_arc(self, var_index: VarIndex, ineq_index: int) -> bool:
        ineq_node = self.get_ineq_node(ineq_index)
        return any(iter_var_index == var_index for iter_var_index, _, _ in ineq_node)

    def add_arc(self, var_index: VarIndex, ineq_index: int, sign: Sign, entry: Any) -> None:
        def ensure_absent(arcs: List[Tuple[Any, Any, Any]], node_index: Any) -> None:
            for iter_node_index, _, _ in arcs:
                if iter_node_index == node_index:
                    raise ValueError("TangentDigraph.add_arc: arc already exists")

        var_node = self.get_var_node(var_index)
        ensure_absent(var_node, ineq_index)
        var_node_new = [(ineq_index, sign, entry)] + var_node
        if var_index[0] == ColKind.AFFINE:
            self.affine_var_node = var_node_new
        else:
            self.var_nodes[var_index[1]] = var_node_new

        ineq_node = self.get_ineq_node(ineq_index)
        ensure_absent(ineq_node, var_index)
        self.ineq_nodes[ineq_index] = [(var_index, sign, entry)] + ineq_node

    def remove_arc(self, var_index: VarIndex, ineq_index: int) -> None:
        def remove_arc_from_list(node_index: Any, lst: List[Tuple[Any, Any, Any]]):
            to_remove = [arc for arc in lst if arc[0] == node_index]
            remaining = [arc for arc in lst if arc[0] != node_index]
            if len(to_remove) == 1:
                return remaining
            if len(to_remove) == 0:
                raise ValueError("TangentDigraph.remove_arc: arc does not exist")
            raise ValueError("TangentDigraph.remove_arc: multiple occurrences of arc")

        var_node = self.get_var_node(var_index)
        new_var_node = remove_arc_from_list(ineq_index, var_node)
        if var_index[0] == ColKind.AFFINE:
            self.affine_var_node = new_var_node
        else:
            self.var_nodes[var_index[1]] = new_var_node

        ineq_node = self.get_ineq_node(ineq_index)
        new_ineq_node = remove_arc_from_list(var_index, ineq_node)
        self.ineq_nodes[ineq_index] = new_ineq_node

    def remove_arcs_of_node(self, node_index: NodeIndex) -> None:
        node_type, payload = node_index
        if node_type == "VarNode":
            arcs = list(self.get_var_node(payload))
            for ineq_index, _, _ in arcs:
                self.remove_arc(payload, ineq_index)
        elif node_type == "IneqNode":
            arcs = list(self.get_ineq_node(payload))
            for var_index, _, _ in arcs:
                self.remove_arc(var_index, payload)
        else:
            raise ValueError(f"TangentDigraph.remove_arcs_of_node: invalid node {node_index}")

    # --- Traversals ---
    def dfs_fold_acyclic_graph(self, f: Callable[[Any, NodeIndex], Any], acc: Any, start_node_index: NodeIndex) -> Any:
        def dfs(node_index: NodeIndex, parent: NodeIndex | None, acc_value: Any) -> Any:
            acc1 = f(acc_value, node_index)
            node_type, payload = node_index
            if node_type == "VarNode":
                arcs = self.get_var_node(payload)
                for ineq_index, _, _ in arcs:
                    neighbor: NodeIndex = ("IneqNode", ineq_index)
                    if parent is not None and neighbor == parent:
                        continue
                    acc1 = dfs(neighbor, node_index, acc1)
            elif node_type == "IneqNode":
                arcs = self.get_ineq_node(payload)
                for var_index, _, _ in arcs:
                    neighbor = ("VarNode", var_index)
                    if parent is not None and neighbor == parent:
                        continue
                    acc1 = dfs(neighbor, node_index, acc1)
            else:
                raise ValueError(f"dfs_fold_acyclic_graph: invalid node {node_index}")
            return acc1

        return dfs(start_node_index, None, acc)

    # --- Connectivity and certificates ---
    def _depth_first_iter(self, visit: Callable[[NodeIndex], None], node_index: NodeIndex) -> None:
        var_seen = [False] * len(self.var_nodes)
        affine_seen = False
        ineq_seen = [False] * len(self.ineq_nodes)

        def is_seen(node: NodeIndex) -> bool:
            node_type, payload = node
            if node_type == "VarNode":
                kind, j = payload
                if kind == ColKind.AFFINE:
                    return affine_seen
                return var_seen[j]
            if node_type == "IneqNode":
                return ineq_seen[payload]
            raise ValueError(f"depth_first_iter: invalid node {node}")

        def mark_seen(node: NodeIndex) -> None:
            nonlocal affine_seen
            node_type, payload = node
            if node_type == "VarNode":
                kind, j = payload
                if kind == ColKind.AFFINE:
                    affine_seen = True
                else:
                    var_seen[j] = True
            elif node_type == "IneqNode":
                ineq_seen[payload] = True
            else:
                raise ValueError(f"depth_first_iter: invalid node {node}")

        def depth_first_iter_aux(node: NodeIndex) -> None:
            mark_seen(node)
            visit(node)
            node_type, payload = node
            if node_type == "VarNode":
                arcs = self.get_var_node(payload)
                for ineq_index, _, _ in arcs:
                    neighbor: NodeIndex = ("IneqNode", ineq_index)
                    if not is_seen(neighbor):
                        depth_first_iter_aux(neighbor)
            elif node_type == "IneqNode":
                arcs = self.get_ineq_node(payload)
                for var_index, _, _ in arcs:
                    neighbor = ("VarNode", var_index)
                    if not is_seen(neighbor):
                        depth_first_iter_aux(neighbor)

        depth_first_iter_aux(node_index)

    def nb_connected_component(self) -> int:
        var_seen = [False] * len(self.var_nodes)
        affine_seen = False
        ineq_seen = [False] * len(self.ineq_nodes)

        def mark_seen(node: NodeIndex) -> None:
            nonlocal affine_seen
            node_type, payload = node
            if node_type == "VarNode":
                kind, j = payload
                if kind == ColKind.AFFINE:
                    affine_seen = True
                else:
                    var_seen[j] = True
            elif node_type == "IneqNode":
                ineq_seen[payload] = True

        def depth_first_iter(node: NodeIndex) -> None:
            def dfs_local(current: NodeIndex) -> None:
                mark_seen(current)
                node_type, payload = current
                if node_type == "VarNode":
                    arcs = self.get_var_node(payload)
                    for ineq_index, _, _ in arcs:
                        if not ineq_seen[ineq_index]:
                            dfs_local(("IneqNode", ineq_index))
                elif node_type == "IneqNode":
                    arcs = self.get_ineq_node(payload)
                    for var_index, _, _ in arcs:
                        kind, j = var_index
                        if kind == ColKind.AFFINE:
                            if not affine_seen:
                                dfs_local(("VarNode", var_index))
                        elif not var_seen[j]:
                            dfs_local(("VarNode", var_index))
            dfs_local(node)

        nb_cc = 0
        for i in range(len(self.ineq_nodes)):
            if not ineq_seen[i]:
                nb_cc += 1
                depth_first_iter(("IneqNode", i))
        for j in range(len(self.var_nodes)):
            if not var_seen[j]:
                nb_cc += 1
                depth_first_iter(("VarNode", (ColKind.VAR, j)))
        if not affine_seen:
            nb_cc += 1
        return nb_cc

    def _degree_of_ineq_node(self, ineq_index: int) -> Tuple[int, int]:
        ineq_node = self.get_ineq_node(ineq_index)
        nb_pos, nb_neg = 0, 0
        for _, sign, _ in ineq_node:
            if sign == Sign.POS:
                nb_pos += 1
            else:
                nb_neg += 1
        return nb_pos, nb_neg

    def _nb_hyp_nodes(self) -> int:
        return sum(1 for i in range(len(self.ineq_nodes)) if self.is_hyp_node(i))

    def _nb_var_nodes(self) -> int:
        return len(self.var_nodes) + 1

    def _nb_ineq_nodes(self) -> int:
        return len(self.ineq_nodes)

    def is_basic_point(self) -> bool:
        nb_hyp_nodes = self._nb_hyp_nodes()
        nb_var_nodes = self._nb_var_nodes()
        nb_ineq_nodes = self._nb_ineq_nodes()

        if nb_hyp_nodes != nb_var_nodes - 1:
            print(
                "Not a tangent digraph of a basic point: nb of hyperplane nodes not equal to nb of variables.",
                file=sys.stderr,
            )
            return False

        def degrees_ok(idx: int) -> bool:
            if idx >= len(self.ineq_nodes):
                return True
            nb_pos, nb_neg = self._degree_of_ineq_node(idx)
            if (nb_pos, nb_neg) in ((1, 1), (0, 0)):
                return degrees_ok(idx + 1)
            return False

        if not degrees_ok(0):
            return False

        nb_cc = self.nb_connected_component()
        if nb_cc == (nb_ineq_nodes - (nb_var_nodes - 1) + 1):
            return True
        print(
            f"Not a tangent digraph of a basic point: there is {nb_cc} connected components instead of {nb_ineq_nodes - (nb_var_nodes - 1) + 1}.",
            file=sys.stderr,
        )
        return False

    def is_breakpoint(self) -> bool:
        nb_hyp_nodes = self._nb_hyp_nodes()
        nb_var_nodes = self._nb_var_nodes()
        nb_ineq_nodes = self._nb_ineq_nodes()

        if nb_hyp_nodes != nb_var_nodes - 2:
            print(
                "Not a tangent digraph of a breakpoint: nb of hyperplane nodes not equal to nb of variables.",
                file=sys.stderr,
            )
            return False

        def degrees_ok(degree_three_found: bool, idx: int) -> bool:
            if idx >= len(self.ineq_nodes):
                return degree_three_found
            nb_pos, nb_neg = self._degree_of_ineq_node(idx)
            if (nb_pos, nb_neg) in ((1, 1), (0, 0)):
                return degrees_ok(degree_three_found, idx + 1)
            if (nb_pos, nb_neg) in ((2, 1), (1, 2)) and not degree_three_found:
                return degrees_ok(True, idx + 1)
            if (nb_pos, nb_neg) in ((2, 1), (1, 2)) and degree_three_found:
                print(
                    "Not a tangent digraph of a breakpoint: found two hyperplane nodes of degree 3.",
                    file=sys.stderr,
                )
            else:
                print(
                    f"Not a tangent digraph of a breakpoint:  hyperplane node {idx} is not of degree 2 or 3.",
                    file=sys.stderr,
                )
            return False

        if not degrees_ok(False, 0):
            return False

        nb_cc = self.nb_connected_component()
        if nb_cc == nb_ineq_nodes - (nb_var_nodes - 1) + 1:
            return True
        print(
            f"Not a tangent digraph of a breakpoint: there is {nb_cc} connected components instead of {nb_ineq_nodes - (nb_var_nodes - 1) + 1}.",
            file=sys.stderr,
        )
        return False

