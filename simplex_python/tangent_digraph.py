from __future__ import annotations
from dataclasses import dataclass
from typing import Any, List, Tuple, Optional, Union
import linear_prog
import group


Sign = linear_prog.Sign
ColIndex = linear_prog.ColIndex
RowIndex = linear_prog.RowIndex
VarIndex = ColIndex
IneqIndex = int

VarNodeIndex = Tuple[str, VarIndex]  # ("VarNode", var_index)
IneqNodeIndex = Tuple[str, int]      # ("IneqNode", i)
NodeIndex = Union[VarNodeIndex, IneqNodeIndex]


@dataclass
class TangentDigraph:
    """Tangent digraph associato a un punto di un tropical LP."""
    var_nodes: List[List[Tuple[IneqIndex, Sign, Any]]]
    affine_var_node: List[Tuple[IneqIndex, Sign, Any]]
    ineq_nodes: List[List[Tuple[VarIndex, Sign, Any]]]

    # === Accessori ===
    def get_ineq_node(self, i: int):
        return self.ineq_nodes[i]

    def get_var_node(self, var_index: VarIndex):
        kind, j = var_index
        if kind == linear_prog.ColKind.AFFINE:
            return self.affine_var_node
        elif kind == linear_prog.ColKind.VAR and j is not None:
            return self.var_nodes[j]
        else:
            raise ValueError(f"Invalid var_index: {var_index}")

    def is_hyp_node(self, i: int) -> bool:
        arcs = self.ineq_nodes[i]
        nb_pos, nb_neg = 0, 0
        for _, sign, _ in arcs:
            if sign == linear_prog.Sign.POS:
                nb_pos += 1
            elif sign == linear_prog.Sign.NEG:
                nb_neg += 1
            if nb_pos > 1 or nb_neg > 1:
                return False
        return nb_pos == 1 and nb_neg == 1

    # === Costruzione ===
    @classmethod
    def compute(cls, lp: linear_prog.LP, point: Any) -> "TangentDigraph":
        """Crea il grafo tangente del LP al punto dato."""
        dim = lp.dim()
        nb_ineq = lp.nb_ineq()

        if len(point) != dim:
            raise ValueError(
                f"dimension mismatch: LP has {dim} vars, point has {len(point)}"
            )

        var_nodes = [[] for _ in range(dim)]
        affine_var_node: List[Tuple[IneqIndex, Sign, Any]] = []
        ineq_nodes = [[] for _ in range(nb_ineq)]

        for i in range(nb_ineq):
            arg = lp.compute_slack_args((linear_prog.RowKind.INEQ, i), point)

            # Debug: uncomment to see which variables are active for each inequality at this point
            # print(linear_prog.RowKind.INEQ, i, "args:", arg)

            # Check if the inequality is saturated (has both + and -)
            has_pos = any(sign == linear_prog.Sign.POS for _, sign, _ in arg)
            has_neg = any(sign == linear_prog.Sign.NEG for _, sign, _ in arg)

            # Allow purely-negative rows (e.g., infinity plane in Phase I) to be initially non-saturated
            if not has_pos:
                all_neg = all(sign == linear_prog.Sign.NEG for _, sign, _ in arg)
                if all_neg:
                    continue
                raise ValueError(f"Error while initializing tangent digraph. "
                               f"Input point does not satisfy inequality indexed by {i}")
            
            if has_pos and has_neg:
                # Inequality is saturated (active) - create a hyp node
                ineq_nodes[i] = arg  # active node
            # If has_pos and not has_neg: inequality is satisfied but not saturated - skip it

        # Build variable nodes
        for i, arcs in enumerate(ineq_nodes):
            for var_index, sign, entry in arcs:
                if var_index[0] == linear_prog.ColKind.AFFINE:
                    affine_var_node.append((i, sign, entry))
                elif var_index[0] == linear_prog.ColKind.VAR:
                    j = var_index[1]
                    if j is not None:
                        var_nodes[j].append((i, sign, entry))

        return cls(var_nodes, affine_var_node, ineq_nodes)

    # === Diagnostica ===
    def nb_connected_component(self) -> int:
        seen_vars = [False] * len(self.var_nodes)
        seen_ineqs = [False] * len(self.ineq_nodes)
        seen_affine = False
        count = 0

        def dfs_var(j):
            nonlocal count
            seen_vars[j] = True
            for i, _, _ in self.var_nodes[j]:
                if not seen_ineqs[i]:
                    dfs_ineq(i)

        def dfs_ineq(i):
            seen_ineqs[i] = True
            for var_index, _, _ in self.ineq_nodes[i]:
                if var_index[0] == linear_prog.ColKind.AFFINE:
                    nonlocal seen_affine
                    seen_affine = True
                elif var_index[0] == linear_prog.ColKind.VAR:
                    j = var_index[1]
                    if j is not None and not seen_vars[j]:
                        dfs_var(j)

        for i in range(len(self.ineq_nodes)):
            if not seen_ineqs[i]:
                count += 1
                dfs_ineq(i)
        for j in range(len(self.var_nodes)):
            if not seen_vars[j]:
                count += 1
                dfs_var(j)
        if not seen_affine:
            count += 1
        return count

    def is_basic_point(self) -> bool:
        """
        Check if the tangent digraph corresponds to a basic point.
        A basic point must saturate exactly dim inequalities (have dim hyp nodes).
        """
        nb_hyp = sum(1 for i in range(len(self.ineq_nodes)) if self.is_hyp_node(i))
        nb_vars = len(self.var_nodes) + 1  # +1 for affine variable
        dim = nb_vars - 1  # dimension = number of non-affine variables
        
        # A basic point must have exactly dim saturated inequalities
        return nb_hyp == dim

    # === Manipolazione del Grafo (necessari per Simplet) ===
    
    def add_arc(self, var_index: VarIndex, ineq_index: int, sign: Sign, entry: Any) -> None:
        """Aggiunge un arco tra var_index e ineq_index con segno e peso dati."""
        # Aggiungi l'arco al nodo disuguaglianza
        self.ineq_nodes[ineq_index].append((var_index, sign, entry))
        
        # Aggiungi l'arco al nodo variabile corrispondente
        if var_index[0] == linear_prog.ColKind.AFFINE:
            self.affine_var_node.append((ineq_index, sign, entry))
        elif var_index[0] == linear_prog.ColKind.VAR:
            j = var_index[1]
            if j is not None:
                self.var_nodes[j].append((ineq_index, sign, entry))

    def remove_arc(self, var_index: VarIndex, ineq_index: int) -> None:
        """Rimuove l'arco tra var_index e ineq_index."""
        # Rimuovi dal nodo disuguaglianza
        self.ineq_nodes[ineq_index] = [
            (v, s, e) for v, s, e in self.ineq_nodes[ineq_index] 
            if v != var_index
        ]
        
        # Rimuovi dal nodo variabile
        if var_index[0] == linear_prog.ColKind.AFFINE:
            self.affine_var_node = [
                (i, s, e) for i, s, e in self.affine_var_node 
                if i != ineq_index
            ]
        elif var_index[0] == linear_prog.ColKind.VAR:
            j = var_index[1]
            if j is not None:
                self.var_nodes[j] = [
                    (i, s, e) for i, s, e in self.var_nodes[j] 
                    if i != ineq_index
                ]

    def remove_arcs_of_node(self, node_index: NodeIndex) -> None:
        """Rimuove tutti gli archi che coinvolgono il nodo specificato."""
        node_type, node_value = node_index
        
        if node_type == "IneqNode":
            # node_value è int per IneqNode
            ineq_index = node_value
            if isinstance(ineq_index, int):
                # Ottieni tutti gli archi da rimuovere
                arcs_to_remove = list(self.ineq_nodes[ineq_index])
                # Rimuovi ogni arco
                for var_index, _, _ in arcs_to_remove:
                    self.remove_arc(var_index, ineq_index)
                
        elif node_type == "VarNode":
            # node_value è VarIndex per VarNode
            var_index = node_value
            if isinstance(var_index, tuple):
                # Ottieni tutti gli archi da rimuovere  
                arcs_to_remove = list(self.get_var_node(var_index))
                # Rimuovi ogni arco
                for ineq_index, _, _ in arcs_to_remove:
                    self.remove_arc(var_index, ineq_index)

    def dfs_fold_acyclic_graph(self, f: Any, acc: Any, start_node: NodeIndex) -> Any:
        """
        Attraversa il grafo aciclico in DFS applicando la funzione f.
        
        Corrisponde a dfs_fold_acyclic_graph nell'originale OCaml.
        Essenziale per il calcolo della permutazione massimizzante in Simplet.
        """
        visited_vars = set()
        visited_ineqs = set()
        visited_affine = False
        
        def visit_node(node: NodeIndex, current_acc):
            nonlocal visited_vars, visited_ineqs, visited_affine
            
            node_type, node_value = node
            
            # Applica la funzione al nodo corrente
            current_acc = f(current_acc, node)
            
            if node_type == "VarNode":
                # Type guard: node_value è VarIndex per VarNode
                if isinstance(node_value, tuple):
                    var_index = node_value
                    
                    # Marca come visitato
                    if var_index[0] == linear_prog.ColKind.AFFINE:
                        visited_affine = True
                    elif var_index[0] == linear_prog.ColKind.VAR:
                        j = var_index[1]
                        if j is not None:
                            visited_vars.add(j)
                    
                    # Visita i nodi disuguaglianza connessi
                    var_node_arcs = self.get_var_node(var_index)
                    for ineq_index, _, _ in var_node_arcs:
                        if ineq_index not in visited_ineqs:
                            current_acc = visit_node(("IneqNode", ineq_index), current_acc)
                        
            elif node_type == "IneqNode":
                # Type guard: node_value è int per IneqNode
                if isinstance(node_value, int):
                    ineq_index = node_value
                    visited_ineqs.add(ineq_index)
                    
                    # Visita i nodi variabile connessi
                    ineq_node_arcs = self.ineq_nodes[ineq_index]
                    for var_index, _, _ in ineq_node_arcs:
                        should_visit = False
                        
                        if var_index[0] == linear_prog.ColKind.AFFINE:
                            should_visit = not visited_affine
                        elif var_index[0] == linear_prog.ColKind.VAR:
                            j = var_index[1]
                            should_visit = j is not None and j not in visited_vars
                        
                        if should_visit:
                            current_acc = visit_node(("VarNode", var_index), current_acc)
            
            return current_acc
        
        return visit_node(start_node, acc)

    def print(self):
        """Stampa il grafo tangente per debug."""
        print("=== Tangent Digraph ===")
        print("Inequality nodes:")
        for i, arcs in enumerate(self.ineq_nodes):
            incoming = [(v, s, e) for v, s, e in arcs if s == linear_prog.Sign.POS]
            outgoing = [(v, s, e) for v, s, e in arcs if s == linear_prog.Sign.NEG]
            hyp = "HYP" if self.is_hyp_node(i) else "NON-HYP"
            print(f"  Ineq {i} ({hyp}):")
            print(f"    incoming: {incoming}")
            print(f"    outgoing: {outgoing}")
        
        print("Variable nodes:")
        print(f"  Affine: {self.affine_var_node}")
        for j, arcs in enumerate(self.var_nodes):
            print(f"  Var {j}: {arcs}")
        
        print(f"Basic point: {self.is_basic_point()}")
        print(f"Connected components: {self.nb_connected_component()}")
        print("========================")
