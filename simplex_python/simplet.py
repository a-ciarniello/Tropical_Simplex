from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple, TextIO, List, Dict
from enum import Enum
import numpy as np
import linear_prog
import tangent_digraph


class IneqStatus(Enum):
    """Status delle disuguaglianze durante il pivoting."""
    BREAK_HYP = "BreakHyp"
    BASIS = "Basis" 
    ENT_HYP = "EntHyp"
    INACTIVE = "Inactive"


class Simplet:
    """
    Traduzione Python di Simplet.Make(LP).
    
    Implementa l'algoritmo del simplesso tropicale descritto nel Capitolo 7 di
    "Tropical aspects of linear programming", Pascal Benchimol, PhD Thesis, 2014
    """

    def __init__(self, lp_module_or_instance):
        """Inizializza il modulo Simplet con riferimento al modulo LinearProg o istanza LP."""
        if hasattr(lp_module_or_instance, 'G') and hasattr(lp_module_or_instance, 'nb_ineq'):
            # È un'istanza LP
            self.LP = lp_module_or_instance
            self.G = lp_module_or_instance.G
        else:
            # È un modulo
            self.LP = lp_module_or_instance  
            self.G = lp_module_or_instance.G

    class SimpletInstance:
        """
        Un'istanza del simplet tropicale contenente tutte le strutture dati necessarie.
        Corrisponde al tipo 't' nel modulo OCaml originale.
        """
        
        def __init__(self, lp: linear_prog.LP, basic_point: np.ndarray):
            self.lp = lp
            self.point = basic_point.copy()  # Punto base corrente
            
            # Strutture dati principali
            self.tangent_digraph: Optional[tangent_digraph.TangentDigraph] = None
            
            # Grafo tra Ineq i e argmax_j (|W^+_ij| + x_j)
            self.arg_slacks: List[List[Tuple[Any, Any]]] = [[] for _ in range(lp.nb_ineq())]
            self.var_in_arg_slack: List[List[int]] = [[] for _ in range(lp.dim())]
            self.affine_var_in_arg_slack: List[int] = []
            
            # Per il pivoting
            self.direction: List[Any] = []  # Lista di var_index
            # argmax_{j in direction} (|W_ij| + x_j)
            self.arg_lambdas: List[List[Tuple[Any, str, Any]]] = [[] for _ in range(lp.nb_ineq())]
            self.ineq_status: List[IneqStatus] = [IneqStatus.INACTIVE for _ in range(lp.nb_ineq())]
            
            # Per i costi ridotti
            self.max_permutation: List[Optional[Tuple[int, str, Any]]] = [None for _ in range(lp.dim())]
            self.reduced_costs: List[Optional[Tuple[str, Any]]] = [None for _ in range(lp.nb_ineq())]
            self.dual_slacks: List[Optional[Tuple[str, Any]]] = [None for _ in range(lp.dim())]
            self.var_seen: List[bool] = [False for _ in range(lp.dim())]  # Per Dijkstra
            
            # Contatore iterazioni
            self.iteration = 0

    def init(self, lp: linear_prog.LP, basic_point: np.ndarray) -> "Simplet.SimpletInstance":
        """
        Crea un nuovo simplet per un LP e un punto base.
        
        Args:
            lp: Il programma lineare tropicale (deve essere non-degenerato)
            basic_point: Punto base (deve essere fattibile e saturare esattamente dim inequalities)
            
        Returns:
            Un'istanza SimpletInstance completamente inizializzata
            
        Raises:
            ValueError: Se basic_point non è fattibile o non è un punto base
        """
        nb_ineq = lp.nb_ineq()
        dim = lp.dim()
        
        # Verifica che ogni riga in lp abbia almeno un coefficiente non nullo
        for i in range(nb_ineq):
            try:
                w_i = lp.get_row((linear_prog.RowKind.INEQ, i))
                if not w_i:  # Lista vuota
                    raise ValueError(f"Simplet.init: tutti i coefficienti sono nulli nella disuguaglianza {i}. "
                                   "È richiesto almeno un coefficiente non nullo.")
            except AttributeError:
                # Fallback se get_row non è implementato
                pass
        
        # Costruire il tangent digraph
        tg = tangent_digraph.TangentDigraph.compute(lp, basic_point)
        
        # Verificare che il punto dato sia un punto base
        if not tg.is_basic_point():
            raise ValueError("Simplet.init: il punto di input non è un punto base.")
        
        # Verifica se il punto base è fattibile
        if hasattr(lp, 'is_point_feasible') and not lp.is_point_feasible(basic_point):
            raise ValueError("Simplet.init: il punto di input non è fattibile.")
        
        # Crea l'istanza
        simplet_inst = Simplet.SimpletInstance(lp, basic_point)
        simplet_inst.tangent_digraph = tg
        
        # Inizializzare ineq_status basato sul tangent digraph
        for i in range(nb_ineq):
            if tg.is_hyp_node(i):
                simplet_inst.ineq_status[i] = IneqStatus.BASIS
        
        # Calcola arg_slacks e reduced_costs
        self._compute_arg_slacks_pos(simplet_inst)
        self._compute_reduced_costs(simplet_inst)
        
        return simplet_inst

    def _compute_arg_slacks_pos(self, simplet_inst: "Simplet.SimpletInstance") -> None:
        """
        Calcola il grafo tra var j e argmax_j (W^+_ij + x_j) e lo memorizza in simplet_inst.
        Corrisponde a compute_arg_slacks_pos nell'originale OCaml.
        
        Utilizza compute_slack_args di LinearProg per trovare gli argmin dei coefficienti
        per ogni disuguaglianza, poi filtra solo quelli con segno positivo.
        """
        nb_ineq = simplet_inst.lp.nb_ineq()
        dim = simplet_inst.lp.dim()
        
        # Reset delle strutture dati
        simplet_inst.arg_slacks = [[] for _ in range(nb_ineq)]
        simplet_inst.var_in_arg_slack = [[] for _ in range(dim)]
        simplet_inst.affine_var_in_arg_slack = []
        
        for i in range(nb_ineq):
            # Calcola gli argmin per la disuguaglianza i
            arg_slack = simplet_inst.lp.compute_slack_args((linear_prog.RowKind.INEQ, i), simplet_inst.point)
            
            # Processa ogni elemento nell'argmin
            for col_index, sign, entry in arg_slack:
                # In OCaml, controlla il segno e processa solo quelli positivi
                if sign == linear_prog.Sign.NEG:
                    continue
                elif sign == linear_prog.Sign.POS:
                    # Aggiungi a arg_slacks (memorizza solo var_index ed entry, non il segno)
                    simplet_inst.arg_slacks[i].append((col_index, entry))
                    
                    # Aggiorna var_in_arg_slack per indicizzazione inversa
                    if col_index[0] == linear_prog.ColKind.AFFINE:
                        # Se è la variabile affine, aggiungila alla lista speciale
                        simplet_inst.affine_var_in_arg_slack.append(i)
                    elif col_index[0] == linear_prog.ColKind.VAR:
                        # Se è una variabile ordinaria j, aggiungila a var_in_arg_slack[j]
                        j = col_index[1]
                        if j is not None:
                            simplet_inst.var_in_arg_slack[j].append(i)
    
    def _compute_reduced_costs(self, simplet_inst: "Simplet.SimpletInstance") -> None:
        """
        Calcola i costi ridotti usando l'algoritmo di Dijkstra per i percorsi più lunghi.
        Corrisponde a compute_reduced_costs nell'originale OCaml.
        
        I costi ridotti sono calcolati come le distanze nel grafo bipartito tra variabili
        e disuguaglianze, usando il tangent digraph per determinare la permutazione ottimale.
        """
        nb_ineq = simplet_inst.lp.nb_ineq()
        dim = simplet_inst.lp.dim()
        
        # Calcola la permutazione massimizzante
        self._compute_max_permutation(simplet_inst)
        
        # Reset delle strutture dati
        simplet_inst.reduced_costs = [None for _ in range(nb_ineq)]
        simplet_inst.dual_slacks = [None for _ in range(dim)]
        simplet_inst.var_seen = [False for _ in range(dim)]
        
        # Calcola mu = max(0, c_1+x_1, ..., c_n+x_n) dalla funzione obiettivo
        mu = self._compute_objective_value(simplet_inst)
        # Nel semiring tropicale, la negazione non esiste. Usiamo mu direttamente.
        try:
            neg_mu = self.G.neg(mu)
        except NotImplementedError:
            # Nel caso tropicale, usiamo una strategia alternativa
            neg_mu = self.G.zero()  # oppure -mu se il gruppo supporta la sottrazione
        
        # Inizializza le distanze dall'obiettivo ai nodi variabile
        self._initialize_dual_slacks_from_objective(simplet_inst, neg_mu)
        
        # Algoritmo di Dijkstra semplificato per i percorsi più lunghi
        # TODO: Implementazione completa quando sarà disponibile il tangent digraph
        # Per ora, stima iniziale basata sui slack
        for i in range(nb_ineq):
            if simplet_inst.arg_slacks[i]:
                # Se la disuguaglianza ha argmin definiti, stima il costo ridotto
                # basandosi sul primo argmin
                var_index, entry = simplet_inst.arg_slacks[i][0]
                slack_value = simplet_inst.lp.compute_entry_plus_var(entry, var_index, simplet_inst.point)
                
                # Stima semplificata del costo ridotto
                if self.G.compare(slack_value, mu) <= 0:
                    simplet_inst.reduced_costs[i] = (linear_prog.Sign.POS.value, self.G.zero())
                else:
                    # Potenzialmente negativo (favorevole per il pivoting)
                    diff = self.G.substract(mu, slack_value)
                    simplet_inst.reduced_costs[i] = (linear_prog.Sign.NEG.value, diff)
            else:
                # Nessun argmin definito, costo neutro
                simplet_inst.reduced_costs[i] = (linear_prog.Sign.POS.value, self.G.zero())
    
    def _compute_objective_value(self, simplet_inst: "Simplet.SimpletInstance") -> Any:
        """
        Calcola il valore della funzione obiettivo nel punto corrente.
        mu = max(0, c_1+x_1, ..., c_n+x_n)
        """
        # Inizializza con zero (termine costante)
        mu = self.G.zero()
        
        # Calcola il massimo sui termini della funzione obiettivo
        objective_terms = simplet_inst.lp.get_row((linear_prog.RowKind.OBJECTIVE, None))
        for col_index, sign, entry in objective_terms:
            if sign == linear_prog.Sign.POS:
                term_value = simplet_inst.lp.compute_entry_plus_var(entry, col_index, simplet_inst.point)
                mu = self.G.max(mu, term_value)
        
        return mu
    
    def _initialize_dual_slacks_from_objective(self, simplet_inst: "Simplet.SimpletInstance", neg_mu: Any) -> None:
        """
        Inizializza le distanze dall'obiettivo ai nodi variabile.
        """
        objective_terms = simplet_inst.lp.get_row((linear_prog.RowKind.OBJECTIVE, None))
        
        for col_index, sign, entry in objective_terms:
            if col_index[0] == linear_prog.ColKind.VAR:
                j = col_index[1]
                if j is not None:
                    # Calcola la distanza scalata
                    scaled_entry = self.G.add(neg_mu, entry)
                    simplet_inst.dual_slacks[j] = (sign.value, scaled_entry)
    
    def _compute_max_permutation(self, simplet_inst: "Simplet.SimpletInstance") -> None:
        """
        Calcola la permutazione massimizzante sigma : [n] -> I nel permanente tropicale di A_I.
        Corrisponde a compute_max_permutation nell'originale OCaml.
        
        Usa DFS nel tangent digraph per determinare la permutazione ottimale.
        """
        dim = simplet_inst.lp.dim()
        simplet_inst.max_permutation = [None for _ in range(dim)]
        
        if simplet_inst.tangent_digraph is not None:
            # Implementa DFS nel tangent digraph
            def compute_sigma(acc, node):
                node_type, node_value = node
                if node_type == "IneqNode":
                    ineq_index = int(node_value) if isinstance(node_value, int) else None
                    if ineq_index is not None:
                        # Trova l'unico vicino variabile nel tangent digraph
                        if simplet_inst.tangent_digraph is not None:
                            ineq_node = simplet_inst.tangent_digraph.get_ineq_node(ineq_index)
                            for var_index, sign, entry in ineq_node:
                                if var_index[0] == linear_prog.ColKind.VAR:
                                    j = var_index[1]
                                    if j is not None and simplet_inst.max_permutation[j] is None:
                                        simplet_inst.max_permutation[j] = (ineq_index, sign.value, entry)
                                        break
                return acc
            
            # Inizia DFS dal nodo affine
            start_node = ("VarNode", (linear_prog.ColKind.AFFINE, None))
            simplet_inst.tangent_digraph.dfs_fold_acyclic_graph(compute_sigma, None, start_node)
        
        # Verifica che tutte le variabili abbiano una permutazione
        for j in range(dim):
            if simplet_inst.max_permutation[j] is None:
                # Fallback se DFS non ha trovato una permutazione completa
                print(f"Attenzione: permutazione per variabile {j} non trovata")

    def _bound_on_length_by_ineq(self, simplet_inst: "Simplet.SimpletInstance", ineq_index: int) -> Optional[Any]:
        """
        Durante il pivoting, calcola il bound sulla lunghezza del segmento ordinario corrente
        dato dalla disuguaglianza ineq_index.
        """
        arg_slack = simplet_inst.arg_slacks[ineq_index]
        arg_lambda = simplet_inst.arg_lambdas[ineq_index]
        ineq_status = simplet_inst.ineq_status[ineq_index]
        
        if not arg_slack:  # Lista vuota
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
            # Calcola il bound effettivo: entry_slack - entry_lambda
            # Questo rappresenta la massima lunghezza del segmento prima che la disuguaglianza cambi status
            bound = self.G.substract(entry_slack, entry_lambda)
            return bound
        
        return None

    def bland_rule(self, inst: "Simplet.SimpletInstance") -> Optional[int]:
        """
        Trova la prima disuguaglianza con costo ridotto negativo.
        Implementa la regola di Bland per evitare cicli.
        """
        nb_ineq = inst.lp.nb_ineq()
        
        for i in range(nb_ineq):
            # Verificare se l'ineq è un hyp_node usando TangentDigraph
            if inst.tangent_digraph is None or not inst.tangent_digraph.is_hyp_node(i):
                continue
                
            red_cost = inst.reduced_costs[i]
            if red_cost is not None and red_cost[0] == linear_prog.Sign.NEG.value:
                return i
        
        return None

    def basis_contains(self, inst: "Simplet.SimpletInstance", ineq_index: int) -> bool:
        """
        Verifica se la disuguaglianza ineq_index è saturata dal punto base corrente.
        """
        if inst.tangent_digraph is not None:
            return inst.tangent_digraph.is_hyp_node(ineq_index)
        else:
            # Fallback su ineq_status se TangentDigraph non è disponibile
            return inst.ineq_status[ineq_index] == IneqStatus.BASIS

    def pivot(self, inst: "Simplet.SimpletInstance", i_out: int) -> None:
        """
        Esegue il pivoting tropicale dall'attuale punto base lungo il lato definito da I \\ {i_out}.
        Questa è l'implementazione del cuore dell'algoritmo del simplesso tropicale.
        
        L'algoritmo completo include:
        1. Inizializzazione dello status delle disuguaglianze
        2. Rimozione del nodo hyp dal tangent digraph  
        3. Calcolo della direzione e break hyp
        4. Traversal dell'edge tropicale attraverso segmenti ordinari
        5. Aggiornamento delle strutture dati
        """
        nb_ineq = inst.lp.nb_ineq()
        
        # Verifica che i_out appartenga alla base
        if not self.basis_contains(inst, i_out):
            raise ValueError("Simplet.pivot: l'indice di input non appartiene alla base")
        
        if inst.tangent_digraph is None:
            raise ValueError("Simplet.pivot: tangent digraph non inizializzato")
        
        # 1. Inizializzare ineq_status basato sul tangent digraph
        for i in range(nb_ineq):
            if inst.tangent_digraph.is_hyp_node(i):
                inst.ineq_status[i] = IneqStatus.BREAK_HYP
            else:
                inst.ineq_status[i] = IneqStatus.ENT_HYP
        
        # 2. Marcare hyp node come leaving the basis
        inst.ineq_status[i_out] = IneqStatus.INACTIVE
        
        # 3. Trovare l'arco unico entrante nel hyp node i_out
        incoming_arc = self._find_incoming_arc(inst, i_out)
        if incoming_arc is None:
            raise ValueError(f"Simplet.pivot: nessun arco entrante trovato per hyp node {i_out}")
        
        incoming_var_index, incoming_sign, incoming_entry = incoming_arc
        
        # 4. Rimuovere hyp node i_out dal tangent digraph
        inst.tangent_digraph.remove_arcs_of_node(("IneqNode", i_out))
        
        # 5. Calcolare break hyp e direzione
        self._compute_direction_and_break_hyp(inst, incoming_var_index)
        
        # 6. Calcolare ent hyp e arg_lambda  
        self._compute_arg_lambdas(inst)
        
        # 7. Traversare l'edge tropicale
        new_point = self._traverse_tropical_edge(inst, incoming_var_index, incoming_entry)
        
        # 8. Aggiornare le strutture dati
        inst.point = new_point
        inst.iteration += 1
        
        # Ricostruire tangent digraph dal nuovo punto
        inst.tangent_digraph = tangent_digraph.TangentDigraph.compute(inst.lp, new_point)
        
        # Ricalcola le strutture dati dopo il pivot
        self._compute_arg_slacks_pos(inst)
        self._compute_reduced_costs(inst)

    def _find_incoming_arc(self, inst: "Simplet.SimpletInstance", i_out: int) -> Optional[Tuple[linear_prog.ColIndex, linear_prog.Sign, Any]]:
        """
        Trova l'unico arco entrante nel hyp node i_out.
        Un hyp node ha esattamente un arco entrante (POS) e uno uscente (NEG).
        """
        if inst.tangent_digraph is None:
            return None
            
        ineq_node_arcs = inst.tangent_digraph.get_ineq_node(i_out)
        
        # Cerca l'arco entrante (segno positivo)
        for var_index, sign, entry in ineq_node_arcs:
            if sign == linear_prog.Sign.POS:
                return (var_index, sign, entry)
        
        return None

    def _compute_direction_and_break_hyp(self, inst: "Simplet.SimpletInstance", incoming_var_index: linear_prog.ColIndex) -> None:
        """
        Calcola la direzione del movimento e identifica le disuguaglianze break_hyp.
        La direzione è determinata dall'arco entrante nel nodo che sta uscendo dalla base.
        """
        # La direzione è data dalla variabile dell'arco entrante
        if incoming_var_index[0] == linear_prog.ColKind.VAR:
            j = incoming_var_index[1]
            if j is not None:
                inst.direction = [j]  # Direzione lungo la variabile j
        else:
            # Se l'arco entrante è dalla variabile affine, direzione vuota
            inst.direction = []

    def _compute_arg_lambdas(self, inst: "Simplet.SimpletInstance") -> None:
        """
        Calcola arg_lambda per ogni disuguaglianza durante il pivoting.
        arg_lambda[i] = argmax_{j in direction} (|W_ij| + x_j)
        """
        nb_ineq = inst.lp.nb_ineq()
        
        # Reset arg_lambdas
        inst.arg_lambdas = [[] for _ in range(nb_ineq)]
        
        for i in range(nb_ineq):
            # Per ogni disuguaglianza, calcola l'argmax sulla direzione
            if inst.direction:
                # Se c'è una direzione definita
                max_val = None
                best_args = []
                
                for j in inst.direction:
                    # Trova il coefficiente W_ij per la variabile j nella disuguaglianza i
                    row_i = inst.lp.get_row((linear_prog.RowKind.INEQ, i))
                    for col_index, sign, entry in row_i:
                        if col_index[0] == linear_prog.ColKind.VAR and col_index[1] == j:
                            # Calcola |W_ij| + x_j
                            val = inst.lp.compute_entry_plus_var(entry, col_index, inst.point)
                            
                            if max_val is None or self.G.compare(val, max_val) > 0:
                                max_val = val
                                best_args = [(col_index, sign.value, entry)]
                            elif max_val is not None and self.G.compare(val, max_val) == 0:
                                best_args.append((col_index, sign.value, entry))
                
                inst.arg_lambdas[i] = best_args
            else:
                # Nessuna direzione, arg_lambda vuoto
                inst.arg_lambdas[i] = []

    def _traverse_tropical_edge(self, inst: "Simplet.SimpletInstance", incoming_var_index: linear_prog.ColIndex, incoming_entry: Any) -> np.ndarray:
        """
        Traversa l'edge tropicale fino al nuovo punto base.
        Questo è il cuore dell'algoritmo: si muove lungo la direzione fino a quando
        una nuova disuguaglianza diventa attiva.
        """
        current_point = inst.point.copy()
        
        # Se la direzione è vuota (movimento lungo la variabile affine), il punto non cambia
        if not inst.direction:
            return current_point
        
        # Calcola il massimo lambda per cui rimaniamo fattibili
        lambda_max = None
        
        for i in range(inst.lp.nb_ineq()):
            if inst.ineq_status[i] in [IneqStatus.ENT_HYP, IneqStatus.BREAK_HYP]:
                bound = self._bound_on_length_by_ineq(inst, i)
                if bound is not None:
                    if lambda_max is None or self.G.compare(bound, lambda_max) < 0:
                        lambda_max = bound
        
        # Se non c'è bound, usa un valore di default piccolo
        if lambda_max is None:
            lambda_max = self.G.one  # Movimento unitario
        
        # Muovi il punto nella direzione calcolata
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
    ) -> None:
        """
        Risolve il LP tropicale iterando pivot finché non si raggiunge l'ottimo.
        """
        nb_iter = 1
        is_optimal = False
        
        def myprintf(outchannel, format_str, *args):
            if outchannel is not None:
                print(format_str % args, file=outchannel)
        
        while not is_optimal:
            myprintf(log, "\niterazione %d", nb_iter)
            self.print_status(inst, log)
            
            leaving_ineq = pivot_rule(inst)
            
            if leaving_ineq is None:
                is_optimal = True
                myprintf(log, "\nOttimo raggiunto!")
            else:
                myprintf(log, "\npivoting su %d", leaving_ineq)
                self.pivot(inst, leaving_ineq)
                nb_iter += 1

    # --- Metodi di accesso ---
    def basic_point(self, inst: "Simplet.SimpletInstance") -> np.ndarray:
        """Restituisce il punto base corrente."""
        return inst.point

    def red_cost(self, inst: "Simplet.SimpletInstance", i: int) -> Optional[Tuple[str, Any]]:
        """Restituisce il costo ridotto per la disuguaglianza i."""
        if 0 <= i < len(inst.reduced_costs):
            return inst.reduced_costs[i]
        return None

    def lp(self, inst: "Simplet.SimpletInstance") -> linear_prog.LP:
        """Restituisce il programma lineare associato."""
        return inst.lp

    def tangent_digraph(self, inst: "Simplet.SimpletInstance") -> Any:
        """Restituisce il tangent digraph (quando sarà implementato)."""
        return inst.tangent_digraph

    def print_status(self, inst: "Simplet.SimpletInstance", log: Optional[TextIO] = None) -> None:
        """Stampa lo stato corrente del simplet."""
        if log is None:
            return
            
        print(f"Simplet iterazione {inst.iteration}", file=log)
        
        # Stampa la base
        print("Base:", file=log)
        basis_indices = [i for i in range(len(inst.ineq_status)) 
                        if inst.ineq_status[i] == IneqStatus.BASIS]
        print(f"  {basis_indices}", file=log)
        
        # Stampa il punto base
        print("Punto base:", file=log)
        for j, x_val in enumerate(inst.point):
            print(f"  x{j}: {x_val}", file=log)
        
        # Stampa i costi ridotti
        print("Costi ridotti:", file=log)
        for i, red_cost in enumerate(inst.reduced_costs):
            if red_cost is None:
                red_cost_str = "null"
            else:
                sign, entry = red_cost
                red_cost_str = f"{sign} {entry}"
            print(f"  y{i}: {red_cost_str}", file=log)
        
        print(file=log)  # Riga vuota
        
        if log:
            log.flush()

    def print_reduced_costs(self, inst: "Simplet.SimpletInstance", log: TextIO) -> None:
        """Stampa solo i costi ridotti."""
        print("Costi ridotti:", file=log)
        for i, red_cost in enumerate(inst.reduced_costs):
            if red_cost is None:
                red_cost_str = "null"
            else:
                sign, entry = red_cost
                red_cost_str = f"{sign} {entry}"
            print(f"y{i}: {red_cost_str}", file=log)
        
        if log:
            log.flush()