"""Tropical abstraction utilities for ReLU networks.

Implements the zone-based abstraction for linear layers from:
"Static analysis of ReLU neural networks with tropical polyhedra",
E. Goubault et al. (see Proposition 3 and Theorem 1).

The main goal is to compute a tropical (max-plus) polyhedral abstraction
of each linear layer, then combine it with exact ReLU abstraction and
interval bound propagation to obtain a network-level abstraction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import argparse
import torch


@dataclass(frozen=True)
class TropicalLinearAbstraction:
    """Zone-based tropical abstraction of a linear layer.

    x_lower, x_upper: bounds for input hypercube.
    y_lower, y_upper: bounds for output layer.
    delta: matrix delta_{i,j} (Proposition 3).
    Delta: matrix Delta_{i1,i2} for output differences.
    d: matrix d_{i1,i2} = Delta_{i1,i2} + m_{i2}.
    tropical_constraints: list of (lhs_terms, rhs_term) for max-plus
        inequalities of Theorem 1 in the form max(lhs_terms) <= rhs_term.
    """

    x_lower: np.ndarray
    x_upper: np.ndarray
    y_lower: np.ndarray
    y_upper: np.ndarray
    delta: np.ndarray
    Delta: np.ndarray
    d: np.ndarray
    tropical_constraints: List[Tuple[List[Tuple[str, int, float]], Tuple[str, int, float]]]


def _compute_m_M(W: np.ndarray, b: np.ndarray, x_lower: np.ndarray, x_upper: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute tight interval bounds for y = W x + b over box [x_lower, x_upper]."""

    W_pos = np.maximum(W, 0.0)
    W_neg = np.minimum(W, 0.0)
    m = W_pos @ x_lower + W_neg @ x_upper + b
    M = W_pos @ x_upper + W_neg @ x_lower + b
    return m, M


def _compute_delta(W: np.ndarray, x_lower: np.ndarray, x_upper: np.ndarray) -> np.ndarray:
    """Compute delta_{i,j} (Proposition 3)."""

    width = x_upper - x_lower
    delta = np.zeros_like(W)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            w = W[i, j]
            if w <= 0:
                delta[i, j] = 0.0
            elif w <= 1:
                delta[i, j] = w * width[j]
            else:
                delta[i, j] = width[j]
    return delta


def _compute_Delta(W: np.ndarray, b: np.ndarray, x_lower: np.ndarray, x_upper: np.ndarray) -> np.ndarray:
    """Compute Delta_{i1,i2} (Proposition 3)."""

    n, m = W.shape
    Delta = np.zeros((n, n), dtype=float)
    for i1 in range(n):
        for i2 in range(n):
            diff = 0.0
            for j in range(m):
                if W[i1, j] < W[i2, j]:
                    diff += (W[i1, j] - W[i2, j]) * x_upper[j]
                elif W[i1, j] > W[i2, j]:
                    diff += (W[i1, j] - W[i2, j]) * x_lower[j]
            Delta[i1, i2] = diff + (b[i1] - b[i2])
    return Delta


def linear_layer_tropical_zone(
    W: np.ndarray,
    b: np.ndarray,
    x_lower: np.ndarray,
    x_upper: np.ndarray,
) -> TropicalLinearAbstraction:
    """Compute the tropical zone abstraction for a linear layer.

    Implements Proposition 3 and Theorem 1 (external representation).
    """

    x_lower = np.asarray(x_lower, dtype=float)
    x_upper = np.asarray(x_upper, dtype=float)
    W = np.asarray(W, dtype=float)
    b = np.asarray(b, dtype=float)

    m, M = _compute_m_M(W, b, x_lower, x_upper)
    delta = _compute_delta(W, x_lower, x_upper)
    Delta = _compute_Delta(W, b, x_lower, x_upper)

    # d_{i1,i2} = Delta_{i1,i2} + m_{i2}
    d = Delta + m[None, :]

    constraints: List[Tuple[List[Tuple[str, int, float]], Tuple[str, int, float]]] = []
    # Theorem 1 constraints (2): max(x_j - x_upper_j, y_i - M_i) <= 0
    for j in range(x_lower.shape[0]):
        constraints.append(([("x", j, -x_upper[j])], ("const", -1, 0.0)))
    for i in range(m.shape[0]):
        constraints.append(([("y", i, -M[i])], ("const", -1, 0.0)))

    # Theorem 1 constraints (3): max(0, y_i - M_i + delta_{i,j}) <= x_j - x_lower_j
    for i in range(m.shape[0]):
        for j in range(x_lower.shape[0]):
            lhs = [("const", -1, 0.0), ("y", i, -M[i] + delta[i, j])]
            rhs = ("x", j, -x_lower[j])
            constraints.append((lhs, rhs))

    # Theorem 1 constraints (4): max(0, x_j - x_upper_j + delta_{i,j}, y_k - d_{i,k}) <= y_i - m_i
    for i in range(m.shape[0]):
        for j in range(x_lower.shape[0]):
            lhs: List[Tuple[str, int, float]] = [("const", -1, 0.0), ("x", j, -x_upper[j] + delta[i, j])]
            for k in range(m.shape[0]):
                lhs.append(("y", k, -d[i, k]))
            rhs = ("y", i, -m[i])
            constraints.append((lhs, rhs))

    return TropicalLinearAbstraction(
        x_lower=x_lower,
        x_upper=x_upper,
        y_lower=m,
        y_upper=M,
        delta=delta,
        Delta=Delta,
        d=d,
        tropical_constraints=constraints,
    )


def interval_propagate_linear(W: np.ndarray, b: np.ndarray, x_lower: np.ndarray, x_upper: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Interval bound propagation for a linear layer."""

    return _compute_m_M(np.asarray(W), np.asarray(b), np.asarray(x_lower), np.asarray(x_upper))


def interval_propagate_relu(x_lower: np.ndarray, x_upper: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Interval bounds for y = max(0, x)."""

    return np.maximum(0.0, x_lower), np.maximum(0.0, x_upper)


@dataclass
class TropicalNetworkAbstraction:
    """Network-level abstraction with linear layers + ReLU."""

    weights: List[np.ndarray]
    biases: List[np.ndarray]

    def abstract(self, x_lower: np.ndarray, x_upper: np.ndarray) -> List[TropicalLinearAbstraction]:
        """Compute layer-by-layer abstractions for a ReLU MLP."""

        abstractions: List[TropicalLinearAbstraction] = []
        cur_l, cur_u = np.asarray(x_lower, dtype=float), np.asarray(x_upper, dtype=float)

        for layer_idx, (W, b) in enumerate(zip(self.weights, self.biases)):
            abs_layer = linear_layer_tropical_zone(W, b, cur_l, cur_u)
            abstractions.append(abs_layer)

            # ReLU on all but last layer
            y_l, y_u = abs_layer.y_lower, abs_layer.y_upper
            if layer_idx < len(self.weights) - 1:
                cur_l, cur_u = interval_propagate_relu(y_l, y_u)
            else:
                cur_l, cur_u = y_l, y_u
        return abstractions


def load_linear_layers_from_state_dict(
    state_dict: dict,
    layer_prefixes: Optional[Sequence[str]] = None,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Extract linear layer weights/biases from a PyTorch state_dict.

    If layer_prefixes is None, tries to auto-detect matching "<prefix>.weight" and
    "<prefix>.bias" pairs and returns them in a numeric-aware sorted order.
    """

    def _numeric_key(s: str) -> List[object]:
        import re

        parts: List[object] = []
        for chunk in re.split(r"(\d+)", s):
            if chunk.isdigit():
                parts.append(int(chunk))
            elif chunk:
                parts.append(chunk)
        return parts

    if layer_prefixes is None:
        weight_keys = [k for k in state_dict.keys() if k.endswith(".weight")]
        prefixes = [k[: -len(".weight")] for k in weight_keys if k[: -len(".weight")] + ".bias" in state_dict]
        prefixes = sorted(prefixes, key=_numeric_key)
    else:
        prefixes = list(layer_prefixes)

    weights: List[np.ndarray] = []
    biases: List[np.ndarray] = []
    for prefix in prefixes:
        w = state_dict[f"{prefix}.weight"]
        b = state_dict[f"{prefix}.bias"]
        weights.append(w.detach().cpu().numpy())
        biases.append(b.detach().cpu().numpy())
    return weights, biases


def load_linear_layers_from_pt(
    pt_path: str,
    layer_prefixes: Optional[Sequence[str]] = None,
    map_location: str = "cpu",
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Load a .pt file and extract linear layers.

    Supports files saved as state_dict or a dict containing a "state_dict" key.
    """

    if torch is None:
        raise RuntimeError("torch is required to load .pt files.")

    try:
        obj = torch.load(pt_path, map_location=map_location, weights_only=True)
    except TypeError:
        obj = torch.load(pt_path, map_location=map_location)
    if isinstance(obj, dict) and "state_dict" in obj:
        state_dict = obj["state_dict"]
    else:
        state_dict = obj
    return load_linear_layers_from_state_dict(state_dict, layer_prefixes=layer_prefixes)


def _parse_float_list(arg: str) -> np.ndarray:
    return np.asarray([float(x) for x in arg.split(",")], dtype=float)


def _expand_bounds(bounds: np.ndarray, dim: int, name: str) -> np.ndarray:
    if bounds.size == 1:
        return np.full(dim, float(bounds.item()), dtype=float)
    if bounds.size != dim:
        raise ValueError(f"{name} must have length {dim} (got {bounds.size}).")
    return bounds


def _format_number(val: float) -> str:
    return f"{val:.6f}"


def _format_term(var_name: Optional[str], offset: float) -> str:
    if var_name is None:
        return _format_number(offset)
    if abs(offset) < 1e-12:
        return var_name
    if offset > 0:
        return f"{var_name} + {_format_number(offset)}"
    return f"{var_name} - {_format_number(abs(offset))}"


def _format_max(terms: List[str]) -> str:
    if len(terms) == 1:
        return terms[0]
    return f"max({', '.join(terms)})"


def export_tropical_lp(
    output_path: str,
    abstractions: List[TropicalLinearAbstraction],
    x_lower: np.ndarray,
    x_upper: np.ndarray,
    include_relu: bool = True,
    objective_var: Optional[str] = None,
    objective_sense: str = "maximize",
    numeric: str = "float",
    semiring: str = "maxplus",
) -> str:
    """Export the abstraction to the custom tropical LP format used by simplex_python."""

    if not abstractions:
        raise ValueError("No abstractions to export.")

    input_dim = x_lower.size
    num_layers = len(abstractions)

    x_vars = [f"x_{i}" for i in range(input_dim)]
    h_vars: List[List[str]] = []
    z_vars: List[List[str]] = []

    for l, abs_layer in enumerate(abstractions, start=1):
        if l == num_layers:
            h_vars.append(["out_layer"]) 
        else:
            h_vars.append([f"h{l}_{i}" for i in range(abs_layer.y_lower.size)])
            z_vars.append([f"z{l}_{i}" for i in range(abs_layer.y_lower.size)])

    vars_list: List[str] = []
    vars_list.extend(x_vars)
    for l in range(num_layers):
        vars_list.extend(h_vars[l])
        if l < num_layers - 1:
            vars_list.extend(z_vars[l])

    if objective_var is None:
        objective_var = h_vars[-1][0]

    lines: List[str] = []
    lines.append(f"numeric: {numeric}")
    lines.append(f"semiring: {semiring}")
    lines.append("vars: " + ",".join(vars_list))
    lines.append("lp:")
    lines.append(f"{objective_sense} {objective_var};")
    lines.append("")

    # Input bounds
    for j in range(input_dim):
        lines.append(f"{x_vars[j]} >= {_format_number(float(x_lower[j]))};")
        lines.append(f"{x_vars[j]} <= {_format_number(float(x_upper[j]))};")

    # Layer constraints + optional ReLU
    for l, abs_layer in enumerate(abstractions, start=1):
        in_vars = x_vars if l == 1 else z_vars[l - 2]
        out_vars = h_vars[l - 1]

        for lhs_terms, rhs_term in abs_layer.tropical_constraints:
            lhs_str_terms: List[str] = []
            for kind, idx, offset in lhs_terms:
                if kind == "const":
                    lhs_str_terms.append(_format_term(None, offset))
                elif kind == "x":
                    lhs_str_terms.append(_format_term(in_vars[idx], offset))
                elif kind == "y":
                    lhs_str_terms.append(_format_term(out_vars[idx], offset))
                else:
                    raise ValueError(f"Unknown term kind '{kind}'")

            r_kind, r_idx, r_offset = rhs_term
            if r_kind == "const":
                rhs_str = _format_term(None, r_offset)
            elif r_kind == "x":
                rhs_str = _format_term(in_vars[r_idx], r_offset)
            elif r_kind == "y":
                rhs_str = _format_term(out_vars[r_idx], r_offset)
            else:
                raise ValueError(f"Unknown rhs kind '{r_kind}'")

            lines.append(f"{_format_max(lhs_str_terms)} <= {rhs_str};")

        if include_relu and l < num_layers:
            relu_out = z_vars[l - 1]
            for i, h_name in enumerate(out_vars):
                z_name = relu_out[i]
                lines.append(f"{z_name} >= {h_name};")
                lines.append(f"{z_name} >= 0;")
                lines.append(f"{z_name} <= max({h_name}, 0);")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return output_path


def main() -> None:
    """CLI entrypoint to load a .pt and build abstractions."""

    parser = argparse.ArgumentParser(description="Tropical abstraction for ReLU MLPs")

    parser.add_argument(
        "pt_path", 
        help="Path to .pt file (state_dict or dict with state_dict)"
                        )
    
    parser.add_argument(
        "--layer-prefixes",
        default=None,
        help="Comma-separated layer prefixes in order (e.g., first_layer,layers.0,output_layer)",
    )
    
    parser.add_argument(
        "-xlb",
        "--x-lower",
        required=True,
        help="Comma-separated lower bounds for input box",
    )
    
    parser.add_argument(
        "-xub",
        "--x-upper",
        required=True,
        help="Comma-separated upper bounds for input box",
    )

    parser.add_argument(
        "-o",
        "--output-lp",
        required=True,
        default=None,
        help="Path or filename of the abstraction in tropical LP format",
    )

    parser.add_argument(
        "--objective-var",
        default=None,
        help="Objective variable name (default: first output neuron)",
    )

    parser.add_argument(
        "-s",
        "--semiring",
        default="maxplus",
        choices=["maxplus", "minplus"],
        help="Semiring for tropical LP (default: maxplus)",
    )

    parser.add_argument(
        "-objs",
        "--objective-sense",
        default="maximize",
        choices=["maximize", "minimize"],
        help="Objective direction",
    )

    parser.add_argument(
        "--no-relu",
        action="store_true",
        help="Do not emit ReLU constraints between layers",
    )

    args = parser.parse_args()
    prefixes = args.layer_prefixes.split(",") if args.layer_prefixes else None

    weights, biases = load_linear_layers_from_pt(args.pt_path, layer_prefixes=prefixes)
    abstraction = TropicalNetworkAbstraction(weights, biases)
    x_lower = _parse_float_list(args.x_lower)
    x_upper = _parse_float_list(args.x_upper)
    input_dim = weights[0].shape[1]
    x_lower = _expand_bounds(x_lower, input_dim, "x_lower")
    x_upper = _expand_bounds(x_upper, input_dim, "x_upper")
    layers = abstraction.abstract(x_lower, x_upper)

    print(f"Loaded {len(weights)} linear layers from {args.pt_path}")
    print(f"Input bounds: [{x_lower.min():.3f}, {x_upper.max():.3f}] (dim={x_lower.size})")
    for i, layer in enumerate(layers, start=1):
        print(f"Layer {i}: y in [{layer.y_lower.min():.3f}, {layer.y_upper.max():.3f}] (dim={layer.y_lower.size})")

    if args.output_lp:
        export_tropical_lp(
            args.output_lp,
            layers,
            x_lower,
            x_upper,
            include_relu=not args.no_relu,
            objective_var=args.objective_var,
            objective_sense=args.objective_sense,
        )
        print(f"LP exported to: {args.output_lp}")


if __name__ == "__main__":
    main()
