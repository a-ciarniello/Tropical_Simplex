"""
Convert a trained tropical MLP checkpoint (.pt) into an LP that the
tropical simplex (python/ocaml) can read. We assume a max-plus network with
layer weights `W` and biases via `tanh(raw_b) * b_scale` (as in TropLayer).

Each neuron output y_i is constrained as the max of its tropical pre-activations:
    y_i >= W_ij + x_j   for all j
    y_i >= b_i

Objective: maximize one chosen output neuron (default index 0).

Input variables are free in [lo, hi] (defaults 0..1). Adjust via CLI flags.

LP format emitted:
    numeric: float
    semiring: maxplus
    vars: v1,v2,...
    
    lp:
    maximize out_0;
    ...constraints...
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch


def load_state_dict(pt_path: Path) -> Dict[str, torch.Tensor]:
    """Load a checkpoint from disk and ensure it is a state_dict mapping."""
    try:
        try:
            state_dict = torch.load(pt_path, map_location="cpu", weights_only=True)
        except TypeError:
            state_dict = torch.load(pt_path, map_location="cpu")
    except Exception as exc:  # pragma: no cover - IO guard
        raise RuntimeError(f"Unable to load '{pt_path}': {exc}") from exc

    if not isinstance(state_dict, dict):
        raise ValueError("Unexpected checkpoint structure: expected a state_dict dictionary")
    return state_dict


def to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy().astype(np.float64)


def fmt(value: float) -> str:
    """Format numbers without scientific notation (fixed 12 decimals)."""
    return f"{value:.12f}"


def add_offset(base: str, coeff: float) -> str:
    """Return a string like `base + v` or `base - v` depending on coeff sign."""
    if coeff < 0:
        return f"{base} - {fmt(abs(coeff))}"
    else:
        return f"{base} + {fmt(coeff)}"


@dataclass
class TropLayerParams:
    name: str
    weight: np.ndarray  # shape (out, in)
    bias: np.ndarray    # shape (out,)


class TropNetConverter:
    def __init__(self, state_dict: Dict[str, torch.Tensor], objective_index: int, input_lo: float, input_hi: float) -> None:
        self.state_dict = state_dict
        self.objective_index = objective_index
        self.input_lo = input_lo
        self.input_hi = input_hi

    def convert(self) -> Tuple[str, Tuple[int, int]]:
        layers = self._collect_layers()
        if not layers:
            raise ValueError("No tropical layers found (expected keys like '*.W' and '*.raw_b').")

        input_size = layers[0].weight.shape[1]

        # Variable names
        inputs = [f"x_{i}" for i in range(input_size)]
        all_vars: List[str] = list(inputs)

        # Bias anchor fixed to 0 to avoid single-term constraints
        bias_anchor = "bias0"
        zero_anchor = "z0"
        all_vars.append(bias_anchor)
        all_vars.append(zero_anchor)

        layer_outputs: List[List[str]] = []

        prev = inputs
        for li, layer in enumerate(layers):
            outs = []
            for j in range(layer.weight.shape[0]):
                v = f"{layer.name}_{j}"
                outs.append(v)
                all_vars.append(v)
            layer_outputs.append(outs)
            prev = outs

        out_vars = layer_outputs[-1]
        if self.objective_index < 0 or self.objective_index >= len(out_vars):
            raise ValueError(f"objective_index {self.objective_index} out of range (0..{len(out_vars)-1})")

        lines: List[str] = []

        # Input bounds via anchors to keep both POS/NEG arcs
        for x in inputs:
            lines.append(f"{x} >= {add_offset(bias_anchor, self.input_lo)};")
            lines.append(f"{x} <= {add_offset(bias_anchor, self.input_hi)};")

        # Fix bias_anchor and zero_anchor to 0 but keep arcs on both sides
        lines.append(f"{bias_anchor} >= {add_offset(zero_anchor, 0.0)};")
        lines.append(f"{bias_anchor} <= {add_offset(zero_anchor, 0.0)};")
        lines.append(f"{zero_anchor} >= {add_offset(bias_anchor, 0.0)};")
        lines.append(f"{zero_anchor} <= {add_offset(bias_anchor, 0.0)};")

        # Layer constraints
        prev_names = inputs
        for layer, outs in zip(layers, layer_outputs):
            W = layer.weight
            b = layer.bias
            for j, out_name in enumerate(outs):
                # bias constraint via anchor variable to keep a POS and NEG arc
                lines.append(f"{out_name} >= {add_offset(bias_anchor, b[j])};")
                # each incoming term
                for k, inp_name in enumerate(prev_names):
                    lines.append(f"{out_name} >= {add_offset(inp_name, W[j, k])};")
            prev_names = outs

        # Objective
        objective_var = out_vars[self.objective_index]

        header = [
            "numeric: float",
            "semiring: maxplus",
            f"vars: {','.join(all_vars)}",
            "",
            "lp:",
            "",
            f"maximize {objective_var};",
            "",
        ]

        lp_text = "\n".join(header + lines) + "\n"
        stats = (len(all_vars), len(lines))
        return lp_text, stats

    def _collect_layers(self) -> List[TropLayerParams]:
        """Collect TropLayer parameters (W, raw_b, b_scale)."""
        layer_bases = []
        for key in self.state_dict.keys():
            if key.endswith(".W"):
                base = key[:-len(".W")]
                if f"{base}.raw_b" in self.state_dict:
                    layer_bases.append(base)

        # Preserve declared order if possible; fall back to sorted
        layer_bases = sorted(layer_bases)

        layers: List[TropLayerParams] = []
        for base in layer_bases:
            W = to_numpy(self.state_dict[f"{base}.W"])
            raw_b = to_numpy(self.state_dict[f"{base}.raw_b"])
            b_scale_tensor = self.state_dict.get(f"{base}.b_scale")
            if b_scale_tensor is None:
                b_scale = 3.0
            else:
                b_scale = float(to_numpy(b_scale_tensor).reshape(()))
            bias = np.tanh(raw_b) * b_scale
            layers.append(TropLayerParams(name=base.replace(".", "_"), weight=W, bias=bias))

        return layers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export tropical MLP to LP (max-plus constraints).")

    parser.add_argument("pt", 
                        type=Path, 
                        help="Checkpoint .pt path")
    
    parser.add_argument("--output",
                        "-o", 
                        type=Path, 
                        default=Path("network.lp"), 
                        help="Destination LP filename")
    
    parser.add_argument("--objective-index", 
                        type=int, 
                        default=0, 
                        help="Index of output neuron to maximize")
    
    parser.add_argument("--input-lo", 
                        type=float, 
                        default=0.0, 
                        help="Lower bound for each input")
    
    parser.add_argument("--input-hi", 
                        type=float, 
                        default=1.0, 
                        help="Upper bound for each input")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    state_dict = load_state_dict(args.pt)

    converter = TropNetConverter(
        state_dict=state_dict,
        objective_index=args.objective_index,
        input_lo=args.input_lo,
        input_hi=args.input_hi,
    )

    lp_text, stats = converter.convert()
    args.output.write_text(lp_text, encoding="ascii")

    print(f"LP written to '{args.output}'.")
    print(f"Variables: {stats[0]} | Constraints: {stats[1]}")


if __name__ == "__main__":
    main()