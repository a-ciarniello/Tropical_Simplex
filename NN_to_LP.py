"""Convert a trained ACAS Xu PyTorch model (.pt) into an LP consumable by the tropical simplex."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch


def load_state_dict(pt_path: Path) -> Dict[str, torch.Tensor]:
    """Load a checkpoint from disk and ensure it is a state_dict mapping."""
    try:
        state_dict = torch.load(pt_path, map_location=torch.device("cpu"))
    except Exception as exc:
        raise RuntimeError(f"Unable to load '{pt_path}': {exc}") from exc

    if not isinstance(state_dict, dict):
        raise ValueError("Unexpected checkpoint structure: expected a state_dict dictionary")
    return state_dict


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Detach ``tensor`` from PyTorch and produce a float64 NumPy array."""
    return tensor.detach().cpu().numpy().astype(np.float64)


def format_affine(constant: float, coeffs: Iterable[Tuple[float, str]], tol: float = 1e-12) -> str:
    """Format an affine expression suitable for LP emission."""
    parts: List[str] = []
    if abs(constant) > tol:
        parts.append(f"{constant:.12g}")
    for coeff, var in coeffs:
        if abs(coeff) <= tol:
            continue
        parts.append(f"{coeff:.12g} * {var}")
    if not parts:
        return "0"
    return " + ".join(parts)


@dataclass
class LPWriter:
    """Helper that accumulates rows and renders them as an LP file."""
    numeric: str = "float"
    semiring: str | None = None

    def __post_init__(self) -> None:
        self._var_list: List[str] = []
        self._var_set: set[str] = set()
        self._objective: Tuple[str, str] | None = None  # (sense, expr)
        self._lines: List[str] = []
        self._constraint_count = 0

    def add_variable(self, name: str) -> None:
        if name in self._var_set:
            return
        self._var_set.add(name)
        self._var_list.append(name)

    def add_comment(self, text: str) -> None:
        self._lines.append(f"/* {text} */")

    def add_constraint(self, lhs: str, operator: str, rhs: str, comment: str | None = None) -> None:
        if comment:
            self.add_comment(comment)
        self._lines.append(f"{lhs} {operator} {rhs};")
        self._constraint_count += 1

    def add_equality(self, lhs: str, rhs: str, comment: str | None = None) -> None:
        lower = comment if comment else None
        upper = f"{comment} (>=)" if comment else None
        self.add_constraint(lhs, "<=", rhs, lower)
        self.add_constraint(lhs, ">=", rhs, upper)

    def add_relu(self, pre_var: str, post_var: str, comment: str) -> None:
        self.add_constraint(post_var, ">=", pre_var, f"{comment} (lower bound)")
        self.add_constraint(post_var, ">=", "0", None)
        self.add_constraint(post_var, "<=", f"max({pre_var}, 0)", f"{comment} (upper bound)")

    def set_objective(self, sense: str, expr: str) -> None:
        if sense not in {"maximize", "minimize"}:
            raise ValueError("Objective sense must be 'maximize' or 'minimize'")
        self._objective = (sense, expr)

    def render(self) -> str:
        if not self._var_list:
            raise ValueError("No variables declared.")
        if self._objective is None:
            raise ValueError("Objective not specified.")

        header: List[str] = []
        header.append(f"numeric: {self.numeric}")
        if self.semiring:
            header.append(f"semiring: {self.semiring}")
        header.append(f"vars: {','.join(self._var_list)}")
        header.append("")
        header.append("lp:")
        header.append("")

        sense, expr = self._objective
        body = [f"{sense} {expr};", ""] + self._lines
        return "\n".join(header + body) + "\n"

    @property
    def stats(self) -> Tuple[int, int]:
        return len(self._var_list), self._constraint_count


@dataclass
class LayerSpec:
    name: str
    weight: np.ndarray
    bias: np.ndarray
    apply_relu: bool


class NetworkLPConverter:
    """Convert PyTorch checkpoints into solver-friendly LP descriptions."""
    def __init__(
        self,
        state_dict: Dict[str, torch.Tensor],
        numeric: str,
        semiring: str | None,
        objective_index: int,
    ) -> None:
        self.state_dict = state_dict
        self.numeric = numeric
        self.semiring = semiring
        self.objective_index = objective_index

    def convert(self) -> Tuple[str, Tuple[int, int], List[str]]:
        """Emit the LP text, basic statistics, and ordered output variable names."""
        writer = LPWriter(numeric=self.numeric, semiring=self.semiring)

        means = tensor_to_numpy(self.state_dict.get("means", torch.zeros(0)))
        ranges = tensor_to_numpy(self.state_dict.get("ranges", torch.ones_like(torch.from_numpy(means))))
        if means.size == 0 or ranges.size == 0:
            raise ValueError("The checkpoint does not provide 'means' and 'ranges' buffers.")

        # Avoid zero division
        ranges[ranges == 0.0] = 1.0

        input_size = means.shape[0]
        raw_inputs = [f"input_{i}" for i in range(input_size)]
        norm_inputs = [f"norm_{i}" for i in range(input_size)]

        for var in raw_inputs + norm_inputs:
            writer.add_variable(var)

        for idx, (raw_var, norm_var, mean, rng) in enumerate(zip(raw_inputs, norm_inputs, means, ranges)):
            rhs = format_affine(mean, [(rng, norm_var)])
            writer.add_equality(raw_var, rhs, f"Normalize input {idx}")

        layers = self._collect_layers()
        previous = norm_inputs
        final_outputs: List[str] = []

        for layer_idx, layer in enumerate(layers):
            prev_count = len(previous)
            if layer.weight.shape[1] != prev_count:
                raise ValueError(
                    f"Layer '{layer.name}' expects {layer.weight.shape[1]} inputs but received {prev_count}."
                )

            layer_outputs: List[str] = []
            for neuron_idx in range(layer.weight.shape[0]):
                pre_var = f"{layer.name}_pre_{neuron_idx}"
                writer.add_variable(pre_var)

                coeffs = [
                    (float(layer.weight[neuron_idx, src_idx]), previous[src_idx])
                    for src_idx in range(prev_count)
                ]
                affine_expr = format_affine(float(layer.bias[neuron_idx]), coeffs)
                writer.add_equality(pre_var, affine_expr, f"Affine map {layer.name}[{neuron_idx}]")

                if layer.apply_relu:
                    post_var = f"{layer.name}_act_{neuron_idx}"
                    writer.add_variable(post_var)
                    writer.add_relu(pre_var, post_var, f"ReLU {layer.name}[{neuron_idx}]")
                else:
                    post_var = f"output_{neuron_idx}"
                    writer.add_variable(post_var)
                    writer.add_equality(post_var, pre_var, f"Output neuron {neuron_idx}")
                layer_outputs.append(post_var)

            previous = layer_outputs
            final_outputs = layer_outputs

        if not final_outputs:
            raise ValueError("No outputs were produced; check the checkpoint contents.")

        if self.objective_index < 0 or self.objective_index >= len(final_outputs):
            raise ValueError(
                f"Objective index {self.objective_index} is out of range for {len(final_outputs)} outputs."
            )

        objective_var = final_outputs[self.objective_index]
        writer.set_objective("maximize", objective_var)
        writer.add_comment("Add property-specific constraints below this line as needed.")

        lp_text = writer.render()
        stats = writer.stats
        return lp_text, stats, final_outputs

    def _collect_layers(self) -> List[LayerSpec]:
        """Gather sequential layer specs from the checkpoint state dictionary."""
        layers: List[LayerSpec] = []
        hidden_idx = 0
        while True:
            w_key = f"hidden_layers.{hidden_idx}.weight"
            b_key = f"hidden_layers.{hidden_idx}.bias"
            if w_key not in self.state_dict:
                break
            weight = tensor_to_numpy(self.state_dict[w_key])
            bias = tensor_to_numpy(self.state_dict[b_key])
            layers.append(LayerSpec(name=f"layer{hidden_idx}", weight=weight, bias=bias, apply_relu=True))
            hidden_idx += 1

        out_w = self.state_dict.get("output_layer.weight")
        out_b = self.state_dict.get("output_layer.bias")
        if out_w is None or out_b is None:
            raise ValueError("Missing output layer weights/bias in the checkpoint.")

        layers.append(
            LayerSpec(
                name="output_layer",
                weight=tensor_to_numpy(out_w),
                bias=tensor_to_numpy(out_b),
                apply_relu=False,
            )
        )
        return layers


def parse_args() -> argparse.Namespace:
    """Configure and parse the CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Export a PyTorch model to a linear program."
        )
    )
    parser.add_argument("pt", type=Path, nargs="?", help="Checkpoint .pt path")

    parser.add_argument(
        "--output", "-o", type=Path, default=Path("network.lp"), help="Destination LP filename"
    )
    parser.add_argument(
        "--objective-index",
        type=int,
        default=0,
        help="Index of the output neuron used in the objective (default: 0)",
    )
    parser.add_argument(
        "--numeric",
        type=str,
        default="float",
        help="numeric: header to emit in the LP. Default: float.",
    )
    parser.add_argument(
        "--semiring",
        type=str,
        default=None,
        help="Optional semiring header (minplus / maxplus). Leave empty to omit.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point: orchestrate conversion and report statistics."""
    args = parse_args()
    state_dict = load_state_dict(args.pt)

    converter = NetworkLPConverter(
        state_dict=state_dict,
        numeric=args.numeric,
        semiring=args.semiring,
        objective_index=args.objective_index,
    )

    lp_text, stats, outputs = converter.convert()
    args.output.write_text(lp_text, encoding="ascii")

    var_count, constraint_count = stats
    print(f"LP written to '{args.output}'.")
    print(f"Variables: {var_count} | Constraints: {constraint_count}")
    print(f"Outputs available: {', '.join(outputs)}")


if __name__ == "__main__":
    main()