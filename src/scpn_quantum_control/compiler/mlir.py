# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- MLIR textual compiler surface
"""Deterministic MLIR-style export for Kuramoto-XY workloads.

The module emits a conservative textual interchange layer for the SCPN
Kuramoto-XY compiler. It does not require an MLIR Python runtime and does not
claim lowering to LLVM, QIR, or provider pulses. The value is a stable,
auditable IR boundary for compiler passes and external tooling.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np

from ..differentiable import CustomDerivativeRule, value_and_custom_jacobian
from ..kuramoto_core import KuramotoProblem, build_kuramoto_problem


@dataclass(frozen=True)
class MLIRCompileConfig:
    """Configuration for Kuramoto-XY MLIR-style export."""

    time: float
    trotter_steps: int = 1
    trotter_order: int = 1
    dialect: str = "scpn_kuramoto"
    include_metadata: bool = True

    def __post_init__(self) -> None:
        if not np.isfinite(self.time) or self.time <= 0.0:
            raise ValueError("time must be finite and positive")
        if not isinstance(self.trotter_steps, int) or self.trotter_steps < 1:
            raise ValueError("trotter_steps must be a positive integer")
        if self.trotter_order not in {1, 2}:
            raise ValueError("trotter_order must be 1 or 2")
        if not self.dialect or not self.dialect.replace("_", "").isalnum():
            raise ValueError("dialect must be a non-empty MLIR-safe identifier")


@dataclass(frozen=True)
class MLIRModule:
    """Textual MLIR module plus deterministic provenance."""

    text: str
    sha256: str
    dialect: str
    resource_counts: Mapping[str, int]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("text must be non-empty")
        expected = hashlib.sha256(self.text.encode("utf-8")).hexdigest()
        if self.sha256 != expected:
            raise ValueError("sha256 must match text")
        object.__setattr__(self, "resource_counts", MappingProxyType(dict(self.resource_counts)))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True)
class DifferentiableMLIRCompileConfig:
    """Configuration for differentiable primitive MLIR-style lowering."""

    dialect: str = "scpn_diff"
    target: str = "mlir"
    include_numeric_payload: bool = True
    include_metadata: bool = True

    def __post_init__(self) -> None:
        if not self.dialect or not self.dialect.replace("_", "").isalnum():
            raise ValueError("dialect must be a non-empty MLIR-safe identifier")
        if self.target not in {"mlir"}:
            raise ValueError(
                "target must be 'mlir'; executable LLVM/JIT lowering is not yet available"
            )
        if not isinstance(self.include_numeric_payload, bool):
            raise ValueError("include_numeric_payload must be a boolean")
        if not isinstance(self.include_metadata, bool):
            raise ValueError("include_metadata must be a boolean")


def compile_kuramoto_to_mlir(
    problem: KuramotoProblem | np.ndarray,
    config: MLIRCompileConfig,
    omega: np.ndarray | None = None,
) -> MLIRModule:
    """Compile a Kuramoto problem into deterministic MLIR-style text.

    ``problem`` may be a validated :class:`KuramotoProblem` or a raw coupling
    matrix when ``omega`` is supplied. Raw arrays are validated through the
    public Kuramoto facade before IR generation.
    """

    if isinstance(problem, KuramotoProblem):
        validated = problem
    else:
        if omega is None:
            raise ValueError("omega is required when problem is a raw coupling matrix")
        validated = build_kuramoto_problem(problem, omega)

    coupling_terms = _coupling_terms(validated.K_nm)
    lines = [
        f'module attributes {{scpn.module = "kuramoto_xy", scpn.dialect = "{config.dialect}", '
        f"scpn.n_oscillators = {validated.n_oscillators}, "
        f"scpn.trotter_steps = {config.trotter_steps}, "
        f"scpn.trotter_order = {config.trotter_order}}} {{",
        "  func.func @main() {",
    ]
    for index, value in enumerate(validated.omega):
        lines.append(f"    scpn.omega %{index} {{value = {_fmt_float(float(value))}}}")
    for term_index, (left, right, value) in enumerate(coupling_terms):
        lines.append(
            "    scpn.coupling "
            f"%c{term_index} {{i = {left}, j = {right}, value = {_fmt_float(value)}}}"
        )
    lines.append(
        "    scpn.trotter_evolve "
        f"{{time = {_fmt_float(config.time)}, steps = {config.trotter_steps}, "
        f"order = {config.trotter_order}}}"
    )
    lines.append("    return")
    lines.append("  }")
    if config.include_metadata and validated.metadata:
        encoded = json.dumps(dict(validated.metadata), sort_keys=True, separators=(",", ":"))
        lines.append(f'  scpn.metadata {{json = "{_escape_mlir_string(encoded)}"}}')
    lines.append("}")
    text = "\n".join(lines) + "\n"
    resource_counts = {
        "n_oscillators": validated.n_oscillators,
        "omega_terms": validated.n_oscillators,
        "coupling_terms": len(coupling_terms),
        "trotter_steps": config.trotter_steps,
        "trotter_order": config.trotter_order,
    }
    return MLIRModule(
        text=text,
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        dialect=config.dialect,
        resource_counts=resource_counts,
        metadata={
            "claim_boundary": "textual MLIR-style IR export; no provider lowering or hardware execution",
            "problem": validated.to_metadata(),
        },
    )


def compile_custom_derivative_rule_to_mlir(
    rule: CustomDerivativeRule,
    values: np.ndarray,
    config: DifferentiableMLIRCompileConfig | None = None,
) -> MLIRModule:
    """Lower an exact custom derivative rule to deterministic MLIR-style text.

    This emits an auditable differentiable-primitive interchange artifact with
    value and Jacobian shape metadata. When numeric payloads are enabled, the
    current value and exact custom Jacobian are embedded as deterministic
    attributes. The function deliberately does not claim executable LLVM or JIT
    code generation.
    """

    if not isinstance(rule, CustomDerivativeRule):
        raise ValueError("differentiable MLIR lowering requires a CustomDerivativeRule")
    compile_config = DifferentiableMLIRCompileConfig() if config is None else config
    jacobian_result = value_and_custom_jacobian(rule, values)
    parameter_count = jacobian_result.jacobian.shape[1]
    output_count = jacobian_result.value.size
    lines = [
        f'module attributes {{scpn.module = "differentiable_primitive", '
        f'scpn.dialect = "{compile_config.dialect}", '
        f'scpn.rule = "{_escape_mlir_string(rule.name)}", '
        f"scpn.n_parameters = {parameter_count}, "
        f"scpn.n_outputs = {output_count}}} {{",
        "  func.func @main() {",
    ]
    for index, (name, trainable) in enumerate(
        zip(jacobian_result.parameter_names, jacobian_result.trainable, strict=True)
    ):
        lines.append(
            "    scpn_diff.parameter "
            f'%p{index} {{name = "{_escape_mlir_string(name)}", trainable = {_fmt_bool(trainable)}}}'
        )
    if compile_config.include_numeric_payload:
        for index, value in enumerate(jacobian_result.value):
            lines.append(f"    scpn_diff.value %{index} {{value = {_fmt_float(float(value))}}}")
        for row in range(output_count):
            for column in range(parameter_count):
                value = float(jacobian_result.jacobian[row, column])
                if abs(value) > 1.0e-15:
                    lines.append(
                        "    scpn_diff.jacobian "
                        f"{{row = {row}, col = {column}, value = {_fmt_float(value)}}}"
                    )
    lines.append(
        "    scpn_diff.custom_rule "
        f"{{jvp = {_fmt_bool(rule.jvp_rule is not None)}, "
        f"vjp = {_fmt_bool(rule.vjp_rule is not None)}, "
        'execution = "interchange_only"}}'
    )
    lines.append("    return")
    lines.append("  }")
    if compile_config.include_metadata:
        metadata = {
            "method": jacobian_result.method,
            "parameter_names": list(jacobian_result.parameter_names),
            "trainable": list(jacobian_result.trainable),
            "target": compile_config.target,
        }
        encoded = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
        lines.append(f'  scpn.metadata {{json = "{_escape_mlir_string(encoded)}"}}')
    lines.append("}")
    text = "\n".join(lines) + "\n"
    return MLIRModule(
        text=text,
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        dialect=compile_config.dialect,
        resource_counts={
            "parameters": parameter_count,
            "outputs": output_count,
            "jacobian_nnz": int(np.count_nonzero(jacobian_result.jacobian)),
            "trainable_parameters": int(sum(jacobian_result.trainable)),
        },
        metadata={
            "claim_boundary": "textual differentiable MLIR-style IR export; no executable LLVM or JIT lowering",
            "rule": rule.name,
            "target": compile_config.target,
            "sha256_source": "module.text",
        },
    )


def _coupling_terms(K_nm: np.ndarray) -> tuple[tuple[int, int, float], ...]:
    terms: list[tuple[int, int, float]] = []
    n_oscillators = K_nm.shape[0]
    for left in range(n_oscillators):
        for right in range(left + 1, n_oscillators):
            value = float(K_nm[left, right])
            if abs(value) > 1e-15:
                terms.append((left, right, value))
    return tuple(terms)


def _fmt_float(value: float) -> str:
    if not np.isfinite(value):
        raise ValueError("MLIR numeric attributes must be finite")
    return format(value, ".17g")


def _fmt_bool(value: bool) -> str:
    return "true" if value else "false"


def _escape_mlir_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


__all__ = [
    "DifferentiableMLIRCompileConfig",
    "MLIRCompileConfig",
    "MLIRModule",
    "compile_custom_derivative_rule_to_mlir",
    "compile_kuramoto_to_mlir",
]
