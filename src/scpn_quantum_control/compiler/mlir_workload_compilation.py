# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — MLIR workload compilation module
# scpn-quantum-control -- MLIR workload compilation
"""Kuramoto and custom-rule executable MLIR compilation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from typing import Any, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from ..differentiable import (
    CustomDerivativeRegistry,
    CustomDerivativeRule,
    PrimitiveIdentity,
    program_ad_linalg_matrix_power_derivative_rule,
    program_ad_linalg_multi_dot_derivative_rule,
)
from ..kuramoto_core import KuramotoProblem, build_kuramoto_problem
from .mlir_executable_kernel import (
    ExecutableCompilerADKernel,
    _compile_scalar_gradient_llvm_ir,
    _verify_executable_ad_kernel,
    compile_custom_derivative_rule_to_mlir,
)
from .mlir_native_primitives import _as_finite_vector, _escape_mlir_string, _fmt_float
from .mlir_records import CompilerADExecutableConfig, MLIRCompileConfig, MLIRModule

FloatArray: TypeAlias = NDArray[np.float64]


def compile_kuramoto_to_mlir(
    problem: KuramotoProblem | FloatArray,
    config: MLIRCompileConfig,
    omega: FloatArray | None = None,
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


def compile_custom_derivative_rule_to_executable(
    rule: CustomDerivativeRule,
    sample_values: Sequence[float] | FloatArray,
    config: CompilerADExecutableConfig | None = None,
    *,
    sample_tangent: Sequence[float] | FloatArray | None = None,
    sample_cotangent: Sequence[float] | FloatArray | None = None,
) -> ExecutableCompilerADKernel:
    """Compile a custom derivative rule into a verified executable AD kernel.

    The executable backend is the dependency-free SCPN MLIR runtime adapter:
    it couples deterministic differentiable MLIR provenance with normalized
    runtime callables for value/JVP/VJP execution and verifies those kernels
    against the source custom derivative rule before returning. Native LLVM/JIT
    kernels use primitive-specific lowering entrypoints.
    """
    if not isinstance(rule, CustomDerivativeRule):
        raise ValueError("executable AD lowering requires a CustomDerivativeRule")
    compile_config = CompilerADExecutableConfig() if config is None else config
    if compile_config.backend != "mlir_runtime":
        raise ValueError(
            "compile_custom_derivative_rule_to_executable requires backend='mlir_runtime'; "
            "use a primitive-specific native LLVM/JIT lowering entrypoint"
        )
    values = _as_finite_vector("sample_values", sample_values)
    mlir_module = compile_custom_derivative_rule_to_mlir(
        rule,
        values,
        compile_config.mlir_config,
    )

    def value_kernel(raw_values: FloatArray) -> FloatArray:
        return _as_finite_vector(
            "value kernel output", rule.value_fn(_as_finite_vector("values", raw_values))
        )

    def jvp_kernel(raw_values: FloatArray, raw_tangent: FloatArray) -> FloatArray:
        if rule.jvp_rule is None:
            raise ValueError(f"rule {rule.name} has no JVP rule")
        checked_values = _as_finite_vector("values", raw_values)
        checked_tangent = _as_finite_vector("tangent", raw_tangent)
        if checked_tangent.shape != checked_values.shape:
            raise ValueError("tangent shape must match values shape")
        return _as_finite_vector(
            "JVP kernel output", rule.jvp_rule(checked_values, checked_tangent)
        )

    def vjp_kernel(raw_values: FloatArray, raw_cotangent: FloatArray) -> FloatArray:
        if rule.vjp_rule is None:
            raise ValueError(f"rule {rule.name} has no VJP rule")
        checked_values = _as_finite_vector("values", raw_values)
        checked_cotangent = _as_finite_vector("cotangent", raw_cotangent)
        return _as_finite_vector(
            "VJP kernel output", rule.vjp_rule(checked_values, checked_cotangent)
        )

    verification = _verify_executable_ad_kernel(
        rule,
        values,
        value_kernel,
        jvp_kernel if rule.jvp_rule is not None else None,
        vjp_kernel if rule.vjp_rule is not None else None,
        compile_config,
        sample_tangent=sample_tangent,
        sample_cotangent=sample_cotangent,
    )
    llvm_gradient_ir = (
        _compile_scalar_gradient_llvm_ir(rule, values, vjp_kernel)
        if verification.gradient_close is True and rule.vjp_rule is not None
        else None
    )
    return ExecutableCompilerADKernel(
        rule_name=rule.name,
        backend=compile_config.backend,
        mlir_module=mlir_module,
        value_kernel=value_kernel,
        jvp_kernel=jvp_kernel if rule.jvp_rule is not None else None,
        vjp_kernel=vjp_kernel if rule.vjp_rule is not None else None,
        verification=verification,
        llvm_gradient_ir=llvm_gradient_ir,
    )


def compile_registered_primitive_to_executable(
    registry: CustomDerivativeRegistry,
    identity: PrimitiveIdentity | str,
    sample_values: Sequence[float] | FloatArray,
    config: CompilerADExecutableConfig | None = None,
    *,
    sample_tangent: Sequence[float] | FloatArray | None = None,
    sample_cotangent: Sequence[float] | FloatArray | None = None,
) -> ExecutableCompilerADKernel:
    """Compile a registered primitive identity into an executable AD kernel."""
    if not isinstance(registry, CustomDerivativeRegistry):
        raise ValueError("registry must be a CustomDerivativeRegistry")
    primitive_identity = PrimitiveIdentity.parse(identity)
    transform = registry.transform_snapshot().get(primitive_identity)
    rule = registry.require(primitive_identity)
    if transform is not None and transform.lowering_rule is not None:
        lowering_rule = cast(Callable[..., Any], transform.lowering_rule)
        try:
            lowered = lowering_rule(
                rule,
                sample_values,
                config,
                sample_tangent=sample_tangent,
                sample_cotangent=sample_cotangent,
            )
        except TypeError:
            lowered = lowering_rule(rule)
        if not isinstance(lowered, ExecutableCompilerADKernel):
            raise ValueError("registered lowering_rule must return an ExecutableCompilerADKernel")
        return lowered
    return compile_custom_derivative_rule_to_executable(
        rule,
        sample_values,
        config,
        sample_tangent=sample_tangent,
        sample_cotangent=sample_cotangent,
    )


def make_program_ad_linalg_matrix_power_executable_lowering_rule(
    power: int | np.integer,
    sample_values: Sequence[float] | FloatArray,
    config: CompilerADExecutableConfig | None = None,
    *,
    sample_tangent: Sequence[float] | FloatArray | None = None,
) -> Callable[[CustomDerivativeRule], ExecutableCompilerADKernel]:
    """Build a verified executable lowering rule for a fixed matrix_power signature."""
    direct_rule = program_ad_linalg_matrix_power_derivative_rule(power)
    values = _as_finite_vector("sample_values", sample_values)
    compile_config = CompilerADExecutableConfig() if config is None else config

    def lowering_rule(_registered_rule: CustomDerivativeRule) -> ExecutableCompilerADKernel:
        if not isinstance(_registered_rule, CustomDerivativeRule):
            raise ValueError("registered_rule must be a CustomDerivativeRule")
        return compile_custom_derivative_rule_to_executable(
            direct_rule,
            values,
            compile_config,
            sample_tangent=sample_tangent,
        )

    return lowering_rule


def make_program_ad_linalg_multi_dot_executable_lowering_rule(
    operand_shapes: Sequence[Sequence[int]],
    sample_values: Sequence[float] | FloatArray,
    config: CompilerADExecutableConfig | None = None,
    *,
    sample_tangent: Sequence[float] | FloatArray | None = None,
) -> Callable[[CustomDerivativeRule], ExecutableCompilerADKernel]:
    """Build a verified executable lowering rule for a fixed multi_dot signature."""
    direct_rule = program_ad_linalg_multi_dot_derivative_rule(operand_shapes)
    values = _as_finite_vector("sample_values", sample_values)
    compile_config = CompilerADExecutableConfig() if config is None else config

    def lowering_rule(_registered_rule: CustomDerivativeRule) -> ExecutableCompilerADKernel:
        if not isinstance(_registered_rule, CustomDerivativeRule):
            raise ValueError("registered_rule must be a CustomDerivativeRule")
        return compile_custom_derivative_rule_to_executable(
            direct_rule,
            values,
            compile_config,
            sample_tangent=sample_tangent,
        )

    return lowering_rule


def _coupling_terms(K_nm: FloatArray) -> tuple[tuple[int, int, float], ...]:
    terms: list[tuple[int, int, float]] = []
    n_oscillators = K_nm.shape[0]
    for left in range(n_oscillators):
        for right in range(left + 1, n_oscillators):
            value = float(K_nm[left, right])
            if abs(value) > 1e-15:
                terms.append((left, right, value))
    return tuple(terms)
