# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — MLIR Native Compilation Integration Tests
"""Integration tests for MLIR facade, Kuramoto, and custom executable compilation."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.compiler.mlir import (
    CompilerADExecutableConfig,
    CompilerADKernelVerification,
    DifferentiableMLIRCompileConfig,
    ExecutableCompilerADKernel,
    MLIRCompileConfig,
    compile_custom_derivative_rule_to_executable,
    compile_custom_derivative_rule_to_mlir,
    compile_kuramoto_to_mlir,
    compile_registered_primitive_to_executable,
)
from scpn_quantum_control.differentiable import (
    CustomDerivativeRegistry,
    CustomDerivativeRule,
    PrimitiveIdentity,
    PrimitiveTransformRule,
)
from scpn_quantum_control.kuramoto_core import KuramotoProblem, build_kuramoto_problem


def _problem() -> KuramotoProblem:
    """Build the deterministic Kuramoto fixture used by MLIR smoke tests."""

    return build_kuramoto_problem(
        np.array(
            [
                [0.0, 0.25, 0.0],
                [0.25, 0.0, -0.5],
                [0.0, -0.5, 0.0],
            ],
            dtype=np.float64,
        ),
        np.array([0.1, -0.2, 0.3], dtype=np.float64),
        metadata={"experiment": "compiler-smoke"},
    )


def test_kuramoto_mlir_emits_deterministic_text_digest_and_resources() -> None:
    """MLIR export should be deterministic and explicit about compiler resources."""

    module = compile_kuramoto_to_mlir(
        _problem(),
        MLIRCompileConfig(time=0.4, trotter_steps=2, trotter_order=2),
    )
    repeat = compile_kuramoto_to_mlir(
        _problem(),
        MLIRCompileConfig(time=0.4, trotter_steps=2, trotter_order=2),
    )

    assert module.text == repeat.text
    assert module.sha256 == repeat.sha256
    assert module.resource_counts["n_oscillators"] == 3
    assert module.resource_counts["coupling_terms"] == 2
    assert module.resource_counts["trotter_steps"] == 2
    assert 'scpn.module = "kuramoto_xy"' in module.text
    assert "scpn.omega" in module.text
    assert "scpn.coupling" in module.text
    assert "scpn.trotter_evolve" in module.text


def test_kuramoto_mlir_rejects_invalid_compile_config() -> None:
    """MLIR config should fail closed before emitting misleading IR."""

    with pytest.raises(ValueError, match="trotter_steps"):
        MLIRCompileConfig(time=0.1, trotter_steps=0)
    with pytest.raises(ValueError, match="time"):
        MLIRCompileConfig(time=float("nan"))


def test_differentiable_mlir_lowers_custom_derivative_rule_deterministically() -> None:
    """Differentiable primitive lowering should be deterministic and auditable."""

    rule = CustomDerivativeRule(
        name="linear_residual",
        value_fn=lambda values: np.array(
            [values[0] + 2.0 * values[1], values[0] - values[1]],
            dtype=np.float64,
        ),
        jvp_rule=lambda values, tangent: np.array(
            [tangent[0] + 2.0 * tangent[1], tangent[0] - tangent[1]],
            dtype=np.float64,
        ),
        parameter_names=("theta", "phi"),
        trainable=(True, False),
    )

    module = compile_custom_derivative_rule_to_mlir(
        rule,
        np.array([1.5, -0.25], dtype=np.float64),
        DifferentiableMLIRCompileConfig(),
    )
    repeat = compile_custom_derivative_rule_to_mlir(
        rule,
        np.array([1.5, -0.25], dtype=np.float64),
        DifferentiableMLIRCompileConfig(),
    )

    assert module.text == repeat.text
    assert module.sha256 == repeat.sha256
    assert module.dialect == "scpn_diff"
    assert module.resource_counts["parameters"] == 2
    assert module.resource_counts["outputs"] == 2
    assert module.resource_counts["jacobian_nnz"] == 2
    assert module.resource_counts["trainable_parameters"] == 1
    assert 'scpn.module = "differentiable_primitive"' in module.text
    assert 'scpn.rule = "linear_residual"' in module.text
    assert "scpn_diff.parameter" in module.text
    assert "scpn_diff.value" in module.text
    assert "scpn_diff.jacobian" in module.text
    assert 'execution = "interchange_only"' in module.text
    assert module.metadata["target"] == "mlir"


def test_custom_derivative_rule_compiles_to_verified_executable_ad_kernel() -> None:
    """Compiler AD should execute differentiated primitive kernels with MLIR provenance."""

    identity = PrimitiveIdentity("scpn.quantum", "rx_expectation", "1")
    rule = CustomDerivativeRule(
        name="rx_expectation_rule",
        value_fn=lambda values: np.array([np.cos(values[0])], dtype=np.float64),
        jvp_rule=lambda values, tangent: np.array(
            [-np.sin(values[0]) * tangent[0]], dtype=np.float64
        ),
        vjp_rule=lambda values, cotangent: np.array(
            [-np.sin(values[0]) * cotangent[0]], dtype=np.float64
        ),
        parameter_names=("theta",),
        trainable=(True,),
    )
    registry = CustomDerivativeRegistry()
    registry.register_transform(
        PrimitiveTransformRule(
            identity=identity,
            derivative_rule=rule,
            lowering_metadata={
                "mlir": "available: executable scpn_diff MLIR-runtime primitive kernel",
                "mlir_op": "scpn_diff.rx_expectation",
                "llvm": "blocked: native LLVM lowering backend not linked",
            },
        )
    )

    kernel = compile_registered_primitive_to_executable(
        registry,
        identity,
        np.array([0.25], dtype=np.float64),
        sample_tangent=np.array([0.5], dtype=np.float64),
        sample_cotangent=np.array([2.0], dtype=np.float64),
    )

    assert isinstance(kernel, ExecutableCompilerADKernel)
    assert isinstance(kernel.verification, CompilerADKernelVerification)
    assert kernel.backend == "mlir_runtime"
    assert kernel.verification.passed is True
    assert kernel.mlir_module.metadata["target"] == "mlir"
    assert "differentiable_primitive" in kernel.mlir_module.text
    np.testing.assert_allclose(kernel.value(np.array([0.25])), [np.cos(0.25)])
    np.testing.assert_allclose(
        kernel.jvp(np.array([0.25]), np.array([0.5])),
        [-np.sin(0.25) * 0.5],
    )
    np.testing.assert_allclose(
        kernel.vjp(np.array([0.25]), np.array([2.0])),
        [-np.sin(0.25) * 2.0],
    )


def test_executable_compiler_ad_kernel_verifies_scalar_gradient_output() -> None:
    """Executable compiler AD should expose verified scalar-output gradients."""

    rule = CustomDerivativeRule(
        name="quadratic_phase_rule",
        value_fn=lambda values: np.array([values[0] ** 2 + np.sin(values[1])], dtype=np.float64),
        jvp_rule=lambda values, tangent: np.array(
            [2.0 * values[0] * tangent[0] + np.cos(values[1]) * tangent[1]],
            dtype=np.float64,
        ),
        vjp_rule=lambda values, cotangent: np.array(
            [2.0 * values[0] * cotangent[0], np.cos(values[1]) * cotangent[0]],
            dtype=np.float64,
        ),
        parameter_names=("theta", "phase"),
        trainable=(True, True),
    )
    values = np.array([0.75, -0.3], dtype=np.float64)

    kernel = compile_custom_derivative_rule_to_executable(
        rule,
        values,
        sample_tangent=np.array([0.25, -0.5], dtype=np.float64),
        sample_cotangent=np.array([1.0], dtype=np.float64),
    )

    assert kernel.backend == "mlir_runtime"
    assert kernel.verification.gradient_close is True
    assert "verified executable MLIR-runtime" in kernel.claim_boundary
    assert "native LLVM/JIT code generation remains fail-closed" in kernel.claim_boundary
    assert kernel.llvm_gradient_ir is not None
    assert 'source = "verified_mlir_runtime_vjp_cotangent_one"' in kernel.llvm_gradient_ir
    assert "define void @quadratic_phase_rule_gradient" in kernel.llvm_gradient_ir
    assert "scpn_diff.custom_rule" in kernel.mlir_module.text
    np.testing.assert_allclose(
        kernel.gradient(values),
        [1.5, np.cos(-0.3)],
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_differentiable_mlir_rejects_executable_target_claims() -> None:
    """LLVM/JIT target names must fail until backed by a real executable backend."""

    with pytest.raises(ValueError, match="target"):
        DifferentiableMLIRCompileConfig(target="llvm")
    with pytest.raises(ValueError, match="backend"):
        CompilerADExecutableConfig(backend="llvm")
    with pytest.raises(ValueError, match="boolean"):
        DifferentiableMLIRCompileConfig(include_numeric_payload=1)  # type: ignore[arg-type]
