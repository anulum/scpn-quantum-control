# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — MLIR Native Compilation Integration Tests
"""Integration tests for verified MLIR native executable compiler kernels."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

import scpn_quantum_control as scpn
import scpn_quantum_control.compiler.mlir as compiler_mlir
from scpn_quantum_control.compiler.mlir import (
    CompilerADExecutableConfig,
    CompilerADKernelVerification,
    DifferentiableMLIRCompileConfig,
    ExecutableCompilerADKernel,
    MLIRCompileConfig,
    build_compiler_ad_transform_plan,
    compile_compiler_ad_transform_plan_to_mlir,
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
    vmap,
)
from scpn_quantum_control.kuramoto_core import KuramotoProblem, build_kuramoto_problem

FloatArray = NDArray[np.float64]


def _eager_batching_rule(
    function: Callable[..., object],
    args: tuple[object, ...],
    axes: tuple[int | None, ...],
    out_axes: int,
) -> object:
    """Small primitive batching rule used by compiler-AD registry tests."""

    del axes, out_axes
    if len(args) != 1:
        raise ValueError("test batching rule expects one batched argument")
    batch = np.asarray(args[0], dtype=np.float64)
    return np.asarray([function(item) for item in batch], dtype=np.float64)


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


def test_native_llvm_jit_scalar_quadratic_kernel_executes_and_marks_plan_native() -> None:
    """Native LLVM/JIT scalar AD kernels should execute and update compiler contracts."""

    identity = PrimitiveIdentity("scpn.compiler_ad.native", "scalar_quadratic", "1")
    rule = CustomDerivativeRule(
        name="native_scalar_quadratic_rule",
        value_fn=lambda values: np.array(
            [3.0 * values[0] ** 2 - 2.0 * values[0] + 0.5],
            dtype=np.float64,
        ),
        jvp_rule=lambda values, tangent: np.array(
            [(6.0 * values[0] - 2.0) * tangent[0]],
            dtype=np.float64,
        ),
        vjp_rule=lambda values, cotangent: np.array(
            [(6.0 * values[0] - 2.0) * cotangent[0]],
            dtype=np.float64,
        ),
        parameter_names=("x",),
        trainable=(True,),
    )
    config = CompilerADExecutableConfig(backend="native_llvm_jit")
    values = np.array([0.75], dtype=np.float64)
    tangent = np.array([-0.25], dtype=np.float64)
    cotangent = np.array([1.5], dtype=np.float64)

    kernel = compiler_mlir.compile_scalar_quadratic_ad_to_native_llvm_jit(
        rule,
        quadratic=3.0,
        linear=-2.0,
        constant=0.5,
        sample_values=values,
        config=config,
        sample_tangent=tangent,
        sample_cotangent=cotangent,
    )

    assert kernel.backend == "native_llvm_jit"
    assert kernel.verification.passed is True
    assert kernel.verification.value_close is True
    assert kernel.verification.jvp_close is True
    assert kernel.verification.vjp_close is True
    assert kernel.verification.gradient_close is True
    assert "verified native LLVM MCJIT" in kernel.claim_boundary
    assert kernel.llvm_gradient_ir is not None
    assert "target triple" in kernel.llvm_gradient_ir
    assert "define void @native_scalar_quadratic_rule_value" in kernel.llvm_gradient_ir
    assert "define void @native_scalar_quadratic_rule_jvp" in kernel.llvm_gradient_ir
    assert "define void @native_scalar_quadratic_rule_vjp" in kernel.llvm_gradient_ir
    assert "define void @native_scalar_quadratic_rule_gradient" in kernel.llvm_gradient_ir
    np.testing.assert_allclose(kernel.value(values), [0.6875], rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(kernel.jvp(values, tangent), [-0.625], rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(kernel.vjp(values, cotangent), [3.75], rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(kernel.gradient(values), [2.5], rtol=1.0e-12, atol=1.0e-12)

    registry = CustomDerivativeRegistry()
    registry.register_transform(
        PrimitiveTransformRule(
            identity=identity,
            derivative_rule=rule,
            batching_rule=_eager_batching_rule,
            lowering_rule=compiler_mlir.make_scalar_quadratic_native_llvm_jit_lowering_rule(
                quadratic=3.0,
                linear=-2.0,
                constant=0.5,
                sample_values=values,
                config=config,
                sample_tangent=tangent,
                sample_cotangent=cotangent,
            ),
            lowering_metadata={
                "mlir": "available: executable scpn_diff MLIR-runtime primitive kernel",
                "mlir_op": "scpn_diff.native_scalar_quadratic",
                "mlir_runtime_verification": (
                    "verified: native LLVM/JIT scalar quadratic sample JVP"
                ),
                "llvm": "available: native LLVM MCJIT scalar AD kernel",
                "jit": "available: native LLVM MCJIT scalar AD kernel",
                "native_backend": "native_llvm_jit",
                "native_backend_verification": (
                    "verified: native LLVM MCJIT value/JVP/VJP/gradient"
                ),
                "static_derivative_factory": "native_scalar_quadratic_llvm_jit",
                "static_signature": "quadratic:f64,linear:f64,constant:f64",
                "nondifferentiable_boundary": "none_scalar_polynomial",
                "nondifferentiable_boundary_policy": "fail_closed",
            },
            shape_rule=lambda _args: (1,),
            dtype_rule=lambda _args: "float64",
            static_argument_rule=lambda args: args,
            nondifferentiable_policy="polynomial_is_smooth_over_reals",
            effect="pure",
        )
    )
    plan = build_compiler_ad_transform_plan(registry)
    module = compile_compiler_ad_transform_plan_to_mlir(plan)
    registered_kernel = compile_registered_primitive_to_executable(registry, identity, values)

    assert registered_kernel.backend == "native_llvm_jit"
    assert plan.executable_backend == "native_llvm_jit"
    assert module.metadata["executable_backend"] == "native_llvm_jit"
    assert module.resource_counts["executable_backends"] == 1
    assert module.resource_counts["llvm_backend_contracts"] == 1
    assert module.resource_counts["jit_backend_contracts"] == 1
    assert module.resource_counts["native_backend_contracts"] == 1
    assert module.resource_counts["primitive_readiness_native_executable"] == 1
    assert module.metadata["llvm_backend_contract_primitives"] == [identity.key]
    assert module.metadata["jit_backend_contract_primitives"] == [identity.key]
    assert module.metadata["native_backend_contract_primitives"] == [identity.key]
    assert module.metadata["primitive_readiness"][identity.key]["llvm_backend_contract"] is True
    assert module.metadata["primitive_readiness"][identity.key]["jit_backend_contract"] is True
    assert module.metadata["primitive_readiness"][identity.key]["native_backend_contract"] is True
    assert module.metadata["primitive_readiness"][identity.key]["verdict"] == "native_executable"
    assert module.metadata["primitive_hard_gaps"][identity.key] == ["rust_backend_contract"]


def test_native_llvm_jit_scalar_unary_sin_kernel_executes_and_marks_plan_native() -> None:
    """Native LLVM/JIT nonlinear scalar unary AD kernels should execute."""

    identity = PrimitiveIdentity("scpn.compiler_ad.native", "scalar_unary_sin", "1")
    rule = CustomDerivativeRule(
        name="native_scalar_unary_sin_rule",
        value_fn=lambda values: np.array([np.sin(values[0])], dtype=np.float64),
        jvp_rule=lambda values, tangent: np.array(
            [np.cos(values[0]) * tangent[0]],
            dtype=np.float64,
        ),
        vjp_rule=lambda values, cotangent: np.array(
            [np.cos(values[0]) * cotangent[0]],
            dtype=np.float64,
        ),
        parameter_names=("x",),
        trainable=(True,),
    )
    config = CompilerADExecutableConfig(backend="native_llvm_jit")
    values = np.array([0.5], dtype=np.float64)
    tangent = np.array([-0.75], dtype=np.float64)
    cotangent = np.array([2.0], dtype=np.float64)

    kernel = compiler_mlir.compile_scalar_unary_elementwise_ad_to_native_llvm_jit(
        rule,
        primitive="sin",
        sample_values=values,
        config=config,
        sample_tangent=tangent,
        sample_cotangent=cotangent,
    )

    expected_gradient = np.cos(values)
    assert kernel.backend == "native_llvm_jit"
    assert kernel.verification.passed is True
    assert kernel.verification.value_close is True
    assert kernel.verification.jvp_close is True
    assert kernel.verification.vjp_close is True
    assert kernel.verification.gradient_close is True
    assert "verified native LLVM MCJIT scalar unary" in kernel.claim_boundary
    assert kernel.llvm_gradient_ir is not None
    assert "declare double @llvm.sin.f64(double)" in kernel.llvm_gradient_ir
    assert "declare double @llvm.cos.f64(double)" in kernel.llvm_gradient_ir
    assert "define void @native_scalar_unary_sin_rule_value" in kernel.llvm_gradient_ir
    assert "define void @native_scalar_unary_sin_rule_jvp" in kernel.llvm_gradient_ir
    assert "define void @native_scalar_unary_sin_rule_vjp" in kernel.llvm_gradient_ir
    assert "define void @native_scalar_unary_sin_rule_gradient" in kernel.llvm_gradient_ir
    np.testing.assert_allclose(kernel.value(values), np.sin(values), rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(
        kernel.jvp(values, tangent), expected_gradient * tangent, rtol=1.0e-12, atol=1.0e-12
    )
    np.testing.assert_allclose(
        kernel.vjp(values, cotangent),
        expected_gradient * cotangent,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        kernel.gradient(values), expected_gradient, rtol=1.0e-12, atol=1.0e-12
    )

    registry = CustomDerivativeRegistry()
    registry.register_transform(
        PrimitiveTransformRule(
            identity=identity,
            derivative_rule=rule,
            batching_rule=_eager_batching_rule,
            lowering_rule=(
                compiler_mlir.make_scalar_unary_elementwise_native_llvm_jit_lowering_rule(
                    primitive="sin",
                    sample_values=values,
                    config=config,
                    sample_tangent=tangent,
                    sample_cotangent=cotangent,
                )
            ),
            lowering_metadata={
                "mlir": "available: executable scpn_diff MLIR-runtime primitive kernel",
                "mlir_op": "scpn_diff.native_scalar_unary_sin",
                "mlir_runtime_verification": "verified: native LLVM/JIT scalar unary sin JVP",
                "llvm": "available: native LLVM MCJIT scalar unary AD kernel",
                "jit": "available: native LLVM MCJIT scalar unary AD kernel",
                "native_backend": "native_llvm_jit",
                "native_backend_verification": (
                    "verified: native LLVM MCJIT scalar unary value/JVP/VJP/gradient"
                ),
                "static_derivative_factory": "native_scalar_unary_sin_llvm_jit",
                "static_signature": "primitive:sin;input:f64",
                "nondifferentiable_boundary": "none_smooth_unary",
                "nondifferentiable_boundary_policy": "fail_closed",
            },
            shape_rule=lambda _args: (1,),
            dtype_rule=lambda _args: "float64",
            static_argument_rule=lambda args: args,
            nondifferentiable_policy="smooth_unary_real_domain",
            effect="pure",
        )
    )
    plan = build_compiler_ad_transform_plan(registry)
    module = compile_compiler_ad_transform_plan_to_mlir(plan)
    registered_kernel = compile_registered_primitive_to_executable(registry, identity, values)

    assert registered_kernel.backend == "native_llvm_jit"
    assert plan.executable_backend == "native_llvm_jit"
    assert module.metadata["executable_backend"] == "native_llvm_jit"
    assert module.resource_counts["native_backend_contracts"] == 1
    assert module.resource_counts["primitive_readiness_native_executable"] == 1
    assert module.metadata["native_backend_contract_primitives"] == [identity.key]
    assert module.metadata["primitive_readiness"][identity.key]["verdict"] == "native_executable"
    assert module.metadata["primitive_hard_gaps"][identity.key] == ["rust_backend_contract"]


def test_native_llvm_jit_scalar_binary_multiply_kernel_executes_and_marks_plan_native() -> None:
    """Native LLVM/JIT scalar binary elementwise AD kernels should execute."""

    identity = PrimitiveIdentity("scpn.compiler_ad.native", "scalar_binary_multiply", "1")
    rule = CustomDerivativeRule(
        name="native_scalar_binary_multiply_rule",
        value_fn=lambda values: np.array([values[0] * values[1]], dtype=np.float64),
        jvp_rule=lambda values, tangent: np.array(
            [values[1] * tangent[0] + values[0] * tangent[1]],
            dtype=np.float64,
        ),
        vjp_rule=lambda values, cotangent: np.array(
            [values[1] * cotangent[0], values[0] * cotangent[0]],
            dtype=np.float64,
        ),
        parameter_names=("x", "y"),
        trainable=(True, True),
    )
    config = CompilerADExecutableConfig(backend="native_llvm_jit")
    values = np.array([1.5, -2.0], dtype=np.float64)
    tangent = np.array([0.25, -0.5], dtype=np.float64)
    cotangent = np.array([1.25], dtype=np.float64)

    kernel = compiler_mlir.compile_scalar_binary_elementwise_ad_to_native_llvm_jit(
        rule,
        primitive="multiply",
        sample_values=values,
        config=config,
        sample_tangent=tangent,
        sample_cotangent=cotangent,
    )

    assert kernel.backend == "native_llvm_jit"
    assert kernel.verification.passed is True
    assert kernel.verification.value_close is True
    assert kernel.verification.jvp_close is True
    assert kernel.verification.vjp_close is True
    assert kernel.verification.gradient_close is True
    assert "verified native LLVM MCJIT scalar binary" in kernel.claim_boundary
    assert kernel.llvm_gradient_ir is not None
    assert "define void @native_scalar_binary_multiply_rule_value" in kernel.llvm_gradient_ir
    assert "define void @native_scalar_binary_multiply_rule_jvp" in kernel.llvm_gradient_ir
    assert "define void @native_scalar_binary_multiply_rule_vjp" in kernel.llvm_gradient_ir
    assert "define void @native_scalar_binary_multiply_rule_gradient" in kernel.llvm_gradient_ir
    np.testing.assert_allclose(kernel.value(values), [-3.0], rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(kernel.jvp(values, tangent), [-1.25], rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(
        kernel.vjp(values, cotangent),
        [-2.5, 1.875],
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        kernel.gradient(values),
        [-2.0, 1.5],
        rtol=1.0e-12,
        atol=1.0e-12,
    )

    registry = CustomDerivativeRegistry()
    registry.register_transform(
        PrimitiveTransformRule(
            identity=identity,
            derivative_rule=rule,
            batching_rule=_eager_batching_rule,
            lowering_rule=(
                compiler_mlir.make_scalar_binary_elementwise_native_llvm_jit_lowering_rule(
                    primitive="multiply",
                    sample_values=values,
                    config=config,
                    sample_tangent=tangent,
                    sample_cotangent=cotangent,
                )
            ),
            lowering_metadata={
                "mlir": "available: executable scpn_diff MLIR-runtime primitive kernel",
                "mlir_op": "scpn_diff.native_scalar_binary_multiply",
                "mlir_runtime_verification": (
                    "verified: native LLVM/JIT scalar binary multiply JVP"
                ),
                "llvm": "available: native LLVM MCJIT scalar binary AD kernel",
                "jit": "available: native LLVM MCJIT scalar binary AD kernel",
                "native_backend": "native_llvm_jit",
                "native_backend_verification": (
                    "verified: native LLVM MCJIT scalar binary value/JVP/VJP/gradient"
                ),
                "static_derivative_factory": "native_scalar_binary_multiply_llvm_jit",
                "static_signature": "primitive:multiply;left:f64;right:f64",
                "nondifferentiable_boundary": "none_smooth_binary",
                "nondifferentiable_boundary_policy": "fail_closed",
            },
            shape_rule=lambda _args: (1,),
            dtype_rule=lambda _args: "float64",
            static_argument_rule=lambda args: args,
            nondifferentiable_policy="smooth_binary_real_domain",
            effect="pure",
        )
    )
    plan = build_compiler_ad_transform_plan(registry)
    module = compile_compiler_ad_transform_plan_to_mlir(plan)
    registered_kernel = compile_registered_primitive_to_executable(registry, identity, values)

    assert registered_kernel.backend == "native_llvm_jit"
    assert plan.executable_backend == "native_llvm_jit"
    assert module.metadata["executable_backend"] == "native_llvm_jit"
    assert module.resource_counts["native_backend_contracts"] == 1
    assert module.resource_counts["primitive_readiness_native_executable"] == 1
    assert module.metadata["native_backend_contract_primitives"] == [identity.key]
    assert module.metadata["primitive_readiness"][identity.key]["verdict"] == "native_executable"
    assert module.metadata["primitive_hard_gaps"][identity.key] == ["rust_backend_contract"]


def test_native_llvm_jit_vector_dot_kernel_executes_and_marks_plan_native() -> None:
    """Native LLVM/JIT vector dot-product AD kernels should execute."""

    identity = PrimitiveIdentity("scpn.compiler_ad.native", "vector_dot", "1")
    rule = CustomDerivativeRule(
        name="native_vector_dot_rule",
        value_fn=lambda values: np.array([np.dot(values[:2], values[2:])], dtype=np.float64),
        jvp_rule=lambda values, tangent: np.array(
            [np.dot(values[2:], tangent[:2]) + np.dot(values[:2], tangent[2:])],
            dtype=np.float64,
        ),
        vjp_rule=lambda values, cotangent: np.concatenate(
            [values[2:] * cotangent[0], values[:2] * cotangent[0]]
        ).astype(np.float64),
        parameter_names=("x0", "x1", "y0", "y1"),
        trainable=(True, True, True, True),
    )
    config = CompilerADExecutableConfig(backend="native_llvm_jit")
    values = np.array([1.0, 2.0, -3.0, 4.0], dtype=np.float64)
    tangent = np.array([0.5, -1.0, 2.0, -0.25], dtype=np.float64)
    cotangent = np.array([1.25], dtype=np.float64)

    kernel = compiler_mlir.compile_vector_dot_ad_to_native_llvm_jit(
        rule,
        dimension=2,
        sample_values=values,
        config=config,
        sample_tangent=tangent,
        sample_cotangent=cotangent,
    )

    assert kernel.backend == "native_llvm_jit"
    assert kernel.verification.passed is True
    assert kernel.verification.value_close is True
    assert kernel.verification.jvp_close is True
    assert kernel.verification.vjp_close is True
    assert kernel.verification.gradient_close is True
    assert "verified native LLVM MCJIT vector dot" in kernel.claim_boundary
    assert kernel.llvm_gradient_ir is not None
    assert "define void @native_vector_dot_rule_value" in kernel.llvm_gradient_ir
    assert "define void @native_vector_dot_rule_jvp" in kernel.llvm_gradient_ir
    assert "define void @native_vector_dot_rule_vjp" in kernel.llvm_gradient_ir
    assert "define void @native_vector_dot_rule_gradient" in kernel.llvm_gradient_ir
    np.testing.assert_allclose(kernel.value(values), [5.0], rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(kernel.jvp(values, tangent), [-4.0], rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(
        kernel.vjp(values, cotangent),
        [-3.75, 5.0, 1.25, 2.5],
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        kernel.gradient(values),
        [-3.0, 4.0, 1.0, 2.0],
        rtol=1.0e-12,
        atol=1.0e-12,
    )

    registry = CustomDerivativeRegistry()
    registry.register_transform(
        PrimitiveTransformRule(
            identity=identity,
            derivative_rule=rule,
            batching_rule=_eager_batching_rule,
            lowering_rule=compiler_mlir.make_vector_dot_native_llvm_jit_lowering_rule(
                dimension=2,
                sample_values=values,
                config=config,
                sample_tangent=tangent,
                sample_cotangent=cotangent,
            ),
            lowering_metadata={
                "mlir": "available: executable scpn_diff MLIR-runtime primitive kernel",
                "mlir_op": "scpn_diff.native_vector_dot",
                "mlir_runtime_verification": "verified: native LLVM/JIT vector dot JVP",
                "llvm": "available: native LLVM MCJIT vector dot AD kernel",
                "jit": "available: native LLVM MCJIT vector dot AD kernel",
                "native_backend": "native_llvm_jit",
                "native_backend_verification": (
                    "verified: native LLVM MCJIT vector dot value/JVP/VJP/gradient"
                ),
                "static_derivative_factory": "native_vector_dot_llvm_jit",
                "static_signature": "primitive:dot;dimension:2;layout:x_then_y",
                "nondifferentiable_boundary": "none_bilinear_vector_dot",
                "nondifferentiable_boundary_policy": "fail_closed",
            },
            shape_rule=lambda _args: (1,),
            dtype_rule=lambda _args: "float64",
            static_argument_rule=lambda args: args,
            nondifferentiable_policy="smooth_bilinear_real_domain",
            effect="pure",
        )
    )
    plan = build_compiler_ad_transform_plan(registry)
    module = compile_compiler_ad_transform_plan_to_mlir(plan)
    registered_kernel = compile_registered_primitive_to_executable(registry, identity, values)

    assert registered_kernel.backend == "native_llvm_jit"
    assert plan.executable_backend == "native_llvm_jit"
    assert module.metadata["executable_backend"] == "native_llvm_jit"
    assert module.resource_counts["native_backend_contracts"] == 1
    assert module.resource_counts["primitive_readiness_native_executable"] == 1
    assert module.metadata["native_backend_contract_primitives"] == [identity.key]
    assert module.metadata["primitive_readiness"][identity.key]["verdict"] == "native_executable"
    assert module.metadata["primitive_hard_gaps"][identity.key] == ["rust_backend_contract"]

    rust_registry = CustomDerivativeRegistry()
    rust_registry.register_transform(
        compiler_mlir.make_vector_dot_native_llvm_jit_primitive_transform(
            identity,
            rule,
            dimension=2,
            sample_values=values,
            config=config,
            sample_tangent=tangent,
            sample_cotangent=cotangent,
        )
    )
    rust_plan = build_compiler_ad_transform_plan(rust_registry)
    rust_module = compile_compiler_ad_transform_plan_to_mlir(rust_plan)
    rust_registered_kernel = compile_registered_primitive_to_executable(
        rust_registry,
        identity,
        values,
    )

    assert (
        scpn.make_vector_dot_native_llvm_jit_primitive_transform
        is compiler_mlir.make_vector_dot_native_llvm_jit_primitive_transform
    )
    assert rust_registered_kernel.backend == "native_llvm_jit"
    assert rust_module.resource_counts["native_backend_contracts"] == 1
    assert rust_module.resource_counts["rust_backend_contracts"] == 1
    assert rust_module.resource_counts["primitive_readiness_native_executable"] == 1
    assert rust_module.resource_counts["primitive_hard_gaps"] == 0
    assert rust_module.metadata["primitive_hard_gaps"][identity.key] == []
    assert rust_module.metadata["rust_backend_contract_primitives"] == [identity.key]
    assert rust_module.metadata["rust_backend_signatures"] == {
        identity.key: "primitive:dot;dimension:2;layout:x_then_y"
    }
    assert rust_module.metadata["rust_backend_functions"] == {
        identity.key: "vector_dot_value,vector_dot_jvp,vector_dot_vjp,vector_dot_gradient"
    }
    assert rust_module.metadata["rust_backend_verification_primitives"] == {
        identity.key: "verified: scpn_quantum_engine vector_dot value/JVP/VJP/gradient parity"
    }
    np.testing.assert_allclose(
        rust_registered_kernel.gradient(values),
        [-3.0, 4.0, 1.0, 2.0],
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_native_llvm_jit_matrix_quadratic_form_kernel_executes_and_marks_plan_native() -> None:
    """Native LLVM/JIT matrix quadratic-form AD kernels should execute."""

    identity = PrimitiveIdentity("scpn.compiler_ad.native", "matrix_quadratic_form", "1")

    def unpack(values: FloatArray) -> tuple[FloatArray, FloatArray]:
        matrix = values[:4].reshape(2, 2)
        vector = values[4:]
        return matrix, vector

    def value_fn(values: FloatArray) -> FloatArray:
        matrix, vector = unpack(values)
        return np.array([vector @ matrix @ vector], dtype=np.float64)

    def jvp_rule(values: FloatArray, tangent: FloatArray) -> FloatArray:
        matrix, vector = unpack(values)
        matrix_tangent, vector_tangent = unpack(tangent)
        return np.array(
            [
                vector @ matrix_tangent @ vector
                + vector_tangent @ matrix @ vector
                + vector @ matrix @ vector_tangent
            ],
            dtype=np.float64,
        )

    def vjp_rule(values: FloatArray, cotangent: FloatArray) -> FloatArray:
        matrix, vector = unpack(values)
        cotangent_value = cotangent[0]
        matrix_gradient = np.outer(vector, vector).reshape(-1)
        vector_gradient = (matrix + matrix.T) @ vector
        return cast(
            FloatArray, np.concatenate([matrix_gradient, vector_gradient]) * cotangent_value
        )

    rule = CustomDerivativeRule(
        name="native_matrix_quadratic_form_rule",
        value_fn=value_fn,
        jvp_rule=jvp_rule,
        vjp_rule=vjp_rule,
        parameter_names=("a00", "a01", "a10", "a11", "x0", "x1"),
        trainable=(True, True, True, True, True, True),
    )
    config = CompilerADExecutableConfig(backend="native_llvm_jit")
    values = np.array([2.0, -1.0, 0.5, 3.0, 1.5, -2.0], dtype=np.float64)
    tangent = np.array([0.1, -0.2, 0.3, 0.4, -0.5, 0.25], dtype=np.float64)
    cotangent = np.array([1.25], dtype=np.float64)

    kernel = compiler_mlir.compile_matrix_quadratic_form_ad_to_native_llvm_jit(
        rule,
        dimension=2,
        sample_values=values,
        config=config,
        sample_tangent=tangent,
        sample_cotangent=cotangent,
    )

    assert kernel.backend == "native_llvm_jit"
    assert kernel.verification.passed is True
    assert kernel.verification.value_close is True
    assert kernel.verification.jvp_close is True
    assert kernel.verification.vjp_close is True
    assert kernel.verification.gradient_close is True
    assert "verified native LLVM MCJIT matrix quadratic form" in kernel.claim_boundary
    assert kernel.llvm_gradient_ir is not None
    assert "define void @native_matrix_quadratic_form_rule_value" in kernel.llvm_gradient_ir
    assert "define void @native_matrix_quadratic_form_rule_jvp" in kernel.llvm_gradient_ir
    assert "define void @native_matrix_quadratic_form_rule_vjp" in kernel.llvm_gradient_ir
    assert "define void @native_matrix_quadratic_form_rule_gradient" in kernel.llvm_gradient_ir
    np.testing.assert_allclose(kernel.value(values), [18.0], rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(kernel.jvp(values, tangent), [-5.1625], rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(
        kernel.vjp(values, cotangent),
        [2.8125, -3.75, -3.75, 5.0, 8.75, -15.9375],
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        kernel.gradient(values),
        [2.25, -3.0, -3.0, 4.0, 7.0, -12.75],
        rtol=1.0e-12,
        atol=1.0e-12,
    )

    registry = CustomDerivativeRegistry()
    registry.register_transform(
        PrimitiveTransformRule(
            identity=identity,
            derivative_rule=rule,
            batching_rule=_eager_batching_rule,
            lowering_rule=compiler_mlir.make_matrix_quadratic_form_native_llvm_jit_lowering_rule(
                dimension=2,
                sample_values=values,
                config=config,
                sample_tangent=tangent,
                sample_cotangent=cotangent,
            ),
            lowering_metadata={
                "mlir": "available: executable scpn_diff MLIR-runtime primitive kernel",
                "mlir_op": "scpn_diff.native_matrix_quadratic_form",
                "mlir_runtime_verification": (
                    "verified: native LLVM/JIT matrix quadratic form JVP"
                ),
                "llvm": "available: native LLVM MCJIT matrix quadratic form AD kernel",
                "jit": "available: native LLVM MCJIT matrix quadratic form AD kernel",
                "native_backend": "native_llvm_jit",
                "native_backend_verification": (
                    "verified: native LLVM MCJIT matrix quadratic form value/JVP/VJP/gradient"
                ),
                "static_derivative_factory": "native_matrix_quadratic_form_llvm_jit",
                "static_signature": "primitive:quadratic_form;dimension:2;layout:matrix_then_vector",
                "nondifferentiable_boundary": "none_smooth_matrix_quadratic_form",
                "nondifferentiable_boundary_policy": "fail_closed",
            },
            shape_rule=lambda _args: (1,),
            dtype_rule=lambda _args: "float64",
            static_argument_rule=lambda args: args,
            nondifferentiable_policy="smooth_matrix_quadratic_form_real_domain",
            effect="pure",
        )
    )
    plan = build_compiler_ad_transform_plan(registry)
    module = compile_compiler_ad_transform_plan_to_mlir(plan)
    registered_kernel = compile_registered_primitive_to_executable(registry, identity, values)

    assert registered_kernel.backend == "native_llvm_jit"
    assert plan.executable_backend == "native_llvm_jit"
    assert module.metadata["executable_backend"] == "native_llvm_jit"
    assert module.resource_counts["native_backend_contracts"] == 1
    assert module.resource_counts["primitive_readiness_native_executable"] == 1
    assert module.metadata["native_backend_contract_primitives"] == [identity.key]
    assert module.metadata["primitive_readiness"][identity.key]["verdict"] == "native_executable"
    assert module.metadata["primitive_hard_gaps"][identity.key] == ["rust_backend_contract"]

    rust_registry = CustomDerivativeRegistry()
    rust_registry.register_transform(
        compiler_mlir.make_matrix_quadratic_form_native_llvm_jit_primitive_transform(
            identity,
            rule,
            dimension=2,
            sample_values=values,
            config=config,
            sample_tangent=tangent,
            sample_cotangent=cotangent,
        )
    )
    rust_plan = build_compiler_ad_transform_plan(rust_registry)
    rust_module = compile_compiler_ad_transform_plan_to_mlir(rust_plan)
    rust_registered_kernel = compile_registered_primitive_to_executable(
        rust_registry,
        identity,
        values,
    )

    assert rust_registered_kernel.backend == "native_llvm_jit"
    assert rust_plan.executable_backend == "native_llvm_jit"
    assert rust_module.metadata["executable_backend"] == "native_llvm_jit"
    assert rust_module.resource_counts["rust_backend_contracts"] == 1
    assert rust_module.resource_counts["rust_backend_incomplete_primitives"] == 0
    assert rust_module.resource_counts["rust_backend_blockers"] == 0
    assert rust_module.resource_counts["primitive_hard_gaps"] == 0
    assert rust_module.metadata["rust_backend_contract_primitives"] == [identity.key]
    assert rust_module.metadata["rust_backend_incomplete_primitives"] == []
    assert rust_module.metadata["rust_backend_blockers"] == {}
    assert rust_module.metadata["rust_backend_signatures"] == {
        identity.key: "primitive:quadratic_form;dimension:2;layout:matrix_then_vector"
    }
    assert rust_module.metadata["rust_backend_functions"] == {
        identity.key: (
            "matrix_quadratic_form_value,matrix_quadratic_form_jvp,"
            "matrix_quadratic_form_vjp,matrix_quadratic_form_gradient"
        )
    }
    assert rust_module.metadata["rust_backend_verification_primitives"] == {
        identity.key: (
            "verified: scpn_quantum_engine matrix_quadratic_form value/JVP/VJP/gradient parity"
        )
    }
    assert rust_module.resource_counts["rust_backend_verifications"] == 1
    assert rust_module.metadata["primitive_hard_gaps"][identity.key] == []
    assert identity.key not in rust_module.metadata["primitive_next_hard_gap"]
    assert rust_module.metadata["primitive_readiness"][identity.key] == {
        "adjoint_contract": True,
        "forward_contract": True,
        "jit_backend_contract": True,
        "llvm_backend_contract": True,
        "mlir_runtime_contract": True,
        "native_backend_contract": True,
        "registry_contract": True,
        "reverse_contract": True,
        "rust_backend_contract": True,
        "transform_contract": True,
        "verdict": "native_executable",
    }
    np.testing.assert_allclose(
        rust_registered_kernel.gradient(values),
        [2.25, -3.0, -3.0, 4.0, 7.0, -12.75],
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    assert scpn.make_matrix_quadratic_form_native_llvm_jit_primitive_transform is (
        compiler_mlir.make_matrix_quadratic_form_native_llvm_jit_primitive_transform
    )


def test_native_llvm_jit_vector_squared_norm_kernel_executes_and_marks_plan_native() -> None:
    """Native LLVM/JIT vector squared-norm AD kernels should execute."""

    identity = PrimitiveIdentity("scpn.compiler_ad.native", "vector_squared_norm", "1")
    rule = CustomDerivativeRule(
        name="native_vector_squared_norm_rule",
        value_fn=lambda values: np.array([np.dot(values, values)], dtype=np.float64),
        jvp_rule=lambda values, tangent: np.array(
            [2.0 * np.dot(values, tangent)],
            dtype=np.float64,
        ),
        vjp_rule=lambda values, cotangent: (2.0 * values * cotangent[0]).astype(np.float64),
        parameter_names=("x0", "x1", "x2"),
        trainable=(True, True, True),
    )
    config = CompilerADExecutableConfig(backend="native_llvm_jit")
    values = np.array([1.5, -2.0, 0.25], dtype=np.float64)
    tangent = np.array([-0.5, 0.75, 2.0], dtype=np.float64)
    cotangent = np.array([1.25], dtype=np.float64)

    kernel = compiler_mlir.compile_vector_squared_norm_ad_to_native_llvm_jit(
        rule,
        dimension=3,
        sample_values=values,
        config=config,
        sample_tangent=tangent,
        sample_cotangent=cotangent,
    )

    assert kernel.backend == "native_llvm_jit"
    assert kernel.verification.passed is True
    assert kernel.verification.value_close is True
    assert kernel.verification.jvp_close is True
    assert kernel.verification.vjp_close is True
    assert kernel.verification.gradient_close is True
    assert "verified native LLVM MCJIT vector squared norm" in kernel.claim_boundary
    assert kernel.llvm_gradient_ir is not None
    assert "define void @native_vector_squared_norm_rule_value" in kernel.llvm_gradient_ir
    assert "define void @native_vector_squared_norm_rule_jvp" in kernel.llvm_gradient_ir
    assert "define void @native_vector_squared_norm_rule_vjp" in kernel.llvm_gradient_ir
    assert "define void @native_vector_squared_norm_rule_gradient" in kernel.llvm_gradient_ir
    np.testing.assert_allclose(kernel.value(values), [6.3125], rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(kernel.jvp(values, tangent), [-3.5], rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(
        kernel.vjp(values, cotangent),
        [3.75, -5.0, 0.625],
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        kernel.gradient(values),
        [3.0, -4.0, 0.5],
        rtol=1.0e-12,
        atol=1.0e-12,
    )

    registry = CustomDerivativeRegistry()
    registry.register_transform(
        PrimitiveTransformRule(
            identity=identity,
            derivative_rule=rule,
            batching_rule=_eager_batching_rule,
            lowering_rule=compiler_mlir.make_vector_squared_norm_native_llvm_jit_lowering_rule(
                dimension=3,
                sample_values=values,
                config=config,
                sample_tangent=tangent,
                sample_cotangent=cotangent,
            ),
            lowering_metadata={
                "mlir": "available: executable scpn_diff MLIR-runtime primitive kernel",
                "mlir_op": "scpn_diff.native_vector_squared_norm",
                "mlir_runtime_verification": "verified: native LLVM/JIT vector squared norm JVP",
                "llvm": "available: native LLVM MCJIT vector squared norm AD kernel",
                "jit": "available: native LLVM MCJIT vector squared norm AD kernel",
                "native_backend": "native_llvm_jit",
                "native_backend_verification": (
                    "verified: native LLVM MCJIT vector squared norm value/JVP/VJP/gradient"
                ),
                "static_derivative_factory": "native_vector_squared_norm_llvm_jit",
                "static_signature": "primitive:squared_norm;dimension:3",
                "nondifferentiable_boundary": "none_smooth_vector_squared_norm",
                "nondifferentiable_boundary_policy": "fail_closed",
            },
            shape_rule=lambda _args: (1,),
            dtype_rule=lambda _args: "float64",
            static_argument_rule=lambda args: args,
            nondifferentiable_policy="smooth_vector_squared_norm_real_domain",
            effect="pure",
        )
    )
    plan = build_compiler_ad_transform_plan(registry)
    module = compile_compiler_ad_transform_plan_to_mlir(plan)
    registered_kernel = compile_registered_primitive_to_executable(registry, identity, values)

    assert registered_kernel.backend == "native_llvm_jit"
    assert plan.executable_backend == "native_llvm_jit"
    assert module.metadata["executable_backend"] == "native_llvm_jit"
    assert module.resource_counts["native_backend_contracts"] == 1
    assert module.resource_counts["primitive_readiness_native_executable"] == 1
    assert module.metadata["native_backend_contract_primitives"] == [identity.key]
    assert module.metadata["primitive_readiness"][identity.key]["verdict"] == "native_executable"
    assert module.metadata["primitive_hard_gaps"][identity.key] == ["rust_backend_contract"]

    rust_registry = CustomDerivativeRegistry()
    rust_registry.register_transform(
        compiler_mlir.make_vector_squared_norm_native_llvm_jit_primitive_transform(
            identity,
            rule,
            dimension=3,
            sample_values=values,
            config=config,
            sample_tangent=tangent,
            sample_cotangent=cotangent,
        )
    )
    rust_plan = build_compiler_ad_transform_plan(rust_registry)
    rust_module = compile_compiler_ad_transform_plan_to_mlir(rust_plan)

    assert (
        scpn.make_vector_squared_norm_native_llvm_jit_primitive_transform
        is compiler_mlir.make_vector_squared_norm_native_llvm_jit_primitive_transform
    )
    assert rust_module.resource_counts["native_backend_contracts"] == 1
    assert rust_module.resource_counts["rust_backend_contracts"] == 1
    assert rust_module.resource_counts["primitive_readiness_native_executable"] == 1
    assert rust_module.metadata["primitive_readiness"][identity.key]["verdict"] == (
        "native_executable"
    )
    assert rust_module.metadata["primitive_hard_gaps"][identity.key] == []
    assert rust_module.metadata["rust_backend_contract_primitives"] == [identity.key]
    assert rust_module.metadata["rust_backend_signatures"] == {
        identity.key: "primitive:squared_norm;dimension:3"
    }
    assert rust_module.metadata["rust_backend_functions"] == {
        identity.key: (
            "vector_squared_norm_value,vector_squared_norm_jvp,"
            "vector_squared_norm_vjp,vector_squared_norm_gradient"
        )
    }


def test_native_llvm_jit_matrix_vector_kernel_executes_and_marks_plan_native() -> None:
    """Native LLVM/JIT matrix-vector AD kernels should execute vector-output AD."""

    identity = PrimitiveIdentity("scpn.compiler_ad.native", "matrix_vector_product", "1")

    def unpack(values: FloatArray) -> tuple[FloatArray, FloatArray]:
        matrix = values[:4].reshape(2, 2)
        vector = values[4:]
        return matrix, vector

    def value_fn(values: FloatArray) -> FloatArray:
        matrix, vector = unpack(values)
        return matrix @ vector

    def jvp_rule(values: FloatArray, tangent: FloatArray) -> FloatArray:
        matrix, vector = unpack(values)
        matrix_tangent, vector_tangent = unpack(tangent)
        return matrix_tangent @ vector + matrix @ vector_tangent

    def vjp_rule(values: FloatArray, cotangent: FloatArray) -> FloatArray:
        matrix, vector = unpack(values)
        matrix_gradient = np.outer(cotangent, vector).reshape(-1)
        vector_gradient = matrix.T @ cotangent
        return np.concatenate([matrix_gradient, vector_gradient])

    rule = CustomDerivativeRule(
        name="native_matrix_vector_product_rule",
        value_fn=value_fn,
        jvp_rule=jvp_rule,
        vjp_rule=vjp_rule,
        parameter_names=("a00", "a01", "a10", "a11", "x0", "x1"),
        trainable=(True, True, True, True, True, True),
    )
    config = CompilerADExecutableConfig(backend="native_llvm_jit")
    values = np.array([2.0, -1.0, 0.5, 3.0, 1.5, -2.0], dtype=np.float64)
    tangent = np.array([0.1, -0.2, 0.3, 0.4, -0.5, 0.25], dtype=np.float64)
    cotangent = np.array([1.25, -0.5], dtype=np.float64)

    kernel = compiler_mlir.compile_matrix_vector_product_ad_to_native_llvm_jit(
        rule,
        dimension=2,
        sample_values=values,
        config=config,
        sample_tangent=tangent,
        sample_cotangent=cotangent,
    )

    assert kernel.backend == "native_llvm_jit"
    assert kernel.verification.passed is True
    assert kernel.verification.value_close is True
    assert kernel.verification.jvp_close is True
    assert kernel.verification.vjp_close is True
    assert kernel.verification.gradient_close is None
    assert "verified native LLVM MCJIT matrix-vector product" in kernel.claim_boundary
    assert kernel.llvm_gradient_ir is not None
    assert "define void @native_matrix_vector_product_rule_value" in kernel.llvm_gradient_ir
    assert "define void @native_matrix_vector_product_rule_jvp" in kernel.llvm_gradient_ir
    assert "define void @native_matrix_vector_product_rule_vjp" in kernel.llvm_gradient_ir
    assert "define void @native_matrix_vector_product_rule_gradient" in kernel.llvm_gradient_ir
    np.testing.assert_allclose(kernel.value(values), [5.0, -5.25], rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(
        kernel.jvp(values, tangent), [-0.7, 0.15], rtol=1.0e-12, atol=1.0e-12
    )
    np.testing.assert_allclose(
        kernel.vjp(values, cotangent),
        [1.875, -2.5, -0.75, 1.0, 2.25, -2.75],
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    with pytest.raises(ValueError, match="requires scalar output"):
        kernel.gradient(values)

    registry = CustomDerivativeRegistry()
    registry.register_transform(
        PrimitiveTransformRule(
            identity=identity,
            derivative_rule=rule,
            batching_rule=_eager_batching_rule,
            lowering_rule=compiler_mlir.make_matrix_vector_product_native_llvm_jit_lowering_rule(
                dimension=2,
                sample_values=values,
                config=config,
                sample_tangent=tangent,
                sample_cotangent=cotangent,
            ),
            lowering_metadata={
                "mlir": "available: executable scpn_diff MLIR-runtime primitive kernel",
                "mlir_op": "scpn_diff.native_matrix_vector_product",
                "mlir_runtime_verification": "verified: native LLVM/JIT matrix-vector JVP",
                "llvm": "available: native LLVM MCJIT matrix-vector AD kernel",
                "jit": "available: native LLVM MCJIT matrix-vector AD kernel",
                "native_backend": "native_llvm_jit",
                "native_backend_verification": (
                    "verified: native LLVM MCJIT matrix-vector value/JVP/VJP"
                ),
                "static_derivative_factory": "native_matrix_vector_product_llvm_jit",
                "static_signature": "primitive:matvec;dimension:2;layout:matrix_then_vector",
                "nondifferentiable_boundary": "none_smooth_matrix_vector_product",
                "nondifferentiable_boundary_policy": "fail_closed",
            },
            shape_rule=lambda _args: (2,),
            dtype_rule=lambda _args: "float64",
            static_argument_rule=lambda args: args,
            nondifferentiable_policy="smooth_matrix_vector_product_real_domain",
            effect="pure",
        )
    )
    plan = build_compiler_ad_transform_plan(registry)
    module = compile_compiler_ad_transform_plan_to_mlir(plan)
    registered_kernel = compile_registered_primitive_to_executable(registry, identity, values)

    assert registered_kernel.backend == "native_llvm_jit"
    assert plan.executable_backend == "native_llvm_jit"
    assert module.metadata["executable_backend"] == "native_llvm_jit"
    assert module.resource_counts["native_backend_contracts"] == 1
    assert module.resource_counts["primitive_readiness_native_executable"] == 1
    assert module.metadata["native_backend_contract_primitives"] == [identity.key]
    assert module.metadata["primitive_readiness"][identity.key]["verdict"] == "native_executable"
    assert module.metadata["primitive_hard_gaps"][identity.key] == ["rust_backend_contract"]

    rust_registry = CustomDerivativeRegistry()
    rust_registry.register_transform(
        compiler_mlir.make_matrix_vector_product_native_llvm_jit_primitive_transform(
            identity,
            rule,
            dimension=2,
            sample_values=values,
            config=config,
            sample_tangent=tangent,
            sample_cotangent=cotangent,
        )
    )
    rust_plan = build_compiler_ad_transform_plan(rust_registry)
    rust_module = compile_compiler_ad_transform_plan_to_mlir(rust_plan)
    rust_registered_kernel = compile_registered_primitive_to_executable(
        rust_registry,
        identity,
        values,
    )

    assert rust_registered_kernel.backend == "native_llvm_jit"
    assert rust_plan.executable_backend == "native_llvm_jit"
    assert rust_module.metadata["executable_backend"] == "native_llvm_jit"
    assert rust_module.resource_counts["rust_backend_contracts"] == 1
    assert rust_module.resource_counts["rust_backend_incomplete_primitives"] == 0
    assert rust_module.resource_counts["rust_backend_blockers"] == 0
    assert rust_module.resource_counts["primitive_hard_gaps"] == 0
    assert rust_module.metadata["rust_backend_contract_primitives"] == [identity.key]
    assert rust_module.metadata["rust_backend_incomplete_primitives"] == []
    assert rust_module.metadata["rust_backend_blockers"] == {}
    assert rust_module.metadata["rust_backend_signatures"] == {
        identity.key: "primitive:matvec;dimension:2;layout:matrix_then_vector"
    }
    assert rust_module.metadata["rust_backend_functions"] == {
        identity.key: (
            "matrix_vector_product_value,matrix_vector_product_jvp,"
            "matrix_vector_product_vjp,matrix_vector_product_sum_gradient"
        )
    }
    assert rust_module.metadata["rust_backend_verification_primitives"] == {
        identity.key: (
            "verified: scpn_quantum_engine matrix_vector_product value/JVP/VJP/sum-gradient parity"
        )
    }
    assert rust_module.resource_counts["rust_backend_verifications"] == 1
    assert rust_module.metadata["primitive_hard_gaps"][identity.key] == []
    assert identity.key not in rust_module.metadata["primitive_next_hard_gap"]
    assert rust_module.metadata["primitive_readiness"][identity.key] == {
        "adjoint_contract": True,
        "forward_contract": True,
        "jit_backend_contract": True,
        "llvm_backend_contract": True,
        "mlir_runtime_contract": True,
        "native_backend_contract": True,
        "registry_contract": True,
        "reverse_contract": True,
        "rust_backend_contract": True,
        "transform_contract": True,
        "verdict": "native_executable",
    }
    batched_value = cast(
        FloatArray,
        vmap(
            lambda row: row,
            primitive_identity=identity,
            registry=rust_registry,
        )(np.vstack([values, values + np.array([0.25, 0.1, 0.05, -0.2, 0.3, -0.4])])),
    )
    np.testing.assert_allclose(
        batched_value[0],
        rust_registered_kernel.value(values),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    assert scpn.make_matrix_vector_product_native_llvm_jit_primitive_transform is (
        compiler_mlir.make_matrix_vector_product_native_llvm_jit_primitive_transform
    )


def test_native_llvm_jit_matrix_matrix_kernel_executes_and_marks_plan_native() -> None:
    """Native LLVM/JIT matrix-matrix AD kernels should execute matrix-output AD."""

    identity = PrimitiveIdentity("scpn.compiler_ad.native", "matrix_matrix_product", "1")

    def unpack(values: FloatArray) -> tuple[FloatArray, FloatArray]:
        left = values[:4].reshape(2, 2)
        right = values[4:].reshape(2, 2)
        return left, right

    def value_fn(values: FloatArray) -> FloatArray:
        left, right = unpack(values)
        return cast(FloatArray, (left @ right).reshape(-1))

    def jvp_rule(values: FloatArray, tangent: FloatArray) -> FloatArray:
        left, right = unpack(values)
        left_tangent, right_tangent = unpack(tangent)
        return cast(FloatArray, (left_tangent @ right + left @ right_tangent).reshape(-1))

    def vjp_rule(values: FloatArray, cotangent: FloatArray) -> FloatArray:
        left, right = unpack(values)
        cotangent_matrix = cotangent.reshape(2, 2)
        left_gradient = cotangent_matrix @ right.T
        right_gradient = left.T @ cotangent_matrix
        return np.concatenate([left_gradient.reshape(-1), right_gradient.reshape(-1)])

    rule = CustomDerivativeRule(
        name="native_matrix_matrix_product_rule",
        value_fn=value_fn,
        jvp_rule=jvp_rule,
        vjp_rule=vjp_rule,
        parameter_names=("a00", "a01", "a10", "a11", "b00", "b01", "b10", "b11"),
        trainable=(True, True, True, True, True, True, True, True),
    )
    config = CompilerADExecutableConfig(backend="native_llvm_jit")
    values = np.array([1.0, -2.0, 0.5, 3.0, 4.0, -1.0, 2.0, 0.25], dtype=np.float64)
    tangent = np.array([0.2, -0.1, 0.3, 0.4, -0.5, 0.75, 0.25, -0.2], dtype=np.float64)
    cotangent = np.array([1.25, -0.5, 0.75, 2.0], dtype=np.float64)

    kernel = compiler_mlir.compile_matrix_matrix_product_ad_to_native_llvm_jit(
        rule,
        dimension=2,
        sample_values=values,
        config=config,
        sample_tangent=tangent,
        sample_cotangent=cotangent,
    )

    assert kernel.backend == "native_llvm_jit"
    assert kernel.verification.passed is True
    assert kernel.verification.value_close is True
    assert kernel.verification.jvp_close is True
    assert kernel.verification.vjp_close is True
    assert kernel.verification.gradient_close is None
    assert "verified native LLVM MCJIT matrix-matrix product" in kernel.claim_boundary
    assert kernel.llvm_gradient_ir is not None
    assert "define void @native_matrix_matrix_product_rule_value" in kernel.llvm_gradient_ir
    assert "define void @native_matrix_matrix_product_rule_jvp" in kernel.llvm_gradient_ir
    assert "define void @native_matrix_matrix_product_rule_vjp" in kernel.llvm_gradient_ir
    assert "define void @native_matrix_matrix_product_rule_gradient" in kernel.llvm_gradient_ir
    np.testing.assert_allclose(
        kernel.value(values),
        [0.0, -1.5, 8.0, 0.25],
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        kernel.jvp(values, tangent),
        [-0.4, 0.925, 2.5, -0.425],
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        kernel.vjp(values, cotangent),
        [5.5, 2.375, 1.0, 2.0, 1.625, 0.5, -0.25, 7.0],
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    with pytest.raises(ValueError, match="requires scalar output"):
        kernel.gradient(values)

    registry = CustomDerivativeRegistry()
    registry.register_transform(
        PrimitiveTransformRule(
            identity=identity,
            derivative_rule=rule,
            batching_rule=_eager_batching_rule,
            lowering_rule=compiler_mlir.make_matrix_matrix_product_native_llvm_jit_lowering_rule(
                dimension=2,
                sample_values=values,
                config=config,
                sample_tangent=tangent,
                sample_cotangent=cotangent,
            ),
            lowering_metadata={
                "mlir": "available: executable scpn_diff MLIR-runtime primitive kernel",
                "mlir_op": "scpn_diff.native_matrix_matrix_product",
                "mlir_runtime_verification": "verified: native LLVM/JIT matrix-matrix JVP",
                "llvm": "available: native LLVM MCJIT matrix-matrix AD kernel",
                "jit": "available: native LLVM MCJIT matrix-matrix AD kernel",
                "native_backend": "native_llvm_jit",
                "native_backend_verification": (
                    "verified: native LLVM MCJIT matrix-matrix value/JVP/VJP"
                ),
                "static_derivative_factory": "native_matrix_matrix_product_llvm_jit",
                "static_signature": "primitive:matmul;dimension:2;layout:left_then_right",
                "nondifferentiable_boundary": "none_smooth_matrix_matrix_product",
                "nondifferentiable_boundary_policy": "fail_closed",
            },
            shape_rule=lambda _args: (2, 2),
            dtype_rule=lambda _args: "float64",
            static_argument_rule=lambda args: args,
            nondifferentiable_policy="smooth_matrix_matrix_product_real_domain",
            effect="pure",
        )
    )
    plan = build_compiler_ad_transform_plan(registry)
    module = compile_compiler_ad_transform_plan_to_mlir(plan)
    registered_kernel = compile_registered_primitive_to_executable(registry, identity, values)

    assert registered_kernel.backend == "native_llvm_jit"
    assert plan.executable_backend == "native_llvm_jit"
    assert module.metadata["executable_backend"] == "native_llvm_jit"
    assert module.resource_counts["native_backend_contracts"] == 1
    assert module.resource_counts["primitive_readiness_native_executable"] == 1
    assert module.metadata["native_backend_contract_primitives"] == [identity.key]
    assert module.metadata["primitive_readiness"][identity.key]["verdict"] == "native_executable"
    assert module.metadata["primitive_hard_gaps"][identity.key] == ["rust_backend_contract"]

    rust_registry = CustomDerivativeRegistry()
    rust_registry.register_transform(
        compiler_mlir.make_matrix_matrix_product_native_llvm_jit_primitive_transform(
            identity,
            rule,
            dimension=2,
            sample_values=values,
            config=config,
            sample_tangent=tangent,
            sample_cotangent=cotangent,
        )
    )
    rust_plan = build_compiler_ad_transform_plan(rust_registry)
    rust_module = compile_compiler_ad_transform_plan_to_mlir(rust_plan)
    rust_registered_kernel = compile_registered_primitive_to_executable(
        rust_registry,
        identity,
        values,
    )

    assert rust_registered_kernel.backend == "native_llvm_jit"
    assert rust_plan.executable_backend == "native_llvm_jit"
    assert rust_module.metadata["executable_backend"] == "native_llvm_jit"
    assert rust_module.resource_counts["rust_backend_contracts"] == 1
    assert rust_module.resource_counts["rust_backend_incomplete_primitives"] == 0
    assert rust_module.resource_counts["rust_backend_blockers"] == 0
    assert rust_module.resource_counts["primitive_hard_gaps"] == 0
    assert rust_module.metadata["rust_backend_contract_primitives"] == [identity.key]
    assert rust_module.metadata["rust_backend_incomplete_primitives"] == []
    assert rust_module.metadata["rust_backend_blockers"] == {}
    assert rust_module.metadata["rust_backend_signatures"] == {
        identity.key: "primitive:matmul;dimension:2;layout:left_then_right"
    }
    assert rust_module.metadata["rust_backend_functions"] == {
        identity.key: (
            "matrix_matrix_product_value,matrix_matrix_product_jvp,"
            "matrix_matrix_product_vjp,matrix_matrix_product_sum_gradient"
        )
    }
    assert rust_module.metadata["rust_backend_verification_primitives"] == {
        identity.key: (
            "verified: scpn_quantum_engine matrix_matrix_product value/JVP/VJP/sum-gradient parity"
        )
    }
    assert rust_module.resource_counts["rust_backend_verifications"] == 1
    assert rust_module.metadata["primitive_hard_gaps"][identity.key] == []
    assert identity.key not in rust_module.metadata["primitive_next_hard_gap"]
    assert rust_module.metadata["primitive_readiness"][identity.key] == {
        "adjoint_contract": True,
        "forward_contract": True,
        "jit_backend_contract": True,
        "llvm_backend_contract": True,
        "mlir_runtime_contract": True,
        "native_backend_contract": True,
        "registry_contract": True,
        "reverse_contract": True,
        "rust_backend_contract": True,
        "transform_contract": True,
        "verdict": "native_executable",
    }
    batched_value = cast(
        FloatArray,
        vmap(
            lambda row: row,
            primitive_identity=identity,
            registry=rust_registry,
        )(np.vstack([values, values + np.array([0.25, 0.1, 0.05, -0.2, 0.3, -0.4, 0.2, 0.1])])),
    )
    np.testing.assert_allclose(
        batched_value[0],
        rust_registered_kernel.value(values),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    assert scpn.make_matrix_matrix_product_native_llvm_jit_primitive_transform is (
        compiler_mlir.make_matrix_matrix_product_native_llvm_jit_primitive_transform
    )


def test_native_llvm_jit_matrix_trace_kernel_executes_and_marks_plan_native() -> None:
    """Native LLVM/JIT matrix trace AD kernels should execute scalar-output AD."""

    identity = PrimitiveIdentity("scpn.compiler_ad.native", "matrix_trace", "1")
    rule = CustomDerivativeRule(
        name="native_matrix_trace_rule",
        value_fn=lambda values: np.array([np.trace(values.reshape(2, 2))], dtype=np.float64),
        jvp_rule=lambda _values, tangent: np.array(
            [np.trace(tangent.reshape(2, 2))],
            dtype=np.float64,
        ),
        vjp_rule=lambda _values, cotangent: np.array(
            [cotangent[0], 0.0, 0.0, cotangent[0]],
            dtype=np.float64,
        ),
        parameter_names=("a00", "a01", "a10", "a11"),
        trainable=(True, True, True, True),
    )
    config = CompilerADExecutableConfig(backend="native_llvm_jit")
    values = np.array([2.0, -1.0, 0.5, 3.0], dtype=np.float64)
    tangent = np.array([0.1, -0.2, 0.3, 0.4], dtype=np.float64)
    cotangent = np.array([1.25], dtype=np.float64)

    kernel = compiler_mlir.compile_matrix_trace_ad_to_native_llvm_jit(
        rule,
        dimension=2,
        sample_values=values,
        config=config,
        sample_tangent=tangent,
        sample_cotangent=cotangent,
    )

    assert kernel.backend == "native_llvm_jit"
    assert kernel.verification.passed is True
    assert kernel.verification.value_close is True
    assert kernel.verification.jvp_close is True
    assert kernel.verification.vjp_close is True
    assert kernel.verification.gradient_close is True
    assert "verified native LLVM MCJIT matrix trace" in kernel.claim_boundary
    assert kernel.llvm_gradient_ir is not None
    assert "define void @native_matrix_trace_rule_value" in kernel.llvm_gradient_ir
    assert "define void @native_matrix_trace_rule_jvp" in kernel.llvm_gradient_ir
    assert "define void @native_matrix_trace_rule_vjp" in kernel.llvm_gradient_ir
    assert "define void @native_matrix_trace_rule_gradient" in kernel.llvm_gradient_ir
    np.testing.assert_allclose(kernel.value(values), [5.0], rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(kernel.jvp(values, tangent), [0.5], rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(
        kernel.vjp(values, cotangent),
        [1.25, 0.0, 0.0, 1.25],
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        kernel.gradient(values),
        [1.0, 0.0, 0.0, 1.0],
        rtol=1.0e-12,
        atol=1.0e-12,
    )

    registry = CustomDerivativeRegistry()
    registry.register_transform(
        PrimitiveTransformRule(
            identity=identity,
            derivative_rule=rule,
            batching_rule=_eager_batching_rule,
            lowering_rule=compiler_mlir.make_matrix_trace_native_llvm_jit_lowering_rule(
                dimension=2,
                sample_values=values,
                config=config,
                sample_tangent=tangent,
                sample_cotangent=cotangent,
            ),
            lowering_metadata={
                "mlir": "available: executable scpn_diff MLIR-runtime primitive kernel",
                "mlir_op": "scpn_diff.native_matrix_trace",
                "mlir_runtime_verification": "verified: native LLVM/JIT matrix trace JVP",
                "llvm": "available: native LLVM MCJIT matrix trace AD kernel",
                "jit": "available: native LLVM MCJIT matrix trace AD kernel",
                "native_backend": "native_llvm_jit",
                "native_backend_verification": (
                    "verified: native LLVM MCJIT matrix trace value/JVP/VJP/gradient"
                ),
                "static_derivative_factory": "native_matrix_trace_llvm_jit",
                "static_signature": "primitive:trace;dimension:2;layout:row_major",
                "nondifferentiable_boundary": "none_smooth_matrix_trace",
                "nondifferentiable_boundary_policy": "fail_closed",
            },
            shape_rule=lambda _args: (1,),
            dtype_rule=lambda _args: "float64",
            static_argument_rule=lambda args: args,
            nondifferentiable_policy="smooth_matrix_trace_real_domain",
            effect="pure",
        )
    )
    plan = build_compiler_ad_transform_plan(registry)
    module = compile_compiler_ad_transform_plan_to_mlir(plan)
    registered_kernel = compile_registered_primitive_to_executable(registry, identity, values)

    assert registered_kernel.backend == "native_llvm_jit"
    assert plan.executable_backend == "native_llvm_jit"
    assert module.metadata["executable_backend"] == "native_llvm_jit"
    assert module.resource_counts["native_backend_contracts"] == 1
    assert module.resource_counts["primitive_readiness_native_executable"] == 1
    assert module.metadata["native_backend_contract_primitives"] == [identity.key]
    assert module.metadata["primitive_readiness"][identity.key]["verdict"] == "native_executable"
    assert module.metadata["primitive_hard_gaps"][identity.key] == ["rust_backend_contract"]

    rust_registry = CustomDerivativeRegistry()
    rust_registry.register_transform(
        compiler_mlir.make_matrix_trace_native_llvm_jit_primitive_transform(
            identity,
            rule,
            dimension=2,
            sample_values=values,
            config=config,
            sample_tangent=tangent,
            sample_cotangent=cotangent,
        )
    )
    rust_plan = build_compiler_ad_transform_plan(rust_registry)
    rust_module = compile_compiler_ad_transform_plan_to_mlir(rust_plan)
    rust_registered_kernel = compile_registered_primitive_to_executable(
        rust_registry,
        identity,
        values,
    )

    assert (
        scpn.make_matrix_trace_native_llvm_jit_primitive_transform
        is compiler_mlir.make_matrix_trace_native_llvm_jit_primitive_transform
    )
    assert rust_registered_kernel.backend == "native_llvm_jit"
    assert rust_module.resource_counts["native_backend_contracts"] == 1
    assert rust_module.resource_counts["rust_backend_contracts"] == 1
    assert rust_module.resource_counts["primitive_readiness_native_executable"] == 1
    assert rust_module.resource_counts["primitive_hard_gaps"] == 0
    assert rust_module.metadata["primitive_hard_gaps"][identity.key] == []
    assert rust_module.metadata["rust_backend_contract_primitives"] == [identity.key]
    assert rust_module.metadata["rust_backend_signatures"] == {
        identity.key: "primitive:trace;dimension:2;layout:row_major"
    }
    assert rust_module.metadata["rust_backend_functions"] == {
        identity.key: (
            "matrix_trace_value,matrix_trace_jvp,matrix_trace_vjp,matrix_trace_gradient"
        )
    }
    assert rust_module.metadata["rust_backend_verification_primitives"] == {
        identity.key: ("verified: scpn_quantum_engine matrix_trace value/JVP/VJP/gradient parity")
    }
    np.testing.assert_allclose(
        rust_registered_kernel.gradient(values),
        [1.0, 0.0, 0.0, 1.0],
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_native_llvm_jit_matrix_frobenius_norm_squared_kernel_executes_and_marks_plan_native() -> (
    None
):
    """Native LLVM/JIT Frobenius-squared AD kernels should execute scalar matrix reductions."""

    identity = PrimitiveIdentity("scpn.compiler_ad.native", "matrix_frobenius_norm_squared", "1")
    rule = CustomDerivativeRule(
        name="native_matrix_frobenius_norm_squared_rule",
        value_fn=lambda values: np.array(
            [np.sum(np.square(values.reshape(2, 2)))],
            dtype=np.float64,
        ),
        jvp_rule=lambda values, tangent: np.array(
            [2.0 * np.sum(values.reshape(2, 2) * tangent.reshape(2, 2))],
            dtype=np.float64,
        ),
        vjp_rule=lambda values, cotangent: 2.0 * cotangent[0] * values,
        parameter_names=("a00", "a01", "a10", "a11"),
        trainable=(True, True, True, True),
    )
    config = CompilerADExecutableConfig(backend="native_llvm_jit")
    values = np.array([2.0, -1.0, 0.5, 3.0], dtype=np.float64)
    tangent = np.array([0.1, -0.2, 0.3, 0.4], dtype=np.float64)
    cotangent = np.array([1.25], dtype=np.float64)

    kernel = compiler_mlir.compile_matrix_frobenius_norm_squared_ad_to_native_llvm_jit(
        rule,
        dimension=2,
        sample_values=values,
        config=config,
        sample_tangent=tangent,
        sample_cotangent=cotangent,
    )

    assert kernel.backend == "native_llvm_jit"
    assert kernel.verification.passed is True
    assert kernel.verification.value_close is True
    assert kernel.verification.jvp_close is True
    assert kernel.verification.vjp_close is True
    assert kernel.verification.gradient_close is True
    assert "verified native LLVM MCJIT matrix Frobenius-squared" in kernel.claim_boundary
    assert kernel.llvm_gradient_ir is not None
    assert (
        "define void @native_matrix_frobenius_norm_squared_rule_value" in kernel.llvm_gradient_ir
    )
    assert "define void @native_matrix_frobenius_norm_squared_rule_jvp" in kernel.llvm_gradient_ir
    assert "define void @native_matrix_frobenius_norm_squared_rule_vjp" in kernel.llvm_gradient_ir
    assert (
        "define void @native_matrix_frobenius_norm_squared_rule_gradient"
        in kernel.llvm_gradient_ir
    )
    np.testing.assert_allclose(kernel.value(values), [14.25], rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(kernel.jvp(values, tangent), [3.5], rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(
        kernel.vjp(values, cotangent),
        [5.0, -2.5, 1.25, 7.5],
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        kernel.gradient(values),
        [4.0, -2.0, 1.0, 6.0],
        rtol=1.0e-12,
        atol=1.0e-12,
    )

    registry = CustomDerivativeRegistry()
    registry.register_transform(
        PrimitiveTransformRule(
            identity=identity,
            derivative_rule=rule,
            batching_rule=_eager_batching_rule,
            lowering_rule=compiler_mlir.make_matrix_frobenius_norm_squared_native_llvm_jit_lowering_rule(
                dimension=2,
                sample_values=values,
                config=config,
                sample_tangent=tangent,
                sample_cotangent=cotangent,
            ),
            lowering_metadata={
                "mlir": "available: executable scpn_diff MLIR-runtime primitive kernel",
                "mlir_op": "scpn_diff.native_matrix_frobenius_norm_squared",
                "mlir_runtime_verification": (
                    "verified: native LLVM/JIT matrix Frobenius-squared JVP"
                ),
                "llvm": "available: native LLVM MCJIT matrix Frobenius-squared AD kernel",
                "jit": "available: native LLVM MCJIT matrix Frobenius-squared AD kernel",
                "native_backend": "native_llvm_jit",
                "native_backend_verification": (
                    "verified: native LLVM MCJIT matrix Frobenius-squared value/JVP/VJP/gradient"
                ),
                "static_derivative_factory": "native_matrix_frobenius_norm_squared_llvm_jit",
                "static_signature": "primitive:frobenius_norm_squared;dimension:2;layout:row_major",
                "nondifferentiable_boundary": "none_smooth_matrix_frobenius_norm_squared",
                "nondifferentiable_boundary_policy": "fail_closed",
            },
            shape_rule=lambda _args: (1,),
            dtype_rule=lambda _args: "float64",
            static_argument_rule=lambda args: args,
            nondifferentiable_policy="smooth_matrix_frobenius_norm_squared_real_domain",
            effect="pure",
        )
    )
    plan = build_compiler_ad_transform_plan(registry)
    module = compile_compiler_ad_transform_plan_to_mlir(plan)
    registered_kernel = compile_registered_primitive_to_executable(registry, identity, values)

    assert registered_kernel.backend == "native_llvm_jit"
    assert plan.executable_backend == "native_llvm_jit"
    assert module.metadata["executable_backend"] == "native_llvm_jit"
    assert module.resource_counts["native_backend_contracts"] == 1
    assert module.resource_counts["primitive_readiness_native_executable"] == 1
    assert module.metadata["native_backend_contract_primitives"] == [identity.key]
    assert module.metadata["primitive_readiness"][identity.key]["verdict"] == "native_executable"
    assert module.metadata["primitive_hard_gaps"][identity.key] == ["rust_backend_contract"]

    rust_registry = CustomDerivativeRegistry()
    rust_registry.register_transform(
        compiler_mlir.make_matrix_frobenius_norm_squared_native_llvm_jit_primitive_transform(
            identity,
            rule,
            dimension=2,
            sample_values=values,
            config=config,
            sample_tangent=tangent,
            sample_cotangent=cotangent,
        )
    )
    rust_plan = build_compiler_ad_transform_plan(rust_registry)
    rust_module = compile_compiler_ad_transform_plan_to_mlir(rust_plan)
    rust_registered_kernel = compile_registered_primitive_to_executable(
        rust_registry,
        identity,
        values,
    )

    assert (
        scpn.make_matrix_frobenius_norm_squared_native_llvm_jit_primitive_transform
        is compiler_mlir.make_matrix_frobenius_norm_squared_native_llvm_jit_primitive_transform
    )
    assert rust_registered_kernel.backend == "native_llvm_jit"
    assert rust_module.resource_counts["native_backend_contracts"] == 1
    assert rust_module.resource_counts["rust_backend_contracts"] == 1
    assert rust_module.resource_counts["primitive_readiness_native_executable"] == 1
    assert rust_module.resource_counts["primitive_hard_gaps"] == 0
    assert rust_module.metadata["primitive_hard_gaps"][identity.key] == []
    assert rust_module.metadata["rust_backend_contract_primitives"] == [identity.key]
    assert rust_module.metadata["rust_backend_signatures"] == {
        identity.key: "primitive:frobenius_norm_squared;dimension:2;layout:row_major"
    }
    assert rust_module.metadata["rust_backend_functions"] == {
        identity.key: (
            "matrix_frobenius_norm_squared_value,matrix_frobenius_norm_squared_jvp,"
            "matrix_frobenius_norm_squared_vjp,matrix_frobenius_norm_squared_gradient"
        )
    }
    assert rust_module.metadata["rust_backend_verification_primitives"] == {
        identity.key: (
            "verified: scpn_quantum_engine matrix_frobenius_norm_squared "
            "value/JVP/VJP/gradient parity"
        )
    }
    np.testing.assert_allclose(
        rust_registered_kernel.gradient(values),
        [4.0, -2.0, 1.0, 6.0],
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_native_llvm_jit_matrix_2x2_determinant_kernel_executes_and_marks_plan_native() -> None:
    """Native LLVM/JIT determinant AD kernels should execute exact scalar matrix AD."""

    identity = PrimitiveIdentity("scpn.compiler_ad.native", "matrix_2x2_determinant", "1")
    rule = CustomDerivativeRule(
        name="native_matrix_2x2_determinant_rule",
        value_fn=lambda values: np.array(
            [values[0] * values[3] - values[1] * values[2]],
            dtype=np.float64,
        ),
        jvp_rule=lambda values, tangent: np.array(
            [
                tangent[0] * values[3]
                + values[0] * tangent[3]
                - tangent[1] * values[2]
                - values[1] * tangent[2]
            ],
            dtype=np.float64,
        ),
        vjp_rule=lambda values, cotangent: (
            cotangent[0]
            * np.array([values[3], -values[2], -values[1], values[0]], dtype=np.float64)
        ),
        parameter_names=("a00", "a01", "a10", "a11"),
        trainable=(True, True, True, True),
    )
    config = CompilerADExecutableConfig(backend="native_llvm_jit")
    values = np.array([2.0, -1.0, 0.5, 3.0], dtype=np.float64)
    tangent = np.array([0.1, -0.2, 0.3, 0.4], dtype=np.float64)
    cotangent = np.array([1.25], dtype=np.float64)

    kernel = compiler_mlir.compile_matrix_2x2_determinant_ad_to_native_llvm_jit(
        rule,
        sample_values=values,
        config=config,
        sample_tangent=tangent,
        sample_cotangent=cotangent,
    )

    assert kernel.backend == "native_llvm_jit"
    assert kernel.verification.passed is True
    assert kernel.verification.value_close is True
    assert kernel.verification.jvp_close is True
    assert kernel.verification.vjp_close is True
    assert kernel.verification.gradient_close is True
    assert "verified native LLVM MCJIT 2x2 determinant" in kernel.claim_boundary
    assert kernel.llvm_gradient_ir is not None
    assert "define void @native_matrix_2x2_determinant_rule_value" in kernel.llvm_gradient_ir
    assert "define void @native_matrix_2x2_determinant_rule_jvp" in kernel.llvm_gradient_ir
    assert "define void @native_matrix_2x2_determinant_rule_vjp" in kernel.llvm_gradient_ir
    assert "define void @native_matrix_2x2_determinant_rule_gradient" in kernel.llvm_gradient_ir
    np.testing.assert_allclose(kernel.value(values), [6.5], rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(kernel.jvp(values, tangent), [1.5], rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(
        kernel.vjp(values, cotangent),
        [3.75, -0.625, 1.25, 2.5],
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        kernel.gradient(values),
        [3.0, -0.5, 1.0, 2.0],
        rtol=1.0e-12,
        atol=1.0e-12,
    )

    registry = CustomDerivativeRegistry()
    registry.register_transform(
        PrimitiveTransformRule(
            identity=identity,
            derivative_rule=rule,
            batching_rule=_eager_batching_rule,
            lowering_rule=compiler_mlir.make_matrix_2x2_determinant_native_llvm_jit_lowering_rule(
                sample_values=values,
                config=config,
                sample_tangent=tangent,
                sample_cotangent=cotangent,
            ),
            lowering_metadata={
                "mlir": "available: executable scpn_diff MLIR-runtime primitive kernel",
                "mlir_op": "scpn_diff.native_matrix_2x2_determinant",
                "mlir_runtime_verification": "verified: native LLVM/JIT 2x2 determinant JVP",
                "llvm": "available: native LLVM MCJIT 2x2 determinant AD kernel",
                "jit": "available: native LLVM MCJIT 2x2 determinant AD kernel",
                "native_backend": "native_llvm_jit",
                "native_backend_verification": (
                    "verified: native LLVM MCJIT 2x2 determinant value/JVP/VJP/gradient"
                ),
                "static_derivative_factory": "native_matrix_2x2_determinant_llvm_jit",
                "static_signature": "primitive:determinant;dimension:2;layout:row_major",
                "nondifferentiable_boundary": "none_polynomial_matrix_2x2_determinant",
                "nondifferentiable_boundary_policy": "fail_closed",
            },
            shape_rule=lambda _args: (1,),
            dtype_rule=lambda _args: "float64",
            static_argument_rule=lambda args: args,
            nondifferentiable_policy="polynomial_matrix_2x2_determinant_real_domain",
            effect="pure",
        )
    )
    plan = build_compiler_ad_transform_plan(registry)
    module = compile_compiler_ad_transform_plan_to_mlir(plan)
    registered_kernel = compile_registered_primitive_to_executable(registry, identity, values)

    assert registered_kernel.backend == "native_llvm_jit"
    assert plan.executable_backend == "native_llvm_jit"
    assert module.metadata["executable_backend"] == "native_llvm_jit"
    assert module.resource_counts["native_backend_contracts"] == 1
    assert module.resource_counts["primitive_readiness_native_executable"] == 1
    assert module.metadata["native_backend_contract_primitives"] == [identity.key]
    assert module.metadata["primitive_readiness"][identity.key]["verdict"] == "native_executable"
    assert module.metadata["primitive_hard_gaps"][identity.key] == ["rust_backend_contract"]

    rust_registry = CustomDerivativeRegistry()
    rust_registry.register_transform(
        compiler_mlir.make_matrix_2x2_determinant_native_llvm_jit_primitive_transform(
            identity,
            rule,
            sample_values=values,
            config=config,
            sample_tangent=tangent,
            sample_cotangent=cotangent,
        )
    )
    rust_plan = build_compiler_ad_transform_plan(rust_registry)
    rust_module = compile_compiler_ad_transform_plan_to_mlir(rust_plan)
    rust_registered_kernel = compile_registered_primitive_to_executable(
        rust_registry,
        identity,
        values,
    )

    assert (
        scpn.make_matrix_2x2_determinant_native_llvm_jit_primitive_transform
        is compiler_mlir.make_matrix_2x2_determinant_native_llvm_jit_primitive_transform
    )
    assert rust_registered_kernel.backend == "native_llvm_jit"
    assert rust_module.resource_counts["native_backend_contracts"] == 1
    assert rust_module.resource_counts["rust_backend_contracts"] == 1
    assert rust_module.resource_counts["primitive_readiness_native_executable"] == 1
    assert rust_module.resource_counts["primitive_hard_gaps"] == 0
    assert rust_module.metadata["primitive_hard_gaps"][identity.key] == []
    assert rust_module.metadata["rust_backend_contract_primitives"] == [identity.key]
    assert rust_module.metadata["rust_backend_signatures"] == {
        identity.key: "primitive:determinant;dimension:2;layout:row_major"
    }
    assert rust_module.metadata["rust_backend_functions"] == {
        identity.key: (
            "matrix_2x2_determinant_value,matrix_2x2_determinant_jvp,"
            "matrix_2x2_determinant_vjp,matrix_2x2_determinant_gradient"
        )
    }
    assert rust_module.metadata["rust_backend_verification_primitives"] == {
        identity.key: (
            "verified: scpn_quantum_engine matrix_2x2_determinant value/JVP/VJP/gradient parity"
        )
    }
    np.testing.assert_allclose(
        rust_registered_kernel.gradient(values),
        [3.0, -0.5, 1.0, 2.0],
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_executable_ad_kernel_batching_rule_dispatches_native_value_jvp_and_vjp() -> None:
    """Primitive vmap batching should dispatch native executable kernels without fallback calls."""

    identity = PrimitiveIdentity("scpn.compiler_ad.native", "batched_matrix_2x2_determinant", "1")
    rule = CustomDerivativeRule(
        name="native_batched_matrix_2x2_determinant_rule",
        value_fn=lambda values: np.array(
            [values[0] * values[3] - values[1] * values[2]],
            dtype=np.float64,
        ),
        jvp_rule=lambda values, tangent: np.array(
            [
                tangent[0] * values[3]
                + values[0] * tangent[3]
                - tangent[1] * values[2]
                - values[1] * tangent[2]
            ],
            dtype=np.float64,
        ),
        vjp_rule=lambda values, cotangent: (
            cotangent[0]
            * np.array([values[3], -values[2], -values[1], values[0]], dtype=np.float64)
        ),
        parameter_names=("a00", "a01", "a10", "a11"),
        trainable=(True, True, True, True),
    )
    config = CompilerADExecutableConfig(backend="native_llvm_jit")
    sample_values = np.array([2.0, -1.0, 0.5, 3.0], dtype=np.float64)
    sample_tangent = np.array([0.1, -0.2, 0.3, 0.4], dtype=np.float64)
    sample_cotangent = np.array([1.25], dtype=np.float64)
    kernel = compiler_mlir.compile_matrix_2x2_determinant_ad_to_native_llvm_jit(
        rule,
        sample_values=sample_values,
        config=config,
        sample_tangent=sample_tangent,
        sample_cotangent=sample_cotangent,
    )
    registry = CustomDerivativeRegistry()
    registry.register_transform(
        PrimitiveTransformRule(
            identity=identity,
            derivative_rule=rule,
            batching_rule=compiler_mlir.make_executable_ad_kernel_batching_rule(kernel),
            lowering_rule=compiler_mlir.make_matrix_2x2_determinant_native_llvm_jit_lowering_rule(
                sample_values=sample_values,
                config=config,
                sample_tangent=sample_tangent,
                sample_cotangent=sample_cotangent,
            ),
            lowering_metadata={
                "mlir": "available: executable scpn_diff MLIR-runtime primitive kernel",
                "mlir_op": "scpn_diff.native_matrix_2x2_determinant",
                "llvm": "available: native LLVM MCJIT 2x2 determinant AD kernel",
                "jit": "available: native LLVM MCJIT 2x2 determinant AD kernel",
                "native_backend": "native_llvm_jit",
                "static_derivative_factory": "native_matrix_2x2_determinant_llvm_jit",
                "static_signature": "primitive:determinant;dimension:2;layout:row_major",
                "nondifferentiable_boundary": "none_polynomial_matrix_2x2_determinant",
                "nondifferentiable_boundary_policy": "fail_closed",
            },
            shape_rule=lambda _args: (1,),
            dtype_rule=lambda _args: "float64",
            static_argument_rule=lambda args: args,
            nondifferentiable_policy="polynomial_matrix_2x2_determinant_real_domain",
            effect="pure",
        )
    )

    values = np.array(
        [
            [2.0, -1.0, 0.5, 3.0],
            [1.5, 0.25, -2.0, 4.0],
        ],
        dtype=np.float64,
    )
    tangents = np.array(
        [
            [0.1, -0.2, 0.3, 0.4],
            [-0.5, 0.75, 0.25, -0.2],
        ],
        dtype=np.float64,
    )
    cotangents = np.array([[1.25], [-0.5]], dtype=np.float64)

    def unreachable(*_args: object) -> FloatArray:
        raise AssertionError("primitive-specific executable batching rule was not used")

    batched_value = vmap(unreachable, primitive_identity=identity, registry=registry)(values)
    batched_jvp = vmap(
        unreachable,
        in_axes=(0, 0),
        primitive_identity=identity,
        registry=registry,
    )(values, tangents)
    batched_vjp = vmap(
        unreachable,
        in_axes=(0, 0),
        primitive_identity=identity,
        registry=registry,
    )(values, cotangents)

    expected_values = np.asarray([kernel.value(row) for row in values])
    expected_jvps = np.asarray(
        [kernel.jvp(row, tangent) for row, tangent in zip(values, tangents)]
    )
    expected_vjps = np.asarray(
        [kernel.vjp(row, cotangent) for row, cotangent in zip(values, cotangents)]
    )
    np.testing.assert_allclose(
        cast(FloatArray, batched_value), expected_values, rtol=1.0e-12, atol=1.0e-12
    )
    np.testing.assert_allclose(
        cast(FloatArray, batched_jvp), expected_jvps, rtol=1.0e-12, atol=1.0e-12
    )
    np.testing.assert_allclose(
        cast(FloatArray, batched_vjp), expected_vjps, rtol=1.0e-12, atol=1.0e-12
    )
    assert registry.require_complete_contract(identity).batching_rule is not None

    with pytest.raises(ValueError, match="ambiguous"):
        ambiguous_rule = CustomDerivativeRule(
            name="ambiguous_rule",
            value_fn=lambda row: np.asarray(row, dtype=np.float64),
            jvp_rule=lambda row, tangent: np.asarray(tangent, dtype=np.float64),
            vjp_rule=lambda row, cotangent: np.asarray(cotangent, dtype=np.float64),
            parameter_names=("x0", "x1"),
            trainable=(True, True),
        )
        ambiguous_kernel = compile_custom_derivative_rule_to_executable(
            ambiguous_rule,
            np.array([1.0, 2.0], dtype=np.float64),
            CompilerADExecutableConfig(),
        )
        compiler_mlir.make_executable_ad_kernel_batching_rule(ambiguous_kernel)(
            unreachable,
            (
                np.array([[1.0, 2.0]], dtype=np.float64),
                np.array([[0.5, -0.25]], dtype=np.float64),
            ),
            (0, 0),
            0,
        )


def test_native_llvm_jit_matrix_2x2_inverse_kernel_executes_and_marks_plan_native() -> None:
    """Native LLVM/JIT inverse AD kernels should execute exact nonsingular matrix AD."""

    def inverse(values: FloatArray) -> FloatArray:
        return np.asarray(np.linalg.inv(values.reshape(2, 2)).reshape(4), dtype=np.float64)

    def inverse_jvp(values: FloatArray, tangent: FloatArray) -> FloatArray:
        inverse_matrix = np.linalg.inv(values.reshape(2, 2))
        tangent_matrix = tangent.reshape(2, 2)
        return cast(FloatArray, (-inverse_matrix @ tangent_matrix @ inverse_matrix).reshape(4))

    def inverse_vjp(values: FloatArray, cotangent: FloatArray) -> FloatArray:
        inverse_matrix = np.linalg.inv(values.reshape(2, 2))
        cotangent_matrix = cotangent.reshape(2, 2)
        return cast(
            FloatArray,
            (-(inverse_matrix.T @ cotangent_matrix @ inverse_matrix.T)).reshape(4),
        )

    identity = PrimitiveIdentity("scpn.compiler_ad.native", "matrix_2x2_inverse", "1")
    rule = CustomDerivativeRule(
        name="native_matrix_2x2_inverse_rule",
        value_fn=inverse,
        jvp_rule=inverse_jvp,
        vjp_rule=inverse_vjp,
        parameter_names=("a00", "a01", "a10", "a11"),
        trainable=(True, True, True, True),
    )
    config = CompilerADExecutableConfig(backend="native_llvm_jit")
    values = np.array([2.0, -1.0, 0.5, 3.0], dtype=np.float64)
    tangent = np.array([0.1, -0.2, 0.3, 0.4], dtype=np.float64)
    cotangent = np.array([0.75, -1.25, 0.5, 2.0], dtype=np.float64)

    kernel = compiler_mlir.compile_matrix_2x2_inverse_ad_to_native_llvm_jit(
        rule,
        sample_values=values,
        config=config,
        sample_tangent=tangent,
        sample_cotangent=cotangent,
    )

    assert kernel.backend == "native_llvm_jit"
    assert kernel.verification.passed is True
    assert kernel.verification.value_close is True
    assert kernel.verification.jvp_close is True
    assert kernel.verification.vjp_close is True
    assert "verified native LLVM MCJIT 2x2 inverse" in kernel.claim_boundary
    assert "public gradient remains scalar-output fail-closed" in kernel.claim_boundary
    assert kernel.llvm_gradient_ir is not None
    assert "define void @native_matrix_2x2_inverse_rule_value" in kernel.llvm_gradient_ir
    assert "define void @native_matrix_2x2_inverse_rule_jvp" in kernel.llvm_gradient_ir
    assert "define void @native_matrix_2x2_inverse_rule_vjp" in kernel.llvm_gradient_ir
    assert "define void @native_matrix_2x2_inverse_rule_gradient" in kernel.llvm_gradient_ir
    np.testing.assert_allclose(kernel.value(values), inverse(values), rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(
        kernel.jvp(values, tangent),
        inverse_jvp(values, tangent),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        kernel.vjp(values, cotangent),
        inverse_vjp(values, cotangent),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    with pytest.raises(ValueError):
        kernel.gradient(values)

    registry = CustomDerivativeRegistry()
    registry.register_transform(
        PrimitiveTransformRule(
            identity=identity,
            derivative_rule=rule,
            batching_rule=_eager_batching_rule,
            lowering_rule=compiler_mlir.make_matrix_2x2_inverse_native_llvm_jit_lowering_rule(
                sample_values=values,
                config=config,
                sample_tangent=tangent,
                sample_cotangent=cotangent,
            ),
            lowering_metadata={
                "mlir": "available: executable scpn_diff MLIR-runtime primitive kernel",
                "mlir_op": "scpn_diff.native_matrix_2x2_inverse",
                "mlir_runtime_verification": "verified: native LLVM/JIT 2x2 inverse JVP",
                "llvm": "available: native LLVM MCJIT 2x2 inverse AD kernel",
                "jit": "available: native LLVM MCJIT 2x2 inverse AD kernel",
                "native_backend": "native_llvm_jit",
                "native_backend_verification": (
                    "verified: native LLVM MCJIT 2x2 inverse value/JVP/VJP"
                ),
                "static_derivative_factory": "native_matrix_2x2_inverse_llvm_jit",
                "static_signature": "primitive:inverse;dimension:2;layout:row_major",
                "nondifferentiable_boundary": "singular_matrix_2x2_inverse",
                "nondifferentiable_boundary_policy": "fail_closed",
            },
            shape_rule=lambda _args: (4,),
            dtype_rule=lambda _args: "float64",
            static_argument_rule=lambda args: args,
            nondifferentiable_policy="nonsingular_matrix_2x2_inverse_real_domain",
            effect="pure",
        )
    )
    plan = build_compiler_ad_transform_plan(registry)
    module = compile_compiler_ad_transform_plan_to_mlir(plan)
    registered_kernel = compile_registered_primitive_to_executable(registry, identity, values)

    assert registered_kernel.backend == "native_llvm_jit"
    assert plan.executable_backend == "native_llvm_jit"
    assert module.metadata["executable_backend"] == "native_llvm_jit"
    assert module.resource_counts["native_backend_contracts"] == 1
    assert module.resource_counts["primitive_readiness_native_executable"] == 1
    assert module.metadata["native_backend_contract_primitives"] == [identity.key]
    assert module.metadata["primitive_readiness"][identity.key]["verdict"] == "native_executable"
    assert module.metadata["primitive_hard_gaps"][identity.key] == ["rust_backend_contract"]

    rust_registry = CustomDerivativeRegistry()
    rust_registry.register_transform(
        compiler_mlir.make_matrix_2x2_inverse_native_llvm_jit_primitive_transform(
            identity,
            rule,
            sample_values=values,
            config=config,
            sample_tangent=tangent,
            sample_cotangent=cotangent,
        )
    )
    rust_plan = build_compiler_ad_transform_plan(rust_registry)
    rust_module = compile_compiler_ad_transform_plan_to_mlir(rust_plan)
    rust_registered_kernel = compile_registered_primitive_to_executable(
        rust_registry,
        identity,
        values,
    )

    assert (
        scpn.make_matrix_2x2_inverse_native_llvm_jit_primitive_transform
        is compiler_mlir.make_matrix_2x2_inverse_native_llvm_jit_primitive_transform
    )
    assert rust_registered_kernel.backend == "native_llvm_jit"
    assert rust_module.resource_counts["native_backend_contracts"] == 1
    assert rust_module.resource_counts["rust_backend_contracts"] == 1
    assert rust_module.resource_counts["primitive_readiness_native_executable"] == 1
    assert rust_module.resource_counts["primitive_hard_gaps"] == 0
    assert rust_module.metadata["primitive_hard_gaps"][identity.key] == []
    assert rust_module.metadata["rust_backend_contract_primitives"] == [identity.key]
    assert rust_module.metadata["rust_backend_signatures"] == {
        identity.key: "primitive:inverse;dimension:2;layout:row_major"
    }
    assert rust_module.metadata["rust_backend_functions"] == {
        identity.key: (
            "matrix_2x2_inverse_value,matrix_2x2_inverse_jvp,"
            "matrix_2x2_inverse_vjp,matrix_2x2_inverse_sum_gradient"
        )
    }
    assert rust_module.metadata["rust_backend_verification_primitives"] == {
        identity.key: (
            "verified: scpn_quantum_engine matrix_2x2_inverse value/JVP/VJP/sum-gradient parity"
        )
    }


def test_native_llvm_jit_matrix_2x2_solve_kernel_executes_and_marks_plan_native() -> None:
    """Native LLVM/JIT solve AD kernels should execute exact nonsingular linear solves."""

    def solve(values: FloatArray) -> FloatArray:
        matrix = values[:4].reshape(2, 2)
        vector = values[4:]
        return np.asarray(np.linalg.solve(matrix, vector), dtype=np.float64)

    def solve_jvp(values: FloatArray, tangent: FloatArray) -> FloatArray:
        matrix = values[:4].reshape(2, 2)
        primal_solution = np.linalg.solve(matrix, values[4:])
        tangent_matrix = tangent[:4].reshape(2, 2)
        tangent_vector = tangent[4:]
        return np.asarray(
            np.linalg.solve(matrix, tangent_vector - tangent_matrix @ primal_solution),
            dtype=np.float64,
        )

    def solve_vjp(values: FloatArray, cotangent: FloatArray) -> FloatArray:
        matrix = values[:4].reshape(2, 2)
        primal_solution = np.linalg.solve(matrix, values[4:])
        adjoint_vector = np.linalg.solve(matrix.T, cotangent)
        return np.concatenate(
            [
                (-np.outer(adjoint_vector, primal_solution)).reshape(4),
                adjoint_vector,
            ]
        )

    identity = PrimitiveIdentity("scpn.compiler_ad.native", "matrix_2x2_solve", "1")
    rule = CustomDerivativeRule(
        name="native_matrix_2x2_solve_rule",
        value_fn=solve,
        jvp_rule=solve_jvp,
        vjp_rule=solve_vjp,
        parameter_names=("a00", "a01", "a10", "a11", "b0", "b1"),
        trainable=(True, True, True, True, True, True),
    )
    config = CompilerADExecutableConfig(backend="native_llvm_jit")
    values = np.array([2.0, -1.0, 0.5, 3.0, 1.5, -2.0], dtype=np.float64)
    tangent = np.array([0.1, -0.2, 0.3, 0.4, -0.5, 0.75], dtype=np.float64)
    cotangent = np.array([1.25, -0.75], dtype=np.float64)

    kernel = compiler_mlir.compile_matrix_2x2_solve_ad_to_native_llvm_jit(
        rule,
        sample_values=values,
        config=config,
        sample_tangent=tangent,
        sample_cotangent=cotangent,
    )

    assert kernel.backend == "native_llvm_jit"
    assert kernel.verification.passed is True
    assert kernel.verification.value_close is True
    assert kernel.verification.jvp_close is True
    assert kernel.verification.vjp_close is True
    assert "verified native LLVM MCJIT 2x2 solve" in kernel.claim_boundary
    assert "public gradient remains scalar-output fail-closed" in kernel.claim_boundary
    assert kernel.llvm_gradient_ir is not None
    assert "define void @native_matrix_2x2_solve_rule_value" in kernel.llvm_gradient_ir
    assert "define void @native_matrix_2x2_solve_rule_jvp" in kernel.llvm_gradient_ir
    assert "define void @native_matrix_2x2_solve_rule_vjp" in kernel.llvm_gradient_ir
    assert "define void @native_matrix_2x2_solve_rule_gradient" in kernel.llvm_gradient_ir
    np.testing.assert_allclose(kernel.value(values), solve(values), rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(
        kernel.jvp(values, tangent),
        solve_jvp(values, tangent),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        kernel.vjp(values, cotangent),
        solve_vjp(values, cotangent),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    with pytest.raises(ValueError):
        kernel.gradient(values)
    with pytest.raises(ValueError, match="nonsingular"):
        compiler_mlir.compile_matrix_2x2_solve_ad_to_native_llvm_jit(
            rule,
            sample_values=np.array([1.0, 2.0, 2.0, 4.0, 1.0, -1.0], dtype=np.float64),
            config=config,
        )

    registry = CustomDerivativeRegistry()
    registry.register_transform(
        PrimitiveTransformRule(
            identity=identity,
            derivative_rule=rule,
            batching_rule=_eager_batching_rule,
            lowering_rule=compiler_mlir.make_matrix_2x2_solve_native_llvm_jit_lowering_rule(
                sample_values=values,
                config=config,
                sample_tangent=tangent,
                sample_cotangent=cotangent,
            ),
            lowering_metadata={
                "mlir": "available: executable scpn_diff MLIR-runtime primitive kernel",
                "mlir_op": "scpn_diff.native_matrix_2x2_solve",
                "mlir_runtime_verification": "verified: native LLVM/JIT 2x2 solve JVP",
                "llvm": "available: native LLVM MCJIT 2x2 solve AD kernel",
                "jit": "available: native LLVM MCJIT 2x2 solve AD kernel",
                "native_backend": "native_llvm_jit",
                "native_backend_verification": (
                    "verified: native LLVM MCJIT 2x2 solve value/JVP/VJP"
                ),
                "static_derivative_factory": "native_matrix_2x2_solve_llvm_jit",
                "static_signature": "primitive:solve;dimension:2;layout:row_major",
                "nondifferentiable_boundary": "singular_matrix_2x2_solve",
                "nondifferentiable_boundary_policy": "fail_closed",
            },
            shape_rule=lambda _args: (2,),
            dtype_rule=lambda _args: "float64",
            static_argument_rule=lambda args: args,
            nondifferentiable_policy="nonsingular_matrix_2x2_solve_real_domain",
            effect="pure",
        )
    )
    plan = build_compiler_ad_transform_plan(registry)
    module = compile_compiler_ad_transform_plan_to_mlir(plan)
    registered_kernel = compile_registered_primitive_to_executable(registry, identity, values)

    assert registered_kernel.backend == "native_llvm_jit"
    assert plan.executable_backend == "native_llvm_jit"
    assert module.metadata["executable_backend"] == "native_llvm_jit"
    assert module.resource_counts["native_backend_contracts"] == 1
    assert module.resource_counts["primitive_readiness_native_executable"] == 1
    assert module.metadata["native_backend_contract_primitives"] == [identity.key]
    assert module.metadata["primitive_readiness"][identity.key]["verdict"] == "native_executable"
    assert module.metadata["primitive_hard_gaps"][identity.key] == ["rust_backend_contract"]

    rust_registry = CustomDerivativeRegistry()
    rust_registry.register_transform(
        compiler_mlir.make_matrix_2x2_solve_native_llvm_jit_primitive_transform(
            identity,
            rule,
            sample_values=values,
            config=config,
            sample_tangent=tangent,
            sample_cotangent=cotangent,
        )
    )
    rust_plan = build_compiler_ad_transform_plan(rust_registry)
    rust_module = compile_compiler_ad_transform_plan_to_mlir(rust_plan)
    rust_registered_kernel = compile_registered_primitive_to_executable(
        rust_registry,
        identity,
        values,
    )

    assert (
        scpn.make_matrix_2x2_solve_native_llvm_jit_primitive_transform
        is compiler_mlir.make_matrix_2x2_solve_native_llvm_jit_primitive_transform
    )
    assert rust_registered_kernel.backend == "native_llvm_jit"
    assert rust_module.resource_counts["native_backend_contracts"] == 1
    assert rust_module.resource_counts["rust_backend_contracts"] == 1
    assert rust_module.resource_counts["primitive_readiness_native_executable"] == 1
    assert rust_module.resource_counts["primitive_hard_gaps"] == 0
    assert rust_module.metadata["primitive_hard_gaps"][identity.key] == []
    assert rust_module.metadata["rust_backend_contract_primitives"] == [identity.key]
    assert rust_module.metadata["rust_backend_signatures"] == {
        identity.key: "primitive:solve;dimension:2;layout:row_major"
    }
    assert rust_module.metadata["rust_backend_functions"] == {
        identity.key: (
            "matrix_2x2_solve_value,matrix_2x2_solve_jvp,"
            "matrix_2x2_solve_vjp,matrix_2x2_solve_sum_gradient"
        )
    }
    assert rust_module.metadata["rust_backend_verification_primitives"] == {
        identity.key: (
            "verified: scpn_quantum_engine matrix_2x2_solve value/JVP/VJP/sum-gradient parity"
        )
    }


def test_native_llvm_jit_symmetric_2x2_cholesky_kernel_executes_and_marks_plan_native() -> None:
    """Native LLVM/JIT Cholesky AD kernels should execute exact SPD factorisation AD."""

    def cholesky(values: FloatArray) -> FloatArray:
        a00, a01, a11 = values
        l00 = np.sqrt(a00)
        l10 = a01 / l00
        l11 = np.sqrt(a11 - l10 * l10)
        return np.array([l00, l10, l11], dtype=np.float64)

    def cholesky_jvp(values: FloatArray, tangent: FloatArray) -> FloatArray:
        a00, a01, a11 = values
        t00, t01, t11 = tangent
        l00 = np.sqrt(a00)
        l10 = a01 / l00
        l11 = np.sqrt(a11 - l10 * l10)
        tangent_l00 = t00 / (2.0 * l00)
        tangent_l10 = t01 / l00 - l10 * tangent_l00 / l00
        tangent_l11 = (t11 - 2.0 * l10 * tangent_l10) / (2.0 * l11)
        return np.array([tangent_l00, tangent_l10, tangent_l11], dtype=np.float64)

    def cholesky_vjp(values: FloatArray, cotangent: FloatArray) -> FloatArray:
        a00, a01, a11 = values
        cotangent_l00, cotangent_l10, cotangent_l11 = cotangent
        l00 = np.sqrt(a00)
        l10 = a01 / l00
        l11 = np.sqrt(a11 - l10 * l10)
        adjoint_schur = cotangent_l11 / (2.0 * l11)
        adjoint_l10 = cotangent_l10 - 2.0 * l10 * adjoint_schur
        adjoint_l00 = cotangent_l00 - adjoint_l10 * a01 / (l00 * l00)
        return np.array(
            [
                adjoint_l00 / (2.0 * l00),
                adjoint_l10 / l00,
                adjoint_schur,
            ],
            dtype=np.float64,
        )

    identity = PrimitiveIdentity("scpn.compiler_ad.native", "symmetric_2x2_cholesky", "1")
    rule = CustomDerivativeRule(
        name="native_symmetric_2x2_cholesky_rule",
        value_fn=cholesky,
        jvp_rule=cholesky_jvp,
        vjp_rule=cholesky_vjp,
        parameter_names=("a00", "a01", "a11"),
        trainable=(True, True, True),
    )
    config = CompilerADExecutableConfig(backend="native_llvm_jit")
    values = np.array([4.0, 1.0, 3.0], dtype=np.float64)
    tangent = np.array([0.2, -0.3, 0.4], dtype=np.float64)
    cotangent = np.array([1.25, -0.75, 0.5], dtype=np.float64)

    kernel = compiler_mlir.compile_symmetric_2x2_cholesky_ad_to_native_llvm_jit(
        rule,
        sample_values=values,
        config=config,
        sample_tangent=tangent,
        sample_cotangent=cotangent,
    )

    assert kernel.backend == "native_llvm_jit"
    assert kernel.verification.passed is True
    assert kernel.verification.value_close is True
    assert kernel.verification.jvp_close is True
    assert kernel.verification.vjp_close is True
    assert "verified native LLVM MCJIT SPD symmetric 2x2 Cholesky" in kernel.claim_boundary
    assert "non-SPD matrices remain fail-closed" in kernel.claim_boundary
    assert kernel.llvm_gradient_ir is not None
    assert "define void @native_symmetric_2x2_cholesky_rule_value" in kernel.llvm_gradient_ir
    assert "define void @native_symmetric_2x2_cholesky_rule_jvp" in kernel.llvm_gradient_ir
    assert "define void @native_symmetric_2x2_cholesky_rule_vjp" in kernel.llvm_gradient_ir
    assert "define void @native_symmetric_2x2_cholesky_rule_gradient" in kernel.llvm_gradient_ir
    np.testing.assert_allclose(
        kernel.value(values),
        cholesky(values),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        kernel.jvp(values, tangent),
        cholesky_jvp(values, tangent),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        kernel.vjp(values, cotangent),
        cholesky_vjp(values, cotangent),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    with pytest.raises(ValueError):
        kernel.gradient(values)
    with pytest.raises(ValueError, match="positive definite"):
        compiler_mlir.compile_symmetric_2x2_cholesky_ad_to_native_llvm_jit(
            rule,
            sample_values=np.array([1.0, 2.0, 1.0], dtype=np.float64),
            config=config,
        )

    registry = CustomDerivativeRegistry()
    registry.register_transform(
        PrimitiveTransformRule(
            identity=identity,
            derivative_rule=rule,
            batching_rule=_eager_batching_rule,
            lowering_rule=compiler_mlir.make_symmetric_2x2_cholesky_native_llvm_jit_lowering_rule(
                sample_values=values,
                config=config,
                sample_tangent=tangent,
                sample_cotangent=cotangent,
            ),
            lowering_metadata={
                "mlir": "available: executable scpn_diff MLIR-runtime primitive kernel",
                "mlir_op": "scpn_diff.native_symmetric_2x2_cholesky",
                "mlir_runtime_verification": (
                    "verified: native LLVM/JIT SPD symmetric 2x2 Cholesky JVP"
                ),
                "llvm": "available: native LLVM MCJIT SPD symmetric 2x2 Cholesky AD kernel",
                "jit": "available: native LLVM MCJIT SPD symmetric 2x2 Cholesky AD kernel",
                "native_backend": "native_llvm_jit",
                "native_backend_verification": (
                    "verified: native LLVM MCJIT SPD symmetric 2x2 Cholesky value/JVP/VJP"
                ),
                "static_derivative_factory": "native_symmetric_2x2_cholesky_llvm_jit",
                "static_signature": "primitive:cholesky;dimension:2;layout:upper_triangle",
                "nondifferentiable_boundary": "non_spd_symmetric_2x2_cholesky",
                "nondifferentiable_boundary_policy": "fail_closed",
            },
            shape_rule=lambda _args: (3,),
            dtype_rule=lambda _args: "float64",
            static_argument_rule=lambda args: args,
            nondifferentiable_policy="symmetric_2x2_spd_cholesky_real_domain",
            effect="pure",
        )
    )
    plan = build_compiler_ad_transform_plan(registry)
    module = compile_compiler_ad_transform_plan_to_mlir(plan)
    registered_kernel = compile_registered_primitive_to_executable(registry, identity, values)

    assert registered_kernel.backend == "native_llvm_jit"
    assert plan.executable_backend == "native_llvm_jit"
    assert module.metadata["executable_backend"] == "native_llvm_jit"
    assert module.resource_counts["native_backend_contracts"] == 1
    assert module.resource_counts["primitive_readiness_native_executable"] == 1
    assert module.metadata["native_backend_contract_primitives"] == [identity.key]
    assert module.metadata["primitive_readiness"][identity.key]["verdict"] == "native_executable"
    assert module.metadata["primitive_hard_gaps"][identity.key] == ["rust_backend_contract"]

    rust_registry = CustomDerivativeRegistry()
    rust_registry.register_transform(
        compiler_mlir.make_symmetric_2x2_cholesky_native_llvm_jit_primitive_transform(
            identity,
            rule,
            sample_values=values,
            config=config,
            sample_tangent=tangent,
            sample_cotangent=cotangent,
        )
    )
    rust_plan = build_compiler_ad_transform_plan(rust_registry)
    rust_module = compile_compiler_ad_transform_plan_to_mlir(rust_plan)
    rust_registered_kernel = compile_registered_primitive_to_executable(
        rust_registry,
        identity,
        values,
    )

    assert (
        scpn.make_symmetric_2x2_cholesky_native_llvm_jit_primitive_transform
        is compiler_mlir.make_symmetric_2x2_cholesky_native_llvm_jit_primitive_transform
    )
    assert rust_registered_kernel.backend == "native_llvm_jit"
    assert rust_module.resource_counts["native_backend_contracts"] == 1
    assert rust_module.resource_counts["rust_backend_contracts"] == 1
    assert rust_module.resource_counts["primitive_readiness_native_executable"] == 1
    assert rust_module.resource_counts["primitive_hard_gaps"] == 0
    assert rust_module.metadata["primitive_hard_gaps"][identity.key] == []
    assert rust_module.metadata["rust_backend_contract_primitives"] == [identity.key]
    assert rust_module.metadata["rust_backend_signatures"] == {
        identity.key: "primitive:cholesky;dimension:2;layout:upper_triangle"
    }
    assert rust_module.metadata["rust_backend_functions"] == {
        identity.key: (
            "symmetric_2x2_cholesky_value,symmetric_2x2_cholesky_jvp,"
            "symmetric_2x2_cholesky_vjp,symmetric_2x2_cholesky_sum_gradient"
        )
    }
    assert rust_module.metadata["rust_backend_verification_primitives"] == {
        identity.key: (
            "verified: scpn_quantum_engine symmetric_2x2_cholesky "
            "value/JVP/VJP/sum-gradient parity"
        )
    }


def test_native_llvm_jit_symmetric_2x2_eigenvalues_kernel_executes_and_marks_plan_native() -> None:
    """Native LLVM/JIT eigenspectrum AD kernels should execute distinct symmetric 2x2 AD."""

    def eigenvalues(values: FloatArray) -> FloatArray:
        a00, a01, a11 = values
        centre = 0.5 * (a00 + a11)
        half_delta = 0.5 * (a00 - a11)
        radius = np.sqrt(half_delta * half_delta + a01 * a01)
        return np.array([centre - radius, centre + radius], dtype=np.float64)

    def eigenvalues_jvp(values: FloatArray, tangent: FloatArray) -> FloatArray:
        a00, a01, a11 = values
        t00, t01, t11 = tangent
        half_delta = 0.5 * (a00 - a11)
        tangent_half_delta = 0.5 * (t00 - t11)
        tangent_centre = 0.5 * (t00 + t11)
        radius = np.sqrt(half_delta * half_delta + a01 * a01)
        tangent_radius = (half_delta * tangent_half_delta + a01 * t01) / radius
        return np.array(
            [tangent_centre - tangent_radius, tangent_centre + tangent_radius],
            dtype=np.float64,
        )

    def eigenvalues_vjp(values: FloatArray, cotangent: FloatArray) -> FloatArray:
        a00, a01, a11 = values
        lower_cotangent, upper_cotangent = cotangent
        half_delta = 0.5 * (a00 - a11)
        radius = np.sqrt(half_delta * half_delta + a01 * a01)
        half_term = half_delta / (2.0 * radius)
        offdiag_term = a01 / radius
        return np.array(
            [
                lower_cotangent * (0.5 - half_term) + upper_cotangent * (0.5 + half_term),
                (upper_cotangent - lower_cotangent) * offdiag_term,
                lower_cotangent * (0.5 + half_term) + upper_cotangent * (0.5 - half_term),
            ],
            dtype=np.float64,
        )

    identity = PrimitiveIdentity("scpn.compiler_ad.native", "symmetric_2x2_eigenvalues", "1")
    rule = CustomDerivativeRule(
        name="native_symmetric_2x2_eigenvalues_rule",
        value_fn=eigenvalues,
        jvp_rule=eigenvalues_jvp,
        vjp_rule=eigenvalues_vjp,
        parameter_names=("a00", "a01", "a11"),
        trainable=(True, True, True),
    )
    config = CompilerADExecutableConfig(backend="native_llvm_jit")
    values = np.array([2.0, 0.5, 3.0], dtype=np.float64)
    tangent = np.array([0.1, -0.2, 0.4], dtype=np.float64)
    cotangent = np.array([1.25, -0.75], dtype=np.float64)

    kernel = compiler_mlir.compile_symmetric_2x2_eigenvalues_ad_to_native_llvm_jit(
        rule,
        sample_values=values,
        config=config,
        sample_tangent=tangent,
        sample_cotangent=cotangent,
    )

    assert kernel.backend == "native_llvm_jit"
    assert kernel.verification.passed is True
    assert kernel.verification.value_close is True
    assert kernel.verification.jvp_close is True
    assert kernel.verification.vjp_close is True
    assert "verified native LLVM MCJIT distinct symmetric 2x2 eigenvalues" in kernel.claim_boundary
    assert "repeated eigenvalues remain fail-closed" in kernel.claim_boundary
    assert kernel.llvm_gradient_ir is not None
    assert "define void @native_symmetric_2x2_eigenvalues_rule_value" in kernel.llvm_gradient_ir
    assert "define void @native_symmetric_2x2_eigenvalues_rule_jvp" in kernel.llvm_gradient_ir
    assert "define void @native_symmetric_2x2_eigenvalues_rule_vjp" in kernel.llvm_gradient_ir
    assert "define void @native_symmetric_2x2_eigenvalues_rule_gradient" in kernel.llvm_gradient_ir
    np.testing.assert_allclose(
        kernel.value(values),
        eigenvalues(values),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        kernel.jvp(values, tangent),
        eigenvalues_jvp(values, tangent),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        kernel.vjp(values, cotangent),
        eigenvalues_vjp(values, cotangent),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    with pytest.raises(ValueError):
        kernel.gradient(values)
    with pytest.raises(ValueError, match="distinct"):
        compiler_mlir.compile_symmetric_2x2_eigenvalues_ad_to_native_llvm_jit(
            rule,
            sample_values=np.array([1.0, 0.0, 1.0], dtype=np.float64),
            config=config,
        )

    registry = CustomDerivativeRegistry()
    registry.register_transform(
        PrimitiveTransformRule(
            identity=identity,
            derivative_rule=rule,
            batching_rule=_eager_batching_rule,
            lowering_rule=compiler_mlir.make_symmetric_2x2_eigenvalues_native_llvm_jit_lowering_rule(
                sample_values=values,
                config=config,
                sample_tangent=tangent,
                sample_cotangent=cotangent,
            ),
            lowering_metadata={
                "mlir": "available: executable scpn_diff MLIR-runtime primitive kernel",
                "mlir_op": "scpn_diff.native_symmetric_2x2_eigenvalues",
                "mlir_runtime_verification": (
                    "verified: native LLVM/JIT distinct symmetric 2x2 eigenvalue JVP"
                ),
                "llvm": "available: native LLVM MCJIT distinct symmetric 2x2 eigenvalue AD kernel",
                "jit": "available: native LLVM MCJIT distinct symmetric 2x2 eigenvalue AD kernel",
                "native_backend": "native_llvm_jit",
                "native_backend_verification": (
                    "verified: native LLVM MCJIT distinct symmetric 2x2 eigenvalue value/JVP/VJP"
                ),
                "static_derivative_factory": "native_symmetric_2x2_eigenvalues_llvm_jit",
                "static_signature": "primitive:eigvalsh;dimension:2;layout:upper_triangle",
                "nondifferentiable_boundary": "repeated_symmetric_2x2_eigenvalue",
                "nondifferentiable_boundary_policy": "fail_closed",
            },
            shape_rule=lambda _args: (2,),
            dtype_rule=lambda _args: "float64",
            static_argument_rule=lambda args: args,
            nondifferentiable_policy="distinct_symmetric_2x2_eigenvalues_real_domain",
            effect="pure",
        )
    )
    plan = build_compiler_ad_transform_plan(registry)
    module = compile_compiler_ad_transform_plan_to_mlir(plan)
    registered_kernel = compile_registered_primitive_to_executable(registry, identity, values)

    assert registered_kernel.backend == "native_llvm_jit"
    assert plan.executable_backend == "native_llvm_jit"
    assert module.metadata["executable_backend"] == "native_llvm_jit"
    assert module.resource_counts["native_backend_contracts"] == 1
    assert module.resource_counts["primitive_readiness_native_executable"] == 1
    assert module.metadata["primitive_hard_gaps"][identity.key] == ["rust_backend_contract"]

    rust_registry = CustomDerivativeRegistry()
    rust_registry.register_transform(
        compiler_mlir.make_symmetric_2x2_eigenvalues_native_llvm_jit_primitive_transform(
            identity,
            rule,
            sample_values=values,
            config=config,
            sample_tangent=tangent,
            sample_cotangent=cotangent,
        )
    )
    rust_plan = build_compiler_ad_transform_plan(rust_registry)
    rust_module = compile_compiler_ad_transform_plan_to_mlir(rust_plan)
    rust_registered_kernel = compile_registered_primitive_to_executable(
        rust_registry,
        identity,
        values,
    )

    assert (
        scpn.make_symmetric_2x2_eigenvalues_native_llvm_jit_primitive_transform
        is compiler_mlir.make_symmetric_2x2_eigenvalues_native_llvm_jit_primitive_transform
    )
    assert rust_registered_kernel.backend == "native_llvm_jit"
    assert rust_module.resource_counts["native_backend_contracts"] == 1
    assert rust_module.resource_counts["rust_backend_contracts"] == 1
    assert rust_module.resource_counts["primitive_readiness_native_executable"] == 1
    assert rust_module.resource_counts["primitive_hard_gaps"] == 0
    assert rust_module.metadata["primitive_hard_gaps"][identity.key] == []
    assert rust_module.metadata["rust_backend_contract_primitives"] == [identity.key]
    assert rust_module.metadata["rust_backend_signatures"] == {
        identity.key: "primitive:eigvalsh;dimension:2;layout:upper_triangle"
    }
    assert rust_module.metadata["rust_backend_functions"] == {
        identity.key: (
            "symmetric_2x2_eigenvalues_value,symmetric_2x2_eigenvalues_jvp,"
            "symmetric_2x2_eigenvalues_vjp,symmetric_2x2_eigenvalues_sum_gradient"
        )
    }
    assert rust_module.metadata["rust_backend_verification_primitives"] == {
        identity.key: (
            "verified: scpn_quantum_engine symmetric_2x2_eigenvalues "
            "value/JVP/VJP/sum-gradient parity"
        )
    }


def test_native_llvm_jit_matrix_2x2_eigenvalues_kernel_executes_and_marks_plan_native() -> None:
    """Native LLVM/JIT eigenspectrum AD should cover real-simple nonsymmetric 2x2 matrices."""

    def eigenvalues(values: FloatArray) -> FloatArray:
        a00, a01, a10, a11 = values
        trace = a00 + a11
        discriminant = (a00 - a11) * (a00 - a11) + 4.0 * a01 * a10
        root = np.sqrt(discriminant)
        return np.array([0.5 * (trace - root), 0.5 * (trace + root)], dtype=np.float64)

    def eigenvalues_jvp(values: FloatArray, tangent: FloatArray) -> FloatArray:
        a00, a01, a10, a11 = values
        t00, t01, t10, t11 = tangent
        delta = a00 - a11
        trace_tangent = t00 + t11
        discriminant = delta * delta + 4.0 * a01 * a10
        root = np.sqrt(discriminant)
        discriminant_tangent = 2.0 * delta * (t00 - t11) + 4.0 * (t01 * a10 + a01 * t10)
        root_tangent = discriminant_tangent / (2.0 * root)
        return np.array(
            [0.5 * (trace_tangent - root_tangent), 0.5 * (trace_tangent + root_tangent)],
            dtype=np.float64,
        )

    def eigenvalues_vjp(values: FloatArray, cotangent: FloatArray) -> FloatArray:
        a00, a01, a10, a11 = values
        lower_cotangent, upper_cotangent = cotangent
        delta = a00 - a11
        discriminant = delta * delta + 4.0 * a01 * a10
        root = np.sqrt(discriminant)
        alpha = 0.5 * (lower_cotangent + upper_cotangent)
        beta = (upper_cotangent - lower_cotangent) / (4.0 * root)
        delta_term = 2.0 * delta * beta
        return np.array(
            [
                alpha + delta_term,
                4.0 * a10 * beta,
                4.0 * a01 * beta,
                alpha - delta_term,
            ],
            dtype=np.float64,
        )

    identity = PrimitiveIdentity("scpn.compiler_ad.native", "matrix_2x2_eigenvalues", "1")
    rule = CustomDerivativeRule(
        name="native_matrix_2x2_eigenvalues_rule",
        value_fn=eigenvalues,
        jvp_rule=eigenvalues_jvp,
        vjp_rule=eigenvalues_vjp,
        parameter_names=("a00", "a01", "a10", "a11"),
        trainable=(True, True, True, True),
    )
    config = CompilerADExecutableConfig(backend="native_llvm_jit")
    values = np.array([2.0, 0.25, 0.75, 1.0], dtype=np.float64)
    tangent = np.array([0.1, -0.2, 0.4, -0.3], dtype=np.float64)
    cotangent = np.array([1.25, -0.75], dtype=np.float64)

    kernel = compiler_mlir.compile_matrix_2x2_eigenvalues_ad_to_native_llvm_jit(
        rule,
        sample_values=values,
        config=config,
        sample_tangent=tangent,
        sample_cotangent=cotangent,
    )

    assert kernel.backend == "native_llvm_jit"
    assert kernel.verification.passed is True
    assert kernel.verification.value_close is True
    assert kernel.verification.jvp_close is True
    assert kernel.verification.vjp_close is True
    assert "verified native LLVM MCJIT real-simple nonsymmetric 2x2 eigenvalues" in (
        kernel.claim_boundary
    )
    assert "complex or repeated eigenvalues remain fail-closed" in kernel.claim_boundary
    assert kernel.llvm_gradient_ir is not None
    assert "define void @native_matrix_2x2_eigenvalues_rule_value" in kernel.llvm_gradient_ir
    assert "define void @native_matrix_2x2_eigenvalues_rule_jvp" in kernel.llvm_gradient_ir
    assert "define void @native_matrix_2x2_eigenvalues_rule_vjp" in kernel.llvm_gradient_ir
    assert "define void @native_matrix_2x2_eigenvalues_rule_gradient" in kernel.llvm_gradient_ir
    np.testing.assert_allclose(
        kernel.value(values), eigenvalues(values), rtol=1.0e-12, atol=1.0e-12
    )
    np.testing.assert_allclose(
        kernel.jvp(values, tangent),
        eigenvalues_jvp(values, tangent),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        kernel.vjp(values, cotangent),
        eigenvalues_vjp(values, cotangent),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    with pytest.raises(ValueError):
        kernel.gradient(values)
    with pytest.raises(ValueError, match="real distinct eigenvalues"):
        compiler_mlir.compile_matrix_2x2_eigenvalues_ad_to_native_llvm_jit(
            rule,
            sample_values=np.array([0.0, -1.0, 1.0, 0.0], dtype=np.float64),
            config=config,
        )
    with pytest.raises(ValueError, match="real distinct eigenvalues"):
        compiler_mlir.compile_matrix_2x2_eigenvalues_ad_to_native_llvm_jit(
            rule,
            sample_values=np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float64),
            config=config,
        )

    registry = CustomDerivativeRegistry()
    registry.register_transform(
        PrimitiveTransformRule(
            identity=identity,
            derivative_rule=rule,
            batching_rule=_eager_batching_rule,
            lowering_rule=compiler_mlir.make_matrix_2x2_eigenvalues_native_llvm_jit_lowering_rule(
                sample_values=values,
                config=config,
                sample_tangent=tangent,
                sample_cotangent=cotangent,
            ),
            lowering_metadata={
                "mlir": "available: executable scpn_diff MLIR-runtime primitive kernel",
                "mlir_op": "scpn_diff.native_matrix_2x2_eigenvalues",
                "mlir_runtime_verification": (
                    "verified: native LLVM/JIT real-simple nonsymmetric 2x2 eigenvalue JVP"
                ),
                "llvm": "available: native LLVM MCJIT real-simple nonsymmetric 2x2 eigenvalue AD kernel",
                "jit": "available: native LLVM MCJIT real-simple nonsymmetric 2x2 eigenvalue AD kernel",
                "native_backend": "native_llvm_jit",
                "native_backend_verification": (
                    "verified: native LLVM MCJIT real-simple nonsymmetric 2x2 eigenvalue value/JVP/VJP"
                ),
                "static_derivative_factory": "native_matrix_2x2_eigenvalues_llvm_jit",
                "static_signature": "primitive:eigvals;dimension:2;layout:row_major;domain:real_simple",
                "nondifferentiable_boundary": "nonreal_or_repeated_matrix_2x2_eigenvalue",
                "nondifferentiable_boundary_policy": "fail_closed",
            },
            shape_rule=lambda _args: (2,),
            dtype_rule=lambda _args: "float64",
            static_argument_rule=lambda args: args,
            nondifferentiable_policy="real_simple_matrix_2x2_eigenvalues_domain",
            effect="pure",
        )
    )
    plan = build_compiler_ad_transform_plan(registry)
    module = compile_compiler_ad_transform_plan_to_mlir(plan)
    registered_kernel = compile_registered_primitive_to_executable(registry, identity, values)

    assert registered_kernel.backend == "native_llvm_jit"
    assert plan.executable_backend == "native_llvm_jit"
    assert module.metadata["executable_backend"] == "native_llvm_jit"
    assert module.resource_counts["native_backend_contracts"] == 1
    assert module.resource_counts["primitive_readiness_native_executable"] == 1
    assert scpn.compile_matrix_2x2_eigenvalues_ad_to_native_llvm_jit is (
        compiler_mlir.compile_matrix_2x2_eigenvalues_ad_to_native_llvm_jit
    )
    assert scpn.make_matrix_2x2_eigenvalues_native_llvm_jit_lowering_rule is (
        compiler_mlir.make_matrix_2x2_eigenvalues_native_llvm_jit_lowering_rule
    )
    assert module.metadata["native_backend_contract_primitives"] == [identity.key]
    assert module.metadata["primitive_readiness"][identity.key]["verdict"] == "native_executable"
    assert module.metadata["primitive_hard_gaps"][identity.key] == ["rust_backend_contract"]

    rust_registry = CustomDerivativeRegistry()
    rust_registry.register_transform(
        compiler_mlir.make_matrix_2x2_eigenvalues_native_llvm_jit_primitive_transform(
            identity,
            rule,
            sample_values=values,
            config=config,
            sample_tangent=tangent,
            sample_cotangent=cotangent,
        )
    )
    rust_plan = build_compiler_ad_transform_plan(rust_registry)
    rust_module = compile_compiler_ad_transform_plan_to_mlir(rust_plan)
    rust_registered_kernel = compile_registered_primitive_to_executable(
        rust_registry,
        identity,
        values,
    )

    assert rust_registered_kernel.backend == "native_llvm_jit"
    assert rust_plan.executable_backend == "native_llvm_jit"
    assert rust_module.metadata["executable_backend"] == "native_llvm_jit"
    assert rust_module.resource_counts["rust_backend_contracts"] == 1
    assert rust_module.resource_counts["rust_backend_incomplete_primitives"] == 0
    assert rust_module.resource_counts["rust_backend_blockers"] == 0
    assert rust_module.resource_counts["primitive_hard_gaps"] == 0
    assert rust_module.metadata["rust_backend_contract_primitives"] == [identity.key]
    assert rust_module.metadata["rust_backend_incomplete_primitives"] == []
    assert rust_module.metadata["rust_backend_blockers"] == {}
    assert rust_module.metadata["rust_backend_signatures"] == {
        identity.key: "primitive:eigvals;dimension:2;layout:row_major;domain:real_simple"
    }
    assert rust_module.metadata["rust_backend_functions"] == {
        identity.key: (
            "matrix_2x2_eigenvalues_value,matrix_2x2_eigenvalues_jvp,"
            "matrix_2x2_eigenvalues_vjp,matrix_2x2_eigenvalues_sum_gradient"
        )
    }
    assert rust_module.metadata["rust_backend_verification_primitives"] == {
        identity.key: (
            "verified: scpn_quantum_engine matrix_2x2_eigenvalues "
            "value/JVP/VJP/sum-gradient parity"
        )
    }
    assert rust_module.resource_counts["rust_backend_verifications"] == 1
    assert rust_module.metadata["primitive_hard_gaps"][identity.key] == []
    assert identity.key not in rust_module.metadata["primitive_next_hard_gap"]
    assert rust_module.metadata["primitive_readiness"][identity.key] == {
        "adjoint_contract": True,
        "forward_contract": True,
        "jit_backend_contract": True,
        "llvm_backend_contract": True,
        "mlir_runtime_contract": True,
        "native_backend_contract": True,
        "registry_contract": True,
        "reverse_contract": True,
        "rust_backend_contract": True,
        "transform_contract": True,
        "verdict": "native_executable",
    }
    batched_value = cast(
        FloatArray,
        vmap(
            lambda row: row,
            primitive_identity=identity,
            registry=rust_registry,
        )(np.vstack([values, values + np.array([0.25, 0.1, 0.05, -0.2], dtype=np.float64)])),
    )
    np.testing.assert_allclose(
        batched_value[0],
        rust_registered_kernel.value(values),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    assert scpn.make_matrix_2x2_eigenvalues_native_llvm_jit_primitive_transform is (
        compiler_mlir.make_matrix_2x2_eigenvalues_native_llvm_jit_primitive_transform
    )


def test_native_llvm_jit_matrix_2x2_eigensystem_kernel_executes_and_marks_plan_native() -> None:
    """Bounded 2x2 nonsymmetric eigensystem AD must execute as native LLVM/JIT."""

    def eigensystem(values: FloatArray) -> FloatArray:
        a00, a01, a10, a11 = values
        trace = a00 + a11
        delta = a00 - a11
        discriminant = delta * delta + 4.0 * a01 * a10
        root = float(np.sqrt(discriminant))
        lower = 0.5 * (trace - root)
        upper = 0.5 * (trace + root)
        q_lower = 0.5 * (-delta - root)
        q_upper = 0.5 * (-delta + root)
        lower_vec = np.array([a01, q_lower], dtype=np.float64)
        upper_vec = np.array([a01, q_upper], dtype=np.float64)
        lower_vec = lower_vec / np.linalg.norm(lower_vec)
        upper_vec = upper_vec / np.linalg.norm(upper_vec)
        return np.array(
            [lower, upper, lower_vec[0], upper_vec[0], lower_vec[1], upper_vec[1]],
            dtype=np.float64,
        )

    def eigensystem_jvp(values: FloatArray, tangent: FloatArray) -> FloatArray:
        a00, a01, a10, a11 = values
        t00, t01, t10, t11 = tangent
        trace_t = t00 + t11
        delta = a00 - a11
        delta_t = t00 - t11
        discriminant = delta * delta + 4.0 * a01 * a10
        root = float(np.sqrt(discriminant))
        discriminant_t = 2.0 * delta * delta_t + 4.0 * (t01 * a10 + a01 * t10)
        root_t = discriminant_t / (2.0 * root)
        lower_t = 0.5 * (trace_t - root_t)
        upper_t = 0.5 * (trace_t + root_t)
        q_lower = 0.5 * (-(a00 - a11) - root)
        q_upper = 0.5 * (-(a00 - a11) + root)
        q_lower_t = lower_t - t00
        q_upper_t = upper_t - t00

        def vector_tangent(q_value: float, q_tangent: float) -> FloatArray:
            raw = np.array([a01, q_value], dtype=np.float64)
            raw_tangent = np.array([t01, q_tangent], dtype=np.float64)
            norm = np.linalg.norm(raw)
            vector = raw / norm
            return (raw_tangent - vector * float(vector @ raw_tangent)) / norm

        lower_vec_t = vector_tangent(float(q_lower), float(q_lower_t))
        upper_vec_t = vector_tangent(float(q_upper), float(q_upper_t))
        return np.array(
            [
                lower_t,
                upper_t,
                lower_vec_t[0],
                upper_vec_t[0],
                lower_vec_t[1],
                upper_vec_t[1],
            ],
            dtype=np.float64,
        )

    def eigensystem_vjp(values: FloatArray, cotangent: FloatArray) -> FloatArray:
        adjoint = np.zeros(4, dtype=np.float64)
        for index in range(4):
            basis = np.zeros(4, dtype=np.float64)
            basis[index] = 1.0
            adjoint[index] = float(cotangent @ eigensystem_jvp(values, basis))
        return adjoint

    import scpn_quantum_control as scpn

    identity = PrimitiveIdentity(
        "scpn.compiler_ad.native",
        "matrix_2x2_eigensystem",
        "1",
    )
    rule = CustomDerivativeRule(
        "native_matrix_2x2_eigensystem_rule",
        eigensystem,
        eigensystem_jvp,
        eigensystem_vjp,
        parameter_names=("a00", "a01", "a10", "a11"),
        trainable=(True, True, True, True),
    )
    values = np.array([2.0, 0.25, 0.75, 1.0], dtype=np.float64)
    tangent = np.array([0.1, -0.2, 0.4, -0.3], dtype=np.float64)
    cotangent = np.array([1.25, -0.75, 0.5, -0.25, 0.3, -0.6], dtype=np.float64)
    config = CompilerADExecutableConfig(backend="native_llvm_jit")

    kernel = compiler_mlir.compile_matrix_2x2_eigensystem_ad_to_native_llvm_jit(
        rule,
        sample_values=values,
        config=config,
        sample_tangent=tangent,
        sample_cotangent=cotangent,
    )

    assert kernel.backend == "native_llvm_jit"
    assert kernel.verification.value_close is True
    assert kernel.verification.jvp_close is True
    assert kernel.verification.vjp_close is True
    assert kernel.verification.gradient_close is None
    assert kernel.claim_boundary == (
        "verified native LLVM MCJIT real-simple nonsymmetric 2x2 eigensystem value/JVP/VJP "
        "kernel with sum-output gradient provenance; complex spectra, repeated eigenvalues, "
        "and zero upper off-diagonal eigenvector charts remain fail-closed"
    )
    assert "matrix_2x2_eigensystem" in (kernel.llvm_gradient_ir or "")
    np.testing.assert_allclose(kernel.value(values), eigensystem(values), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        kernel.jvp(values, tangent),
        eigensystem_jvp(values, tangent),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        kernel.vjp(values, cotangent),
        eigensystem_vjp(values, cotangent),
        rtol=1e-12,
        atol=1e-12,
    )
    with pytest.raises(ValueError, match="scalar output"):
        kernel.gradient(values)
    with pytest.raises(ValueError, match="real distinct eigenvalues"):
        kernel.value(np.array([0.0, -1.0, 1.0, 0.0], dtype=np.float64))
    with pytest.raises(ValueError, match="real distinct eigenvalues"):
        kernel.value(np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float64))
    with pytest.raises(ValueError, match="upper off-diagonal eigenvector chart"):
        kernel.value(np.array([2.0, 0.0, 1.0, 1.0], dtype=np.float64))

    registry = CustomDerivativeRegistry()
    registry.register_transform(
        PrimitiveTransformRule(
            identity=identity,
            derivative_rule=rule,
            batching_rule=_eager_batching_rule,
            lowering_rule=compiler_mlir.make_matrix_2x2_eigensystem_native_llvm_jit_lowering_rule(
                sample_values=values,
                config=config,
                sample_tangent=tangent,
                sample_cotangent=cotangent,
            ),
            lowering_metadata={
                "mlir": "available: executable scpn_diff MLIR-runtime primitive kernel",
                "mlir_op": "scpn_diff.native_matrix_2x2_eigensystem",
                "mlir_runtime_verification": (
                    "verified: native LLVM/JIT real-simple nonsymmetric 2x2 eigensystem JVP"
                ),
                "llvm": "available: native LLVM MCJIT real-simple nonsymmetric 2x2 eigensystem AD kernel",
                "jit": "available: native LLVM MCJIT real-simple nonsymmetric 2x2 eigensystem AD kernel",
                "native_backend": "native_llvm_jit",
                "native_backend_verification": (
                    "verified: native LLVM MCJIT real-simple nonsymmetric 2x2 eigensystem value/JVP/VJP"
                ),
                "static_derivative_factory": "native_matrix_2x2_eigensystem_llvm_jit",
                "static_signature": (
                    "primitive:eig;dimension:2;layout:row_major;domain:real_simple_upper_chart"
                ),
                "nondifferentiable_boundary": (
                    "nonreal_repeated_or_zero_upper_chart_matrix_2x2_eigensystem"
                ),
                "nondifferentiable_boundary_policy": "fail_closed",
            },
            shape_rule=lambda _args: (6,),
            dtype_rule=lambda _args: "float64",
            static_argument_rule=lambda args: args,
            nondifferentiable_policy="real_simple_upper_chart_matrix_2x2_eigensystem_domain",
            effect="pure",
        )
    )
    plan = build_compiler_ad_transform_plan(registry)
    module = compile_compiler_ad_transform_plan_to_mlir(plan)
    registered_kernel = compile_registered_primitive_to_executable(registry, identity, values)

    assert registered_kernel.backend == "native_llvm_jit"
    assert plan.executable_backend == "native_llvm_jit"
    assert module.metadata["executable_backend"] == "native_llvm_jit"
    assert module.resource_counts["native_backend_contracts"] == 1
    assert module.resource_counts["primitive_readiness_native_executable"] == 1
    assert scpn.compile_matrix_2x2_eigensystem_ad_to_native_llvm_jit is (
        compiler_mlir.compile_matrix_2x2_eigensystem_ad_to_native_llvm_jit
    )
    assert scpn.make_matrix_2x2_eigensystem_native_llvm_jit_lowering_rule is (
        compiler_mlir.make_matrix_2x2_eigensystem_native_llvm_jit_lowering_rule
    )
    assert module.metadata["native_backend_contract_primitives"] == [identity.key]
    assert module.metadata["primitive_readiness"][identity.key]["verdict"] == "native_executable"
    assert module.metadata["primitive_hard_gaps"][identity.key] == ["rust_backend_contract"]

    rust_registry = CustomDerivativeRegistry()
    rust_registry.register_transform(
        compiler_mlir.make_matrix_2x2_eigensystem_native_llvm_jit_primitive_transform(
            identity,
            rule,
            sample_values=values,
            config=config,
            sample_tangent=tangent,
            sample_cotangent=cotangent,
        )
    )
    rust_plan = build_compiler_ad_transform_plan(rust_registry)
    rust_module = compile_compiler_ad_transform_plan_to_mlir(rust_plan)
    rust_registered_kernel = compile_registered_primitive_to_executable(
        rust_registry,
        identity,
        values,
    )

    assert rust_registered_kernel.backend == "native_llvm_jit"
    assert rust_plan.executable_backend == "native_llvm_jit"
    assert rust_module.metadata["executable_backend"] == "native_llvm_jit"
    assert rust_module.resource_counts["rust_backend_contracts"] == 1
    assert rust_module.resource_counts["rust_backend_incomplete_primitives"] == 0
    assert rust_module.resource_counts["rust_backend_blockers"] == 0
    assert rust_module.resource_counts["primitive_hard_gaps"] == 0
    assert rust_module.metadata["rust_backend_contract_primitives"] == [identity.key]
    assert rust_module.metadata["rust_backend_incomplete_primitives"] == []
    assert rust_module.metadata["rust_backend_blockers"] == {}
    assert rust_module.metadata["rust_backend_signatures"] == {
        identity.key: "primitive:eig;dimension:2;layout:row_major;domain:real_simple_upper_chart"
    }
    assert rust_module.metadata["rust_backend_functions"] == {
        identity.key: (
            "matrix_2x2_eigensystem_value,matrix_2x2_eigensystem_jvp,"
            "matrix_2x2_eigensystem_vjp,matrix_2x2_eigensystem_sum_gradient"
        )
    }
    assert rust_module.metadata["rust_backend_verification_primitives"] == {
        identity.key: (
            "verified: scpn_quantum_engine matrix_2x2_eigensystem "
            "value/JVP/VJP/sum-gradient parity"
        )
    }
    assert rust_module.resource_counts["rust_backend_verifications"] == 1
    assert rust_module.metadata["primitive_hard_gaps"][identity.key] == []
    assert identity.key not in rust_module.metadata["primitive_next_hard_gap"]
    assert rust_module.metadata["primitive_readiness"][identity.key] == {
        "adjoint_contract": True,
        "forward_contract": True,
        "jit_backend_contract": True,
        "llvm_backend_contract": True,
        "mlir_runtime_contract": True,
        "native_backend_contract": True,
        "registry_contract": True,
        "reverse_contract": True,
        "rust_backend_contract": True,
        "transform_contract": True,
        "verdict": "native_executable",
    }
    batched_value = cast(
        FloatArray,
        vmap(
            lambda row: row,
            primitive_identity=identity,
            registry=rust_registry,
        )(np.vstack([values, values + np.array([0.25, 0.1, 0.05, -0.2], dtype=np.float64)])),
    )
    np.testing.assert_allclose(
        batched_value[0],
        rust_registered_kernel.value(values),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    assert scpn.make_matrix_2x2_eigensystem_native_llvm_jit_primitive_transform is (
        compiler_mlir.make_matrix_2x2_eigensystem_native_llvm_jit_primitive_transform
    )


def test_differentiable_mlir_rejects_executable_target_claims() -> None:
    """LLVM/JIT target names must fail until backed by a real executable backend."""

    with pytest.raises(ValueError, match="target"):
        DifferentiableMLIRCompileConfig(target="llvm")
    with pytest.raises(ValueError, match="backend"):
        CompilerADExecutableConfig(backend="llvm")
    with pytest.raises(ValueError, match="boolean"):
        DifferentiableMLIRCompileConfig(include_numeric_payload=1)  # type: ignore[arg-type]
