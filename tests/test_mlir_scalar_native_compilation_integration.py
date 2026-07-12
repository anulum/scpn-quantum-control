# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — MLIR Scalar Native Compilation Integration Tests
"""Integration tests for scalar MLIR native compiler kernels."""

from __future__ import annotations

import numpy as np
from _mlir_native_compilation_test_helpers import _eager_batching_rule

import scpn_quantum_control.compiler.mlir as compiler_mlir
from scpn_quantum_control.compiler.mlir import (
    CompilerADExecutableConfig,
    build_compiler_ad_transform_plan,
    compile_compiler_ad_transform_plan_to_mlir,
    compile_registered_primitive_to_executable,
)
from scpn_quantum_control.differentiable import (
    CustomDerivativeRegistry,
    CustomDerivativeRule,
    PrimitiveIdentity,
    PrimitiveTransformRule,
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
