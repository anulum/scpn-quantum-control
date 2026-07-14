# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — MLIR Vector Native Compilation Integration Tests
"""Integration tests for vector MLIR native compiler kernels."""

from __future__ import annotations

import numpy as np
from _mlir_native_compilation_test_helpers import _eager_batching_rule

import scpn_quantum_control as scpn
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
