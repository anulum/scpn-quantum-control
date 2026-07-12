# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — MLIR Matrix Native Compilation Integration Tests
"""Integration tests for dimension-generic matrix MLIR native kernels."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest
from _mlir_native_compilation_test_helpers import FloatArray, _eager_batching_rule

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
    vmap,
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
