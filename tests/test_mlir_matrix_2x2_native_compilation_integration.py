# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — MLIR Matrix 2x2 Native Compilation Integration Tests
"""Integration tests for general 2x2 matrix MLIR native kernels."""

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
