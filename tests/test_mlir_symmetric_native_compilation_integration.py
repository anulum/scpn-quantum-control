# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — MLIR Symmetric Native Compilation Integration Tests
"""Integration tests for symmetric 2x2 MLIR native compiler kernels."""

from __future__ import annotations

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
)


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
