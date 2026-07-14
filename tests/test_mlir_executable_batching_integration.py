# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — MLIR Executable Batching Integration Tests
"""Integration tests for native executable value, JVP, and VJP batching."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest
from _mlir_native_compilation_test_helpers import FloatArray

import scpn_quantum_control.compiler.mlir as compiler_mlir
from scpn_quantum_control.compiler.mlir import (
    CompilerADExecutableConfig,
    compile_custom_derivative_rule_to_executable,
)
from scpn_quantum_control.differentiable import (
    CustomDerivativeRegistry,
    CustomDerivativeRule,
    PrimitiveIdentity,
    PrimitiveTransformRule,
    vmap,
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
