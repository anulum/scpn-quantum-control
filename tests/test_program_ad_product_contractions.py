# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Program AD product contraction tests
"""Tests for Program AD tensor contraction registry and direct rules."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from scpn_quantum_control.differentiable import (
    Parameter,
    primitive_contract_for,
    program_ad_product_einsum_derivative_rule,
    program_adjoint_gradient,
    whole_program_value_and_grad,
)


def _assert_allclose(
    actual: object, expected: object, *, rtol: float = 1.0e-7, atol: float = 0.0
) -> None:
    """Assert NumPy closeness across dynamically typed Program AD result payloads."""

    cast(Any, np.testing.assert_allclose)(actual, expected, rtol=rtol, atol=atol)


def test_program_ad_einsum_handles_explicit_ranked_tensor_contractions() -> None:
    """Program AD np.einsum should support explicit static tensor contractions."""

    weights = np.array([[1.0, -2.0], [0.5, 3.0]], dtype=np.float64)
    vector = np.array([2.0, -0.25], dtype=np.float64)

    def objective(values: Any) -> object:
        tensor = np.reshape(values, (2, 2, 2))
        contracted = np.einsum("abc,c,ab->", tensor, vector, weights)
        projected = np.einsum("abc,c->ab", tensor, vector)
        diagonal = np.einsum("aab,b->a", tensor, vector)
        return contracted + np.sum(projected * weights) + np.sum(diagonal)

    values = np.array([0.5, -0.25, 1.0, -1.5, 2.0, 0.75, -0.5, 1.25], dtype=np.float64)
    result = whole_program_value_and_grad(
        objective,
        values,
        parameters=tuple(Parameter(f"x{index}") for index in range(values.size)),
    )

    expected = np.zeros((2, 2, 2), dtype=np.float64)
    for a_index in range(2):
        for b_index in range(2):
            for c_index in range(2):
                expected[a_index, b_index, c_index] += (
                    2.0 * weights[a_index, b_index] * vector[c_index]
                )
                if a_index == b_index:
                    expected[a_index, b_index, c_index] += vector[c_index]

    assert result.value == pytest.approx(float(cast(Any, objective(values))))
    _assert_allclose(result.gradient, expected.reshape(-1), rtol=1.0e-12, atol=1.0e-12)
    _assert_allclose(
        program_adjoint_gradient(result), expected.reshape(-1), rtol=1.0e-12, atol=1.0e-12
    )


def test_program_ad_einsum_registry_contract_and_direct_rule() -> None:
    """Program AD einsum should be registry-gated and expose exact fixed-shape rules."""

    subscripts = "abc,c,ab->"
    shapes = ((2, 2, 2), (2,), (2, 2))
    tensor = np.arange(1.0, 9.0, dtype=np.float64).reshape(shapes[0])
    vector = np.array([2.0, -0.25], dtype=np.float64)
    weights = np.array([[1.0, -2.0], [0.5, 3.0]], dtype=np.float64)
    values = np.concatenate([tensor.reshape(-1), vector.reshape(-1), weights.reshape(-1)])
    tangent = np.linspace(-0.4, 0.7, values.size, dtype=np.float64)
    cotangent = np.array([1.25], dtype=np.float64)

    contract = primitive_contract_for("scpn.program_ad.product:einsum")
    shape_rule = cast(Any, contract.shape_rule)
    static_argument_rule = cast(Any, contract.static_argument_rule)
    assert contract.shape_rule is not None
    assert contract.dtype_rule is not None
    assert contract.static_argument_rule is not None
    assert contract.lowering_metadata["static_derivative_factory"] == (
        "program_ad_product_einsum_derivative_rule"
    )
    assert shape_rule((subscripts, tensor, vector, weights)) == ()
    assert static_argument_rule((subscripts, tensor, vector, weights)) == (
        subscripts,
        shapes,
        ("abc", "c", "ab"),
        "",
    )

    rule = program_ad_product_einsum_derivative_rule(subscripts, shapes)
    rule_jvp = rule.jvp_rule
    rule_vjp = rule.vjp_rule
    assert rule_jvp is not None
    assert rule_vjp is not None
    tangent_operands = (
        tangent[: tensor.size].reshape(tensor.shape),
        tangent[tensor.size : tensor.size + vector.size].reshape(vector.shape),
        tangent[tensor.size + vector.size :].reshape(weights.shape),
    )
    expected_jvp = (
        np.einsum(subscripts, tangent_operands[0], vector, weights)
        + np.einsum(subscripts, tensor, tangent_operands[1], weights)
        + np.einsum(subscripts, tensor, vector, tangent_operands[2])
    )
    expected_vjp = np.concatenate(
        [
            (cotangent[0] * vector[None, None, :] * weights[:, :, None]).reshape(-1),
            (cotangent[0] * np.sum(tensor * weights[:, :, None], axis=(0, 1))).reshape(-1),
            (cotangent[0] * np.sum(tensor * vector[None, None, :], axis=2)).reshape(-1),
        ]
    )

    _assert_allclose(rule.value_fn(values), [np.einsum(subscripts, tensor, vector, weights)])
    _assert_allclose(rule_jvp(values, tangent), [expected_jvp])
    _assert_allclose(rule_vjp(values, cotangent), expected_vjp)
