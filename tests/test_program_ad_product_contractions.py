# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Program AD product contraction tests
"""Tests for Program AD tensor contraction registry and direct rules."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import numpy as np
import pytest

import scpn_quantum_control.program_ad_product_primitives as product_primitives
from scpn_quantum_control.differentiable import (
    DEFAULT_CUSTOM_DERIVATIVE_REGISTRY,
    Parameter,
    PrimitiveContract,
    PrimitiveIdentity,
    PrimitiveTransformRule,
    custom_derivative_rule_for,
    primitive_complete_contract_for,
    primitive_contract_for,
    program_ad_product_einsum_derivative_rule,
    program_ad_product_inner_derivative_rule,
    program_ad_product_matmul_derivative_rule,
    program_ad_product_outer_derivative_rule,
    program_ad_product_tensordot_derivative_rule,
    program_adjoint_gradient,
    whole_program_value_and_grad,
)


def _assert_allclose(
    actual: object, expected: object, *, rtol: float = 1.0e-7, atol: float = 0.0
) -> None:
    """Assert NumPy closeness across dynamically typed Program AD result payloads."""

    cast(Any, np.testing.assert_allclose)(actual, expected, rtol=rtol, atol=atol)


def _transform_rule_from_contract(contract: PrimitiveContract) -> PrimitiveTransformRule:
    """Return a mutable registry transform that exactly mirrors a contract."""

    return PrimitiveTransformRule(
        identity=contract.identity,
        derivative_rule=contract.derivative_rule,
        batching_rule=contract.batching_rule,
        lowering_rule=contract.lowering_rule,
        lowering_metadata=contract.lowering_metadata,
        shape_rule=contract.shape_rule,
        dtype_rule=contract.dtype_rule,
        static_argument_rule=contract.static_argument_rule,
        nondifferentiable_policy=contract.nondifferentiable_policy,
        effect=contract.effect,
    )


def test_program_ad_product_factories_remain_facade_compatible() -> None:
    """Product factories should re-export the extracted module implementations."""

    assert (
        program_ad_product_einsum_derivative_rule
        is product_primitives.program_ad_product_einsum_derivative_rule
    )
    assert (
        program_ad_product_inner_derivative_rule
        is product_primitives.program_ad_product_inner_derivative_rule
    )
    assert (
        program_ad_product_matmul_derivative_rule
        is product_primitives.program_ad_product_matmul_derivative_rule
    )
    assert (
        program_ad_product_outer_derivative_rule
        is product_primitives.program_ad_product_outer_derivative_rule
    )
    assert (
        program_ad_product_tensordot_derivative_rule
        is product_primitives.program_ad_product_tensordot_derivative_rule
    )


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


def test_program_ad_product_primitives_are_registry_policy_gated() -> None:
    """Dot, vdot, inner, outer, and matmul should expose primitive registry contracts."""

    left_vector = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    right_vector = np.array([4.0, 5.0, 6.0], dtype=np.float64)
    matrix = np.arange(6.0, dtype=np.float64).reshape(2, 3)

    dot_contract = primitive_contract_for("scpn.program_ad.product:dot")
    assert dot_contract.identity == PrimitiveIdentity("scpn.program_ad.product", "dot", "1")
    assert dot_contract.nondifferentiable_policy == "program_ad_trace_exact_fail_closed"
    assert dot_contract.effect == "pure"
    assert dot_contract.lowering_metadata["mlir_op"] == "scpn_diff.product.dot"
    assert dot_contract.lowering_metadata["static_derivative_factory"] == "not_required"
    assert dot_contract.lowering_metadata["static_signature"] == "none"
    assert dot_contract.lowering_metadata["nondifferentiable_boundary"] == (
        "inner_dimension_alignment"
    )
    assert dot_contract.shape_rule is not None
    assert dot_contract.shape_rule((left_vector, right_vector)) == ()
    assert dot_contract.dtype_rule is not None
    assert dot_contract.dtype_rule((left_vector, right_vector)) == "float64"
    assert dot_contract.static_argument_rule is not None
    assert dot_contract.static_argument_rule((left_vector, right_vector)) == ()
    with pytest.raises(ValueError, match="incomplete primitive contract"):
        primitive_complete_contract_for(dot_contract.identity)

    vdot_contract = primitive_contract_for("scpn.program_ad.product:vdot")
    assert vdot_contract.identity == PrimitiveIdentity("scpn.program_ad.product", "vdot", "1")
    assert vdot_contract.lowering_metadata["mlir_op"] == "scpn_diff.product.vdot"
    assert vdot_contract.lowering_metadata["static_derivative_factory"] == "not_required"
    assert vdot_contract.lowering_metadata["static_signature"] == "none"
    assert vdot_contract.lowering_metadata["nondifferentiable_boundary"] == (
        "flattened_size_alignment"
    )
    assert vdot_contract.nondifferentiable_policy == "program_ad_trace_exact_fail_closed"
    assert vdot_contract.effect == "pure"
    assert vdot_contract.shape_rule is not None
    assert vdot_contract.shape_rule((matrix, np.arange(6.0, dtype=np.float64))) == ()
    assert vdot_contract.dtype_rule is not None
    assert vdot_contract.dtype_rule((matrix, np.arange(6.0, dtype=np.float64))) == "float64"
    assert vdot_contract.static_argument_rule is not None
    assert vdot_contract.static_argument_rule((matrix, np.arange(6.0, dtype=np.float64))) == ()
    with pytest.raises(ValueError, match="incomplete primitive contract"):
        primitive_complete_contract_for(vdot_contract.identity)

    inner_contract = primitive_contract_for("scpn.program_ad.product:inner")
    assert inner_contract.identity == PrimitiveIdentity("scpn.program_ad.product", "inner", "1")
    assert inner_contract.nondifferentiable_policy == "program_ad_trace_exact_fail_closed"
    assert inner_contract.effect == "pure"
    assert inner_contract.lowering_metadata["mlir_op"] == "scpn_diff.product.inner"
    assert (
        inner_contract.lowering_metadata["static_derivative_factory"]
        == "program_ad_product_inner_derivative_rule"
    )
    assert inner_contract.lowering_metadata["static_signature"] == (
        "left_shape:ranked_tensor_shape;right_shape:ranked_tensor_shape"
    )
    assert inner_contract.shape_rule is not None
    assert inner_contract.shape_rule((matrix, matrix)) == (2, 2)
    assert inner_contract.dtype_rule is not None
    assert inner_contract.dtype_rule((matrix, matrix)) == "float64"
    assert inner_contract.static_argument_rule is not None
    assert inner_contract.static_argument_rule((matrix, matrix)) == ()
    with pytest.raises(ValueError, match="incomplete primitive contract"):
        primitive_complete_contract_for(inner_contract.identity)

    outer_contract = primitive_contract_for("scpn.program_ad.product:outer")
    assert outer_contract.identity == PrimitiveIdentity("scpn.program_ad.product", "outer", "1")
    assert outer_contract.nondifferentiable_policy == "program_ad_trace_exact_fail_closed"
    assert outer_contract.effect == "pure"
    assert outer_contract.lowering_metadata["mlir_op"] == "scpn_diff.product.outer"
    assert (
        outer_contract.lowering_metadata["static_derivative_factory"]
        == "program_ad_product_outer_derivative_rule"
    )
    assert outer_contract.lowering_metadata["static_signature"] == (
        "left_shape:ranked_tensor_shape;right_shape:ranked_tensor_shape"
    )
    assert outer_contract.shape_rule is not None
    assert outer_contract.shape_rule((left_vector, matrix)) == (3, 6)
    assert outer_contract.dtype_rule is not None
    assert outer_contract.dtype_rule((left_vector, matrix)) == "float64"
    assert outer_contract.static_argument_rule is not None
    assert outer_contract.static_argument_rule((left_vector, matrix)) == ()
    with pytest.raises(ValueError, match="incomplete primitive contract"):
        primitive_complete_contract_for(outer_contract.identity)

    matmul_contract = primitive_contract_for("scpn.program_ad.product:matmul")
    assert matmul_contract.identity == PrimitiveIdentity("scpn.program_ad.product", "matmul", "1")
    assert matmul_contract.nondifferentiable_policy == "program_ad_trace_exact_fail_closed"
    assert matmul_contract.effect == "pure"
    assert matmul_contract.lowering_metadata["mlir_op"] == "scpn_diff.product.matmul"
    assert (
        matmul_contract.lowering_metadata["static_derivative_factory"]
        == "program_ad_product_matmul_derivative_rule"
    )
    assert matmul_contract.lowering_metadata["static_signature"] == (
        "left_shape:ranked_tensor_shape;right_shape:ranked_tensor_shape"
    )
    assert matmul_contract.shape_rule is not None
    assert matmul_contract.shape_rule((matrix, right_vector)) == (2,)
    assert matmul_contract.dtype_rule is not None
    assert matmul_contract.dtype_rule((matrix, right_vector)) == "float64"
    assert matmul_contract.static_argument_rule is not None
    assert matmul_contract.static_argument_rule((matrix, right_vector)) == ()
    with pytest.raises(ValueError, match="incomplete primitive contract"):
        primitive_complete_contract_for(matmul_contract.identity)


def test_program_ad_vdot_flattens_operands_with_exact_adjoint() -> None:
    """Program AD vdot should apply exact flattened real inner-product semantics."""

    left_weights = np.linspace(-1.5, 2.5, 6, dtype=np.float64).reshape(2, 3)
    right_weights = np.linspace(-2.0, 1.0, 6, dtype=np.float64).reshape(2, 3)

    def objective(values: Any) -> object:
        left = np.reshape(values[:6], (2, 3))
        right = np.reshape(values[6:], (3, 2))
        return np.vdot(left, right) + np.vdot(left * left_weights, right_weights)

    values = np.linspace(-0.8, 1.2, 12, dtype=np.float64)
    result = whole_program_value_and_grad(
        objective,
        values,
        parameters=tuple(Parameter(f"theta_{index}") for index in range(values.size)),
    )

    expected = np.zeros(12, dtype=np.float64)
    expected[:6] = values[6:] + left_weights.reshape(-1) * right_weights.reshape(-1)
    expected[6:] = values[:6]
    _assert_allclose(result.gradient, expected, rtol=1.0e-12, atol=1.0e-12)
    _assert_allclose(program_adjoint_gradient(result), expected, rtol=1.0e-12, atol=1.0e-12)


def test_program_ad_vdot_fails_closed_size_mismatch() -> None:
    """Program AD vdot should reject flattened size mismatches explicitly."""

    with pytest.raises(ValueError, match="vdot flattened operands must have matching size"):
        whole_program_value_and_grad(
            lambda values: np.vdot(values[:2], values[2:5]),
            np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64),
        )


def test_program_ad_product_boundary_metadata_is_explicit() -> None:
    """Product contracts should expose fail-closed contraction boundaries."""

    expected_factories = {
        "dot": "not_required",
        "vdot": "not_required",
        "inner": "program_ad_product_inner_derivative_rule",
        "outer": "program_ad_product_outer_derivative_rule",
        "matmul": "program_ad_product_matmul_derivative_rule",
        "tensordot": "program_ad_product_tensordot_derivative_rule",
        "einsum": "program_ad_product_einsum_derivative_rule",
    }
    expected_static_signatures = {
        "dot": "none",
        "vdot": "none",
        "inner": "left_shape:ranked_tensor_shape;right_shape:ranked_tensor_shape",
        "outer": "left_shape:ranked_tensor_shape;right_shape:ranked_tensor_shape",
        "matmul": "left_shape:ranked_tensor_shape;right_shape:ranked_tensor_shape",
        "tensordot": "left_shape:ranked_tensor_shape;right_shape:ranked_tensor_shape;axes",
        "einsum": "subscripts:explicit_static;operand_shapes:ranked_tensor_shapes",
    }
    expected_boundaries = {
        "dot": "inner_dimension_alignment",
        "vdot": "flattened_size_alignment",
        "inner": "last_dimension_alignment",
        "outer": "flattened_outer_product",
        "matmul": "core_dimension_alignment",
        "tensordot": "static_axes_tensor_contraction",
        "einsum": "explicit_static_tensor_contraction",
    }
    for name, boundary in expected_boundaries.items():
        metadata = primitive_contract_for(
            PrimitiveIdentity("scpn.program_ad.product", name, "1")
        ).lowering_metadata
        assert metadata["static_derivative_factory"] == expected_factories[name]
        assert metadata["static_signature"] == expected_static_signatures[name]
        assert metadata["nondifferentiable_boundary"] == boundary
        assert metadata["nondifferentiable_boundary_policy"] == "fail_closed"


def test_program_ad_product_primitives_validate_registry_rules_at_dispatch() -> None:
    """Supported product primitives must execute through registry validation rules."""

    originals = {
        name: primitive_contract_for(f"scpn.program_ad.product:{name}")
        for name in ("dot", "vdot", "inner", "outer", "matmul", "tensordot", "einsum")
    }
    calls: dict[str, set[str]] = {name: set() for name in originals}

    for name, original in originals.items():
        assert original.shape_rule is not None
        assert original.dtype_rule is not None
        assert original.static_argument_rule is not None
        original_shape_rule = original.shape_rule
        original_dtype_rule = original.dtype_rule
        original_static_argument_rule = original.static_argument_rule

        def shape_rule(
            args: tuple[object, ...],
            *,
            primitive_name: str = name,
            wrapped_rule: Callable[[tuple[object, ...]], tuple[int, ...]] = original_shape_rule,
        ) -> tuple[int, ...]:
            calls[primitive_name].add("shape")
            return wrapped_rule(args)

        def dtype_rule(
            args: tuple[object, ...],
            *,
            primitive_name: str = name,
            wrapped_rule: Callable[[tuple[object, ...]], str] = original_dtype_rule,
        ) -> str:
            calls[primitive_name].add("dtype")
            return wrapped_rule(args)

        def static_argument_rule(
            args: tuple[object, ...],
            *,
            primitive_name: str = name,
            wrapped_rule: Callable[
                [tuple[object, ...]], tuple[object, ...]
            ] = original_static_argument_rule,
        ) -> tuple[object, ...]:
            calls[primitive_name].add("static")
            return wrapped_rule(args)

        DEFAULT_CUSTOM_DERIVATIVE_REGISTRY.register_transform(
            PrimitiveTransformRule(
                identity=original.identity,
                derivative_rule=original.derivative_rule,
                batching_rule=original.batching_rule,
                lowering_rule=original.lowering_rule,
                lowering_metadata=original.lowering_metadata,
                shape_rule=shape_rule,
                dtype_rule=dtype_rule,
                static_argument_rule=static_argument_rule,
                nondifferentiable_policy=original.nondifferentiable_policy,
                effect=original.effect,
            ),
            overwrite=True,
        )
    try:
        result = whole_program_value_and_grad(
            lambda values: (
                np.dot(values[:3], np.array([2.0, -1.0, 0.5]))
                + np.vdot(np.reshape(values, (2, 3)), np.arange(1.0, 7.0))
                + np.inner(values[:3], np.array([0.5, -2.0, 1.5]))
                + np.sum(np.outer(values[:2], values[2:4]) * np.array([[1.0, -0.5], [0.25, 2.0]]))
                + np.sum(np.matmul(np.reshape(values, (2, 3)), np.array([1.0, 2.0, -1.0])))
                + np.sum(
                    np.tensordot(
                        np.reshape(values, (2, 3)),
                        np.array([[0.5, -1.0], [1.5, 0.25], [-0.75, 2.0]]),
                        axes=([1], [0]),
                    )
                )
                + np.sum(
                    np.einsum(
                        "ij,ij->i",
                        np.reshape(values, (2, 3)),
                        np.array([[1.0, -2.0, 0.5], [0.25, 1.5, -1.0]]),
                    )
                )
            ),
            np.array([0.25, 0.5, 0.75, 1.0, 1.25, 1.5], dtype=np.float64),
        )
    finally:
        for original in originals.values():
            DEFAULT_CUSTOM_DERIVATIVE_REGISTRY.register_transform(
                _transform_rule_from_contract(original), overwrite=True
            )

    values = np.array([0.25, 0.5, 0.75, 1.0, 1.25, 1.5], dtype=np.float64)
    expected_value = float(
        np.dot(values[:3], np.array([2.0, -1.0, 0.5]))
        + np.vdot(np.reshape(values, (2, 3)), np.arange(1.0, 7.0))
        + np.inner(values[:3], np.array([0.5, -2.0, 1.5]))
        + np.sum(np.outer(values[:2], values[2:4]) * np.array([[1.0, -0.5], [0.25, 2.0]]))
        + np.sum(np.matmul(np.reshape(values, (2, 3)), np.array([1.0, 2.0, -1.0])))
        + np.sum(
            np.tensordot(
                np.reshape(values, (2, 3)),
                np.array([[0.5, -1.0], [1.5, 0.25], [-0.75, 2.0]]),
                axes=([1], [0]),
            )
        )
        + np.sum(
            np.einsum(
                "ij,ij->i",
                np.reshape(values, (2, 3)),
                np.array([[1.0, -2.0, 0.5], [0.25, 1.5, -1.0]]),
            )
        )
    )
    assert result.value == pytest.approx(expected_value)
    assert calls == {
        "dot": {"shape", "dtype", "static"},
        "vdot": {"shape", "dtype", "static"},
        "inner": {"shape", "dtype", "static"},
        "outer": {"shape", "dtype", "static"},
        "matmul": {"shape", "dtype", "static"},
        "tensordot": {"shape", "dtype", "static"},
        "einsum": {"shape", "dtype", "static"},
    }


def test_program_ad_product_primitives_expose_direct_value_jvp_kernels() -> None:
    """Flat product primitive contracts should expose exact direct value/JVP rules."""

    left = np.array([1.0, -2.0, 3.0], dtype=np.float64)
    right = np.array([0.5, 4.0, -1.5], dtype=np.float64)
    left_tangent = np.array([0.2, -0.1, 0.4], dtype=np.float64)
    right_tangent = np.array([-0.3, 0.6, 0.25], dtype=np.float64)
    vector_values = np.concatenate((left, right))
    vector_tangent = np.concatenate((left_tangent, right_tangent))

    dot_rule = custom_derivative_rule_for(PrimitiveIdentity("scpn.program_ad.product", "dot", "1"))
    vdot_rule = custom_derivative_rule_for(
        PrimitiveIdentity("scpn.program_ad.product", "vdot", "1")
    )
    inner_rule = custom_derivative_rule_for(
        PrimitiveIdentity("scpn.program_ad.product", "inner", "1")
    )
    outer_rule = custom_derivative_rule_for(
        PrimitiveIdentity("scpn.program_ad.product", "outer", "1")
    )

    assert dot_rule.name == "program_ad_product_dot_direct_rule"
    assert vdot_rule.name == "program_ad_product_vdot_direct_rule"
    assert inner_rule.name == "program_ad_product_inner_direct_rule"
    assert outer_rule.name == "program_ad_product_outer_direct_rule"
    assert dot_rule.jvp_rule is not None
    assert vdot_rule.jvp_rule is not None
    assert inner_rule.jvp_rule is not None
    assert outer_rule.jvp_rule is not None
    assert dot_rule.vjp_rule is not None
    assert vdot_rule.vjp_rule is not None
    assert inner_rule.vjp_rule is not None
    assert outer_rule.vjp_rule is not None

    expected_inner = np.array([np.dot(left, right)], dtype=np.float64)
    expected_inner_jvp = np.array(
        [np.dot(left_tangent, right) + np.dot(left, right_tangent)], dtype=np.float64
    )
    expected_inner_vjp = np.concatenate((2.5 * right, 2.5 * left))
    _assert_allclose(dot_rule.value_fn(vector_values), expected_inner)
    _assert_allclose(dot_rule.jvp_rule(vector_values, vector_tangent), expected_inner_jvp)
    _assert_allclose(dot_rule.vjp_rule(vector_values, np.array([2.5])), expected_inner_vjp)
    _assert_allclose(vdot_rule.value_fn(vector_values), expected_inner)
    _assert_allclose(vdot_rule.jvp_rule(vector_values, vector_tangent), expected_inner_jvp)
    _assert_allclose(vdot_rule.vjp_rule(vector_values, np.array([2.5])), expected_inner_vjp)
    _assert_allclose(inner_rule.value_fn(vector_values), expected_inner)
    _assert_allclose(inner_rule.jvp_rule(vector_values, vector_tangent), expected_inner_jvp)
    _assert_allclose(inner_rule.vjp_rule(vector_values, np.array([2.5])), expected_inner_vjp)
    expected_outer = np.asarray(np.outer(left, right), dtype=np.float64)
    expected_outer_jvp = np.outer(left_tangent, right) + np.outer(left, right_tangent)
    expected_outer_vjp = np.concatenate((expected_outer @ right, expected_outer.T @ left))
    _assert_allclose(outer_rule.value_fn(vector_values), expected_outer.reshape(-1))
    _assert_allclose(
        outer_rule.jvp_rule(vector_values, vector_tangent), expected_outer_jvp.reshape(-1)
    )
    _assert_allclose(
        outer_rule.vjp_rule(vector_values, expected_outer.reshape(-1)), expected_outer_vjp
    )

    left_matrix = np.array([[1.0, -2.0], [0.5, 3.0]], dtype=np.float64)
    right_matrix = np.array([[2.0, 1.5], [-1.0, 0.25]], dtype=np.float64)
    left_matrix_tangent = np.array([[0.2, -0.3], [0.4, 0.1]], dtype=np.float64)
    right_matrix_tangent = np.array([[-0.5, 0.6], [0.25, -0.2]], dtype=np.float64)
    matrix_values = np.concatenate((left_matrix.reshape(-1), right_matrix.reshape(-1)))
    matrix_tangent = np.concatenate(
        (left_matrix_tangent.reshape(-1), right_matrix_tangent.reshape(-1))
    )
    matmul_rule = custom_derivative_rule_for(
        PrimitiveIdentity("scpn.program_ad.product", "matmul", "1")
    )

    assert matmul_rule.name == "program_ad_product_matmul_direct_rule"
    assert matmul_rule.jvp_rule is not None
    assert matmul_rule.vjp_rule is not None
    _assert_allclose(matmul_rule.value_fn(matrix_values), (left_matrix @ right_matrix).reshape(-1))
    _assert_allclose(
        matmul_rule.jvp_rule(matrix_values, matrix_tangent),
        (left_matrix_tangent @ right_matrix + left_matrix @ right_matrix_tangent).reshape(-1),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    matrix_cotangent = np.array([[1.5, -0.5], [0.75, 2.0]], dtype=np.float64)
    expected_matmul_vjp = np.concatenate(
        (
            (matrix_cotangent @ right_matrix.T).reshape(-1),
            (left_matrix.T @ matrix_cotangent).reshape(-1),
        )
    )
    _assert_allclose(
        matmul_rule.vjp_rule(matrix_values, matrix_cotangent.reshape(-1)),
        expected_matmul_vjp,
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_program_ad_product_matmul_static_derivative_factory_supports_rank1_rank2() -> None:
    """Static matmul factories should support vector and rectangular matrix contracts."""

    matrix = np.array([[1.0, -2.0, 0.5], [0.75, 3.0, -1.25]], dtype=np.float64)
    vector = np.array([0.25, -1.5, 2.0], dtype=np.float64)
    tangent_matrix = np.array([[0.2, -0.3, 0.4], [0.1, 0.5, -0.25]], dtype=np.float64)
    tangent_vector = np.array([-0.5, 0.75, 1.25], dtype=np.float64)

    matvec_rule = program_ad_product_matmul_derivative_rule((2, 3), (3,))
    assert matvec_rule.name == "program_ad_product_matmul_2x3_by_3_direct_rule"
    assert matvec_rule.jvp_rule is not None
    assert matvec_rule.vjp_rule is not None
    matvec_values = np.concatenate((matrix.reshape(-1), vector))
    matvec_tangent = np.concatenate((tangent_matrix.reshape(-1), tangent_vector))
    matvec_cotangent = np.array([1.25, -0.5], dtype=np.float64)
    _assert_allclose(matvec_rule.value_fn(matvec_values), matrix @ vector)
    _assert_allclose(
        matvec_rule.jvp_rule(matvec_values, matvec_tangent),
        tangent_matrix @ vector + matrix @ tangent_vector,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    _assert_allclose(
        matvec_rule.vjp_rule(matvec_values, matvec_cotangent),
        np.concatenate(
            (np.outer(matvec_cotangent, vector).reshape(-1), matrix.T @ matvec_cotangent)
        ),
        rtol=1.0e-12,
        atol=1.0e-12,
    )

    row = np.array([1.5, -0.75], dtype=np.float64)
    right = np.array([[2.0, -1.0, 0.25], [0.5, 1.25, -2.0]], dtype=np.float64)
    tangent_row = np.array([-0.25, 0.5], dtype=np.float64)
    tangent_right = np.array([[0.1, -0.3, 0.7], [0.4, -0.2, 0.6]], dtype=np.float64)
    vecmat_rule = program_ad_product_matmul_derivative_rule((2,), (2, 3))
    assert vecmat_rule.jvp_rule is not None
    assert vecmat_rule.vjp_rule is not None
    vecmat_values = np.concatenate((row, right.reshape(-1)))
    vecmat_tangent = np.concatenate((tangent_row, tangent_right.reshape(-1)))
    vecmat_cotangent = np.array([0.25, -1.5, 2.0], dtype=np.float64)
    _assert_allclose(vecmat_rule.value_fn(vecmat_values), row @ right)
    _assert_allclose(
        vecmat_rule.jvp_rule(vecmat_values, vecmat_tangent),
        tangent_row @ right + row @ tangent_right,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    _assert_allclose(
        vecmat_rule.vjp_rule(vecmat_values, vecmat_cotangent),
        np.concatenate((right @ vecmat_cotangent, np.outer(row, vecmat_cotangent).reshape(-1))),
        rtol=1.0e-12,
        atol=1.0e-12,
    )

    rectangular_left = np.array([[1.0, -2.0, 0.5], [0.75, 3.0, -1.25]], dtype=np.float64)
    rectangular_right = np.array([[0.25, -1.0], [1.5, 0.75], [-0.5, 2.0]], dtype=np.float64)
    rect_rule = program_ad_product_matmul_derivative_rule((2, 3), (3, 2))
    assert rect_rule.name == "program_ad_product_matmul_2x3_by_3x2_direct_rule"
    assert rect_rule.vjp_rule is not None
    rect_values = np.concatenate((rectangular_left.reshape(-1), rectangular_right.reshape(-1)))
    rect_cotangent = np.array([[1.25, -0.5], [0.75, 2.0]], dtype=np.float64)
    _assert_allclose(
        rect_rule.value_fn(rect_values),
        (rectangular_left @ rectangular_right).reshape(-1),
    )
    _assert_allclose(
        rect_rule.vjp_rule(rect_values, rect_cotangent.reshape(-1)),
        np.concatenate(
            (
                (rect_cotangent @ rectangular_right.T).reshape(-1),
                (rectangular_left.T @ rect_cotangent).reshape(-1),
            )
        ),
        rtol=1.0e-12,
        atol=1.0e-12,
    )

    with pytest.raises(ValueError, match="dimensions must align"):
        program_ad_product_matmul_derivative_rule((2, 3), (2, 2))
    with pytest.raises(ValueError, match="rank-1 or rank-2"):
        program_ad_product_matmul_derivative_rule((1, 2, 3), (3,))


def test_program_ad_product_inner_outer_static_derivative_factories() -> None:
    """Static inner and outer factories should expose exact product adjoints."""

    left = np.array([[1.0, -2.0, 0.5], [0.75, 3.0, -1.25]], dtype=np.float64)
    right = np.array([[0.25, -1.0, 1.5], [1.25, 0.5, -0.75]], dtype=np.float64)
    tangent_left = np.array([[0.2, -0.3, 0.4], [0.1, 0.5, -0.25]], dtype=np.float64)
    tangent_right = np.array([[-0.5, 0.6, 0.25], [0.75, -0.2, 0.1]], dtype=np.float64)
    values = np.concatenate((left.reshape(-1), right.reshape(-1)))
    tangent = np.concatenate((tangent_left.reshape(-1), tangent_right.reshape(-1)))

    inner_rule = program_ad_product_inner_derivative_rule(left.shape, right.shape)
    assert inner_rule.name == "program_ad_product_inner_2x3_by_2x3_direct_rule"
    assert inner_rule.jvp_rule is not None
    assert inner_rule.vjp_rule is not None
    expected_inner = np.inner(left, right)
    expected_inner_jvp = np.inner(tangent_left, right) + np.inner(left, tangent_right)
    inner_cotangent = np.array([[1.5, -0.5], [0.75, 2.0]], dtype=np.float64)
    expected_left_adjoint = inner_cotangent @ right
    expected_right_adjoint = inner_cotangent.T @ left
    _assert_allclose(inner_rule.value_fn(values), expected_inner.reshape(-1))
    _assert_allclose(
        inner_rule.jvp_rule(values, tangent),
        expected_inner_jvp.reshape(-1),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    _assert_allclose(
        inner_rule.vjp_rule(values, inner_cotangent.reshape(-1)),
        np.concatenate((expected_left_adjoint.reshape(-1), expected_right_adjoint.reshape(-1))),
        rtol=1.0e-12,
        atol=1.0e-12,
    )

    outer_rule = program_ad_product_outer_derivative_rule((2,), (3,))
    assert outer_rule.name == "program_ad_product_outer_2_by_3_direct_rule"
    assert outer_rule.jvp_rule is not None
    assert outer_rule.vjp_rule is not None
    outer_left = np.array([1.0, -2.0], dtype=np.float64)
    outer_right = np.array([0.25, -1.0, 1.5], dtype=np.float64)
    outer_left_tangent = np.array([0.2, -0.3], dtype=np.float64)
    outer_right_tangent = np.array([-0.5, 0.6, 0.25], dtype=np.float64)
    outer_values = np.concatenate((outer_left, outer_right))
    outer_tangent = np.concatenate((outer_left_tangent, outer_right_tangent))
    outer_cotangent = np.array([[1.25, -0.5, 0.75], [-1.0, 0.25, 1.5]], dtype=np.float64)
    _assert_allclose(
        outer_rule.value_fn(outer_values), np.outer(outer_left, outer_right).reshape(-1)
    )
    _assert_allclose(
        outer_rule.jvp_rule(outer_values, outer_tangent),
        (
            np.outer(outer_left_tangent, outer_right) + np.outer(outer_left, outer_right_tangent)
        ).reshape(-1),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    _assert_allclose(
        outer_rule.vjp_rule(outer_values, outer_cotangent.reshape(-1)),
        np.concatenate((outer_cotangent @ outer_right, outer_cotangent.T @ outer_left)),
        rtol=1.0e-12,
        atol=1.0e-12,
    )

    with pytest.raises(ValueError, match="last dimensions must align"):
        program_ad_product_inner_derivative_rule((2, 3), (2, 2))


def test_program_ad_product_tensordot_registry_contract_and_direct_rule() -> None:
    """Static tensordot contracts should expose exact contraction JVP/VJP rules."""

    left = np.linspace(-0.7, 1.6, 24, dtype=np.float64).reshape(2, 3, 4)
    right = np.linspace(1.2, -0.5, 24, dtype=np.float64).reshape(4, 3, 2)
    tangent_left = np.linspace(0.3, -0.2, 24, dtype=np.float64).reshape(left.shape)
    tangent_right = np.linspace(-0.4, 0.6, 24, dtype=np.float64).reshape(right.shape)
    axes = ((2, 1), (0, 1))
    values = np.concatenate((left.reshape(-1), right.reshape(-1)))
    tangent = np.concatenate((tangent_left.reshape(-1), tangent_right.reshape(-1)))
    cotangent = np.array([[0.5, -1.25], [1.75, 0.25]], dtype=np.float64)

    contract = primitive_contract_for(
        PrimitiveIdentity("scpn.program_ad.product", "tensordot", "1")
    )
    assert contract.identity == PrimitiveIdentity("scpn.program_ad.product", "tensordot", "1")
    assert contract.nondifferentiable_policy == "program_ad_trace_exact_fail_closed"
    assert contract.lowering_metadata["mlir_op"] == "scpn_diff.product.tensordot"
    assert (
        contract.lowering_metadata["static_derivative_factory"]
        == "program_ad_product_tensordot_derivative_rule"
    )
    assert (
        contract.lowering_metadata["static_signature"]
        == "left_shape:ranked_tensor_shape;right_shape:ranked_tensor_shape;axes"
    )
    assert contract.shape_rule is not None
    assert contract.shape_rule((left, right, axes)) == (2, 2)
    assert contract.dtype_rule is not None
    assert contract.dtype_rule((left, right, axes)) == "float64"
    assert contract.static_argument_rule is not None
    assert contract.static_argument_rule((left, right, axes)) == (
        left.shape,
        right.shape,
        ((2, 1), (0, 1)),
    )

    rule = program_ad_product_tensordot_derivative_rule(left.shape, right.shape, axes=axes)
    assert rule.name == "program_ad_product_tensordot_2x3x4_by_4x3x2_axes_2_1_by_0_1_direct_rule"
    assert rule.jvp_rule is not None
    assert rule.vjp_rule is not None
    _assert_allclose(rule.value_fn(values), np.tensordot(left, right, axes=axes).reshape(-1))
    _assert_allclose(
        rule.jvp_rule(values, tangent),
        (
            np.tensordot(tangent_left, right, axes=axes)
            + np.tensordot(left, tangent_right, axes=axes)
        ).reshape(-1),
        rtol=1.0e-12,
        atol=1.0e-12,
    )

    expected_adjoint = np.zeros(values.size, dtype=np.float64)
    for index in range(values.size):
        basis = np.zeros_like(values)
        basis[index] = 1.0
        basis_left = basis[: left.size].reshape(left.shape)
        basis_right = basis[left.size :].reshape(right.shape)
        basis_result = np.tensordot(basis_left, right, axes=axes) + np.tensordot(
            left,
            basis_right,
            axes=axes,
        )
        expected_adjoint[index] = float(np.sum(basis_result * cotangent))
    _assert_allclose(
        rule.vjp_rule(values, cotangent.reshape(-1)),
        expected_adjoint,
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_program_ad_tensordot_static_axes_preserves_high_rank_adjoint() -> None:
    """Program AD should differentiate static high-rank tensordot contractions exactly."""

    left_shape = (2, 3, 2)
    right_shape = (2, 3, 2)
    axes = ((2, 1), (0, 1))
    weights = np.array([[0.5, -1.25], [1.75, 0.25]], dtype=np.float64)
    values = np.linspace(-0.8, 1.5, 24, dtype=np.float64)

    def objective(trace_values: Any) -> object:
        left = np.reshape(trace_values[:12], left_shape)
        right = np.reshape(trace_values[12:], right_shape)
        return np.sum(np.tensordot(left, right, axes=axes) * weights)

    result = whole_program_value_and_grad(
        objective,
        values,
        parameters=tuple(Parameter(f"x{index}") for index in range(values.size)),
    )

    expected = np.zeros_like(values)
    for index in range(values.size):
        basis = np.zeros_like(values)
        basis[index] = 1.0
        basis_left = basis[:12].reshape(left_shape)
        basis_right = basis[12:].reshape(right_shape)
        left = values[:12].reshape(left_shape)
        right = values[12:].reshape(right_shape)
        expected[index] = np.sum(
            (
                np.tensordot(basis_left, right, axes=axes)
                + np.tensordot(left, basis_right, axes=axes)
            )
            * weights
        )
    _assert_allclose(result.gradient, expected, rtol=1.0e-12, atol=1.0e-12)
