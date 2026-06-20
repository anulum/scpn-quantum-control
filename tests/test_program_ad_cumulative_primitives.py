# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Program AD cumulative primitive tests
"""Tests for Program AD cumulative primitive contracts and direct rules."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import numpy as np
import pytest

from scpn_quantum_control.differentiable import (
    DEFAULT_CUSTOM_DERIVATIVE_REGISTRY,
    PrimitiveContract,
    PrimitiveIdentity,
    PrimitiveTransformRule,
    custom_derivative_rule_for,
    primitive_complete_contract_for,
    primitive_contract_for,
    program_ad_cumulative_cumprod_derivative_rule,
    program_ad_cumulative_cumsum_derivative_rule,
    program_ad_cumulative_diff_derivative_rule,
    whole_program_value_and_grad,
)


def _assert_allclose(
    actual: object, expected: object, *, rtol: float = 1.0e-7, atol: float = 0.0
) -> None:
    """Assert NumPy closeness across dynamically typed Program AD payloads."""

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


def test_program_ad_cumulative_primitives_are_registry_policy_gated() -> None:
    """Cumsum, cumprod, and diff should expose primitive registry contracts."""

    matrix = np.arange(6.0, dtype=np.float64).reshape(2, 3)
    cumsum_contract = primitive_contract_for("scpn.program_ad.cumulative:cumsum")
    assert cumsum_contract.identity == PrimitiveIdentity(
        "scpn.program_ad.cumulative", "cumsum", "1"
    )
    assert cumsum_contract.nondifferentiable_policy == "program_ad_trace_exact_fail_closed"
    assert cumsum_contract.effect == "pure"
    assert cumsum_contract.lowering_metadata["mlir_op"] == "scpn_diff.cumulative.cumsum"
    assert (
        cumsum_contract.lowering_metadata["static_derivative_factory"]
        == "program_ad_cumulative_cumsum_derivative_rule"
    )
    assert cumsum_contract.lowering_metadata["static_signature"] == (
        "source_shape:ranked_tensor_shape;axis"
    )
    assert cumsum_contract.shape_rule is not None
    assert cumsum_contract.shape_rule((matrix, None)) == (6,)
    assert cumsum_contract.shape_rule((matrix, 1)) == (2, 3)
    assert cumsum_contract.dtype_rule is not None
    assert cumsum_contract.dtype_rule((matrix, 1)) == "float64"
    assert cumsum_contract.static_argument_rule is not None
    assert cumsum_contract.static_argument_rule((matrix, 1)) == (1,)
    with pytest.raises(ValueError, match="incomplete primitive contract"):
        primitive_complete_contract_for(cumsum_contract.identity)

    cumprod_contract = primitive_contract_for("scpn.program_ad.cumulative:cumprod")
    assert cumprod_contract.identity == PrimitiveIdentity(
        "scpn.program_ad.cumulative", "cumprod", "1"
    )
    assert cumprod_contract.nondifferentiable_policy == "program_ad_trace_exact_fail_closed"
    assert cumprod_contract.effect == "pure"
    assert cumprod_contract.lowering_metadata["mlir_op"] == "scpn_diff.cumulative.cumprod"
    assert (
        cumprod_contract.lowering_metadata["static_derivative_factory"]
        == "program_ad_cumulative_cumprod_derivative_rule"
    )
    assert cumprod_contract.lowering_metadata["static_signature"] == (
        "source_shape:ranked_tensor_shape;axis"
    )
    assert cumprod_contract.shape_rule is not None
    assert cumprod_contract.shape_rule((matrix, None)) == (6,)
    assert cumprod_contract.shape_rule((matrix, 1)) == (2, 3)
    assert cumprod_contract.dtype_rule is not None
    assert cumprod_contract.dtype_rule((matrix, 1)) == "float64"
    assert cumprod_contract.static_argument_rule is not None
    assert cumprod_contract.static_argument_rule((matrix, 1)) == (1,)
    with pytest.raises(ValueError, match="incomplete primitive contract"):
        primitive_complete_contract_for(cumprod_contract.identity)

    diff_contract = primitive_contract_for("scpn.program_ad.cumulative:diff")
    assert diff_contract.identity == PrimitiveIdentity("scpn.program_ad.cumulative", "diff", "1")
    assert diff_contract.nondifferentiable_policy == "program_ad_trace_exact_fail_closed"
    assert diff_contract.effect == "pure"
    assert diff_contract.lowering_metadata["mlir_op"] == "scpn_diff.cumulative.diff"
    assert (
        diff_contract.lowering_metadata["static_derivative_factory"]
        == "program_ad_cumulative_diff_derivative_rule"
    )
    assert diff_contract.lowering_metadata["static_signature"] == (
        "source_shape:ranked_tensor_shape;order_axis"
    )
    assert diff_contract.shape_rule is not None
    assert diff_contract.shape_rule((matrix, 2, 1)) == (2, 1)
    assert diff_contract.dtype_rule is not None
    assert diff_contract.dtype_rule((matrix, 2, 1)) == "float64"
    assert diff_contract.static_argument_rule is not None
    assert diff_contract.static_argument_rule((matrix, 2, 1)) == (2, 1)
    with pytest.raises(ValueError, match="incomplete primitive contract"):
        primitive_complete_contract_for(diff_contract.identity)


def test_program_ad_cumulative_boundary_metadata_is_explicit() -> None:
    """Cumulative contracts should expose fail-closed sequence boundaries."""

    expected_boundaries = {
        "cumsum": "ordered_axis_sequence",
        "cumprod": "ordered_axis_zero_factor_sensitive",
        "diff": "finite_difference_order_and_spacing",
    }
    for name, boundary in expected_boundaries.items():
        metadata = primitive_contract_for(
            PrimitiveIdentity("scpn.program_ad.cumulative", name, "1")
        ).lowering_metadata
        assert metadata["nondifferentiable_boundary"] == boundary
        assert metadata["nondifferentiable_boundary_policy"] == "fail_closed"


def test_program_ad_cumulative_primitives_validate_registry_rules_at_dispatch() -> None:
    """Supported cumulative primitives must execute through registry validation rules."""

    originals = {
        name: primitive_contract_for(f"scpn.program_ad.cumulative:{name}")
        for name in ("cumsum", "cumprod", "diff")
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
                np.cumsum(values)[3]
                + np.cumprod(values + 2.0)[2]
                + np.diff(np.reshape(values, (2, 3)), n=2, axis=1)[0, 0]
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
        np.cumsum(values)[3]
        + np.cumprod(values + 2.0)[2]
        + np.diff(np.reshape(values, (2, 3)), n=2, axis=1)[0, 0]
    )
    assert result.value == pytest.approx(expected_value)
    assert calls == {
        "cumsum": {"shape", "dtype", "static"},
        "cumprod": {"shape", "dtype", "static"},
        "diff": {"shape", "dtype", "static"},
    }


def test_program_ad_cumulative_primitives_expose_direct_value_jvp_kernels() -> None:
    """Flat cumulative primitive contracts should expose exact direct value/JVP rules."""

    values = np.array([2.0, 0.0, -3.0, 4.0], dtype=np.float64)
    tangent = np.array([0.5, -1.0, 0.25, 2.0], dtype=np.float64)

    cumsum_rule = custom_derivative_rule_for(
        PrimitiveIdentity("scpn.program_ad.cumulative", "cumsum", "1")
    )
    cumprod_rule = custom_derivative_rule_for(
        PrimitiveIdentity("scpn.program_ad.cumulative", "cumprod", "1")
    )
    diff_rule = custom_derivative_rule_for(
        PrimitiveIdentity("scpn.program_ad.cumulative", "diff", "1")
    )

    assert cumsum_rule.name == "program_ad_cumulative_cumsum_direct_rule"
    assert cumprod_rule.name == "program_ad_cumulative_cumprod_direct_rule"
    assert diff_rule.name == "program_ad_cumulative_diff_direct_rule"
    assert cumsum_rule.jvp_rule is not None
    assert cumprod_rule.jvp_rule is not None
    assert diff_rule.jvp_rule is not None
    assert cumsum_rule.vjp_rule is not None
    assert cumprod_rule.vjp_rule is not None
    assert diff_rule.vjp_rule is not None

    _assert_allclose(cumsum_rule.value_fn(values), np.cumsum(values))
    _assert_allclose(cumsum_rule.jvp_rule(values, tangent), np.cumsum(tangent))
    cotangent = np.array([1.0, -0.5, 0.25, 2.0], dtype=np.float64)
    _assert_allclose(
        cumsum_rule.vjp_rule(values, cotangent), np.flip(np.cumsum(np.flip(cotangent)))
    )

    expected_cumprod = np.cumprod(values)
    expected_cumprod_jvp = np.array(
        [
            tangent[0],
            tangent[0] * values[1] + values[0] * tangent[1],
            tangent[0] * values[1] * values[2]
            + values[0] * tangent[1] * values[2]
            + values[0] * values[1] * tangent[2],
            tangent[0] * values[1] * values[2] * values[3]
            + values[0] * tangent[1] * values[2] * values[3]
            + values[0] * values[1] * tangent[2] * values[3]
            + values[0] * values[1] * values[2] * tangent[3],
        ],
        dtype=np.float64,
    )
    _assert_allclose(cumprod_rule.value_fn(values), expected_cumprod)
    _assert_allclose(
        cumprod_rule.jvp_rule(values, tangent),
        expected_cumprod_jvp,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    expected_cumprod_vjp = np.array(
        [
            cotangent[0]
            + cotangent[1] * values[1]
            + cotangent[2] * values[1] * values[2]
            + cotangent[3] * values[1] * values[2] * values[3],
            cotangent[1] * values[0]
            + cotangent[2] * values[0] * values[2]
            + cotangent[3] * values[0] * values[2] * values[3],
            cotangent[2] * values[0] * values[1]
            + cotangent[3] * values[0] * values[1] * values[3],
            cotangent[3] * values[0] * values[1] * values[2],
        ],
        dtype=np.float64,
    )
    _assert_allclose(
        cumprod_rule.vjp_rule(values, cotangent),
        expected_cumprod_vjp,
        rtol=1.0e-12,
        atol=1.0e-12,
    )

    _assert_allclose(diff_rule.value_fn(values), np.diff(values))
    _assert_allclose(diff_rule.jvp_rule(values, tangent), np.diff(tangent))
    _assert_allclose(
        diff_rule.vjp_rule(values, cotangent[:3]),
        np.array(
            [-cotangent[0], cotangent[0] - cotangent[1], cotangent[1] - cotangent[2], cotangent[2]]
        ),
    )


def test_program_ad_cumulative_static_derivative_factories_are_axis_aware() -> None:
    """Static cumulative factories should expose exact axis-aware JVP and VJP rules."""

    matrix = np.array([[1.0, 2.0, 0.5], [3.0, -1.0, 4.0]], dtype=np.float64)
    tangent = np.array([[0.25, -0.5, 1.0], [1.5, -0.75, 0.5]], dtype=np.float64)
    values = matrix.reshape(-1)
    tangent_values = tangent.reshape(-1)

    cumsum_rule = program_ad_cumulative_cumsum_derivative_rule((2, 3), axis=1)
    assert cumsum_rule.name == "program_ad_cumulative_cumsum_2x3_axis_1_direct_rule"
    assert cumsum_rule.jvp_rule is not None
    assert cumsum_rule.vjp_rule is not None
    cumsum_cotangent = np.array([[1.0, -0.5, 2.0], [0.25, 1.5, -1.0]], dtype=np.float64)
    _assert_allclose(cumsum_rule.value_fn(values), np.cumsum(matrix, axis=1).reshape(-1))
    _assert_allclose(
        cumsum_rule.jvp_rule(values, tangent_values),
        np.cumsum(tangent, axis=1).reshape(-1),
    )
    _assert_allclose(
        cumsum_rule.vjp_rule(values, cumsum_cotangent.reshape(-1)),
        np.flip(np.cumsum(np.flip(cumsum_cotangent, axis=1), axis=1), axis=1).reshape(-1),
    )

    cumprod_rule = program_ad_cumulative_cumprod_derivative_rule((2, 3), axis=1)
    assert cumprod_rule.name == "program_ad_cumulative_cumprod_2x3_axis_1_direct_rule"
    assert cumprod_rule.jvp_rule is not None
    assert cumprod_rule.vjp_rule is not None
    _assert_allclose(cumprod_rule.value_fn(values), np.cumprod(matrix, axis=1).reshape(-1))
    expected_cumprod_jvp = np.array(
        [
            [
                0.25,
                0.25 * 2.0 + 1.0 * -0.5,
                0.25 * 2.0 * 0.5 + 1.0 * -0.5 * 0.5 + 1.0 * 2.0 * 1.0,
            ],
            [
                1.5,
                1.5 * -1.0 + 3.0 * -0.75,
                1.5 * -1.0 * 4.0 + 3.0 * -0.75 * 4.0 + 3.0 * -1.0 * 0.5,
            ],
        ],
        dtype=np.float64,
    )
    _assert_allclose(
        cumprod_rule.jvp_rule(values, tangent_values),
        expected_cumprod_jvp.reshape(-1),
    )

    diff_rule = program_ad_cumulative_diff_derivative_rule((2, 3), order=2, axis=1)
    assert diff_rule.name == "program_ad_cumulative_diff_2x3_order_2_axis_1_direct_rule"
    assert diff_rule.jvp_rule is not None
    assert diff_rule.vjp_rule is not None
    diff_cotangent = np.array([[1.5], [-2.0]], dtype=np.float64)
    _assert_allclose(diff_rule.value_fn(values), np.diff(matrix, n=2, axis=1).reshape(-1))
    _assert_allclose(
        diff_rule.jvp_rule(values, tangent_values),
        np.diff(tangent, n=2, axis=1).reshape(-1),
    )
    _assert_allclose(
        diff_rule.vjp_rule(values, diff_cotangent.reshape(-1)),
        np.array([[1.5, -3.0, 1.5], [-2.0, 4.0, -2.0]], dtype=np.float64).reshape(-1),
    )

    with pytest.raises(ValueError, match="out of bounds"):
        program_ad_cumulative_cumsum_derivative_rule((2, 3), axis=2)
    with pytest.raises(ValueError, match="non-negative integer"):
        program_ad_cumulative_diff_derivative_rule((2, 3), order=-1, axis=1)
