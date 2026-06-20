# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Program AD reduction and norm tests
"""Tests for Program AD product, statistical, norm, and cumulative reductions."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, cast

import numpy as np
import pytest

from scpn_quantum_control.differentiable import (
    DEFAULT_CUSTOM_DERIVATIVE_REGISTRY,
    Parameter,
    PrimitiveContract,
    PrimitiveIdentity,
    PrimitiveTransformRule,
    custom_derivative_rule_for,
    primitive_complete_contract_for,
    primitive_contract_for,
    program_ad_reduction_max_derivative_rule,
    program_ad_reduction_mean_derivative_rule,
    program_ad_reduction_median_derivative_rule,
    program_ad_reduction_min_derivative_rule,
    program_ad_reduction_percentile_derivative_rule,
    program_ad_reduction_prod_derivative_rule,
    program_ad_reduction_quantile_derivative_rule,
    program_ad_reduction_std_derivative_rule,
    program_ad_reduction_sum_derivative_rule,
    program_ad_reduction_var_derivative_rule,
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


def test_program_ad_reduction_primitives_are_registry_policy_gated() -> None:
    """Scalar reductions should expose primitive registry contracts."""

    matrix = np.arange(6.0, dtype=np.float64).reshape(2, 3)
    expected_shapes = {
        "sum": (2,),
        "prod": (2,),
        "mean": (2,),
        "var": (2,),
        "std": (2,),
        "max": (2,),
        "min": (2,),
        "median": (2,),
        "quantile": (2,),
        "percentile": (2,),
    }
    expected_factories = {
        "sum": "program_ad_reduction_sum_derivative_rule",
        "prod": "program_ad_reduction_prod_derivative_rule",
        "mean": "program_ad_reduction_mean_derivative_rule",
        "var": "program_ad_reduction_var_derivative_rule",
        "std": "program_ad_reduction_std_derivative_rule",
        "max": "program_ad_reduction_max_derivative_rule",
        "min": "program_ad_reduction_min_derivative_rule",
        "median": "program_ad_reduction_median_derivative_rule",
        "quantile": "program_ad_reduction_quantile_derivative_rule",
        "percentile": "program_ad_reduction_percentile_derivative_rule",
    }
    expected_boundaries = {
        "sum": "static_axis_and_stable_output_shape",
        "prod": "static_axis_zero_factor_sensitive",
        "mean": "static_axis_nonempty_reduction",
        "var": "static_axis_ddof_positive_denominator",
        "std": "static_axis_ddof_positive_denominator_nonzero_variance",
        "max": "static_axis_unique_max_selector",
        "min": "static_axis_unique_min_selector",
        "median": "static_axis_strict_order_selection",
        "quantile": "static_scalar_q_axis_method_strict_order_selection",
        "percentile": "static_scalar_q_axis_method_strict_order_selection",
    }
    expected_signatures = {
        "sum": "source_shape:ranked_tensor_shape;axis",
        "prod": "source_shape:ranked_tensor_shape;axis",
        "mean": "source_shape:ranked_tensor_shape;axis",
        "var": "source_shape:ranked_tensor_shape;axis;ddof",
        "std": "source_shape:ranked_tensor_shape;axis;ddof",
        "max": "source_shape:ranked_tensor_shape;axis",
        "min": "source_shape:ranked_tensor_shape;axis",
        "median": "source_shape:ranked_tensor_shape;axis",
        "quantile": "source_shape:ranked_tensor_shape;q;axis;method",
        "percentile": "source_shape:ranked_tensor_shape;q;axis;method",
    }

    for name, expected_shape in expected_shapes.items():
        args = (
            (matrix, 0.25, 1, "linear")
            if name == "quantile"
            else (matrix, 75.0, 1, "linear")
            if name == "percentile"
            else (matrix, 1, 1)
            if name in {"var", "std"}
            else (matrix, 1)
        )
        contract = primitive_contract_for(f"scpn.program_ad.reduction:{name}")
        assert contract.identity == PrimitiveIdentity("scpn.program_ad.reduction", name, "1")
        assert contract.nondifferentiable_policy == "program_ad_trace_exact_fail_closed"
        assert contract.effect == "pure"
        assert contract.lowering_metadata["mlir_op"] == f"scpn_diff.reduction.{name}"
        assert contract.lowering_metadata["static_derivative_factory"] == expected_factories[name]
        assert contract.lowering_metadata["static_signature"] == expected_signatures[name]
        assert (
            contract.lowering_metadata["nondifferentiable_boundary"] == expected_boundaries[name]
        )
        assert contract.lowering_metadata["nondifferentiable_boundary_policy"] == "fail_closed"
        assert contract.shape_rule is not None
        assert contract.shape_rule(args) == expected_shape
        assert contract.dtype_rule is not None
        assert contract.dtype_rule(args) == "float64"
        assert contract.static_argument_rule is not None
        if name == "quantile":
            assert contract.static_argument_rule(args) == (0.25, 1, "linear")
            assert contract.shape_rule((matrix, 0.25, None, "linear")) == ()
        elif name == "percentile":
            assert contract.static_argument_rule(args) == (0.75, 1, "linear")
            assert contract.shape_rule((matrix, 75.0, None, "linear")) == ()
        elif name in {"var", "std"}:
            assert contract.static_argument_rule(args) == (1, 1)
            assert contract.shape_rule((matrix, None, 1)) == ()
        else:
            assert contract.static_argument_rule((matrix, 1)) == (1,)
            assert contract.static_argument_rule((matrix, None)) == (None,)
            assert contract.shape_rule((matrix, None)) == ()
        with pytest.raises(ValueError, match="incomplete primitive contract"):
            primitive_complete_contract_for(contract.identity)


def test_program_ad_reduction_boundary_metadata_is_explicit() -> None:
    """Reduction contracts should expose fail-closed static-axis boundaries."""

    expected_boundaries = {
        "sum": "static_axis_and_stable_output_shape",
        "prod": "static_axis_zero_factor_sensitive",
        "mean": "static_axis_nonempty_reduction",
        "var": "static_axis_ddof_positive_denominator",
        "std": "static_axis_ddof_positive_denominator_nonzero_variance",
        "max": "static_axis_unique_max_selector",
        "min": "static_axis_unique_min_selector",
        "median": "static_axis_strict_order_selection",
        "quantile": "static_scalar_q_axis_method_strict_order_selection",
        "percentile": "static_scalar_q_axis_method_strict_order_selection",
        "trapezoid": "static_axis_and_static_grid_spacing",
    }
    for name, boundary in expected_boundaries.items():
        metadata = primitive_contract_for(
            PrimitiveIdentity("scpn.program_ad.reduction", name, "1")
        ).lowering_metadata
        assert metadata["nondifferentiable_boundary"] == boundary
        assert metadata["nondifferentiable_boundary_policy"] == "fail_closed"


def test_program_ad_reduction_primitives_validate_registry_rules_at_dispatch() -> None:
    """Supported reduction primitives must execute through registry validation rules."""

    originals = {
        name: primitive_contract_for(f"scpn.program_ad.reduction:{name}")
        for name in (
            "sum",
            "prod",
            "mean",
            "var",
            "std",
            "max",
            "min",
            "median",
            "quantile",
            "percentile",
        )
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
                np.sum(np.reshape(values, (2, 3)), axis=0)[0]
                + np.prod(np.reshape(values, (2, 3)), axis=1)[1]
                + np.mean(np.reshape(values, (2, 3)), axis=1)[0]
                + np.var(np.reshape(values, (2, 3)), axis=1, ddof=1)[0]
                + np.std(np.reshape(values, (2, 3)), axis=0, ddof=1)[2]
                + np.max(np.reshape(values, (2, 3)), axis=0)[1]
                + np.min(np.reshape(values, (2, 3)), axis=1)[1]
                + np.median(values)
                + np.quantile(np.reshape(values, (2, 3)), 0.25, axis=1)[0]
                + np.percentile(np.reshape(values, (2, 3)), 75.0, axis=0)[2]
            ),
            np.array([1.0, 2.0, 3.0, 6.0, 8.0, 11.0], dtype=np.float64),
        )
    finally:
        for original in originals.values():
            DEFAULT_CUSTOM_DERIVATIVE_REGISTRY.register_transform(
                _transform_rule_from_contract(original), overwrite=True
            )

    assert result.value == pytest.approx(567.0 + math.sqrt(32.0))
    assert calls == {
        "std": {"shape", "dtype", "static"},
        "var": {"shape", "dtype", "static"},
        "max": {"shape", "dtype", "static"},
        "min": {"shape", "dtype", "static"},
        "median": {"shape", "dtype", "static"},
        "sum": {"shape", "dtype", "static"},
        "prod": {"shape", "dtype", "static"},
        "mean": {"shape", "dtype", "static"},
        "percentile": {"shape", "dtype", "static"},
        "quantile": {"shape", "dtype", "static"},
    }


def test_program_ad_reduction_static_derivative_factories_are_direct_kernels() -> None:
    """Static reduction factories should expose exact axis-aware JVP and VJP rules."""

    matrix = np.array([[1.0, 2.0, 3.0], [4.0, 0.0, -2.0]], dtype=np.float64)
    values = matrix.reshape(-1)
    tangent = np.array([0.5, -1.0, 0.25, 2.0, -0.75, 1.25], dtype=np.float64)
    row_cotangent = np.array([1.5, -2.0], dtype=np.float64)

    sum_rule = program_ad_reduction_sum_derivative_rule((2, 3), axis=1)
    assert sum_rule.name == "program_ad_reduction_sum_2x3_axis_1_direct_rule"
    assert sum_rule.jvp_rule is not None
    assert sum_rule.vjp_rule is not None
    _assert_allclose(sum_rule.value_fn(values), np.sum(matrix, axis=1))
    _assert_allclose(
        sum_rule.jvp_rule(values, tangent),
        np.sum(tangent.reshape(2, 3), axis=1),
    )
    _assert_allclose(
        sum_rule.vjp_rule(values, row_cotangent),
        np.array([1.5, 1.5, 1.5, -2.0, -2.0, -2.0], dtype=np.float64),
    )

    mean_rule = program_ad_reduction_mean_derivative_rule((2, 3), axis=1)
    assert mean_rule.name == "program_ad_reduction_mean_2x3_axis_1_direct_rule"
    assert mean_rule.jvp_rule is not None
    assert mean_rule.vjp_rule is not None
    _assert_allclose(mean_rule.value_fn(values), np.mean(matrix, axis=1))
    _assert_allclose(
        mean_rule.jvp_rule(values, tangent),
        np.mean(tangent.reshape(2, 3), axis=1),
    )
    _assert_allclose(
        mean_rule.vjp_rule(values, row_cotangent),
        np.array([0.5, 0.5, 0.5, -2.0 / 3.0, -2.0 / 3.0, -2.0 / 3.0]),
    )

    prod_rule = program_ad_reduction_prod_derivative_rule((2, 3), axis=1)
    assert prod_rule.name == "program_ad_reduction_prod_2x3_axis_1_direct_rule"
    assert prod_rule.jvp_rule is not None
    assert prod_rule.vjp_rule is not None
    _assert_allclose(prod_rule.value_fn(values), np.prod(matrix, axis=1))
    _assert_allclose(
        prod_rule.jvp_rule(values, tangent),
        np.array(
            [
                tangent[0] * 2.0 * 3.0 + 1.0 * tangent[1] * 3.0 + 1.0 * 2.0 * tangent[2],
                tangent[3] * 0.0 * -2.0 + 4.0 * tangent[4] * -2.0 + 4.0 * 0.0 * tangent[5],
            ],
            dtype=np.float64,
        ),
    )
    _assert_allclose(
        prod_rule.vjp_rule(values, row_cotangent),
        np.array([9.0, 4.5, 3.0, 0.0, 16.0, 0.0], dtype=np.float64),
    )

    flat_rule = program_ad_reduction_sum_derivative_rule((2, 3), axis=None)
    assert flat_rule.name == "program_ad_reduction_sum_2x3_axis_flat_direct_rule"
    assert flat_rule.vjp_rule is not None
    _assert_allclose(flat_rule.value_fn(values), [np.sum(matrix)])
    _assert_allclose(flat_rule.vjp_rule(values, np.array([2.0])), np.full(6, 2.0))

    with pytest.raises(ValueError, match="out of bounds"):
        program_ad_reduction_sum_derivative_rule((2, 3), axis=2)
    with pytest.raises(ValueError, match="at least one value"):
        program_ad_reduction_mean_derivative_rule((0, 3), axis=None)


def test_program_ad_variance_static_derivative_factories_are_direct_kernels() -> None:
    """Variance/std factories should expose exact axis-aware JVP and VJP rules."""

    matrix = np.array([[1.0, 3.0, 6.0], [2.0, 5.0, 9.0]], dtype=np.float64)
    values = matrix.reshape(-1)
    tangent = np.array([0.25, -0.5, 1.0, -1.5, 0.75, 2.0], dtype=np.float64)
    row_cotangent = np.array([1.5, -0.75], dtype=np.float64)

    var_rule = program_ad_reduction_var_derivative_rule((2, 3), axis=1, ddof=1)
    assert var_rule.name == "program_ad_reduction_var_2x3_axis_1_ddof_1_direct_rule"
    assert var_rule.jvp_rule is not None
    assert var_rule.vjp_rule is not None
    _assert_allclose(var_rule.value_fn(values), np.var(matrix, axis=1, ddof=1))
    _assert_allclose(
        var_rule.jvp_rule(values, tangent),
        [
            np.dot(matrix[0] - np.mean(matrix[0]), tangent[:3]),
            np.dot(matrix[1] - np.mean(matrix[1]), tangent[3:]),
        ],
    )
    expected_var_vjp = np.zeros_like(matrix)
    expected_var_vjp[0] = row_cotangent[0] * (matrix[0] - np.mean(matrix[0]))
    expected_var_vjp[1] = row_cotangent[1] * (matrix[1] - np.mean(matrix[1]))
    _assert_allclose(var_rule.vjp_rule(values, row_cotangent), expected_var_vjp.reshape(-1))

    std_rule = program_ad_reduction_std_derivative_rule((2, 3), axis=1, ddof=1)
    assert std_rule.name == "program_ad_reduction_std_2x3_axis_1_ddof_1_direct_rule"
    assert std_rule.jvp_rule is not None
    assert std_rule.vjp_rule is not None
    std_gradient = np.vstack(
        [(row - np.mean(row)) / (2.0 * np.std(row, ddof=1)) for row in matrix]
    )
    _assert_allclose(std_rule.value_fn(values), np.std(matrix, axis=1, ddof=1))
    _assert_allclose(
        std_rule.jvp_rule(values, tangent),
        np.sum(std_gradient * tangent.reshape(matrix.shape), axis=1),
    )
    _assert_allclose(
        std_rule.vjp_rule(values, row_cotangent),
        (std_gradient * row_cotangent[:, None]).reshape(-1),
    )

    with pytest.raises(ValueError, match="positive denominator"):
        program_ad_reduction_var_derivative_rule((2,), ddof=2)
    with pytest.raises(ValueError, match="zero variance"):
        program_ad_reduction_std_derivative_rule((2,)).value_fn(
            np.array([1.0, 1.0], dtype=np.float64)
        )


def test_program_ad_order_statistic_static_derivative_factories_are_direct_kernels() -> None:
    """Order-statistic factories should expose exact strict-order JVP and VJP rules."""

    matrix = np.array([[3.0, -1.0, 2.0], [0.5, 4.0, -2.0]], dtype=np.float64)
    values = matrix.reshape(-1)
    tangent = np.array([0.25, -0.5, 1.5, -1.0, 0.75, 2.0], dtype=np.float64)
    row_cotangent = np.array([1.2, -0.4], dtype=np.float64)

    max_rule = program_ad_reduction_max_derivative_rule((2, 3), axis=1)
    assert max_rule.name == "program_ad_reduction_max_2x3_axis_1_q_1_0_direct_rule"
    assert max_rule.jvp_rule is not None
    assert max_rule.vjp_rule is not None
    _assert_allclose(max_rule.value_fn(values), np.max(matrix, axis=1))
    _assert_allclose(max_rule.jvp_rule(values, tangent), [tangent[0], tangent[4]])
    _assert_allclose(
        max_rule.vjp_rule(values, row_cotangent),
        np.array([1.2, 0.0, 0.0, 0.0, -0.4, 0.0], dtype=np.float64),
    )

    min_rule = program_ad_reduction_min_derivative_rule((2, 3), axis=0)
    assert min_rule.name == "program_ad_reduction_min_2x3_axis_0_q_0_0_direct_rule"
    assert min_rule.jvp_rule is not None
    assert min_rule.vjp_rule is not None
    column_cotangent = np.array([0.5, -1.0, 2.0], dtype=np.float64)
    _assert_allclose(min_rule.value_fn(values), np.min(matrix, axis=0))
    _assert_allclose(min_rule.jvp_rule(values, tangent), [tangent[3], tangent[1], tangent[5]])
    _assert_allclose(
        min_rule.vjp_rule(values, column_cotangent),
        np.array([0.0, -1.0, 0.0, 0.5, 0.0, 2.0], dtype=np.float64),
    )

    median_rule = program_ad_reduction_median_derivative_rule((2, 3), axis=1)
    assert median_rule.name == "program_ad_reduction_median_2x3_axis_1_q_0_5_direct_rule"
    assert median_rule.jvp_rule is not None
    assert median_rule.vjp_rule is not None
    _assert_allclose(median_rule.value_fn(values), np.median(matrix, axis=1))
    _assert_allclose(
        median_rule.jvp_rule(values, tangent),
        np.array([tangent[2], tangent[3]], dtype=np.float64),
    )
    _assert_allclose(
        median_rule.vjp_rule(values, row_cotangent),
        np.array([0.0, 0.0, 1.2, -0.4, 0.0, 0.0], dtype=np.float64),
    )

    quantile_rule = program_ad_reduction_quantile_derivative_rule((2, 3), q=0.25, axis=1)
    assert quantile_rule.name == "program_ad_reduction_quantile_2x3_axis_1_q_0_25_direct_rule"
    assert quantile_rule.jvp_rule is not None
    assert quantile_rule.vjp_rule is not None
    _assert_allclose(quantile_rule.value_fn(values), np.quantile(matrix, 0.25, axis=1))
    _assert_allclose(
        quantile_rule.jvp_rule(values, tangent),
        np.array([0.5 * tangent[1] + 0.5 * tangent[2], 0.5 * tangent[5] + 0.5 * tangent[3]]),
    )
    _assert_allclose(
        quantile_rule.vjp_rule(values, row_cotangent),
        np.array([0.0, 0.6, 0.6, -0.2, 0.0, -0.2], dtype=np.float64),
    )

    percentile_rule = program_ad_reduction_percentile_derivative_rule((2, 3), q=75.0, axis=0)
    assert percentile_rule.name == (
        "program_ad_reduction_percentile_2x3_axis_0_q_0_75_direct_rule"
    )
    assert percentile_rule.vjp_rule is not None
    _assert_allclose(
        percentile_rule.value_fn(values),
        np.percentile(matrix, 75.0, axis=0),
    )
    _assert_allclose(
        percentile_rule.vjp_rule(values, column_cotangent),
        np.array([0.375, -0.25, 1.5, 0.125, -0.75, 0.5], dtype=np.float64),
    )

    with pytest.raises(ValueError, match="strictly ordered"):
        program_ad_reduction_median_derivative_rule((3,)).value_fn(
            np.array([1.0, 1.0, 2.0], dtype=np.float64)
        )


def test_program_ad_reduction_primitives_expose_direct_value_jvp_kernels() -> None:
    """Flat reduction primitive contracts should expose exact direct value/JVP rules."""

    values = np.array([2.0, 0.0, -3.0, 4.0], dtype=np.float64)
    tangent = np.array([0.5, -1.0, 0.25, 2.0], dtype=np.float64)

    sum_rule = custom_derivative_rule_for(
        PrimitiveIdentity("scpn.program_ad.reduction", "sum", "1")
    )
    prod_rule = custom_derivative_rule_for(
        PrimitiveIdentity("scpn.program_ad.reduction", "prod", "1")
    )
    mean_rule = custom_derivative_rule_for(
        PrimitiveIdentity("scpn.program_ad.reduction", "mean", "1")
    )

    assert sum_rule.name == "program_ad_reduction_sum_direct_rule"
    assert prod_rule.name == "program_ad_reduction_prod_direct_rule"
    assert mean_rule.name == "program_ad_reduction_mean_direct_rule"
    assert sum_rule.jvp_rule is not None
    assert prod_rule.jvp_rule is not None
    assert mean_rule.jvp_rule is not None
    assert sum_rule.vjp_rule is not None
    assert prod_rule.vjp_rule is not None
    assert mean_rule.vjp_rule is not None

    _assert_allclose(sum_rule.value_fn(values), [np.sum(values)])
    _assert_allclose(sum_rule.jvp_rule(values, tangent), [np.sum(tangent)])
    _assert_allclose(sum_rule.vjp_rule(values, np.array([1.75])), np.full(4, 1.75))

    expected_prod_jvp = np.array(
        [
            tangent[0] * values[1] * values[2] * values[3]
            + values[0] * tangent[1] * values[2] * values[3]
            + values[0] * values[1] * tangent[2] * values[3]
            + values[0] * values[1] * values[2] * tangent[3]
        ],
        dtype=np.float64,
    )
    _assert_allclose(prod_rule.value_fn(values), [np.prod(values)])
    _assert_allclose(
        prod_rule.jvp_rule(values, tangent),
        expected_prod_jvp,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    _assert_allclose(
        prod_rule.vjp_rule(values, np.array([-2.0])),
        -2.0
        * np.array(
            [
                values[1] * values[2] * values[3],
                values[0] * values[2] * values[3],
                values[0] * values[1] * values[3],
                values[0] * values[1] * values[2],
            ],
            dtype=np.float64,
        ),
        rtol=1.0e-12,
        atol=1.0e-12,
    )

    _assert_allclose(mean_rule.value_fn(values), [np.mean(values)])
    _assert_allclose(mean_rule.jvp_rule(values, tangent), [np.mean(tangent)])
    _assert_allclose(mean_rule.vjp_rule(values, np.array([2.0])), np.full(4, 0.5))


def test_program_ad_product_reductions_match_product_rule_adjoint() -> None:
    """Program AD product reductions should preserve exact product-rule adjoints."""

    def objective(values: Any) -> object:
        matrix = np.reshape(values, (2, 3))
        row_products = np.prod(matrix, axis=1)
        return np.prod(values[:3]) + row_products[0] - 2.0 * row_products[1]

    result = whole_program_value_and_grad(
        objective,
        np.array([2.0, -3.0, 4.0, 5.0, -2.0, 0.5], dtype=np.float64),
        parameters=(
            Parameter("x0"),
            Parameter("x1"),
            Parameter("x2"),
            Parameter("x3"),
            Parameter("x4"),
            Parameter("x5"),
        ),
    )

    assert result.value == pytest.approx(-38.0)
    _assert_allclose(
        result.gradient,
        [-24.0, 16.0, -12.0, 2.0, -5.0, 20.0],
        atol=1.0e-12,
    )
    _assert_allclose(program_adjoint_gradient(result), result.gradient, atol=1.0e-12)


def test_program_ad_product_reduction_methods_handle_zero_factor() -> None:
    """Trace-array prod methods should handle single-zero factors without finite differences."""

    def objective(values: Any) -> object:
        matrix = np.reshape(values, (2, 2))
        return matrix.prod(axis=0)[0] + matrix.prod()

    result = whole_program_value_and_grad(
        objective,
        np.array([0.0, 2.0, 3.0, -4.0], dtype=np.float64),
        parameters=(
            Parameter("x00"),
            Parameter("x01"),
            Parameter("x10"),
            Parameter("x11"),
        ),
    )

    assert result.value == pytest.approx(0.0)
    _assert_allclose(result.gradient, [-21.0, 0.0, 0.0, 0.0], atol=1.0e-12)


def test_program_ad_variance_and_std_reductions_match_analytic_gradients() -> None:
    """Program AD variance and standard deviation should use exact differentiable reductions."""

    def objective(values: Any) -> object:
        matrix = np.reshape(values, (2, 2))
        return np.var(values) + matrix.var(axis=0)[1] + np.std(values[:2])

    result = whole_program_value_and_grad(
        objective,
        np.array([1.0, 3.0, 5.0, 7.0], dtype=np.float64),
        parameters=(
            Parameter("x0"),
            Parameter("x1"),
            Parameter("x2"),
            Parameter("x3"),
        ),
    )

    assert result.value == pytest.approx(10.0)
    _assert_allclose(
        result.gradient,
        [-2.0, -2.0, 0.5, 3.5],
        atol=1.0e-12,
    )


def test_program_ad_variance_and_std_reject_invalid_ddof() -> None:
    """Program AD variance/std should fail closed on unsupported or singular ddof."""

    with pytest.raises(ValueError, match="integer ddof"):
        whole_program_value_and_grad(
            lambda values: np.var(values, ddof=0.5),
            np.array([1.0, 2.0], dtype=np.float64),
        )
    with pytest.raises(ValueError, match="ddof must leave"):
        whole_program_value_and_grad(
            lambda values: np.std(values, ddof=2),
            np.array([1.0, 2.0], dtype=np.float64),
        )


def test_program_ad_axis_norms_match_euclidean_adjoint() -> None:
    """Program AD axis-aware Euclidean norms should replay exact vector adjoints."""

    row_weights = np.array([1.25, -0.5], dtype=np.float64)
    column_weights = np.array([0.75, -1.5, 0.25], dtype=np.float64)

    def objective(values: Any) -> object:
        matrix = np.reshape(values, (2, 3))
        row_norms = np.linalg.norm(matrix, 2, axis=1)
        column_norms = np.linalg.norm(matrix, None, 0)
        flat_norm = np.linalg.norm(values)
        return (
            np.sum(row_norms * row_weights)
            + np.sum(column_norms * column_weights)
            + 0.125 * flat_norm
        )

    values = np.array([1.0, 2.0, 2.0, 4.0, -1.0, 2.0], dtype=np.float64)
    result = whole_program_value_and_grad(
        objective,
        values,
        parameters=tuple(Parameter(f"x{index}") for index in range(values.size)),
    )

    matrix = values.reshape(2, 3)
    expected = np.zeros_like(matrix)
    expected += row_weights[:, None] * matrix / np.linalg.norm(matrix, axis=1)[:, None]
    expected += column_weights[None, :] * matrix / np.linalg.norm(matrix, axis=0)[None, :]
    expected += 0.125 * matrix / np.linalg.norm(values)

    assert result.value == pytest.approx(float(cast(Any, objective(values))))
    _assert_allclose(result.gradient, expected.reshape(-1), atol=1.0e-12)
    _assert_allclose(program_adjoint_gradient(result), expected.reshape(-1), atol=1.0e-12)


def test_program_ad_axis_norms_fail_closed_on_unsupported_contracts() -> None:
    """Program AD axis norms should reject non-Euclidean, dynamic, and singular contracts."""

    with pytest.raises(ValueError, match="Euclidean norm"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.linalg.norm(np.reshape(values, (2, 2)), ord=1, axis=1)),
            np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        )
    with pytest.raises(ValueError, match="axis must be a static integer"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.linalg.norm(np.reshape(values, (2, 2)), axis=True)),
            np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        )
    with pytest.raises(ValueError, match="requires non-zero Euclidean norms"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.linalg.norm(np.reshape(values, (2, 2)), axis=1)),
            np.array([0.0, 0.0, 3.0, 4.0], dtype=np.float64),
        )


def test_program_ad_frobenius_matrix_norms_match_exact_adjoint() -> None:
    """Program AD Frobenius matrix norms should replay exact static two-axis adjoints."""

    batch_weights = np.array([0.75, -1.25], dtype=np.float64)

    def objective(values: Any) -> object:
        tensor = np.reshape(values, (2, 2, 3))
        batched_norms = np.linalg.norm(tensor, "fro", axis=(1, 2))
        leading_matrix_norm = np.linalg.norm(tensor[0], None, axis=(-2, -1))
        return np.sum(batched_norms * batch_weights) + 0.5 * leading_matrix_norm

    values = np.array(
        [1.0, -2.0, 2.0, 0.5, -1.5, 2.5, 3.0, -1.0, 4.0, -2.0, 0.75, 1.25],
        dtype=np.float64,
    )
    result = whole_program_value_and_grad(
        objective,
        values,
        parameters=tuple(Parameter(f"x{index}") for index in range(values.size)),
    )

    tensor = values.reshape(2, 2, 3)
    norms = np.linalg.norm(tensor, ord="fro", axis=(1, 2))
    expected = batch_weights[:, None, None] * tensor / norms[:, None, None]
    expected[0] += 0.5 * tensor[0] / norms[0]

    assert result.value == pytest.approx(float(cast(Any, objective(values))))
    _assert_allclose(result.gradient, expected.reshape(-1), atol=1.0e-12)
    _assert_allclose(program_adjoint_gradient(result), expected.reshape(-1), atol=1.0e-12)


def test_program_ad_frobenius_matrix_norms_fail_closed_on_unsupported_contracts() -> None:
    """Program AD Frobenius norms should reject unsupported matrix-norm boundaries."""

    with pytest.raises(ValueError, match="matrix norms support only Frobenius"):
        whole_program_value_and_grad(
            lambda values: np.linalg.norm(np.reshape(values, (2, 2)), ord=1, axis=(0, 1)),
            np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        )
    with pytest.raises(ValueError, match="axes must be distinct"):
        whole_program_value_and_grad(
            lambda values: np.linalg.norm(np.reshape(values, (2, 2)), ord="fro", axis=(1, 1)),
            np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        )
    with pytest.raises(ValueError, match="requires non-zero Frobenius norms"):
        whole_program_value_and_grad(
            lambda values: np.linalg.norm(np.reshape(values, (2, 2)), ord="fro", axis=(0, 1)),
            np.zeros(4, dtype=np.float64),
        )


def test_program_ad_cumulative_sum_matches_prefix_adjoint() -> None:
    """Program AD cumulative sums should accumulate prefix adjoints exactly."""

    def objective(values: Any) -> object:
        matrix = np.reshape(values, (2, 3))
        flat_prefix = np.cumsum(values)
        row_prefix = matrix.cumsum(axis=1)
        return flat_prefix[3] + row_prefix[1, 2] - 2.0 * row_prefix[0, 1]

    result = whole_program_value_and_grad(
        objective,
        np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64),
        parameters=(
            Parameter("x0"),
            Parameter("x1"),
            Parameter("x2"),
            Parameter("x3"),
            Parameter("x4"),
            Parameter("x5"),
        ),
    )

    assert result.value == pytest.approx(19.0)
    _assert_allclose(
        result.gradient,
        [-1.0, -1.0, 1.0, 2.0, 1.0, 1.0],
        atol=1.0e-12,
    )
    _assert_allclose(program_adjoint_gradient(result), result.gradient, atol=1.0e-12)


def test_program_ad_cumulative_product_matches_prefix_product_adjoint() -> None:
    """Program AD cumulative products should preserve exact product-rule adjoints."""

    def objective(values: Any) -> object:
        matrix = np.reshape(values, (2, 3))
        flat_prefix = np.cumprod(values)
        row_prefix = matrix.cumprod(axis=1)
        return flat_prefix[2] + row_prefix[1, 2] - row_prefix[0, 1]

    result = whole_program_value_and_grad(
        objective,
        np.array([2.0, -3.0, 4.0, 5.0, -2.0, 0.5], dtype=np.float64),
        parameters=(
            Parameter("x0"),
            Parameter("x1"),
            Parameter("x2"),
            Parameter("x3"),
            Parameter("x4"),
            Parameter("x5"),
        ),
    )

    assert result.value == pytest.approx(-23.0)
    _assert_allclose(
        result.gradient,
        [-9.0, 6.0, -6.0, -1.0, 2.5, -10.0],
        atol=1.0e-12,
    )
    _assert_allclose(program_adjoint_gradient(result), result.gradient, atol=1.0e-12)


def test_program_ad_cumulative_product_method_handles_zero_factor() -> None:
    """Trace-array cumulative product methods should differentiate single-zero prefixes."""

    def objective(values: Any) -> object:
        matrix = np.reshape(values, (2, 2))
        return np.cumprod(values)[3] + matrix.cumprod(axis=1)[0, 1]

    result = whole_program_value_and_grad(
        objective,
        np.array([0.0, 2.0, 3.0, -4.0], dtype=np.float64),
        parameters=(
            Parameter("x00"),
            Parameter("x01"),
            Parameter("x10"),
            Parameter("x11"),
        ),
    )

    assert result.value == pytest.approx(0.0)
    _assert_allclose(result.gradient, [-22.0, 0.0, 0.0, 0.0], atol=1.0e-12)
