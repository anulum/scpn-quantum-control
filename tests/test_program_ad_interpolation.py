# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Program AD interpolation tests
"""Tests for Program AD static interpolation semantics and registry contracts."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control import differentiable as differentiable_facade
from scpn_quantum_control import program_ad_interpolation_primitives
from scpn_quantum_control.differentiable import (
    Parameter,
    PrimitiveIdentity,
    primitive_contract_for,
    program_ad_interpolation_interp_derivative_rule,
    program_adjoint_gradient,
    whole_program_value_and_grad,
)
from scpn_quantum_control.program_ad_interpolation_primitives import (
    program_ad_interpolation_interp_derivative_rule as module_interp_derivative_rule,
)

FloatArray = NDArray[np.float64]


def _assert_allclose(
    actual: object, expected: object, *, rtol: float = 1.0e-7, atol: float = 0.0
) -> None:
    """Assert NumPy closeness across dynamically typed Program AD result payloads."""

    cast(Any, np.testing.assert_allclose)(actual, expected, rtol=rtol, atol=atol)


def test_program_ad_interp_matches_static_grid_piecewise_adjoint() -> None:
    """Program AD np.interp should replay exact static-grid piecewise adjoints."""

    grid = np.array([0.0, 1.0, 2.5, 4.0], dtype=np.float64)
    sample_weights = np.array([0.7, -1.1, 0.6], dtype=np.float64)
    static_weights = np.array([1.2, -0.8, 0.4], dtype=np.float64)
    boundary_weights = np.array([0.3, -0.7], dtype=np.float64)
    static_values = np.array([0.5, -0.25, 1.25, -1.5], dtype=np.float64)

    def objective(values: Any) -> object:
        samples = values[:3]
        controls = values[3:]
        dynamic_values = np.interp(samples, grid, controls)
        static_grid_values = np.interp(samples + 0.05, grid, static_values)
        boundary_values = np.interp(np.array([-0.25, 4.25], dtype=np.float64), grid, controls)
        return (
            np.sum(dynamic_values * sample_weights)
            + 0.2 * np.sum(static_grid_values * static_weights)
            + 0.13 * np.sum(boundary_values * boundary_weights)
        )

    values = np.array([0.4, 1.8, 3.2, -1.0, 2.0, 0.5, 3.0], dtype=np.float64)
    result = whole_program_value_and_grad(
        objective,
        values,
        parameters=tuple(Parameter(f"x{index}") for index in range(values.size)),
    )

    expected = np.zeros_like(values)
    samples = values[:3]
    controls = values[3:]
    for sample_index, sample in enumerate(samples):
        segment = int(np.searchsorted(grid, sample, side="right") - 1)
        width = grid[segment + 1] - grid[segment]
        position = (sample - grid[segment]) / width
        expected[sample_index] += (
            sample_weights[sample_index] * (controls[segment + 1] - controls[segment]) / width
        )
        expected[3 + segment] += sample_weights[sample_index] * (1.0 - position)
        expected[3 + segment + 1] += sample_weights[sample_index] * position

        static_sample = sample + 0.05
        static_segment = int(np.searchsorted(grid, static_sample, side="right") - 1)
        static_width = grid[static_segment + 1] - grid[static_segment]
        expected[sample_index] += (
            0.2
            * static_weights[sample_index]
            * (static_values[static_segment + 1] - static_values[static_segment])
            / static_width
        )
    expected[3] += 0.13 * boundary_weights[0]
    expected[6] += 0.13 * boundary_weights[1]

    assert result.value == pytest.approx(float(cast(Any, objective(values))))
    _assert_allclose(result.gradient, expected, rtol=1.0e-12, atol=1.0e-12)
    _assert_allclose(program_adjoint_gradient(result), expected, atol=1.0e-12)


def test_program_ad_interp_fails_closed_invalid_static_contracts() -> None:
    """Program AD np.interp should reject dynamic grids and singular piecewise cases."""

    with pytest.raises(ValueError, match="xp grid must be static"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.interp(np.array([0.5]), values[:3], values[3:])),
            np.array([0.0, 1.0, 2.0, -1.0, 0.5, 2.0], dtype=np.float64),
        )

    with pytest.raises(ValueError, match="strictly increasing"):
        whole_program_value_and_grad(
            lambda values: np.sum(
                np.interp(values[:1], np.array([0.0, 1.0, 1.0]), np.array([0.5, 1.0, 1.5]))
            ),
            np.array([0.5], dtype=np.float64),
        )

    with pytest.raises(ValueError, match="avoid grid knots"):
        whole_program_value_and_grad(
            lambda values: np.sum(
                np.interp(values[:1], np.array([0.0, 1.0, 2.0]), np.array([0.5, 1.0, 1.5]))
            ),
            np.array([1.0], dtype=np.float64),
        )

    with pytest.raises(ValueError, match="period is not supported"):
        whole_program_value_and_grad(
            lambda values: np.sum(
                np.interp(
                    values[:1],
                    np.array([0.0, 1.0, 2.0]),
                    np.array([0.5, 1.0, 1.5]),
                    period=2.0,
                )
            ),
            np.array([0.5], dtype=np.float64),
        )

    with pytest.raises(ValueError, match="fp values must match"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.interp(values[:1], np.array([0.0, 1.0, 2.0]), values[1:])),
            np.array([0.5, -1.0, 1.0], dtype=np.float64),
        )

    with pytest.raises(ValueError, match="left boundary must be static"):
        whole_program_value_and_grad(
            lambda values: np.sum(
                np.interp(
                    np.array([-0.5], dtype=np.float64),
                    np.array([0.0, 1.0, 2.0]),
                    values[:3],
                    left=values[3],
                )
            ),
            np.array([-1.0, 0.5, 2.0, 3.0], dtype=np.float64),
        )


def test_program_ad_interp_primitive_contract_and_direct_rule() -> None:
    """Static interpolation contracts should expose exact piecewise JVP/VJP rules."""

    samples = np.array([0.25, 1.75, 4.5], dtype=np.float64)
    grid = np.array([0.0, 1.0, 2.5, 4.0], dtype=np.float64)
    values = np.array([-1.0, 2.0, 0.5, 3.0], dtype=np.float64)
    tangent_samples = np.array([0.5, -0.25, 1.0], dtype=np.float64)
    tangent_values = np.array([1.25, -0.75, 0.5, -1.5], dtype=np.float64)
    cotangent = np.array([1.5, -0.5, 2.0], dtype=np.float64)
    contract = primitive_contract_for("scpn.program_ad.interpolation:interp")

    assert contract.identity == PrimitiveIdentity("scpn.program_ad.interpolation", "interp", "1")
    assert contract.nondifferentiable_policy == "program_ad_trace_exact_fail_closed"
    assert contract.effect == "pure"
    assert contract.lowering_metadata["mlir_op"] == "scpn_diff.interpolation.interp"
    assert (
        contract.lowering_metadata["static_derivative_factory"]
        == "program_ad_interpolation_interp_derivative_rule"
    )
    assert contract.lowering_metadata["rust"] == (
        "available: bounded compact Program AD Rust value+gradient replay"
    )
    assert contract.lowering_metadata["rust_backend"] == "rust_pyo3"
    assert contract.lowering_metadata["rust_backend_signature"] == (
        "sample_shape:ranked_tensor_shape;xp_grid;fp_shape;left_right_period"
    )
    assert (
        contract.lowering_metadata["rust_backend_functions"]
        == "program_ad_effect_ir_interpret_value_and_gradient"
    )
    assert contract.lowering_metadata["static_signature"] == (
        "sample_shape:ranked_tensor_shape;xp_grid;fp_shape;left_right_period"
    )
    assert (
        contract.lowering_metadata["nondifferentiable_boundary"]
        == "static_grid_knot_and_period_boundary"
    )
    assert contract.lowering_metadata["nondifferentiable_boundary_policy"] == "fail_closed"
    assert contract.shape_rule is not None
    assert contract.shape_rule((samples, grid, values, None, None, None)) == samples.shape
    assert contract.dtype_rule is not None
    assert contract.dtype_rule((samples, grid, values, None, None, None)) == "float64"
    assert contract.static_argument_rule is not None
    assert contract.static_argument_rule((samples, grid, values, None, None, None)) == (
        samples.shape,
        ("xp", (4,), (0.0, 1.0, 2.5, 4.0)),
        (4,),
        None,
        None,
        None,
    )

    rule = program_ad_interpolation_interp_derivative_rule(samples.shape, grid, values.shape)
    assert rule.name == "program_ad_interpolation_interp_x3_grid4_static_direct_rule"
    assert rule.jvp_rule is not None
    assert rule.vjp_rule is not None
    source = np.concatenate([samples, values])
    tangent = np.concatenate([tangent_samples, tangent_values])
    _assert_allclose(rule.value_fn(source), np.interp(samples, grid, values))

    expected_jvp = np.array(
        [
            3.0 * tangent_samples[0] + 0.75 * tangent_values[0] + 0.25 * tangent_values[1],
            -1.0 * tangent_samples[1] + 0.5 * tangent_values[1] + 0.5 * tangent_values[2],
            tangent_values[-1],
        ],
        dtype=np.float64,
    )
    _assert_allclose(rule.jvp_rule(source, tangent), expected_jvp)

    expected_vjp = np.zeros_like(source)
    expected_vjp[0] += cotangent[0] * 3.0
    expected_vjp[3] += cotangent[0] * 0.75
    expected_vjp[4] += cotangent[0] * 0.25
    expected_vjp[1] += cotangent[1] * -1.0
    expected_vjp[4] += cotangent[1] * 0.5
    expected_vjp[5] += cotangent[1] * 0.5
    expected_vjp[6] += cotangent[2]
    _assert_allclose(rule.vjp_rule(source, cotangent), expected_vjp)


def test_program_ad_interp_direct_rule_is_exposed_from_extracted_module() -> None:
    """The interpolation facade should delegate fixed-grid direct rules to the module."""

    samples = np.array([0.25, 1.75], dtype=np.float64)
    grid = np.array([0.0, 1.0, 2.5], dtype=np.float64)
    values = np.array([-1.0, 2.0, 0.5], dtype=np.float64)
    source = np.concatenate([samples, values])
    facade_rule = program_ad_interpolation_interp_derivative_rule(
        samples.shape, grid, values.shape
    )
    module_rule = module_interp_derivative_rule(samples.shape, grid, values.shape)

    assert facade_rule.name == module_rule.name
    _assert_allclose(facade_rule.value_fn(source), module_rule.value_fn(source))


def test_program_ad_interp_contract_helpers_are_exposed_from_extracted_module() -> None:
    """The interpolation facade should reuse the extracted registry-contract helpers."""

    facade_exports = vars(differentiable_facade)
    assert (
        facade_exports["_register_program_ad_interpolation_primitive_contracts"]
        is program_ad_interpolation_primitives._register_program_ad_interpolation_primitive_contracts
    )
    assert (
        facade_exports["_require_program_ad_interpolation_contract"]
        is program_ad_interpolation_primitives._require_program_ad_interpolation_contract
    )
    assert program_ad_interpolation_interp_derivative_rule is module_interp_derivative_rule


def test_program_ad_interp_direct_rule_rejects_invalid_static_boundaries() -> None:
    """Extracted interpolation direct rules should fail closed for invalid signatures."""

    trace_value = type("TraceADScalar", (), {})()
    grid = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    values = np.array([-1.0, 2.0, 0.5], dtype=np.float64)

    with pytest.raises(ValueError, match="xp grid must be static"):
        module_interp_derivative_rule((1,), trace_value, values.shape)
    with pytest.raises(ValueError, match="one-dimensional"):
        module_interp_derivative_rule((1,), np.array([[0.0, 1.0]], dtype=np.float64), (2,))
    with pytest.raises(ValueError, match="at least two samples"):
        module_interp_derivative_rule((1,), np.array([0.0], dtype=np.float64), (1,))
    with pytest.raises(ValueError, match="finite values"):
        module_interp_derivative_rule((1,), np.array([0.0, np.inf], dtype=np.float64), (2,))
    with pytest.raises(ValueError, match="left boundary must be static"):
        module_interp_derivative_rule((1,), grid, values.shape, left=trace_value)
    with pytest.raises(ValueError, match="left must be finite"):
        module_interp_derivative_rule((1,), grid, values.shape, left=np.nan)
    with pytest.raises(ValueError, match="right must be finite"):
        module_interp_derivative_rule((1,), grid, values.shape, right=np.inf)
    with pytest.raises(ValueError, match="does not support period"):
        module_interp_derivative_rule((1,), grid, values.shape, period=2.0)
    with pytest.raises(ValueError, match="fp shape to match xp"):
        module_interp_derivative_rule((1,), grid, (2,))

    rule = module_interp_derivative_rule((2,), grid, values.shape, left=-5.0, right=7.0)
    assert rule.jvp_rule is not None
    assert rule.vjp_rule is not None
    boundary_source = np.array([-0.5, 2.5, -1.0, 2.0, 0.5], dtype=np.float64)
    boundary_tangent = np.array([1.0, -2.0, 0.25, -0.5, 1.5], dtype=np.float64)
    _assert_allclose(rule.value_fn(boundary_source), np.array([-5.0, 7.0], dtype=np.float64))
    _assert_allclose(rule.jvp_rule(boundary_source, boundary_tangent), np.zeros(2))
    _assert_allclose(rule.vjp_rule(boundary_source, np.array([3.0, -4.0])), np.zeros(5))

    default_boundary_rule = module_interp_derivative_rule((2,), grid, values.shape)
    assert default_boundary_rule.vjp_rule is not None
    default_boundary_source = np.array([-0.5, 2.5, -1.0, 2.0, 0.5], dtype=np.float64)
    _assert_allclose(
        default_boundary_rule.value_fn(default_boundary_source),
        np.array([-1.0, 0.5], dtype=np.float64),
    )
    _assert_allclose(
        default_boundary_rule.vjp_rule(default_boundary_source, np.array([3.0, -4.0])),
        np.array([0.0, 0.0, 3.0, 0.0, -4.0], dtype=np.float64),
    )

    interior_rule = module_interp_derivative_rule((1,), grid, values.shape)
    assert interior_rule.vjp_rule is not None
    with pytest.raises(ValueError, match="4 values"):
        interior_rule.value_fn(np.array([0.5, -1.0, 2.0], dtype=np.float64))
    with pytest.raises(ValueError, match="samples must be finite"):
        interior_rule.value_fn(np.array([np.nan, -1.0, 2.0, 0.5], dtype=np.float64))
    with pytest.raises(ValueError, match="avoid grid knots"):
        interior_rule.value_fn(np.array([1.0, -1.0, 2.0, 0.5], dtype=np.float64))
    with pytest.raises(ValueError, match="cotangent matching sample size"):
        interior_rule.vjp_rule(
            np.array([0.5, -1.0, 2.0, 0.5], dtype=np.float64),
            np.array([1.0, 2.0], dtype=np.float64),
        )


def test_program_ad_interp_batching_rule_maps_sample_batches() -> None:
    """Interpolation batching should map sample batches while keeping grid data static."""

    samples = np.array([[0.25, 1.75], [0.5, 3.25]], dtype=np.float64)
    grid = np.array([0.0, 1.0, 2.5, 4.0], dtype=np.float64)
    values = np.array([-1.0, 2.0, 0.5, 3.0], dtype=np.float64)
    contract = primitive_contract_for("scpn.program_ad.interpolation:interp")
    assert contract.batching_rule is not None

    def interp_fn(
        x: FloatArray,
        xp: FloatArray,
        fp: FloatArray,
        left: float | None,
        right: float | None,
        period: float | None,
    ) -> FloatArray:
        return np.interp(x, xp, fp, left=left, right=right, period=period)

    batched = contract.batching_rule(
        interp_fn,
        (samples, grid, values, None, None, None),
        (0, None, None, None, None, None),
        0,
    )
    expected = np.stack(
        [np.interp(samples[index], grid, values) for index in range(samples.shape[0])],
        axis=0,
    )
    _assert_allclose(batched, expected, rtol=1.0e-12, atol=1.0e-12)

    with pytest.raises(ValueError, match="keeps xp, fp, left, right, and period static"):
        contract.batching_rule(
            interp_fn,
            (samples, grid, values, None, None, None),
            (0, None, 0, None, None, None),
            0,
        )
