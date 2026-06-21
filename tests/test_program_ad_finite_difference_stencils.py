# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Program AD finite-difference stencil tests
"""Tests for Program AD finite-difference and static-gradient stencil semantics."""

from __future__ import annotations

from typing import Any, Literal, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control.differentiable import (
    Parameter,
    PrimitiveIdentity,
    primitive_contract_for,
    program_ad_stencil_gradient_derivative_rule,
    program_adjoint_gradient,
    whole_program_value_and_grad,
)
from scpn_quantum_control.program_ad_stencil_primitives import (
    program_ad_stencil_gradient_derivative_rule as module_stencil_gradient_derivative_rule,
)


def _assert_allclose(
    actual: object, expected: object, *, rtol: float = 1.0e-7, atol: float = 0.0
) -> None:
    """Assert NumPy closeness across dynamically typed Program AD result payloads."""

    cast(Any, np.testing.assert_allclose)(actual, expected, rtol=rtol, atol=atol)


def test_program_ad_stencil_gradient_contract_and_direct_rule() -> None:
    """Static gradient stencils should expose exact linear JVP/VJP rules."""

    matrix = np.array([[1.0, 2.0, 4.0], [0.5, -1.5, 3.0]], dtype=np.float64)
    tangent = np.array([[0.25, -0.5, 1.0], [1.5, -0.75, 0.5]], dtype=np.float64)
    x_grid = np.array([0.0, 0.25, 1.0], dtype=np.float64)
    cotangent = np.array([[1.5, -2.0, 0.75], [-0.25, 0.5, 1.25]], dtype=np.float64)
    contract = primitive_contract_for("scpn.program_ad.stencil:gradient")

    assert contract.identity == PrimitiveIdentity("scpn.program_ad.stencil", "gradient", "1")
    assert contract.nondifferentiable_policy == "program_ad_trace_exact_fail_closed"
    assert contract.effect == "pure"
    assert contract.lowering_metadata["mlir_op"] == "scpn_diff.stencil.gradient"
    assert (
        contract.lowering_metadata["static_derivative_factory"]
        == "program_ad_stencil_gradient_derivative_rule"
    )
    assert contract.lowering_metadata["static_signature"] == (
        "source_shape:ranked_tensor_shape;spacing_axis_edge_order"
    )
    assert (
        contract.lowering_metadata["nondifferentiable_boundary"]
        == "static_spacing_axis_edge_order"
    )
    assert contract.lowering_metadata["nondifferentiable_boundary_policy"] == "fail_closed"
    assert contract.shape_rule is not None
    assert contract.shape_rule((matrix, (x_grid,), 1, 1)) == matrix.shape
    assert contract.dtype_rule is not None
    assert contract.dtype_rule((matrix, (x_grid,), 1, 1)) == "float64"
    assert contract.static_argument_rule is not None
    assert contract.static_argument_rule((matrix, (x_grid,), 1, 1)) == (
        matrix.shape,
        (("coordinates", (3,), (0.0, 0.25, 1.0)),),
        (1,),
        1,
    )

    rule = program_ad_stencil_gradient_derivative_rule(matrix.shape, (x_grid,), axis=1)
    assert rule.name == "program_ad_stencil_gradient_2x3_axes_1_edge_1_direct_rule"
    assert rule.jvp_rule is not None
    assert rule.vjp_rule is not None
    _assert_allclose(
        rule.value_fn(matrix.reshape(-1)),
        np.gradient(matrix, x_grid, axis=1).reshape(-1),
    )
    _assert_allclose(
        rule.jvp_rule(matrix.reshape(-1), tangent.reshape(-1)),
        np.gradient(tangent, x_grid, axis=1).reshape(-1),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    expected_vjp = np.zeros(matrix.size, dtype=np.float64)
    for source_index in range(matrix.size):
        basis = np.zeros_like(matrix)
        basis.reshape(-1)[source_index] = 1.0
        expected_vjp[source_index] = float(np.sum(np.gradient(basis, x_grid, axis=1) * cotangent))
    _assert_allclose(
        rule.vjp_rule(matrix.reshape(-1), cotangent.reshape(-1)),
        expected_vjp,
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_program_ad_stencil_gradient_rule_is_exposed_from_extracted_module() -> None:
    """The stencil facade should delegate fixed-shape direct rules to the module."""

    source_shape = (2, 3)
    facade_rule = program_ad_stencil_gradient_derivative_rule(source_shape, axis=1)
    module_rule = module_stencil_gradient_derivative_rule(source_shape, axis=1)
    values = np.arange(6.0, dtype=np.float64)

    assert facade_rule.name == module_rule.name
    assert facade_rule.value_fn(values) == pytest.approx(module_rule.value_fn(values))


def test_program_ad_stencil_gradient_direct_rule_rejects_invalid_static_boundaries() -> None:
    """Static stencil direct rules should fail closed for unsupported signatures."""

    class InvalidArrayProtocol:
        def __array__(self, dtype: object = None, copy: object = None) -> NDArray[np.float64]:
            raise ValueError("invalid spacing")

    trace_spacing = type("TraceADArray", (), {})()
    scalar_edge_two_rule = module_stencil_gradient_derivative_rule((3,), edge_order=2)

    with pytest.raises(ValueError, match="edge_order"):
        module_stencil_gradient_derivative_rule((3,), edge_order=True)
    with pytest.raises(ValueError, match="axis out of bounds"):
        module_stencil_gradient_derivative_rule((3,), axis=-2)
    with pytest.raises(ValueError, match="axis must be a static integer"):
        module_stencil_gradient_derivative_rule((3,), axis=())
    with pytest.raises(ValueError, match="axis must be a static integer"):
        module_stencil_gradient_derivative_rule((3,), axis=(0, "x"))
    with pytest.raises(ValueError, match="spacing must be static real numeric"):
        module_stencil_gradient_derivative_rule((3,), (trace_spacing,), axis=0)
    with pytest.raises(ValueError, match="spacing count must match axes"):
        module_stencil_gradient_derivative_rule((2, 2), (trace_spacing,), axis=(0, 1))
    with pytest.raises(ValueError, match="spacing count must match axes"):
        module_stencil_gradient_derivative_rule((2, 2), ([1.0, 2.0],), axis=(0, 1))
    with pytest.raises(ValueError, match="spacing count must match axes"):
        module_stencil_gradient_derivative_rule((2, 2), (1.0, 1.0, 1.0), axis=(0, 1))
    with pytest.raises(ValueError, match="spacing count must match axes"):
        module_stencil_gradient_derivative_rule((2, 2), (InvalidArrayProtocol(),), axis=(0, 1))
    with pytest.raises(ValueError, match="spacing must be non-zero"):
        module_stencil_gradient_derivative_rule((3,), (0.0,), axis=0)
    with pytest.raises(ValueError, match="coordinates must match"):
        module_stencil_gradient_derivative_rule((3,), (np.array([0.0, 1.0]),), axis=0)
    with pytest.raises(ValueError, match="finite values"):
        module_stencil_gradient_derivative_rule((3,), (np.array([0.0, np.inf, 2.0]),), axis=0)
    with pytest.raises(ValueError, match="positive source dimensions"):
        module_stencil_gradient_derivative_rule((3, 0), axis=0)
    with pytest.raises(ValueError, match="3 values"):
        scalar_edge_two_rule.value_fn(np.array([1.0, 2.0], dtype=np.float64))
    assert scalar_edge_two_rule.vjp_rule is not None
    with pytest.raises(ValueError, match="3 cotangent values"):
        scalar_edge_two_rule.vjp_rule(
            np.array([1.0, 2.0, 4.0], dtype=np.float64),
            np.array([1.0, 2.0], dtype=np.float64),
        )

    expected_edge_two = np.gradient(np.array([1.0, 2.0, 4.0], dtype=np.float64), edge_order=2)
    _assert_allclose(
        scalar_edge_two_rule.value_fn(np.array([1.0, 2.0, 4.0], dtype=np.float64)),
        expected_edge_two,
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_program_ad_stencil_gradient_batching_rule_maps_outer_axes() -> None:
    """Gradient stencils should vmap only over non-differentiated batch axes."""

    cube = np.array(
        [
            [[1.0, 2.0, 4.0], [0.5, -1.5, 3.0]],
            [[-0.25, 0.75, 1.25], [2.5, -0.5, 0.0]],
        ],
        dtype=np.float64,
    )
    x_grid = np.array([0.0, 0.25, 1.0], dtype=np.float64)
    contract = primitive_contract_for("scpn.program_ad.stencil:gradient")
    assert contract.batching_rule is not None
    batching_rule = contract.batching_rule

    def gradient_fn(
        values: NDArray[np.float64],
        spacings: tuple[NDArray[np.float64], ...],
        axis: int,
        edge_order: Literal[1, 2],
    ) -> NDArray[np.float64]:
        return cast(
            NDArray[np.float64],
            np.gradient(values, *spacings, axis=axis, edge_order=edge_order),
        )

    batched = batching_rule(
        gradient_fn,
        (cube, (x_grid,), 2, 1),
        (0, None, None, None),
        0,
    )
    expected = np.stack(
        [np.gradient(cube[index], x_grid, axis=1) for index in range(cube.shape[0])],
        axis=0,
    )
    _assert_allclose(batched, expected, rtol=1.0e-12, atol=1.0e-12)

    with pytest.raises(ValueError, match="cannot batch over a differentiated axis"):
        batching_rule(
            gradient_fn,
            (cube, (x_grid,), 2, 1),
            (2, None, None, None),
            0,
        )


def test_program_ad_stencil_gradient_direct_rule_handles_multi_axis_edge_order_two() -> None:
    """Multi-axis second-order stencils should expose the exact transpose adjoint."""

    matrix = np.array(
        [[1.0, 2.5, 4.0], [0.25, -1.0, 3.5], [2.0, 0.75, -0.5]],
        dtype=np.float64,
    )
    tangent = np.array(
        [[0.5, -0.25, 1.5], [1.0, 0.75, -0.5], [-1.25, 0.25, 0.0]],
        dtype=np.float64,
    )
    row_grid = np.array([0.0, 0.5, 1.5], dtype=np.float64)
    col_grid = np.array([-1.0, 0.25, 2.0], dtype=np.float64)
    cotangent_components = (
        np.array([[1.0, -0.5, 0.25], [0.75, 1.25, -1.5], [0.5, -0.25, 2.0]]),
        np.array([[-0.75, 0.5, 1.5], [1.0, -1.25, 0.25], [0.0, 0.75, -0.5]]),
    )

    rule = program_ad_stencil_gradient_derivative_rule(
        matrix.shape,
        (row_grid, col_grid),
        axis=(0, 1),
        edge_order=2,
    )
    assert rule.jvp_rule is not None
    assert rule.vjp_rule is not None
    expected_value = np.concatenate(
        [
            component.reshape(-1)
            for component in np.gradient(matrix, row_grid, col_grid, edge_order=2)
        ]
    )
    expected_jvp = np.concatenate(
        [
            component.reshape(-1)
            for component in np.gradient(tangent, row_grid, col_grid, edge_order=2)
        ]
    )
    cotangent = np.concatenate([component.reshape(-1) for component in cotangent_components])
    expected_vjp = np.zeros(matrix.size, dtype=np.float64)
    for source_index in range(matrix.size):
        basis = np.zeros_like(matrix)
        basis.reshape(-1)[source_index] = 1.0
        basis_gradient = np.gradient(basis, row_grid, col_grid, edge_order=2)
        expected_vjp[source_index] = float(
            sum(
                np.sum(component * component_cotangent)
                for component, component_cotangent in zip(
                    basis_gradient,
                    cotangent_components,
                    strict=True,
                )
            )
        )

    _assert_allclose(rule.value_fn(matrix.reshape(-1)), expected_value)
    _assert_allclose(
        rule.jvp_rule(matrix.reshape(-1), tangent.reshape(-1)),
        expected_jvp,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    _assert_allclose(
        rule.vjp_rule(matrix.reshape(-1), cotangent),
        expected_vjp,
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_program_ad_finite_differences_match_linear_adjoint() -> None:
    """Program AD finite differences should preserve exact linear adjoints."""

    def objective(values: Any) -> object:
        matrix = np.reshape(values, (2, 3))
        first_order = np.diff(values)
        second_order_rows = np.diff(matrix, n=2, axis=1)
        return first_order[2] - 2.0 * first_order[4] + second_order_rows[0, 0]

    result = whole_program_value_and_grad(
        objective,
        np.array([1.0, 3.0, 6.0, 10.0, 15.0, 21.0], dtype=np.float64),
        parameters=(
            Parameter("x0"),
            Parameter("x1"),
            Parameter("x2"),
            Parameter("x3"),
            Parameter("x4"),
            Parameter("x5"),
        ),
    )

    assert result.value == pytest.approx(-7.0)
    _assert_allclose(
        result.gradient,
        [1.0, -2.0, 0.0, 1.0, 2.0, -2.0],
        atol=1.0e-12,
    )
    _assert_allclose(program_adjoint_gradient(result), result.gradient, atol=1.0e-12)


def test_program_ad_finite_differences_reject_boundary_extensions() -> None:
    """Program AD finite differences should fail closed for boundary-extension modes."""

    with pytest.raises(ValueError, match="non-negative integer n"):
        whole_program_value_and_grad(
            lambda values: np.diff(values, n=-1)[0],
            np.array([1.0, 2.0], dtype=np.float64),
        )
    with pytest.raises(ValueError, match="prepend/append"):
        whole_program_value_and_grad(
            lambda values: np.diff(values, prepend=0.0)[0],
            np.array([1.0, 2.0], dtype=np.float64),
        )


def test_program_ad_gradient_matches_static_spacing_adjoint() -> None:
    """Program AD np.gradient should replay exact static finite-difference adjoints."""

    row_grid = np.array([0.0, 0.5, 1.5], dtype=np.float64)
    column_grid = np.array([-0.25, 0.0, 0.75, 1.25], dtype=np.float64)
    row_weights = np.linspace(-1.5, 2.0, 12, dtype=np.float64).reshape(3, 4)
    column_weights = np.linspace(0.5, -2.5, 12, dtype=np.float64).reshape(3, 4)
    flat_weights = np.array([0.25, -0.5, 1.0, -1.5], dtype=np.float64)

    def objective(values: Any) -> object:
        matrix = np.reshape(values[:12], (3, 4))
        row_gradient, column_gradient = np.gradient(
            matrix,
            row_grid,
            column_grid,
            axis=(0, 1),
            edge_order=2,
        )
        flat_gradient = np.gradient(values[12:], 0.5, edge_order=1)
        return (
            np.sum(row_gradient * row_weights)
            + np.sum(column_gradient * column_weights)
            + np.sum(flat_gradient * flat_weights)
        )

    values = np.array(
        [
            1.0,
            -2.0,
            0.5,
            3.0,
            -1.5,
            2.0,
            4.0,
            -0.25,
            0.75,
            -3.0,
            1.5,
            2.5,
            0.5,
            -1.0,
            2.0,
            4.0,
        ],
        dtype=np.float64,
    )
    result = whole_program_value_and_grad(
        objective,
        values,
        parameters=tuple(Parameter(f"x{index}") for index in range(values.size)),
    )

    expected = np.zeros_like(values)
    for source_index in range(12):
        basis = np.zeros((3, 4), dtype=np.float64)
        basis.reshape(-1)[source_index] = 1.0
        basis_row_gradient, basis_column_gradient = np.gradient(
            basis,
            row_grid,
            column_grid,
            axis=(0, 1),
            edge_order=2,
        )
        expected[source_index] = np.sum(basis_row_gradient * row_weights) + np.sum(
            basis_column_gradient * column_weights
        )
    for source_index in range(4):
        flat_basis = np.zeros(4, dtype=np.float64)
        flat_basis[source_index] = 1.0
        expected[12 + source_index] = np.sum(
            np.gradient(flat_basis, 0.5, edge_order=1) * flat_weights
        )

    assert result.value == pytest.approx(float(cast(Any, objective(values))))
    _assert_allclose(result.gradient, expected, rtol=1.0e-12, atol=1.0e-12)
    _assert_allclose(program_adjoint_gradient(result), expected, atol=1.0e-12)


def test_program_ad_gradient_fails_closed_invalid_static_contracts() -> None:
    """Program AD np.gradient should reject unsupported dynamic or singular grids."""

    with pytest.raises(ValueError, match="spacing must be static"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.gradient(values, values)),
            np.array([1.0, 2.0, 3.0], dtype=np.float64),
        )

    with pytest.raises(ValueError, match="edge_order must be 1 or 2"):
        whole_program_value_and_grad(
            lambda values: np.sum(cast(Any, np.gradient)(values, edge_order=3)),
            np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        )

    with pytest.raises(ValueError, match="axis must be a static integer"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.gradient(np.reshape(values, (2, 2)), axis=True)),
            np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        )

    with pytest.raises(ValueError, match="axes must be unique"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.gradient(np.reshape(values, (2, 2)), axis=(0, 0))[0]),
            np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        )

    with pytest.raises(ValueError, match="strictly monotonic"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.gradient(values, np.array([0.0, 1.0, 1.0]))),
            np.array([1.0, 2.0, 3.0], dtype=np.float64),
        )

    with pytest.raises(ValueError, match="at least 3 samples"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.gradient(values, edge_order=2)),
            np.array([1.0, 2.0], dtype=np.float64),
        )
