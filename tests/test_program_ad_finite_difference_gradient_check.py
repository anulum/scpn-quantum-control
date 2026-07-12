# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — program AD finite difference gradient check tests
# scpn-quantum-control -- Program AD primitive finite-difference gradient checks
"""Independent finite-difference cross-checks of Program AD derivative rules.

Every Program AD primitive family declares a closed-form forward (``jvp``) and
reverse (``vjp``) derivative rule. The per-family registry tests assert those
rules against hand-derived expected arrays; a hand-derivation error can leave a
rule and its expected array wrong in the same way. These tests add an
independent anchor: they perturb the primitive inputs and compare each analytic
rule against a central finite-difference Jacobian computed from the primitive's
own ``value_fn``.

The three-way agreement asserted for each primitive is:

* ``jvp`` Jacobian (assembled column by column from unit tangents) versus the
  finite-difference Jacobian;
* ``vjp`` Jacobian (assembled row by row from unit cotangents) versus the same
  finite-difference Jacobian;

which simultaneously catches a wrong forward rule, a wrong reverse rule, and any
forward/reverse inconsistency. Inputs are conditioned to keep each primitive at
a differentiable point (no reduction ties, interior clip/interp arguments,
symmetric perturbation directions for the symmetric-eigenvalue rules whose
derivative is defined only on the symmetric matrix manifold).
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control.differentiable_finite_difference import finite_difference_jacobian
from scpn_quantum_control.program_ad_cumulative_primitives import (
    program_ad_cumulative_cumprod_derivative_rule,
    program_ad_cumulative_cumsum_derivative_rule,
    program_ad_cumulative_diff_derivative_rule,
)
from scpn_quantum_control.program_ad_elementwise_primitives import (
    _program_ad_elementwise_derivative_rule,
    program_ad_elementwise_binary_derivative_rule,
)
from scpn_quantum_control.program_ad_interpolation_primitives import (
    program_ad_interpolation_interp_derivative_rule,
)
from scpn_quantum_control.program_ad_linalg_primitives import (
    program_ad_linalg_diag_derivative_rule,
    program_ad_linalg_diagflat_derivative_rule,
    program_ad_linalg_eigvals_derivative_rule,
    program_ad_linalg_eigvalsh_derivative_rule,
    program_ad_linalg_matrix_power_derivative_rule,
    program_ad_linalg_multi_dot_derivative_rule,
    program_ad_linalg_pinv_derivative_rule,
    program_ad_linalg_solve_derivative_rule,
    program_ad_linalg_svdvals_derivative_rule,
    program_ad_linalg_trace_derivative_rule,
)
from scpn_quantum_control.program_ad_product_primitives import (
    program_ad_product_einsum_derivative_rule,
    program_ad_product_inner_derivative_rule,
    program_ad_product_matmul_derivative_rule,
    program_ad_product_outer_derivative_rule,
    program_ad_product_tensordot_derivative_rule,
)
from scpn_quantum_control.program_ad_reduction_primitives import (
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
)
from scpn_quantum_control.program_ad_registry import CustomDerivativeRule
from scpn_quantum_control.program_ad_selection_primitives import (
    program_ad_selection_clip_derivative_rule,
    program_ad_selection_where_derivative_rule,
)
from scpn_quantum_control.program_ad_signal_primitives import (
    program_ad_signal_convolve_derivative_rule,
    program_ad_signal_correlate_derivative_rule,
)
from scpn_quantum_control.program_ad_stencil_primitives import (
    program_ad_stencil_gradient_derivative_rule,
)
from scpn_quantum_control.program_ad_trapezoid_primitives import (
    program_ad_reduction_trapezoid_derivative_rule,
)


def _flat(*arrays: object) -> NDArray[np.float64]:
    """Return the flat concatenation used by multi-operand primitive packing."""

    return np.concatenate([np.asarray(array, dtype=np.float64).reshape(-1) for array in arrays])


def _analytic_jvp_jacobian(
    rule: CustomDerivativeRule, values: NDArray[np.float64], out_size: int
) -> NDArray[np.float64]:
    """Assemble the forward-rule Jacobian one column per unit input tangent."""

    assert rule.jvp_rule is not None
    jacobian = np.zeros((out_size, values.size), dtype=np.float64)
    for column in range(values.size):
        tangent = np.zeros(values.size, dtype=np.float64)
        tangent[column] = 1.0
        jacobian[:, column] = np.asarray(rule.jvp_rule(values, tangent), dtype=np.float64).reshape(
            -1
        )
    return jacobian


def _analytic_vjp_jacobian(
    rule: CustomDerivativeRule, values: NDArray[np.float64], out_size: int
) -> NDArray[np.float64]:
    """Assemble the reverse-rule Jacobian one row per unit output cotangent."""

    assert rule.vjp_rule is not None
    jacobian = np.zeros((out_size, values.size), dtype=np.float64)
    for row in range(out_size):
        cotangent = np.zeros(out_size, dtype=np.float64)
        cotangent[row] = 1.0
        jacobian[row, :] = np.asarray(rule.vjp_rule(values, cotangent), dtype=np.float64).reshape(
            -1
        )
    return jacobian


def _assert_rule_matches_finite_difference(
    rule: CustomDerivativeRule,
    values: NDArray[np.float64],
    *,
    rtol: float = 1e-5,
    atol: float = 1e-6,
    step: float = 1e-6,
) -> None:
    """Assert forward and reverse rules match the unconstrained FD Jacobian."""

    values = np.asarray(values, dtype=np.float64).reshape(-1)
    out = np.asarray(rule.value_fn(values), dtype=np.float64).reshape(-1)
    finite_difference = finite_difference_jacobian(rule.value_fn, values, step=step)
    analytic_forward = _analytic_jvp_jacobian(rule, values, out.size)
    analytic_reverse = _analytic_vjp_jacobian(rule, values, out.size)
    np.testing.assert_allclose(analytic_forward, finite_difference, rtol=rtol, atol=atol)
    np.testing.assert_allclose(analytic_reverse, finite_difference, rtol=rtol, atol=atol)
    # Forward and reverse must agree to numerical exactness (transpose identity).
    np.testing.assert_allclose(analytic_forward, analytic_reverse, rtol=1e-9, atol=1e-9)


def _symmetric_directions(size: int) -> list[NDArray[np.float64]]:
    """Return flattened basis directions spanning symmetric ``size×size`` matrices."""

    directions: list[NDArray[np.float64]] = []
    for row in range(size):
        for column in range(row, size):
            direction = np.zeros((size, size), dtype=np.float64)
            direction[row, column] = 1.0
            direction[column, row] = 1.0
            directions.append(direction.reshape(-1))
    return directions


def _assert_symmetric_eigenvalue_rule(
    rule: CustomDerivativeRule,
    matrix: NDArray[np.float64],
    *,
    rtol: float = 1e-5,
    atol: float = 1e-6,
    step: float = 1e-6,
) -> None:
    """Assert an eigenvalue rule matches directional FD on symmetric directions.

    The symmetric-eigenvalue derivative is defined on the symmetric matrix
    manifold; an arbitrary entrywise perturbation both violates the rule's
    symmetric-tangent contract and can push a symmetric matrix off the real
    spectrum. Restricting the check to symmetric directions keeps every
    perturbation inside the differentiable domain while still spanning it.
    """

    values = np.asarray(matrix, dtype=np.float64).reshape(-1)
    assert rule.jvp_rule is not None
    assert rule.vjp_rule is not None
    out_size = np.asarray(rule.value_fn(values), dtype=np.float64).reshape(-1).size
    reverse = _analytic_vjp_jacobian(rule, values, out_size)
    for direction in _symmetric_directions(matrix.shape[0]):
        analytic = np.asarray(rule.jvp_rule(values, direction), dtype=np.float64).reshape(-1)
        plus = np.asarray(rule.value_fn(values + step * direction), dtype=np.float64).reshape(-1)
        minus = np.asarray(rule.value_fn(values - step * direction), dtype=np.float64).reshape(-1)
        finite_difference = (plus - minus) / (2.0 * step)
        np.testing.assert_allclose(analytic, finite_difference, rtol=rtol, atol=atol)
        # Reverse rule contracted with the same direction reproduces the JVP.
        np.testing.assert_allclose(reverse @ direction, analytic, rtol=1e-9, atol=1e-9)


_RNG = np.random.default_rng(20260701)


# --------------------------------------------------------------------------- #
# Elementwise unary
# --------------------------------------------------------------------------- #
_UNARY_INPUTS: dict[str, NDArray[np.float64]] = {
    "sin": np.array([0.2, -0.7, 1.1, 0.4]),
    "cos": np.array([0.2, -0.7, 1.1, 0.4]),
    "exp": np.array([0.2, -0.7, 0.9, 0.4]),
    "expm1": np.array([0.2, -0.7, 0.9, 0.4]),
    "log": np.array([0.3, 0.8, 1.7, 2.4]),
    "log1p": np.array([0.3, 0.8, 1.7, 2.4]),
    "sqrt": np.array([0.3, 0.8, 1.7, 2.4]),
    "tan": np.array([0.2, -0.7, 0.9, 0.4]),
    "tanh": np.array([0.2, -0.7, 1.1, 0.4]),
    "arcsin": np.array([0.1, -0.4, 0.6, 0.3]),
    "arccos": np.array([0.1, -0.4, 0.6, 0.3]),
    "reciprocal": np.array([0.3, -0.8, 1.7, 0.4]),
    "square": np.array([0.2, -0.7, 1.1, 0.4]),
    "abs": np.array([0.2, -0.7, 1.1, 0.4]),
    "negative": np.array([0.2, -0.7, 1.1, 0.4]),
}


@pytest.mark.parametrize("name", sorted(_UNARY_INPUTS))
def test_elementwise_unary_rule_matches_finite_difference(name: str) -> None:
    rule = _program_ad_elementwise_derivative_rule(name)
    _assert_rule_matches_finite_difference(rule, _UNARY_INPUTS[name])


# --------------------------------------------------------------------------- #
# Elementwise binary
# --------------------------------------------------------------------------- #
_BINARY_INPUTS: dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]] = {
    "add": (np.array([0.4, -0.6, 1.2, 0.3]), np.array([0.7, 0.2, -0.9, 1.1])),
    "subtract": (np.array([0.4, -0.6, 1.2, 0.3]), np.array([0.7, 0.2, -0.9, 1.1])),
    "multiply": (np.array([0.4, -0.6, 1.2, 0.3]), np.array([0.7, 0.2, -0.9, 1.1])),
    "divide": (np.array([0.4, -0.6, 1.2, 0.3]), np.array([0.7, 0.5, -0.9, 1.1])),
    "power": (np.array([0.4, 0.6, 1.2, 0.3]), np.array([0.7, 1.5, 0.9, 2.1])),
    "maximum": (np.array([0.4, -0.6, 1.2, 0.3]), np.array([0.7, 0.2, -0.9, 1.1])),
    "minimum": (np.array([0.4, -0.6, 1.2, 0.3]), np.array([0.7, 0.2, -0.9, 1.1])),
}


@pytest.mark.parametrize("name", sorted(_BINARY_INPUTS))
def test_elementwise_binary_rule_matches_finite_difference(name: str) -> None:
    left, right = _BINARY_INPUTS[name]
    rule = program_ad_elementwise_binary_derivative_rule(name, left.shape, right.shape)
    _assert_rule_matches_finite_difference(rule, _flat(left, right))


# --------------------------------------------------------------------------- #
# Cumulative
# --------------------------------------------------------------------------- #
def test_cumsum_rule_matches_finite_difference() -> None:
    rule = program_ad_cumulative_cumsum_derivative_rule((5,))
    _assert_rule_matches_finite_difference(rule, _RNG.normal(size=5))


def test_cumprod_rule_matches_finite_difference() -> None:
    rule = program_ad_cumulative_cumprod_derivative_rule((5,))
    _assert_rule_matches_finite_difference(rule, _RNG.uniform(0.5, 1.5, size=5))


def test_cumprod_axis_rule_matches_finite_difference() -> None:
    rule = program_ad_cumulative_cumprod_derivative_rule((3, 4), 1)
    _assert_rule_matches_finite_difference(rule, _RNG.uniform(0.5, 1.5, size=(3, 4)))


def test_diff_rule_matches_finite_difference() -> None:
    rule = program_ad_cumulative_diff_derivative_rule((6,))
    _assert_rule_matches_finite_difference(rule, _RNG.normal(size=6))


# --------------------------------------------------------------------------- #
# Reduction
# --------------------------------------------------------------------------- #
def test_sum_rule_matches_finite_difference() -> None:
    _assert_rule_matches_finite_difference(
        program_ad_reduction_sum_derivative_rule((6,)), _RNG.normal(size=6)
    )


def test_mean_rule_matches_finite_difference() -> None:
    _assert_rule_matches_finite_difference(
        program_ad_reduction_mean_derivative_rule((6,)), _RNG.normal(size=6)
    )


def test_var_rule_matches_finite_difference() -> None:
    _assert_rule_matches_finite_difference(
        program_ad_reduction_var_derivative_rule((6,)), _RNG.normal(size=6)
    )


def test_std_rule_matches_finite_difference() -> None:
    _assert_rule_matches_finite_difference(
        program_ad_reduction_std_derivative_rule((6,)), _RNG.normal(size=6) + 3.0
    )


def test_max_rule_matches_finite_difference() -> None:
    _assert_rule_matches_finite_difference(
        program_ad_reduction_max_derivative_rule((6,)),
        np.array([0.1, 0.9, 0.3, 0.7, 0.2, 0.5]),
    )


def test_min_rule_matches_finite_difference() -> None:
    _assert_rule_matches_finite_difference(
        program_ad_reduction_min_derivative_rule((6,)),
        np.array([0.4, 0.9, 0.1, 0.7, 0.6, 0.5]),
    )


def test_median_rule_matches_finite_difference() -> None:
    _assert_rule_matches_finite_difference(
        program_ad_reduction_median_derivative_rule((7,)),
        np.array([0.4, 0.9, 0.1, 0.7, 0.6, 0.2, 0.8]),
    )


def test_prod_rule_matches_finite_difference() -> None:
    _assert_rule_matches_finite_difference(
        program_ad_reduction_prod_derivative_rule((6,)), _RNG.uniform(0.5, 1.5, size=6)
    )


def test_quantile_rule_matches_finite_difference() -> None:
    rule = program_ad_reduction_quantile_derivative_rule((7,), q=0.4)
    _assert_rule_matches_finite_difference(rule, np.array([0.4, 0.9, 0.1, 0.7, 0.6, 0.2, 0.8]))


def test_percentile_rule_matches_finite_difference() -> None:
    rule = program_ad_reduction_percentile_derivative_rule((7,), q=40.0)
    _assert_rule_matches_finite_difference(rule, np.array([0.4, 0.9, 0.1, 0.7, 0.6, 0.2, 0.8]))


def test_trapezoid_rule_matches_finite_difference() -> None:
    _assert_rule_matches_finite_difference(
        program_ad_reduction_trapezoid_derivative_rule((6,)), _RNG.normal(size=6)
    )


# --------------------------------------------------------------------------- #
# Product / contraction
# --------------------------------------------------------------------------- #
def test_matmul_rule_matches_finite_difference() -> None:
    rule = program_ad_product_matmul_derivative_rule((2, 3), (3, 2))
    _assert_rule_matches_finite_difference(
        rule, _flat(_RNG.normal(size=(2, 3)), _RNG.normal(size=(3, 2)))
    )


def test_outer_rule_matches_finite_difference() -> None:
    rule = program_ad_product_outer_derivative_rule((3,), (4,))
    _assert_rule_matches_finite_difference(rule, _flat(_RNG.normal(size=3), _RNG.normal(size=4)))


def test_inner_rule_matches_finite_difference() -> None:
    rule = program_ad_product_inner_derivative_rule((4,), (4,))
    _assert_rule_matches_finite_difference(rule, _flat(_RNG.normal(size=4), _RNG.normal(size=4)))


def test_tensordot_rule_matches_finite_difference() -> None:
    rule = program_ad_product_tensordot_derivative_rule((2, 3), (3, 2), axes=1)
    _assert_rule_matches_finite_difference(
        rule, _flat(_RNG.normal(size=(2, 3)), _RNG.normal(size=(3, 2)))
    )


def test_einsum_rule_matches_finite_difference() -> None:
    rule = program_ad_product_einsum_derivative_rule("ij,jk->ik", ((2, 3), (3, 2)))
    _assert_rule_matches_finite_difference(
        rule, _flat(_RNG.normal(size=(2, 3)), _RNG.normal(size=(3, 2)))
    )


# --------------------------------------------------------------------------- #
# Linear algebra (non-eigenvalue)
# --------------------------------------------------------------------------- #
def _well_conditioned(size: int) -> NDArray[np.float64]:
    return _RNG.normal(size=(size, size)) + float(size) * np.eye(size)


def test_solve_rule_matches_finite_difference() -> None:
    rule = program_ad_linalg_solve_derivative_rule((3, 3), (3,))
    _assert_rule_matches_finite_difference(rule, _flat(_well_conditioned(3), _RNG.normal(size=3)))


def test_trace_rule_matches_finite_difference() -> None:
    _assert_rule_matches_finite_difference(
        program_ad_linalg_trace_derivative_rule((3, 3)), _RNG.normal(size=(3, 3))
    )


def test_diag_rule_matches_finite_difference() -> None:
    _assert_rule_matches_finite_difference(
        program_ad_linalg_diag_derivative_rule((4, 4)), _RNG.normal(size=(4, 4))
    )


def test_diagflat_rule_matches_finite_difference() -> None:
    _assert_rule_matches_finite_difference(
        program_ad_linalg_diagflat_derivative_rule((4,)), _RNG.normal(size=4)
    )


def test_matrix_power_rule_matches_finite_difference() -> None:
    rule = program_ad_linalg_matrix_power_derivative_rule(2)
    _assert_rule_matches_finite_difference(rule, _well_conditioned(3), rtol=1e-4, atol=1e-5)


def test_multi_dot_rule_matches_finite_difference() -> None:
    rule = program_ad_linalg_multi_dot_derivative_rule(((2, 3), (3, 2), (2, 4)))
    _assert_rule_matches_finite_difference(
        rule,
        _flat(_RNG.normal(size=(2, 3)), _RNG.normal(size=(3, 2)), _RNG.normal(size=(2, 4))),
    )


def test_svdvals_rule_matches_finite_difference() -> None:
    _assert_rule_matches_finite_difference(
        program_ad_linalg_svdvals_derivative_rule((3, 3)),
        _RNG.normal(size=(3, 3)),
        rtol=1e-4,
        atol=1e-5,
    )


def test_pinv_rule_matches_finite_difference() -> None:
    _assert_rule_matches_finite_difference(
        program_ad_linalg_pinv_derivative_rule((3, 3)),
        _well_conditioned(3),
        rtol=1e-4,
        atol=1e-5,
    )


def test_eigvalsh_rule_matches_symmetric_finite_difference() -> None:
    base = _RNG.normal(size=(3, 3))
    symmetric = base + base.T + np.diag([3.0, 6.0, 9.0])
    _assert_symmetric_eigenvalue_rule(
        program_ad_linalg_eigvalsh_derivative_rule((3, 3)), symmetric, rtol=1e-4, atol=1e-5
    )


def test_eigvals_rule_matches_symmetric_finite_difference() -> None:
    base = _RNG.normal(size=(3, 3))
    symmetric = base + base.T + np.diag([3.0, 6.0, 9.0])
    _assert_symmetric_eigenvalue_rule(
        program_ad_linalg_eigvals_derivative_rule((3, 3)), symmetric, rtol=1e-4, atol=1e-5
    )


# --------------------------------------------------------------------------- #
# Signal
# --------------------------------------------------------------------------- #
def test_convolve_rule_matches_finite_difference() -> None:
    rule = program_ad_signal_convolve_derivative_rule((4,), (3,))
    _assert_rule_matches_finite_difference(rule, _flat(_RNG.normal(size=4), _RNG.normal(size=3)))


def test_correlate_rule_matches_finite_difference() -> None:
    rule = program_ad_signal_correlate_derivative_rule((4,), (3,))
    _assert_rule_matches_finite_difference(rule, _flat(_RNG.normal(size=4), _RNG.normal(size=3)))


# --------------------------------------------------------------------------- #
# Interpolation
# --------------------------------------------------------------------------- #
def test_interp_rule_matches_finite_difference() -> None:
    grid = np.array([0.0, 1.0, 2.0, 3.0])
    query = np.array([0.3, 0.9, 1.4, 2.1, 2.7])
    fp = _RNG.normal(size=4)
    rule = program_ad_interpolation_interp_derivative_rule(query.shape, grid, fp.shape)
    _assert_rule_matches_finite_difference(rule, _flat(query, fp))


# --------------------------------------------------------------------------- #
# Stencil
# --------------------------------------------------------------------------- #
def test_gradient_rule_matches_finite_difference() -> None:
    _assert_rule_matches_finite_difference(
        program_ad_stencil_gradient_derivative_rule((6,)), _RNG.normal(size=6)
    )


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #
def test_where_rule_matches_finite_difference() -> None:
    condition = np.array([True, False, True, False, True])
    rule = program_ad_selection_where_derivative_rule(condition, (5,), (5,))
    _assert_rule_matches_finite_difference(rule, _flat(_RNG.normal(size=5), _RNG.normal(size=5)))


def test_clip_interior_rule_matches_finite_difference() -> None:
    rule = program_ad_selection_clip_derivative_rule((6,), lower_shape=(1,), upper_shape=(1,))
    # Mix of interior, below-lower, and above-upper entries; none at a boundary.
    source = np.array([-1.2, -0.2, 0.1, 0.3, 0.9, 1.5])
    _assert_rule_matches_finite_difference(rule, _flat(source, [-0.5], [0.6]))
