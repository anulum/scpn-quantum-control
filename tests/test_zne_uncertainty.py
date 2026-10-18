# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — ZNE uncertainty propagation tests
"""Tests for uncertainty propagation through zero-noise extrapolation."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control.mitigation.zne import zne_extrapolate
from scpn_quantum_control.mitigation.zne_uncertainty import (
    ZNEUncertaintyResult,
    zne_extrapolate_with_uncertainty,
)

# --- point estimate fidelity ---------------------------------------------------


def test_ols_point_estimate_matches_zne_extrapolate() -> None:
    """With no weights the linear estimate equals the existing extrapolator."""
    scales = [1, 3, 5]
    values = [0.60, 0.50, 0.45]
    reference = zne_extrapolate(scales, values, order=1)
    result = zne_extrapolate_with_uncertainty(scales, values, order=1)
    assert result.zero_noise_estimate == pytest.approx(reference.zero_noise_estimate)
    assert result.fit_residual == pytest.approx(reference.fit_residual)
    assert result.method == "ols-delta"


def test_quadratic_point_estimate_matches_zne_extrapolate() -> None:
    """The order-2 estimate also matches the existing extrapolator."""
    scales = [1, 3, 5, 7]
    values = [0.60, 0.50, 0.45, 0.43]
    reference = zne_extrapolate(scales, values, order=2)
    result = zne_extrapolate_with_uncertainty(scales, values, order=2)
    assert result.zero_noise_estimate == pytest.approx(reference.zero_noise_estimate)


# --- weighted (shot-noise) propagation -----------------------------------------


def test_wls_standard_error_matches_closed_form_two_point_linear() -> None:
    """Two-point linear intercept variance is 1.5^2 s1^2 + 0.5^2 s2^2 at x=[1,3]."""
    scales = [1, 3]
    values = [0.50, 0.40]
    errors = [0.10, 0.10]
    result = zne_extrapolate_with_uncertainty(scales, values, standard_errors=errors, order=1)
    # poly(0) = 1.5*y1 - 0.5*y2  =>  Var = 1.5^2 s1^2 + 0.5^2 s2^2
    expected_se = float(np.sqrt(1.5**2 * 0.1**2 + 0.5**2 * 0.1**2))
    assert result.standard_error == pytest.approx(expected_se)
    assert result.zero_noise_estimate == pytest.approx(1.5 * 0.50 - 0.5 * 0.40)
    assert result.method == "wls-delta"
    assert result.n_points == 2


def test_wls_smaller_errors_give_smaller_standard_error() -> None:
    """Tighter per-point shot noise propagates to a tighter zero-noise error."""
    scales = [1, 3, 5]
    values = [0.60, 0.50, 0.45]
    loose = zne_extrapolate_with_uncertainty(scales, values, standard_errors=[0.05, 0.05, 0.05])
    tight = zne_extrapolate_with_uncertainty(scales, values, standard_errors=[0.01, 0.01, 0.01])
    assert tight.standard_error < loose.standard_error


def test_wls_allows_exactly_determined_fit() -> None:
    """Weighted propagation needs only order + 1 points (input-uncertainty driven)."""
    result = zne_extrapolate_with_uncertainty(
        [1, 3], [0.5, 0.4], standard_errors=[0.1, 0.1], order=1
    )
    assert result.fit_residual == pytest.approx(0.0)
    assert result.standard_error > 0.0


# --- ordinary-least-squares path -----------------------------------------------


def test_ols_standard_error_positive_with_spare_point() -> None:
    """OLS with a spare degree of freedom yields a positive residual-based error."""
    result = zne_extrapolate_with_uncertainty([1, 3, 5], [0.60, 0.50, 0.46], order=1)
    assert result.standard_error > 0.0
    assert result.low <= result.zero_noise_estimate <= result.high


def test_ols_requires_more_points_than_coefficients() -> None:
    """OLS standard error needs at least order + 2 points."""
    with pytest.raises(ValueError, match="order \\+ 2 points"):
        zne_extrapolate_with_uncertainty([1, 3], [0.5, 0.4], order=1)


# --- interval behaviour --------------------------------------------------------


def test_interval_orders_and_width() -> None:
    """The interval brackets the estimate and width equals high - low."""
    result = zne_extrapolate_with_uncertainty(
        [1, 3, 5], [0.60, 0.50, 0.45], standard_errors=[0.02, 0.02, 0.02]
    )
    assert result.low <= result.zero_noise_estimate <= result.high
    assert result.width == pytest.approx(result.high - result.low)


def test_wider_coverage_widens_interval() -> None:
    """A higher coverage target widens the interval."""
    errors = [0.02, 0.02, 0.02]
    narrow = zne_extrapolate_with_uncertainty(
        [1, 3, 5], [0.6, 0.5, 0.45], standard_errors=errors, coverage=0.80
    )
    wide = zne_extrapolate_with_uncertainty(
        [1, 3, 5], [0.6, 0.5, 0.45], standard_errors=errors, coverage=0.99
    )
    assert wide.width > narrow.width


# --- validation ----------------------------------------------------------------


def test_rejects_bad_coverage() -> None:
    """Coverage must lie in the open unit interval."""
    with pytest.raises(ValueError, match="coverage must lie in"):
        zne_extrapolate_with_uncertainty([1, 3, 5], [0.6, 0.5, 0.45], coverage=1.0)


def test_rejects_order_below_one() -> None:
    """Order must be at least one."""
    with pytest.raises(ValueError, match="order must be an integer >= 1"):
        zne_extrapolate_with_uncertainty([1, 3, 5], [0.6, 0.5, 0.45], order=0)


def test_rejects_length_mismatch() -> None:
    """Scales and values must align."""
    with pytest.raises(ValueError, match="same length"):
        zne_extrapolate_with_uncertainty([1, 3, 5], [0.6, 0.5])


def test_rejects_non_finite_values() -> None:
    """Non-finite values are rejected."""
    with pytest.raises(ValueError, match="must be finite"):
        zne_extrapolate_with_uncertainty([1, 3, 5], [0.6, np.inf, 0.45])


def test_rejects_scale_below_one() -> None:
    """Noise scales below one are unphysical."""
    with pytest.raises(ValueError, match="noise_scales must be >= 1"):
        zne_extrapolate_with_uncertainty([0, 3, 5], [0.6, 0.5, 0.45])


def test_rejects_duplicate_scales() -> None:
    """Noise scales must be distinct."""
    with pytest.raises(ValueError, match="must be distinct"):
        zne_extrapolate_with_uncertainty([3, 3, 5], [0.6, 0.5, 0.45])


def test_rejects_too_few_points_for_order() -> None:
    """At least order + 1 points are required for the fit."""
    with pytest.raises(ValueError, match="points for an order-2 fit"):
        zne_extrapolate_with_uncertainty([1, 3], [0.6, 0.5], standard_errors=[0.1, 0.1], order=2)


def test_rejects_mismatched_standard_errors() -> None:
    """Per-point standard errors must match the point count."""
    with pytest.raises(ValueError, match="standard_errors must match"):
        zne_extrapolate_with_uncertainty([1, 3, 5], [0.6, 0.5, 0.45], standard_errors=[0.1, 0.1])


def test_rejects_non_positive_standard_errors() -> None:
    """Standard errors must be strictly positive."""
    with pytest.raises(ValueError, match="standard_errors must be positive"):
        zne_extrapolate_with_uncertainty(
            [1, 3, 5], [0.6, 0.5, 0.45], standard_errors=[0.1, 0.0, 0.1]
        )


def test_rejects_non_finite_standard_errors() -> None:
    """Standard errors must be finite."""
    with pytest.raises(ValueError, match="standard_errors must be finite"):
        zne_extrapolate_with_uncertainty(
            [1, 3, 5], [0.6, 0.5, 0.45], standard_errors=[0.1, np.nan, 0.1]
        )


def test_singular_design_matrix_is_wrapped(monkeypatch: pytest.MonkeyPatch) -> None:
    """A singular normal matrix surfaces as a ValueError, not a raw LinAlgError."""

    def _raise(_matrix: object) -> object:
        raise np.linalg.LinAlgError("singular")

    monkeypatch.setattr(np.linalg, "inv", _raise)
    with pytest.raises(ValueError, match="design matrix is singular"):
        zne_extrapolate_with_uncertainty(
            [1, 3, 5], [0.6, 0.5, 0.45], standard_errors=[0.1, 0.1, 0.1]
        )


def test_rejects_two_dimensional_input() -> None:
    """Inputs must be one-dimensional arrays."""
    with pytest.raises(ValueError, match="one-dimensional"):
        zne_extrapolate_with_uncertainty(cast(Any, [[1, 3], [5, 7]]), [0.6, 0.5, 0.45])


def test_result_is_frozen() -> None:
    """The result dataclass is immutable."""
    result = zne_extrapolate_with_uncertainty(
        [1, 3, 5], [0.6, 0.5, 0.45], standard_errors=[0.02, 0.02, 0.02]
    )
    assert isinstance(result, ZNEUncertaintyResult)
    with pytest.raises(AttributeError):
        result.standard_error = 0.0  # type: ignore[misc]


def test_full_covariance_retains_shared_offset_uncertainty() -> None:
    """A common additive error survives extrapolation with its original variance."""
    values = np.array([0.6, 0.49, 0.45])
    covariance = 0.01 * np.eye(3) + 0.04 * np.ones((3, 3))
    original = covariance.copy()
    result = zne_extrapolate_with_uncertainty([1, 3, 5], values.tolist(), covariance=covariance)
    weights = np.array([13 / 12, 1 / 3, -5 / 12])
    assert result.method == "gls-delta"
    assert result.zero_noise_estimate == pytest.approx(float(weights @ values), abs=1e-12)
    assert result.standard_error**2 == pytest.approx(0.01 * float(weights @ weights) + 0.04)
    np.testing.assert_array_equal(covariance, original)


def test_diagonal_full_covariance_matches_independent_standard_errors() -> None:
    """GLS reduces to the existing WLS result for independent supplied noise."""
    errors = np.array([0.02, 0.04, 0.03])
    independent = zne_extrapolate_with_uncertainty(
        [1, 3, 5], [0.6, 0.51, 0.43], standard_errors=errors.tolist()
    )
    full = zne_extrapolate_with_uncertainty(
        [1, 3, 5], [0.6, 0.51, 0.43], covariance=np.diag(errors**2)
    )
    assert full.zero_noise_estimate == pytest.approx(independent.zero_noise_estimate)
    assert full.standard_error == pytest.approx(independent.standard_error)
    assert full.low == pytest.approx(independent.low)
    assert full.high == pytest.approx(independent.high)


@pytest.mark.parametrize(
    "covariance,message",
    [
        (np.eye(3), "one row and column"),
        (np.array([[1.0, np.nan], [np.nan, 1.0]]), "finite"),
        (np.array([[1.0, 0.2], [0.0, 1.0]]), "symmetric"),
        (np.array([[1.0, 2.0], [2.0, 1.0]]), "positive definite"),
        (np.ones((2, 2)), "positive definite"),
    ],
)
def test_invalid_full_covariance_refuses(covariance: NDArray[np.float64], message: str) -> None:
    """Reject malformed or unidentifiable noise without clipping eigenvalues."""
    original = covariance.copy()
    with pytest.raises(ValueError, match=message):
        zne_extrapolate_with_uncertainty([1, 3], [0.6, 0.5], covariance=covariance)
    np.testing.assert_array_equal(covariance, original)


def test_full_covariance_and_standard_errors_cannot_both_be_supplied() -> None:
    """Do not silently select one of two conflicting uncertainty descriptions."""
    with pytest.raises(ValueError, match="mutually exclusive"):
        zne_extrapolate_with_uncertainty(
            [1, 3], [0.6, 0.5], standard_errors=[0.1, 0.1], covariance=np.eye(2)
        )


def test_full_covariance_obeys_declared_dense_allowance() -> None:
    """Refuse GLS buffers exceeding their allowance before constructing the design."""
    from scpn_quantum_control.dense_budget import DenseAllocationError

    with pytest.raises(DenseAllocationError, match="ZNE covariance buffers"):
        zne_extrapolate_with_uncertainty(
            [1, 3], [0.6, 0.5], covariance=np.eye(2), max_dense_gib=1e-8
        )


def test_actual_unresolvable_design_refuses_full_covariance_fit() -> None:
    """Distinct float64 scales can still produce a singular normal design."""
    with pytest.raises(ValueError, match="design matrix is singular"):
        zne_extrapolate_with_uncertainty([1.0, 1.0 + 2**-51], [0.6, 0.5], covariance=np.eye(2))


@pytest.mark.parametrize(
    "scales,covariance",
    [
        ([1.0, 3.0], np.eye(2) * 1e-310),
        ([1.0, 3.0], np.eye(2) * 1e308),
        ([1e154, 2e154], np.eye(2)),
    ],
)
def test_unrepresentable_full_covariance_fit_refuses(
    scales: list[float], covariance: NDArray[np.float64]
) -> None:
    """Finite inputs cannot turn overflow into a successful confidence result."""
    with pytest.raises(ValueError, match="finite"):
        zne_extrapolate_with_uncertainty(scales, [0.6, 0.5], covariance=covariance)


def test_unrepresentable_gls_fit_residual_refuses() -> None:
    """A finite normal matrix cannot certify an overflowing residual computation."""
    with pytest.raises(ValueError, match="finite"):
        zne_extrapolate_with_uncertainty([1, 3], [5e307, -5e307], covariance=np.eye(2))


def test_complex_full_covariance_refuses_imaginary_data_loss() -> None:
    """Reject complex noise instead of casting its imaginary component away."""
    covariance = np.eye(2, dtype=np.complex128) * (1 + 0.1j)
    with pytest.raises(ValueError, match="real"):
        zne_extrapolate_with_uncertainty([1, 3], [0.6, 0.5], covariance=cast(Any, covariance))


@pytest.mark.parametrize("position", ["scales", "values"])
def test_complex_gls_observations_refuse_imaginary_data_loss(position: str) -> None:
    """The real GLS contract cannot silently project a complex observation."""
    scales: object = [1, 3]
    values: object = [0.6, 0.5]
    if position == "scales":
        scales = np.array([1 + 0.1j, 3 + 0.1j], dtype=np.complex128)
    else:
        values = np.array([0.6 + 0.1j, 0.5 + 0.1j], dtype=np.complex128)
    with pytest.raises(ValueError, match="real"):
        zne_extrapolate_with_uncertainty(
            cast(Any, scales), cast(Any, values), covariance=np.eye(2)
        )
