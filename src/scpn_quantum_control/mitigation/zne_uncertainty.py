# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Uncertainty propagation for zero-noise extrapolation
"""Uncertainty propagation for zero-noise extrapolation (ZNE).

:func:`~scpn_quantum_control.mitigation.zne.zne_extrapolate` returns a point
zero-noise estimate from a polynomial fit over noise-scaled expectation values; it
carries no error bar, so a mitigated result looks exact when it is not. This module
propagates the per-scale shot-noise uncertainties (for example the Z-basis
magnetisation proxy standard errors from
:mod:`scpn_quantum_control.analysis.sync_uncertainty`) through the same polynomial
extrapolation to attach a defensible standard error and coverage interval to the
zero-noise estimate.

The zero-noise estimate is the constant term of the least-squares polynomial, so its
variance is the constant-term entry of the coefficient covariance. With per-point
variances ``s_i**2`` the weighted-least-squares covariance is ``(Vᵀ W V)⁻¹`` with
``W = diag(1/s_i**2)`` and ``V`` the Vandermonde matrix; without per-point
uncertainties an ordinary-least-squares covariance ``sigma_hat**2 (Vᵀ V)⁻¹`` is used
with ``sigma_hat**2`` the residual variance (which needs at least one degree of
freedom). With equal or absent weights the point estimate matches ``zne_extrapolate``.

The computation is small dense linear algebra on an ``(order+1)``-square system
(``order`` is 1 or 2 in practice), pure vectorised NumPy with no Python hot loop, so
the NumPy path is the optimised path and no separate compiled kernel is warranted.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ZNEUncertaintyResult:
    """A zero-noise estimate with a propagated error bar and coverage interval.

    Parameters
    ----------
    zero_noise_estimate:
        Constant term of the least-squares polynomial (the extrapolated value).
    standard_error:
        Propagated standard error of the zero-noise estimate.
    low, high:
        Coverage-interval bounds (``low <= zero_noise_estimate <= high``).
    coverage:
        Target coverage probability of ``[low, high]``.
    order:
        Polynomial degree used (1 = linear, 2 = quadratic).
    fit_residual:
        Root-mean-square fit residual, matching ``zne_extrapolate``.
    method:
        ``"wls-delta"`` when per-point uncertainties were supplied, else
        ``"ols-delta"``.
    n_points:
        Number of noise-scale data points.
    """

    zero_noise_estimate: float
    standard_error: float
    low: float
    high: float
    coverage: float
    order: int
    fit_residual: float
    method: str
    n_points: int

    @property
    def width(self) -> float:
        """Width ``high - low`` of the coverage interval."""
        return self.high - self.low


def _validate_inputs(
    noise_scales: Sequence[float],
    expectation_values: Sequence[float],
    standard_errors: Sequence[float] | None,
    order: int,
    coverage: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64] | None]:
    """Validate and convert ZNE-uncertainty inputs to finite float arrays."""
    if not 0.0 < coverage < 1.0:
        raise ValueError(f"coverage must lie in (0, 1), got {coverage}")
    if not isinstance(order, int) or isinstance(order, bool) or order < 1:
        raise ValueError(f"order must be an integer >= 1, got {order!r}")

    x = np.asarray(noise_scales, dtype=np.float64)
    y = np.asarray(expectation_values, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("noise_scales and expectation_values must be one-dimensional")
    if x.shape != y.shape:
        raise ValueError("noise_scales and expectation_values must have the same length")
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        raise ValueError("noise_scales and expectation_values must be finite")
    if np.any(x < 1.0):
        raise ValueError("noise_scales must be >= 1 (physical noise amplification)")
    if np.unique(x).size != x.size:
        raise ValueError("noise_scales must be distinct")
    if x.size < order + 1:
        raise ValueError(f"need >= {order + 1} points for an order-{order} fit, got {x.size}")

    errors: NDArray[np.float64] | None = None
    if standard_errors is not None:
        errors = np.asarray(standard_errors, dtype=np.float64)
        if errors.shape != x.shape:
            raise ValueError("standard_errors must match the number of points")
        if not np.all(np.isfinite(errors)):
            raise ValueError("standard_errors must be finite")
        if np.any(errors <= 0.0):
            raise ValueError("standard_errors must be positive")
    return x, y, errors


def zne_extrapolate_with_uncertainty(
    noise_scales: Sequence[float],
    expectation_values: Sequence[float],
    *,
    standard_errors: Sequence[float] | None = None,
    order: int = 1,
    coverage: float = 0.95,
) -> ZNEUncertaintyResult:
    """Zero-noise extrapolation with a propagated uncertainty on the estimate.

    Parameters
    ----------
    noise_scales:
        Noise amplification factors (each ``>= 1``, distinct), as fed to
        :func:`~scpn_quantum_control.mitigation.zne.zne_extrapolate`.
    expectation_values:
        Measured expectation value at each noise scale.
    standard_errors:
        Optional per-scale standard errors (for example Z-basis proxy shot-noise errors from
        :mod:`scpn_quantum_control.analysis.sync_uncertainty`). When given, a
        weighted least-squares covariance propagates them; when omitted, an
        ordinary least-squares residual variance is used and at least
        ``order + 2`` points are required.
    order:
        Polynomial degree (1 = linear, 2 = quadratic).
    coverage:
        Target coverage of the reported interval, in ``(0, 1)``.

    Returns
    -------
    ZNEUncertaintyResult
        The extrapolated estimate with standard error and coverage interval.

    Raises
    ------
    ValueError
        On malformed inputs, too few points, or a singular design matrix.
    """
    x, y, errors = _validate_inputs(
        noise_scales, expectation_values, standard_errors, order, coverage
    )
    vander = np.vander(x, order + 1)  # constant term is the final column

    if errors is not None:
        weights = 1.0 / errors**2
        wv = vander * weights[:, None]
        normal_matrix = vander.T @ wv
        rhs = wv.T @ y
        method = "wls-delta"
    else:
        if x.size < order + 2:
            raise ValueError(
                "ordinary least-squares standard error needs >= order + 2 points; "
                "supply standard_errors for an exactly-determined fit"
            )
        normal_matrix = vander.T @ vander
        rhs = vander.T @ y
        method = "ols-delta"

    try:
        covariance = np.linalg.inv(normal_matrix)
    except np.linalg.LinAlgError as exc:
        raise ValueError("noise-scale design matrix is singular") from exc

    coeffs = covariance @ rhs
    poly = np.poly1d(coeffs)
    zero_noise_estimate = float(coeffs[-1])  # poly(0) = constant term
    fit_residual = float(np.sqrt(np.mean((poly(x) - y) ** 2)))

    if errors is None:
        degrees_of_freedom = x.size - (order + 1)
        residual_sum = float(np.sum((poly(x) - y) ** 2))
        sigma_squared = residual_sum / degrees_of_freedom
        intercept_variance = sigma_squared * float(covariance[-1, -1])
    else:
        intercept_variance = float(covariance[-1, -1])

    standard_error = float(np.sqrt(max(intercept_variance, 0.0)))
    z = NormalDist().inv_cdf(0.5 + coverage / 2.0)
    half_width = z * standard_error
    return ZNEUncertaintyResult(
        zero_noise_estimate=zero_noise_estimate,
        standard_error=standard_error,
        low=zero_noise_estimate - half_width,
        high=zero_noise_estimate + half_width,
        coverage=coverage,
        order=order,
        fit_residual=fit_residual,
        method=method,
        n_points=int(x.size),
    )
