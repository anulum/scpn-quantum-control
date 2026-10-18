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
freedom). A supplied full observation covariance uses generalised least squares,
including its off-diagonal correlations. With equal or absent weights the point
estimate matches ``zne_extrapolate``.

Coefficient fitting uses an ``(order+1)``-square system. The full-covariance path
also solves a system with one row per observation and admits its dense buffers.
Both reuse vectorised NumPy linear algebra; no separate compiled kernel is used.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
from numpy.typing import NDArray

from ..dense_budget import DenseAllocationError, dense_budget_bytes


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
        ``"gls-delta"`` for a full observation covariance, ``"wls-delta"`` for
        independent per-point uncertainties, or ``"ols-delta"`` for residual error.
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
    covariance: NDArray[np.float64] | None = None,
    order: int = 1,
    coverage: float = 0.95,
    max_dense_gib: float | None = None,
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
        weighted least-squares covariance propagates them; when both uncertainty
        descriptions are omitted, an
        ordinary least-squares residual variance is used and at least
        ``order + 2`` points are required.
    covariance:
        Optional full covariance of the supplied expectation values, including
        correlations from shared calibration. Must be real, finite, symmetric and
        positive definite, with one row and column per noise scale. Mutually
        exclusive with ``standard_errors``. Generalised least squares propagates
        this supplied covariance; the normal interval remains a delta-method
        approximation and excludes unmodelled bias and drift.
    order:
        Polynomial degree (1 = linear, 2 = quadratic).
    coverage:
        Target coverage of the reported interval, in ``(0, 1)``.
    max_dense_gib:
        Dense-buffer allowance in GiB for the full-covariance GLS path, or
        ``None`` for the existing process budget. Counts declared numerical
        buffers; it does not reserve process memory.

    Returns
    -------
    ZNEUncertaintyResult
        The extrapolated estimate with standard error and coverage interval.

    Raises
    ------
    ValueError
        On malformed inputs, conflicting uncertainty inputs, a nonpositive
        covariance, too few points, a singular design matrix, or an
        unrepresentable full-covariance fit.
    DenseAllocationError
        If declared full-covariance GLS buffers exceed the active allowance.

    """
    if covariance is not None and any(
        np.iscomplexobj(value) for value in (covariance, noise_scales, expectation_values)
    ):
        raise ValueError("GLS inputs must be real")
    x, y, errors = _validate_inputs(
        noise_scales, expectation_values, standard_errors, order, coverage
    )
    if covariance is not None:
        required_bytes = 8 * (6 * x.size**2 + 6 * x.size * (order + 1) + 6 * (order + 1) ** 2)
        if required_bytes > dense_budget_bytes(max_dense_gib):
            raise DenseAllocationError("ZNE covariance buffers exceed the active dense budget")
    vander = np.vander(x, order + 1)  # constant term is the final column

    if covariance is not None:
        if errors is not None:
            raise ValueError("covariance and standard_errors are mutually exclusive")
        supplied = np.asarray(covariance, dtype=np.float64)
        if supplied.shape != (x.size, x.size):
            raise ValueError("covariance must have one row and column per noise scale")
        if not np.all(np.isfinite(supplied)):
            raise ValueError("covariance must be finite")
        if not np.array_equal(supplied, supplied.T):
            raise ValueError("covariance must be symmetric")
        try:
            np.linalg.cholesky(supplied)
        except np.linalg.LinAlgError as exc:
            raise ValueError("covariance must be positive definite") from exc
        with np.errstate(over="ignore", invalid="ignore"):
            weighted_design = np.linalg.solve(supplied, vander)
            normal_matrix = vander.T @ weighted_design
            rhs = weighted_design.T @ y
        if not all(np.all(np.isfinite(value)) for value in (normal_matrix, rhs)):
            raise ValueError("GLS design and weighted observations must remain finite")
        method = "gls-delta"
    elif errors is not None:
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
        coefficient_covariance = np.linalg.inv(normal_matrix)
    except np.linalg.LinAlgError as exc:
        raise ValueError("noise-scale design matrix is singular") from exc

    if covariance is not None and not np.all(np.isfinite(coefficient_covariance)):
        raise ValueError("GLS coefficient covariance must remain finite")
    with np.errstate(over="ignore", invalid="ignore") if covariance is not None else np.errstate():
        coeffs = coefficient_covariance @ rhs
        poly = np.poly1d(coeffs)
        zero_noise_estimate = float(coeffs[-1])  # poly(0) = constant term
        fit_residual = float(np.sqrt(np.mean((poly(x) - y) ** 2)))

    if errors is None and covariance is None:
        degrees_of_freedom = x.size - (order + 1)
        residual_sum = float(np.sum((poly(x) - y) ** 2))
        sigma_squared = residual_sum / degrees_of_freedom
        intercept_variance = sigma_squared * float(coefficient_covariance[-1, -1])
    else:
        intercept_variance = float(coefficient_covariance[-1, -1])

    if covariance is not None and (
        not np.all(np.isfinite([zero_noise_estimate, fit_residual, intercept_variance]))
        or intercept_variance <= 0
    ):
        raise ValueError("GLS result must be finite with positive intercept variance")
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
