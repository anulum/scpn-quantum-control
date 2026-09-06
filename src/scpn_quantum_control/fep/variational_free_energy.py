# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Variational Free Energy
"""Variational free energy computation for the SCPN.

The Free Energy Principle (Friston 2010) states that self-organising
systems minimise variational free energy F, which bounds the negative
log-evidence (surprise):

    F = E_q[log q(z) − log p(z,x)]
      = KL[q(z) || p(z|x)] − log p(x)
      ≥ −log p(x)  (because KL ≥ 0)

For Gaussian beliefs and generative models:
    q(z) = N(μ, Σ)
    p(z,x) = p(x|z) p(z) with p(z) = N(0, Π⁻¹)

    F = ½ (μᵀ Π μ + tr(Π Σ) − log|Σ| − log|Π| − n)
      + ½ (x − g(μ))ᵀ Γ (x − g(μ)) − ½ log|Γ|

where Π = precision of prior (maps to K_nm in SCPN),
      Γ = precision of likelihood (sensory precision),
      g(μ) = generative model prediction.

SCPN mapping:
    μ = oscillator phases θ (sufficient statistics)
    Π = K_nm coupling matrix (prior precision)
    Γ = identity (perfect observation in simulation)
    g(μ) = predicted phases at lower layer (forward model)
    x = observed phases (data from quantum measurement)

A generative model is a pair. ``g`` and its Jacobian ``∂g/∂μ`` are supplied
together or not at all: an absent Jacobian is not replaced by the identity,
because that returns the gradient of a different model rather than an error.
Models need not be square; for ``g: R^n -> R^m`` the Jacobian is ``(m, n)`` and
the sensory precision is ``(m, m)``.

Ref:
    - Friston, Nature Reviews Neuroscience 11, 127 (2010)
    - Friston, J. R. Soc. Interface 10, 20130475 (2013)
    - Buckley et al., Entropy 19, 318 (2017) — tutorial
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import cho_factor, cho_solve

try:
    from scpn_quantum_engine import (
        free_energy_gradient_rust as _grad_rust,
    )

    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False


@dataclass
class FreeEnergyResult:
    """Variational free energy decomposition."""

    free_energy: float  # F
    complexity: float  # KL[q || prior]
    accuracy: float  # −E_q[log p(x|z)] = prediction error energy
    elbo: float  # −F = evidence lower bound
    surprise_bound: float  # F ≥ −log p(x)


PRECISION_RIDGE: Final[float] = 1e-10
"""Ridge added to a precision matrix before inversion.

The prior covariance is the inverse of the precision matrix, so a singular or
marginally indefinite precision would otherwise make the prior undefined. The
ridge is a declared part of the contract, not a silent repair: the regularised
matrix must still be positive definite, and a precision far from admissible is
rejected rather than nudged into range.
"""

COVARIANCE_SYMMETRY_ATOL: Final[float] = 1e-10
"""Absolute tolerance admitted between a covariance and its transpose."""


def _validated_mean(mu: NDArray[np.float64], name: str) -> NDArray[np.float64]:
    """Return ``mu`` as a finite one-dimensional float vector.

    Parameters
    ----------
    mu
        Candidate mean vector.
    name
        Argument name used in error messages.

    Returns
    -------
    numpy.ndarray
        Shape ``(n,)`` float64 copy of ``mu``.

    Raises
    ------
    ValueError
        If ``mu`` is not one-dimensional or contains a non-finite entry.

    """
    vector = np.asarray(mu, dtype=np.float64)
    if vector.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional vector, got shape {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be finite")
    return vector


def _cholesky_of_covariance(
    sigma: NDArray[np.float64], size: int, name: str
) -> tuple[NDArray[np.float64], bool]:
    """Factorise a covariance, rejecting anything that is not one.

    A covariance must be finite, square, symmetric and positive definite. Each
    condition is checked explicitly rather than left to surface as a downstream
    numerical artefact: a negative-definite matrix otherwise produces a negative
    "KL divergence", which is impossible by definition, because
    ``numpy.linalg.slogdet`` reports the magnitude and the sign separately and
    the sign is easy to drop.

    Parameters
    ----------
    sigma
        Candidate covariance matrix.
    size
        Dimension the matrix must match.
    name
        Argument name used in error messages.

    Returns
    -------
    tuple
        The ``scipy.linalg.cho_factor`` result for ``sigma``.

    Raises
    ------
    ValueError
        If ``sigma`` is not a finite, ``size``-by-``size``, symmetric,
        positive-definite matrix.

    """
    matrix = np.asarray(sigma, dtype=np.float64)
    if matrix.shape != (size, size):
        raise ValueError(f"{name} must have shape ({size}, {size}), got {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be finite")
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=COVARIANCE_SYMMETRY_ATOL):
        raise ValueError(f"{name} must be symmetric within {COVARIANCE_SYMMETRY_ATOL}")
    try:
        # scipy carries no stubs here, so the factor is given its declared type
        # at this boundary rather than leaking Any into every caller.
        factor: tuple[NDArray[np.float64], bool] = cho_factor(
            matrix, lower=True, check_finite=False
        )
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"{name} must be positive definite") from exc
    return factor


def _log_determinant(factor: tuple[NDArray[np.float64], bool]) -> float:
    """Return ``log|Σ|`` from a Cholesky factor.

    Computed as ``2 Σ log L_ii``, which is exact for a positive-definite matrix
    and never needs a sign correction.

    Parameters
    ----------
    factor
        A ``scipy.linalg.cho_factor`` result.

    Returns
    -------
    float
        Natural logarithm of the determinant.

    """
    return 2.0 * float(np.sum(np.log(np.diagonal(factor[0]))))


def _validated_prediction(
    predicted: NDArray[np.float64], expected_size: int, name: str
) -> NDArray[np.float64]:
    """Return a generative model's output as a finite prediction vector.

    A model that returns a scalar, a column vector or a vector of the wrong
    length would otherwise broadcast against the observation and yield a
    prediction error of a shape the caller never described, so the shape is
    established here rather than left to NumPy broadcasting.

    Parameters
    ----------
    predicted
        Value returned by the generative model, in the units of ``x_observed``.
    expected_size
        Number of observations the prediction must explain.
    name
        Expression named in error messages, for example ``generative_fn(mu)``.

    Returns
    -------
    numpy.ndarray
        Shape ``(expected_size,)`` float64 view of ``predicted``.

    Raises
    ------
    ValueError
        If the prediction is not a one-dimensional, finite vector of length
        ``expected_size``.

    """
    vector = np.asarray(predicted, dtype=np.float64)
    if vector.ndim != 1:
        raise ValueError(f"{name} must return a one-dimensional vector, got shape {vector.shape}")
    if vector.size != expected_size:
        raise ValueError(f"{name} must return {expected_size} values, got {vector.size}")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must return finite values")
    return vector


def _validated_jacobian(
    matrix: NDArray[np.float64], rows: int, columns: int, name: str
) -> NDArray[np.float64]:
    """Return a generative model's Jacobian as a finite ``(rows, columns)`` matrix.

    The Jacobian of a model ``g: R^n -> R^m`` has one row per prediction and one
    column per belief coordinate. A one-dimensional or mis-shaped return value
    contracts against the prediction error into something that still has the
    dimension of a gradient, so the shape is checked rather than inferred from
    whether the product happens to succeed.

    Parameters
    ----------
    matrix
        Value returned by the Jacobian callable.
    rows
        Number of predictions ``m``.
    columns
        Number of belief coordinates ``n``.
    name
        Expression named in error messages, for example ``generative_jac(mu)``.

    Returns
    -------
    numpy.ndarray
        Shape ``(rows, columns)`` float64 view of ``matrix``.

    Raises
    ------
    ValueError
        If the Jacobian does not have shape ``(rows, columns)`` or is not finite.

    """
    array = np.asarray(matrix, dtype=np.float64)
    if array.shape != (rows, columns):
        raise ValueError(f"{name} must return shape ({rows}, {columns}), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must return finite values")
    return array


def _validated_square_matrix(
    matrix: NDArray[np.float64], size: int, name: str
) -> NDArray[np.float64]:
    """Return a finite ``(size, size)`` matrix.

    Parameters
    ----------
    matrix
        Candidate matrix.
    size
        Dimension the matrix must match.
    name
        Argument name used in error messages.

    Returns
    -------
    numpy.ndarray
        Shape ``(size, size)`` float64 view of ``matrix``.

    Raises
    ------
    ValueError
        If the matrix is not ``size``-by-``size`` or is not finite.

    """
    array = np.asarray(matrix, dtype=np.float64)
    if array.shape != (size, size):
        raise ValueError(f"{name} must have shape ({size}, {size}), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def _validated_precision(
    sensory_precision: NDArray[np.float64] | None, size: int, name: str
) -> NDArray[np.float64]:
    """Resolve an optional sensory precision to a finite ``(size, size)`` matrix.

    ``size`` is the number of predictions, not the number of belief
    coordinates: for a model ``g: R^n -> R^m`` the sensory precision weights the
    prediction error and is ``m``-by-``m``.

    Definiteness is not asserted here. The likelihood precision is documented as
    positive definite, but this function admits any finite square matrix, so a
    caller supplying an indefinite precision receives a weighted error rather
    than a rejection.

    Parameters
    ----------
    sensory_precision
        Candidate precision matrix, or ``None`` for the identity.
    size
        Number of predictions ``m``.
    name
        Argument name used in error messages.

    Returns
    -------
    numpy.ndarray
        Shape ``(size, size)`` float64 matrix.

    Raises
    ------
    ValueError
        If a supplied precision is not ``size``-by-``size`` or is not finite.

    """
    if sensory_precision is None:
        return np.eye(size)
    return _validated_square_matrix(sensory_precision, size, name)


def kl_divergence_gaussian(
    mu_q: NDArray[np.float64],
    sigma_q: NDArray[np.float64],
    mu_p: NDArray[np.float64],
    sigma_p: NDArray[np.float64],
) -> float:
    """KL divergence between two multivariate Gaussians.

    ``KL[N(μ_q, Σ_q) || N(μ_p, Σ_p)] = ½ (tr(Σ_p⁻¹ Σ_q) + (μ_p−μ_q)ᵀ Σ_p⁻¹ (μ_p−μ_q)
    − n + log(|Σ_p| / |Σ_q|))``.

    Both covariances are factorised with Cholesky and the solves go through that
    factor, so no explicit inverse is formed and the log-determinants carry no
    separate sign. Inputs that are not covariances are rejected rather than
    producing a value that cannot be a divergence.

    Parameters
    ----------
    mu_q, mu_p
        Finite mean vectors of shape ``(n,)``.
    sigma_q, sigma_p
        Finite symmetric positive-definite covariances of shape ``(n, n)``.

    Returns
    -------
    float
        The divergence in nats. Non-negative for valid inputs, up to floating
        point rounding near zero; the value is never clamped.

    Raises
    ------
    ValueError
        If a mean is not finite and one-dimensional, if the shapes disagree, or
        if a covariance is not finite, symmetric and positive definite.

    """
    mean_q = _validated_mean(mu_q, "mu_q")
    mean_p = _validated_mean(mu_p, "mu_p")
    n = mean_q.size
    if mean_p.size != n:
        raise ValueError(f"mu_q and mu_p must share a dimension, got {n} and {mean_p.size}")

    factor_q = _cholesky_of_covariance(sigma_q, n, "sigma_q")
    factor_p = _cholesky_of_covariance(sigma_p, n, "sigma_p")

    trace_term = float(
        np.trace(cho_solve(factor_p, np.asarray(sigma_q, dtype=np.float64), check_finite=False))
    )
    diff = mean_p - mean_q
    mahalanobis = float(diff @ cho_solve(factor_p, diff, check_finite=False))
    log_det_ratio = _log_determinant(factor_p) - _log_determinant(factor_q)

    return 0.5 * (trace_term + mahalanobis - n + log_det_ratio)


def _complexity_term(
    mu: NDArray[np.float64],
    sigma: NDArray[np.float64],
    K_precision: NDArray[np.float64],
) -> float:
    """KL[q(z) || prior] where prior = N(0, K⁻¹)."""
    n = len(mu)
    K_reg = np.asarray(K_precision, dtype=np.float64) + PRECISION_RIDGE * np.eye(n)
    prior_factor = _cholesky_of_covariance(K_reg, n, "K_precision + ridge")
    prior_cov = cho_solve(prior_factor, np.eye(n), check_finite=False)
    prior_cov = 0.5 * (prior_cov + prior_cov.T)
    return kl_divergence_gaussian(mu, sigma, np.zeros(n), prior_cov)


def _accuracy_term(
    mu: NDArray[np.float64],
    x_observed: NDArray[np.float64],
    sensory_precision: NDArray[np.float64] | None = None,
    generative_fn: Callable[..., NDArray[np.float64]] | None = None,
) -> float:
    """Prediction error energy ``½ (x − g(μ))ᵀ Γ (x − g(μ))``.

    Parameters
    ----------
    mu
        Belief mean of shape ``(n,)``.
    x_observed
        Observations of shape ``(m,)``; ``m`` equals ``n`` for the identity
        model.
    sensory_precision
        Likelihood precision ``Γ`` of shape ``(m, m)``, or ``None`` for the
        identity.
    generative_fn
        Forward model ``g``, or ``None`` for the identity model.

    Returns
    -------
    float
        The prediction error energy in nats.

    Raises
    ------
    ValueError
        If ``mu`` or ``x_observed`` is not a finite vector, if the model does
        not return a finite vector of length ``m``, or if ``sensory_precision``
        is not a finite ``(m, m)`` matrix.

    """
    mean = _validated_mean(mu, "mu")
    observation = _validated_mean(x_observed, "x_observed")
    if generative_fn is None:
        if observation.size != mean.size:
            raise ValueError(
                f"x_observed must have length {mean.size} for the identity generative "
                f"model, got {observation.size}"
            )
        predicted = mean
    else:
        predicted = _validated_prediction(
            generative_fn(mean), observation.size, "generative_fn(mu)"
        )
    gamma = _validated_precision(sensory_precision, predicted.size, "sensory_precision")
    error = observation - predicted
    return 0.5 * float(error @ gamma @ error)


def variational_free_energy(
    mu: NDArray[np.float64],
    sigma: NDArray[np.float64],
    x_observed: NDArray[np.float64],
    K_precision: NDArray[np.float64],
    sensory_precision: NDArray[np.float64] | None = None,
    generative_fn: Callable[..., NDArray[np.float64]] | None = None,
) -> FreeEnergyResult:
    """Compute variational free energy ``F = complexity + accuracy``.

    Parameters
    ----------
    mu
        Belief mean of shape ``(n,)``.
    sigma
        Belief covariance of shape ``(n, n)``: finite, symmetric and positive
        definite.
    x_observed
        Observations of shape ``(m,)``, where ``m`` is the number of values the
        generative model predicts and equals ``n`` for the identity model.
    K_precision
        Prior precision of shape ``(n, n)``. ``PRECISION_RIDGE`` is added before
        it is inverted to form the prior covariance.
    sensory_precision
        Likelihood precision ``Γ`` of shape ``(m, m)``, or ``None`` for the
        identity. Note that ``m`` follows the model output, not ``mu``.
    generative_fn
        Forward model ``g``, or ``None`` for the identity model. It must return
        a finite one-dimensional vector of length ``m``; a scalar or column
        vector is rejected rather than broadcast against the observation.

    Returns
    -------
    FreeEnergyResult
        The free energy and its complexity/accuracy decomposition, in nats.

    Raises
    ------
    ValueError
        If any argument violates the shape, finiteness or definiteness contract
        above, or if the generative model does not return a length-``m`` finite
        vector.

    """
    complexity = _complexity_term(mu, sigma, K_precision)
    accuracy = _accuracy_term(mu, x_observed, sensory_precision, generative_fn)
    free_energy = complexity + accuracy

    return FreeEnergyResult(
        free_energy=free_energy,
        complexity=complexity,
        accuracy=accuracy,
        elbo=-free_energy,
        surprise_bound=free_energy,
    )


def evidence_lower_bound(
    mu: NDArray[np.float64],
    sigma: NDArray[np.float64],
    x_observed: NDArray[np.float64],
    K_precision: NDArray[np.float64],
) -> float:
    """ELBO = −F (shorthand for optimisation targets).

    Parameters
    ----------
    mu
        Belief mean of shape ``(n,)``.
    sigma
        Belief covariance of shape ``(n, n)``.
    x_observed
        Observations of shape ``(n,)``; this shorthand uses the identity model.
    K_precision
        Prior precision of shape ``(n, n)``.

    Returns
    -------
    float
        The evidence lower bound in nats.

    Raises
    ------
    ValueError
        Propagated from :func:`variational_free_energy`.

    """
    result = variational_free_energy(mu, sigma, x_observed, K_precision)
    return result.elbo


def free_energy_gradient(
    mu: NDArray[np.float64],
    sigma: NDArray[np.float64],
    x_observed: NDArray[np.float64],
    K_precision: NDArray[np.float64],
    sensory_precision: NDArray[np.float64] | None = None,
    generative_fn: Callable[..., NDArray[np.float64]] | None = None,
    generative_jac: Callable[..., NDArray[np.float64]] | None = None,
) -> NDArray[np.float64]:
    """Gradient ``∂F/∂μ`` for belief update dynamics.

    ``∂F/∂μ = Π_z μ − Jᵀ Γ (x − g(μ))``, where ``Π_z`` is the ridged prior
    precision, ``J = ∂g/∂μ`` and ``Γ`` is the sensory precision. Belief dynamics
    follow ``dμ/dt = −∂F/∂μ``.

    A generative model and its Jacobian form one contract and must be supplied
    together. Neither is inferred from the other: substituting the identity for
    an absent Jacobian returns the gradient of a different model, which is a
    plausible vector rather than an error, and applying a Jacobian to the
    identity prediction returns the gradient of neither model. An incomplete
    pair is refused before the model is called, so no belief state is updated
    from it.

    Models are not required to be square. For ``g: R^n -> R^m`` the Jacobian is
    ``(m, n)`` and the sensory precision is ``(m, m)``; the returned gradient
    always has length ``n``.

    ``sigma`` is accepted for signature symmetry with
    :func:`variational_free_energy` and does not enter the gradient: the terms
    of ``F`` that carry the belief covariance are constant in ``μ``.

    The Rust engine implements the identity model only and is used when no model
    is supplied; the arguments are validated here first, so both tiers see the
    same admitted domain.

    Parameters
    ----------
    mu
        Belief mean of shape ``(n,)``, finite.
    sigma
        Belief covariance of shape ``(n, n)``. Unused; see above.
    x_observed
        Observations of shape ``(m,)``, finite.
    K_precision
        Prior precision of shape ``(n, n)``, finite. ``PRECISION_RIDGE`` is
        added to its diagonal.
    sensory_precision
        Likelihood precision ``Γ`` of shape ``(m, m)``, or ``None`` for the
        identity.
    generative_fn
        Forward model ``g`` returning a finite vector of length ``m``. Requires
        ``generative_jac``.
    generative_jac
        Jacobian ``∂g/∂μ`` returning a finite ``(m, n)`` matrix. Requires
        ``generative_fn``.

    Returns
    -------
    numpy.ndarray
        Gradient of shape ``(n,)``.

    Raises
    ------
    ValueError
        If exactly one of ``generative_fn`` and ``generative_jac`` is supplied,
        if ``mu`` or ``x_observed`` is not a finite vector, if the identity
        model is used and the observations do not match ``mu``, if the model or
        its Jacobian returns a non-finite or mis-shaped value, or if
        ``K_precision`` or ``sensory_precision`` has the wrong shape or is not
        finite.

    """
    mean = _validated_mean(mu, "mu")
    observation = _validated_mean(x_observed, "x_observed")
    n = mean.size

    if generative_fn is None or generative_jac is None:
        if generative_jac is not None:
            raise ValueError(
                "generative_jac requires generative_fn: a Jacobian without its model "
                "would differentiate the identity prediction"
            )
        if generative_fn is not None:
            raise ValueError(
                "generative_fn requires generative_jac: no derivative is inferred, and "
                "an identity Jacobian returns the gradient of a different model"
            )
        if observation.size != n:
            raise ValueError(
                f"x_observed must have length {n} for the identity generative model, "
                f"got {observation.size}"
            )
        predicted = mean
        jacobian = np.eye(n)
    else:
        predicted = _validated_prediction(
            generative_fn(mean), observation.size, "generative_fn(mu)"
        )
        jacobian = _validated_jacobian(
            generative_jac(mean), predicted.size, n, "generative_jac(mu)"
        )

    prior_precision = _validated_square_matrix(K_precision, n, "K_precision")
    gamma = _validated_precision(sensory_precision, predicted.size, "sensory_precision")

    if _HAS_RUST and generative_fn is None:
        return np.asarray(
            _grad_rust(mean, observation, prior_precision, gamma, PRECISION_RIDGE),
            dtype=np.float64,
        )

    grad = (prior_precision + PRECISION_RIDGE * np.eye(n)) @ mean
    grad -= jacobian.T @ gamma @ (observation - predicted)

    return np.asarray(grad, dtype=np.float64)
