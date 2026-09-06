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
    sensory_precision: NDArray[np.float64],
    generative_fn: Callable[..., NDArray[np.float64]] | None = None,
) -> float:
    """Prediction error energy: 0.5 × (x − g(μ))ᵀ Γ (x − g(μ))."""
    predicted = generative_fn(mu) if generative_fn is not None else mu
    error = x_observed - predicted
    return 0.5 * float(error @ sensory_precision @ error)


def variational_free_energy(
    mu: NDArray[np.float64],
    sigma: NDArray[np.float64],
    x_observed: NDArray[np.float64],
    K_precision: NDArray[np.float64],
    sensory_precision: NDArray[np.float64] | None = None,
    generative_fn: Callable[..., NDArray[np.float64]] | None = None,
) -> FreeEnergyResult:
    """Compute variational free energy F = complexity + accuracy."""
    n = len(mu)
    if sensory_precision is None:
        sensory_precision = np.eye(n)

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
    """ELBO = −F (shorthand for optimisation targets)."""
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
    """Gradient ∂F/∂μ for belief update dynamics.

    dμ/dt = −∂F/∂μ = −Π_z μ + Jᵀ Γ (x − g(μ))

    where Π_z = prior precision (K_nm), J = ∂g/∂μ (Jacobian),
    Γ = sensory precision.

    With identity generative model: ∂F/∂μ = Π_z μ − Γ(x − μ)
    Uses Rust engine when available (identity generative model only).
    """
    n = len(mu)
    if sensory_precision is None:
        sensory_precision = np.eye(n)

    # Rust path for identity generative model
    if _HAS_RUST and generative_fn is None and generative_jac is None:
        return np.asarray(
            _grad_rust(mu, x_observed, K_precision, sensory_precision, 1e-10),
            dtype=np.float64,
        )

    K_reg = K_precision + 1e-10 * np.eye(n)

    # Prior contribution
    grad = K_reg @ mu

    # Likelihood contribution
    if generative_fn is not None:
        predicted = generative_fn(mu)
        if generative_jac is not None:
            J = generative_jac(mu)
        else:
            J = np.eye(n)
    else:
        predicted = mu
        J = np.eye(n)

    error = x_observed - predicted
    grad -= J.T @ sensory_precision @ error

    return np.asarray(grad, dtype=np.float64)
