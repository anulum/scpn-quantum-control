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

import numpy as np

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


def kl_divergence_gaussian(
    mu_q: np.ndarray,
    sigma_q: np.ndarray,
    mu_p: np.ndarray,
    sigma_p: np.ndarray,
) -> float:
    """KL divergence between two multivariate Gaussians.

    KL[N(μ_q, Σ_q) || N(μ_p, Σ_p)] =
        ½ (tr(Σ_p⁻¹ Σ_q) + (μ_p−μ_q)ᵀ Σ_p⁻¹ (μ_p−μ_q)
           − n + log(|Σ_p| / |Σ_q|))
    """
    n = len(mu_q)
    sigma_p_inv = np.linalg.inv(sigma_p)

    trace_term = float(np.trace(sigma_p_inv @ sigma_q))
    diff = mu_p - mu_q
    mahalanobis = float(diff @ sigma_p_inv @ diff)
    log_det_ratio = float(np.linalg.slogdet(sigma_p)[1] - np.linalg.slogdet(sigma_q)[1])

    return 0.5 * (trace_term + mahalanobis - n + log_det_ratio)


def _complexity_term(
    mu: np.ndarray,
    sigma: np.ndarray,
    K_precision: np.ndarray,
) -> float:
    """KL[q(z) || prior] where prior = N(0, K⁻¹)."""
    n = len(mu)
    K_reg = K_precision + 1e-10 * np.eye(n)
    prior_cov = np.linalg.inv(K_reg)
    return kl_divergence_gaussian(mu, sigma, np.zeros(n), prior_cov)


def _accuracy_term(
    mu: np.ndarray,
    x_observed: np.ndarray,
    sensory_precision: np.ndarray,
    generative_fn: Callable[..., np.ndarray] | None = None,
) -> float:
    """Prediction error energy: 0.5 × (x − g(μ))ᵀ Γ (x − g(μ))."""
    predicted = generative_fn(mu) if generative_fn is not None else mu
    error = x_observed - predicted
    return 0.5 * float(error @ sensory_precision @ error)


def variational_free_energy(
    mu: np.ndarray,
    sigma: np.ndarray,
    x_observed: np.ndarray,
    K_precision: np.ndarray,
    sensory_precision: np.ndarray | None = None,
    generative_fn: Callable[..., np.ndarray] | None = None,
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
    mu: np.ndarray,
    sigma: np.ndarray,
    x_observed: np.ndarray,
    K_precision: np.ndarray,
) -> float:
    """ELBO = −F (shorthand for optimisation targets)."""
    result = variational_free_energy(mu, sigma, x_observed, K_precision)
    return result.elbo


def free_energy_gradient(
    mu: np.ndarray,
    sigma: np.ndarray,
    x_observed: np.ndarray,
    K_precision: np.ndarray,
    sensory_precision: np.ndarray | None = None,
    generative_fn: Callable[..., np.ndarray] | None = None,
    generative_jac: Callable[..., np.ndarray] | None = None,
) -> np.ndarray:
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
        return np.asarray(_grad_rust(mu, x_observed, K_precision, sensory_precision, 1e-10))

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

    return np.asarray(grad)
