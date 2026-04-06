# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Hierarchical Predictive Coding
"""Predictive coding across SCPN layers.

Each layer i predicts the state of the layer below via a generative
model g_i: μ_{i+1} → x̂_i. Prediction errors ε_i = x_i − g_i(μ_{i+1})
propagate upward, weighted by precision Π_i (derived from K_nm).
Beliefs μ_i are updated to minimise variational free energy.

The message-passing scheme follows Friston (2005):
    ε_i = Π_i (x_i − g_i(μ_{i+1}))         — precision-weighted error
    dμ_i/dt = ε_i − ε_{i+1} × ∂g_i/∂μ_i    — belief update

For the SCPN:
    x_i = phase observations at layer i (from quantum measurement)
    g_i = phase prediction from layer i+1 (coupling through K_nm)
    Π_i = precision at layer i (diagonal of K_nm column sums)

This maps the abstract predictive coding hierarchy directly onto
the 15+1 SCPN layer structure, where higher layers predict lower
layers' dynamics and prediction errors drive belief updates.

Ref:
    - Friston, Phil. Trans. R. Soc. B 360, 1249 (2005)
    - Bastos et al., Neuron 76, 695 (2012)
    - Bogacz, J. Math. Psych. 76, 198 (2017) — tutorial
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .variational_free_energy import free_energy_gradient, variational_free_energy

try:
    from scpn_quantum_engine import (
        hierarchical_prediction_error_rust as _pe_rust,
    )

    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False


@dataclass
class PredictiveCodingResult:
    """Result of one predictive coding cycle across layers."""

    prediction_errors: np.ndarray  # ε_i per layer
    beliefs: np.ndarray  # updated μ_i per layer
    free_energy: float  # total F across all layers
    total_error_norm: float  # ||ε||


def hierarchical_prediction_error(
    observations: np.ndarray,
    beliefs: np.ndarray,
    K: np.ndarray,
) -> np.ndarray:
    """Compute precision-weighted prediction errors across layers.

    For each layer i, the prediction from layer i+1 is:
        x̂_i = Σ_j K[i,j] × μ_j / Σ_j K[i,j]  (coupling-weighted mean)

    This is the simplest generative model: each layer's state is
    predicted as the precision-weighted average of connected layers.

    Prediction error: ε_i = x_i − x̂_i, weighted by local precision
    Π_i = Σ_j K[i,j] (total coupling strength).

    Args:
        observations: measured phases x_i, shape (n,)
        beliefs: current beliefs μ_i, shape (n,)
        K: coupling matrix, shape (n, n)

    Returns:
        Precision-weighted prediction errors ε_i, shape (n,)
    """
    if _HAS_RUST:
        return np.asarray(_pe_rust(observations, beliefs, K))

    n = len(observations)
    errors = np.zeros(n)

    for i in range(n):
        k_row = K[i, :]
        total_coupling = np.sum(k_row)
        if total_coupling < 1e-15:
            errors[i] = observations[i] - beliefs[i]
            continue

        # Coupling-weighted prediction from connected layers
        prediction = np.sum(k_row * beliefs) / total_coupling
        precision = total_coupling
        errors[i] = precision * (observations[i] - prediction)

    return errors


def predictive_coding_step(
    observations: np.ndarray,
    beliefs: np.ndarray,
    K: np.ndarray,
    learning_rate: float = 0.01,
    sigma: np.ndarray | None = None,
) -> PredictiveCodingResult:
    """Single predictive coding update step.

    1. Compute prediction errors ε_i
    2. Update beliefs: μ_i ← μ_i − lr × ∂F/∂μ_i
    3. Compute total free energy

    Args:
        observations: measured phases, shape (n,)
        beliefs: current beliefs, shape (n,)
        K: K_nm coupling matrix, shape (n, n)
        learning_rate: gradient step size
        sigma: belief covariance (default: 0.1 × I)

    Returns:
        PredictiveCodingResult with updated beliefs and errors.
    """
    n = len(observations)
    if sigma is None:
        sigma = 0.1 * np.eye(n)

    errors = hierarchical_prediction_error(observations, beliefs, K)
    grad = free_energy_gradient(mu=beliefs, sigma=sigma, x_observed=observations, K_precision=K)
    new_beliefs = beliefs - learning_rate * grad
    fe = variational_free_energy(
        mu=new_beliefs, sigma=sigma, x_observed=observations, K_precision=K
    )

    return PredictiveCodingResult(
        prediction_errors=errors,
        beliefs=new_beliefs,
        free_energy=fe.free_energy,
        total_error_norm=float(np.linalg.norm(errors)),
    )
