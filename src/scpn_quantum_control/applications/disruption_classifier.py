# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Quantum disruption classifier for tokamak plasma.

Uses the quantum kernel (quantum_kernel.py) to classify plasma states
as "disruption" or "stable" from MHD mode features.

Pipeline:
    1. Extract features from plasma diagnostics (synthetic here)
    2. Encode features via K_nm-informed quantum kernel
    3. Classify using kernel ridge regression / SVM

The quantum kernel maps plasma features through the Kuramoto-XY
Hamiltonian, naturally respecting the coupling topology of MHD
mode interactions.

Note: This uses SYNTHETIC data. Real ITER disruption data requires
partnership with fusion facilities (JET, ITER, DIII-D). The synthetic
data mimics disruption precursor signatures (mode locking, beta
collapse, locked mode amplitude growth).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..bridge.knm_hamiltonian import build_knm_paper27
from .quantum_kernel import compute_kernel_matrix


@dataclass
class DisruptionClassifierResult:
    """Disruption classifier result."""

    accuracy: float
    n_train: int
    n_test: int
    predictions: np.ndarray
    labels: np.ndarray
    kernel_n_qubits: int


def generate_synthetic_disruption_data(
    n_samples: int = 50,
    n_features: int = 5,
    disruption_fraction: float = 0.3,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic tokamak disruption data.

    Features mimic:
        0: locked mode amplitude (high → disruption)
        1: beta_N (drops before disruption)
        2: q95 (safety factor, low → disruption)
        3: plasma current derivative (negative → disruption)
        4: radiated power fraction (high → disruption)

    Labels: 0 = stable, 1 = disruption.
    """
    rng = np.random.default_rng(seed)
    n_disrupt = int(n_samples * disruption_fraction)
    n_stable = n_samples - n_disrupt

    # Stable plasmas
    X_stable = rng.normal(loc=[0.1, 2.5, 3.5, 0.0, 0.3], scale=0.3, size=(n_stable, n_features))
    X_stable = np.clip(X_stable, 0, 5)

    # Disrupting plasmas
    X_disrupt = rng.normal(loc=[0.8, 1.2, 2.0, -0.5, 0.7], scale=0.3, size=(n_disrupt, n_features))
    X_disrupt = np.clip(X_disrupt, 0, 5)

    X = np.vstack([X_stable, X_disrupt])
    y = np.array([0] * n_stable + [1] * n_disrupt)

    # Shuffle
    idx = rng.permutation(n_samples)
    return X[idx], y[idx]


def train_disruption_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_qubits: int = 4,
    alpha: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Train quantum kernel classifier for disruption prediction.

    Uses kernel ridge regression: w = (K + αI)^{-1} y.
    Returns (weights, kernel_matrix).
    """
    K_coupling = build_knm_paper27(L=n_qubits)
    kernel_result = compute_kernel_matrix(X_train, K_coupling, n_qubits)
    K_mat = kernel_result.kernel_matrix

    n = K_mat.shape[0]
    weights = np.linalg.solve(K_mat + alpha * np.eye(n), y_train.astype(float))
    return weights, K_mat


def predict_disruption(
    X_test: np.ndarray,
    X_train: np.ndarray,
    weights: np.ndarray,
    n_qubits: int = 4,
) -> np.ndarray:
    """Predict disruption using trained quantum kernel classifier."""
    from .quantum_kernel import quantum_kernel_entry

    K_coupling = build_knm_paper27(L=n_qubits)
    n_test = X_test.shape[0]
    n_train = X_train.shape[0]

    # Compute test-train kernel entries
    K_test = np.zeros((n_test, n_train))
    for i in range(n_test):
        for j in range(n_train):
            K_test[i, j] = quantum_kernel_entry(X_test[i], X_train[j], K_coupling, n_qubits)

    scores = K_test @ weights
    predictions: np.ndarray = (scores > 0.5).astype(int)
    return predictions


def run_disruption_benchmark(
    n_train: int = 30,
    n_test: int = 20,
    n_qubits: int = 4,
    seed: int = 42,
) -> DisruptionClassifierResult:
    """Full disruption classification benchmark on synthetic data."""
    X, y = generate_synthetic_disruption_data(n_samples=n_train + n_test, seed=seed)
    X_train, y_train = X[:n_train], y[:n_train]
    X_test, y_test = X[n_train:], y[n_train:]

    weights, _ = train_disruption_classifier(X_train, y_train, n_qubits)
    preds = predict_disruption(X_test, X_train, weights, n_qubits)

    accuracy = float(np.mean(preds == y_test))

    return DisruptionClassifierResult(
        accuracy=accuracy,
        n_train=n_train,
        n_test=n_test,
        predictions=preds,
        labels=y_test,
        kernel_n_qubits=n_qubits,
    )
