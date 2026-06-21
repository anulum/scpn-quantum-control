# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Disruption Classifier
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
from numpy.typing import NDArray

from ..bridge.knm_hamiltonian import build_knm_paper27
from .quantum_kernel import compute_kernel_matrix


@dataclass
class DisruptionClassifierResult:
    """Disruption classifier result."""

    accuracy: float
    n_train: int
    n_test: int
    predictions: NDArray[np.int64]
    labels: NDArray[np.float64]
    kernel_n_qubits: int
    source_mode: str
    publication_safe: bool


def _validated_feature_matrix(X: NDArray[np.float64], name: str) -> NDArray[np.float64]:
    features = np.asarray(X, dtype=float)
    if features.ndim != 2:
        raise ValueError(f"{name} must be a 2-D feature matrix.")
    if features.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one sample.")
    if features.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one feature.")
    if not np.all(np.isfinite(features)):
        raise ValueError(f"{name} must contain only finite values.")
    return features


def _validated_binary_labels(y_train: NDArray[np.float64], n_samples: int) -> NDArray[np.float64]:
    labels = np.asarray(y_train)
    if labels.ndim != 1:
        raise ValueError("y_train must be a 1-D label vector.")
    if labels.shape[0] != n_samples:
        raise ValueError("y_train must match X_train sample count.")
    if not np.all(np.isfinite(labels.astype(float))):
        raise ValueError("y_train must contain only finite labels.")
    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("y_train labels must be binary 0/1 values.")
    return labels.astype(float)


def _validated_positive_float(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return value


def _validated_weights(weights: NDArray[np.float64], n_samples: int) -> NDArray[np.float64]:
    weight_vector = np.asarray(weights, dtype=float)
    if weight_vector.ndim != 1:
        raise ValueError("weights must be a 1-D vector.")
    if weight_vector.shape[0] != n_samples:
        raise ValueError("weights must match X_train sample count.")
    if not np.all(np.isfinite(weight_vector)):
        raise ValueError("weights must contain only finite values.")
    return weight_vector


def generate_synthetic_disruption_data(
    n_samples: int = 50,
    n_features: int = 5,
    disruption_fraction: float = 0.3,
    seed: int = 42,
    *,
    allow_synthetic: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Generate synthetic tokamak disruption data.

    Features mimic:
        0: locked mode amplitude (high → disruption)
        1: beta_N (drops before disruption)
        2: q95 (safety factor, low → disruption)
        3: plasma current derivative (negative → disruption)
        4: radiated power fraction (high → disruption)

    Labels: 0 = stable, 1 = disruption.
    """
    if not allow_synthetic:
        raise RuntimeError(
            "Refusing generated disruption data without allow_synthetic=True. "
            "Use measured plasma diagnostics for publication-safe claims."
        )
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    if n_features != 5:
        raise ValueError("n_features must be 5 for the documented tokamak feature schema.")
    if not 0.0 < disruption_fraction < 1.0:
        raise ValueError("disruption_fraction must be strictly between 0 and 1.")

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
    X_train: NDArray[np.float64],
    y_train: NDArray[np.float64],
    n_qubits: int = 4,
    alpha: float = 1.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Train quantum kernel classifier for disruption prediction.

    Uses kernel ridge regression: w = (K + αI)^{-1} y.
    Returns (weights, kernel_matrix).
    """
    X_train = _validated_feature_matrix(X_train, "X_train")
    y_train = _validated_binary_labels(y_train, X_train.shape[0])
    alpha = _validated_positive_float(alpha, "alpha")

    K_coupling = build_knm_paper27(L=n_qubits)
    kernel_result = compute_kernel_matrix(X_train, K_coupling, n_qubits)
    K_mat = kernel_result.kernel_matrix

    n = K_mat.shape[0]
    weights = np.linalg.solve(K_mat + alpha * np.eye(n), y_train).astype(np.float64)
    return weights, K_mat


def predict_disruption(
    X_test: NDArray[np.float64],
    X_train: NDArray[np.float64],
    weights: NDArray[np.float64],
    n_qubits: int = 4,
) -> NDArray[np.int64]:
    """Predict disruption using trained quantum kernel classifier."""
    from .quantum_kernel import quantum_kernel_entry

    X_test = _validated_feature_matrix(X_test, "X_test")
    X_train = _validated_feature_matrix(X_train, "X_train")
    if X_test.shape[1] != X_train.shape[1]:
        raise ValueError("X_test feature dimension must match X_train.")
    weights = _validated_weights(weights, X_train.shape[0])

    K_coupling = build_knm_paper27(L=n_qubits)
    n_test = X_test.shape[0]
    n_train = X_train.shape[0]

    # Compute test-train kernel entries
    K_test = np.zeros((n_test, n_train))
    for i in range(n_test):
        for j in range(n_train):
            K_test[i, j] = quantum_kernel_entry(X_test[i], X_train[j], K_coupling, n_qubits)

    scores = K_test @ weights
    predictions: NDArray[np.int64] = (scores > 0.5).astype(np.int64)
    return predictions


def run_disruption_benchmark(
    n_train: int = 30,
    n_test: int = 20,
    n_qubits: int = 4,
    seed: int = 42,
    *,
    allow_synthetic: bool = False,
) -> DisruptionClassifierResult:
    """Full disruption classification benchmark on explicitly generated data."""
    X, y = generate_synthetic_disruption_data(
        n_samples=n_train + n_test,
        seed=seed,
        allow_synthetic=allow_synthetic,
    )
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
        source_mode="synthetic",
        publication_safe=False,
    )
