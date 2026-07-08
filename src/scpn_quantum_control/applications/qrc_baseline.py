# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — QRC Baseline Comparison
"""QRC comparison helpers with a deterministic classical ESN baseline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .quantum_reservoir import reservoir_feature_matrix


@dataclass(frozen=True)
class ClassicalESNReadoutResult:
    """Classical echo-state readout fitted on a fixed reservoir."""

    features: NDArray[np.float64]
    weights: NDArray[np.float64]
    predictions: NDArray[np.float64]
    mse: float
    n_reservoir_features: int
    spectral_radius: float


@dataclass(frozen=True)
class QRCBaselineComparison:
    """Matched-feature comparison between QRC and a classical ESN readout."""

    quantum_predictions: NDArray[np.float64]
    esn_predictions: NDArray[np.float64]
    quantum_mse: float
    esn_mse: float
    n_quantum_features: int
    n_esn_features: int
    mse_delta: float


def _validated_training_matrix(X: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return a finite non-empty two-dimensional training matrix."""
    x_array = np.asarray(X, dtype=np.float64)
    if x_array.ndim != 2:
        raise ValueError("X must be a 2-D training matrix.")
    if x_array.shape[0] == 0:
        raise ValueError("X must contain at least one sample.")
    if x_array.shape[1] == 0:
        raise ValueError("X must contain at least one feature.")
    if not np.all(np.isfinite(x_array)):
        raise ValueError("X must contain only finite values.")
    return x_array


def _validated_targets(y: NDArray[np.float64], n_samples: int) -> NDArray[np.float64]:
    """Return a finite target vector with one value per sample."""
    y_array = np.asarray(y, dtype=np.float64)
    if y_array.ndim != 1 or y_array.shape != (n_samples,):
        raise ValueError("y must be a vector matching the number of rows in X.")
    if not np.all(np.isfinite(y_array)):
        raise ValueError("y must contain only finite values.")
    return y_array


def _validated_positive_float(value: float, *, name: str) -> float:
    """Return a finite positive float."""
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return result


def _validated_leak_rate(leak_rate: float) -> float:
    """Return a finite leak rate in the open-closed interval ``(0, 1]``."""
    result = float(leak_rate)
    if not np.isfinite(result) or result <= 0.0 or result > 1.0:
        raise ValueError("leak_rate must be finite and in the interval (0, 1].")
    return result


def _validated_reservoir_size(reservoir_size: int) -> int:
    """Return a positive integer reservoir size."""
    if not isinstance(reservoir_size, int):
        raise TypeError("reservoir_size must be an integer.")
    if reservoir_size <= 0:
        raise ValueError("reservoir_size must be positive.")
    return reservoir_size


def _ridge_readout(
    features: NDArray[np.float64],
    targets: NDArray[np.float64],
    alpha: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    """Fit a ridge readout and return weights, predictions, and MSE."""
    regularisation = _validated_positive_float(alpha, name="alpha")
    n_features = features.shape[1]
    weights = np.linalg.solve(
        features.T @ features + regularisation * np.eye(n_features),
        features.T @ targets,
    ).astype(np.float64)
    predictions = (features @ weights).astype(np.float64)
    mse = float(np.mean((predictions - targets) ** 2))
    return weights, predictions, mse


def classical_esn_feature_matrix(
    X: NDArray[np.float64],
    *,
    reservoir_size: int,
    spectral_radius: float = 0.9,
    input_scale: float = 0.5,
    leak_rate: float = 1.0,
    seed: int = 0,
) -> NDArray[np.float64]:
    """Return deterministic echo-state features for a sample sequence.

    Parameters
    ----------
    X:
        Input samples with shape ``(n_samples, n_features)``.
    reservoir_size:
        Number of classical reservoir units. Use the QRC feature count for a
        matched-feature comparison.
    spectral_radius:
        Target spectral radius of the recurrent matrix.
    input_scale:
        Uniform input-weight scale.
    leak_rate:
        Leaky integration rate in ``(0, 1]``.
    seed:
        Seed for the deterministic reservoir weights.

    Returns
    -------
    numpy.ndarray
        Feature matrix with shape ``(n_samples, reservoir_size)``.
    """
    x_array = _validated_training_matrix(X)
    n_reservoir = _validated_reservoir_size(reservoir_size)
    radius = _validated_positive_float(spectral_radius, name="spectral_radius")
    scale = _validated_positive_float(input_scale, name="input_scale")
    leak = _validated_leak_rate(leak_rate)

    rng = np.random.default_rng(seed)
    w_input = rng.uniform(-scale, scale, size=(n_reservoir, x_array.shape[1] + 1))
    w_recurrent = rng.standard_normal(size=(n_reservoir, n_reservoir))
    raw_radius = float(np.max(np.abs(np.linalg.eigvals(w_recurrent))))
    if raw_radius > 0.0:
        w_recurrent = (w_recurrent / raw_radius) * radius

    state = np.zeros(n_reservoir, dtype=np.float64)
    features = np.zeros((x_array.shape[0], n_reservoir), dtype=np.float64)
    for row_index, row in enumerate(x_array):
        drive = w_input @ np.concatenate((np.array([1.0], dtype=np.float64), row))
        candidate = np.tanh(drive + w_recurrent @ state)
        state = ((1.0 - leak) * state + leak * candidate).astype(np.float64)
        features[row_index] = state
    return features


def classical_esn_ridge_regression(
    X_train: NDArray[np.float64],
    y_train: NDArray[np.float64],
    *,
    reservoir_size: int,
    alpha: float = 1.0,
    spectral_radius: float = 0.9,
    input_scale: float = 0.5,
    leak_rate: float = 1.0,
    seed: int = 0,
) -> ClassicalESNReadoutResult:
    """Fit a ridge readout on deterministic classical ESN features.

    Parameters
    ----------
    X_train:
        Input samples with shape ``(n_samples, n_features)``.
    y_train:
        Target values with one value per sample.
    reservoir_size:
        Number of classical reservoir units.
    alpha:
        Ridge regularisation strength.
    spectral_radius:
        Target recurrent spectral radius.
    input_scale:
        Uniform input-weight scale.
    leak_rate:
        Leaky integration rate in ``(0, 1]``.
    seed:
        Seed for deterministic reservoir weights.

    Returns
    -------
    ClassicalESNReadoutResult
        Features, readout weights, training predictions, and MSE.
    """
    x_array = _validated_training_matrix(X_train)
    y_array = _validated_targets(y_train, x_array.shape[0])
    features = classical_esn_feature_matrix(
        x_array,
        reservoir_size=reservoir_size,
        spectral_radius=spectral_radius,
        input_scale=input_scale,
        leak_rate=leak_rate,
        seed=seed,
    )
    weights, predictions, mse = _ridge_readout(features, y_array, alpha)
    return ClassicalESNReadoutResult(
        features=features,
        weights=weights,
        predictions=predictions,
        mse=mse,
        n_reservoir_features=features.shape[1],
        spectral_radius=float(spectral_radius),
    )


def compare_quantum_reservoir_to_esn(
    X_train: NDArray[np.float64],
    y_train: NDArray[np.float64],
    K: NDArray[np.float64],
    *,
    omega: NDArray[np.float64] | None = None,
    alpha: float = 1.0,
    max_weight: int = 1,
    reservoir_size: int | None = None,
    spectral_radius: float = 0.9,
    input_scale: float = 0.5,
    leak_rate: float = 1.0,
    seed: int = 0,
) -> QRCBaselineComparison:
    """Compare the shipped QRC feature map against a classical ESN baseline.

    The default ESN size matches the quantum reservoir feature count. The
    comparison reports training-set MSE only; it is a bounded capability
    adjudication surface, not a general performance claim.

    Parameters
    ----------
    X_train:
        Input samples with shape ``(n_samples, n_features)``.
    y_train:
        Target values with one value per sample.
    K:
        Kuramoto-XY coupling matrix consumed by the existing QRC feature map.
    omega:
        Optional natural-frequency vector for the QRC feature map.
    alpha:
        Ridge regularisation strength used by both readouts.
    max_weight:
        Maximum Pauli-string weight used by the QRC feature map.
    reservoir_size:
        Classical ESN feature count. When omitted, it matches the QRC count.
    spectral_radius:
        Target ESN recurrent spectral radius.
    input_scale:
        Uniform ESN input-weight scale.
    leak_rate:
        ESN leaky integration rate.
    seed:
        Seed for deterministic ESN weights.

    Returns
    -------
    QRCBaselineComparison
        Matched-feature predictions and MSE values for the two readouts.
    """
    x_array = _validated_training_matrix(X_train)
    y_array = _validated_targets(y_train, x_array.shape[0])
    quantum_features = reservoir_feature_matrix(
        x_array,
        K,
        omega=omega,
        max_weight=max_weight,
    )
    _, quantum_predictions, quantum_mse = _ridge_readout(quantum_features, y_array, alpha)

    esn_size = quantum_features.shape[1] if reservoir_size is None else reservoir_size
    esn = classical_esn_ridge_regression(
        x_array,
        y_array,
        reservoir_size=esn_size,
        alpha=alpha,
        spectral_radius=spectral_radius,
        input_scale=input_scale,
        leak_rate=leak_rate,
        seed=seed,
    )
    return QRCBaselineComparison(
        quantum_predictions=quantum_predictions,
        esn_predictions=esn.predictions,
        quantum_mse=quantum_mse,
        esn_mse=esn.mse,
        n_quantum_features=quantum_features.shape[1],
        n_esn_features=esn.n_reservoir_features,
        mse_delta=quantum_mse - esn.mse,
    )


__all__ = [
    "ClassicalESNReadoutResult",
    "QRCBaselineComparison",
    "classical_esn_feature_matrix",
    "classical_esn_ridge_regression",
    "compare_quantum_reservoir_to_esn",
]
