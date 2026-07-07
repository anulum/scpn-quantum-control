# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Quantum Reservoir
"""Quantum reservoir computing for exponential state space expansion.

A quantum reservoir maps n input qubits to 2^n measurement features,
providing exponential compression: 4 qubits → 16 features, 7 → 128.

For the Kuramoto-XY system, the reservoir uses the Hamiltonian
evolution as the nonlinear dynamical map:

    Input: x (classical feature vector, n_features)
    Encoding: |ψ(x)> = U_K(x) |0>
    Reservoir: Evolve under H(K) for time t
    Features: <P_i> for all Pauli strings up to weight k

With n qubits and weight-k Pauli features:
    n_features = Σ_{w=0}^{k} C(n,w) × 3^w

For k=2 (pairwise): n_features = 1 + 3n + 9n(n-1)/2

The readout is a classical linear layer:
    y = W_out × features

Training: only W_out is trained (reservoir is fixed), making
this equivalent to kernel ridge regression with a quantum kernel.

Ref: Fujii & Nakajima, Phys. Rev. Applied 8, 024030 (2017).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
from numpy.typing import NDArray
from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.synthesis import LieTrotter

from ..bridge.knm_hamiltonian import knm_to_hamiltonian


@dataclass
class ReservoirResult:
    """Quantum reservoir computation result."""

    features: NDArray[np.float64]  # (n_features,) Pauli expectation values
    n_qubits: int
    n_features: int
    feature_labels: list[str]


def _validated_coupling_matrix(K: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return a finite square coupling matrix."""
    K_array = np.asarray(K, dtype=float)
    if K_array.ndim != 2 or K_array.shape[0] != K_array.shape[1]:
        raise ValueError("K must be a square 2-D coupling matrix.")
    if K_array.shape[0] == 0:
        raise ValueError("K must contain at least one oscillator.")
    if not np.all(np.isfinite(K_array)):
        raise ValueError("K must contain only finite values.")
    return K_array


def _validated_feature_vector(x: NDArray[np.float64], *, name: str = "x") -> NDArray[np.float64]:
    """Return a finite non-empty 1-D feature vector."""
    x_array = np.asarray(x, dtype=float)
    if x_array.ndim != 1:
        raise ValueError(f"{name} must be a 1-D feature vector.")
    if x_array.size == 0:
        raise ValueError(f"{name} must contain at least one feature.")
    if not np.all(np.isfinite(x_array)):
        raise ValueError(f"{name} must contain only finite values.")
    return x_array


def _validated_omega(omega: NDArray[np.float64] | None, n: int) -> NDArray[np.float64]:
    """Return a finite frequency vector matching ``K``."""
    if omega is None:
        return np.zeros(n)
    omega_array = np.asarray(omega, dtype=float)
    if omega_array.ndim != 1 or omega_array.shape != (n,):
        raise ValueError("omega must be a vector matching K.")
    if not np.all(np.isfinite(omega_array)):
        raise ValueError("omega must contain only finite values.")
    return omega_array


def _validated_max_weight(max_weight: int, n: int) -> int:
    """Return a valid Pauli feature weight limit."""
    if not isinstance(max_weight, int):
        raise TypeError("max_weight must be an integer.")
    if max_weight < 1 or max_weight > n:
        raise ValueError("max_weight must be between 1 and n_qubits.")
    return max_weight


def _validated_feature_matrix(X: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return a finite non-empty 2-D feature matrix."""
    X_array = np.asarray(X, dtype=float)
    if X_array.ndim != 2:
        raise ValueError("X must be a 2-D feature matrix.")
    if X_array.shape[0] == 0:
        raise ValueError("X must contain at least one sample.")
    if X_array.shape[1] == 0:
        raise ValueError("X must contain at least one feature.")
    if not np.all(np.isfinite(X_array)):
        raise ValueError("X must contain only finite values.")
    return X_array


def _pauli_feature_set(n: int, max_weight: int = 2) -> list[str]:
    """Generate Pauli feature labels up to given weight."""
    max_weight = _validated_max_weight(max_weight, n)
    labels: list[str] = []
    paulis = ["I", "X", "Y", "Z"]
    for combo in product(paulis, repeat=n):
        weight = sum(1 for p in combo if p != "I")
        if 0 < weight <= max_weight:
            labels.append("".join(combo))
    return labels


def reservoir_features(
    x: NDArray[np.float64],
    K: NDArray[np.float64],
    omega: NDArray[np.float64] | None = None,
    t: float = 1.0,
    max_weight: int = 2,
) -> ReservoirResult:
    """Compute quantum reservoir features for input x.

    Args:
        x: input feature vector (length n_qubits or less)
        K: coupling matrix (n × n)
        omega: natural frequencies
        t: reservoir evolution time
        max_weight: maximum Pauli weight for features
    """
    K = _validated_coupling_matrix(K)
    n = K.shape[0]
    x = _validated_feature_vector(x)
    omega = _validated_omega(omega, n)
    max_weight = _validated_max_weight(max_weight, n)

    H = knm_to_hamiltonian(K, omega)

    # Encode input
    qc = QuantumCircuit(n)
    for i in range(min(len(x), n)):
        qc.ry(float(x[i]) * np.pi, i)

    # Reservoir evolution
    synth = LieTrotter(reps=2)
    evo = PauliEvolutionGate(H, time=t, synthesis=synth)
    qc.append(evo, range(n))

    sv = Statevector.from_instruction(qc)

    # Measure Pauli features
    labels = _pauli_feature_set(n, max_weight)
    features = np.zeros(len(labels))
    for i, label in enumerate(labels):
        op = SparsePauliOp(label)
        features[i] = float(sv.expectation_value(op).real)

    return ReservoirResult(
        features=features,
        n_qubits=n,
        n_features=len(labels),
        feature_labels=labels,
    )


def reservoir_feature_matrix(
    X: NDArray[np.float64],
    K: NDArray[np.float64],
    omega: NDArray[np.float64] | None = None,
    t: float = 1.0,
    max_weight: int = 2,
) -> NDArray[np.float64]:
    """Compute reservoir features for multiple inputs.

    Args:
        X: (n_samples, n_input_features) matrix
        K: coupling matrix
        omega: natural frequencies
        t: reservoir evolution time
        max_weight: Pauli weight limit

    Returns
    -------
        (n_samples, n_reservoir_features) feature matrix
    """
    X = _validated_feature_matrix(X)
    K = _validated_coupling_matrix(K)
    omega = _validated_omega(omega, K.shape[0])
    max_weight = _validated_max_weight(max_weight, K.shape[0])
    n_samples = X.shape[0]
    first = reservoir_features(X[0], K, omega, t, max_weight)
    n_feat = first.n_features

    F = np.zeros((n_samples, n_feat))
    F[0] = first.features

    for i in range(1, n_samples):
        r = reservoir_features(X[i], K, omega, t, max_weight)
        F[i] = r.features

    result: NDArray[np.float64] = F
    return result


def reservoir_ridge_regression(
    X_train: NDArray[np.float64],
    y_train: NDArray[np.float64],
    K: NDArray[np.float64],
    omega: NDArray[np.float64] | None = None,
    alpha: float = 1.0,
    max_weight: int = 2,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Train quantum reservoir with ridge regression readout.

    Returns (weights, predictions_on_train).
    """
    X_train = _validated_feature_matrix(X_train)
    y_array = np.asarray(y_train, dtype=float)
    if y_array.ndim != 1 or y_array.shape != (X_train.shape[0],):
        raise ValueError("y_train must be a vector matching the number of rows in X_train.")
    if not np.all(np.isfinite(y_array)):
        raise ValueError("y_train must contain only finite values.")
    if not np.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("alpha must be finite and positive.")

    F = reservoir_feature_matrix(X_train, K, omega, max_weight=max_weight)
    # Ridge: W = (F^T F + αI)^{-1} F^T y
    n_feat = F.shape[1]
    W = np.linalg.solve(F.T @ F + alpha * np.eye(n_feat), F.T @ y_array)
    preds: NDArray[np.float64] = F @ W
    return W, preds
