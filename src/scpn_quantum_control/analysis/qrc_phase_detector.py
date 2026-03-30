# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Qrc Phase Detector
"""Quantum reservoir computing as synchronization phase detector.

The reservoir IS the Kuramoto-XY Hamiltonian: the dynamics that
produce the phase transition also serve as the nonlinear map for
phase classification. Self-probing architecture.

Protocol:
1. For each K_base, evolve |0⟩ under H(K) for time t
2. Extract Pauli expectation features ⟨P_i⟩ from the evolved state
3. Label: K > K_c → "synchronized" (1), K < K_c → "desynchronized" (0)
4. Train a linear readout W on the reservoir features
5. The classification accuracy measures how well the reservoir
   distinguishes the two phases

Prior art: Kobayashi/Motome Nat Comms (2025) used QRP for QPTs
in spin chains. Nobody has used the SAME Hamiltonian as both
reservoir and the system being probed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qiskit.quantum_info import SparsePauliOp, Statevector

from ..bridge.knm_hamiltonian import knm_to_hamiltonian


@dataclass
class QRCPhaseResult:
    """QRC phase detection result."""

    accuracy: float
    n_train: int
    n_test: int
    n_features: int
    weights: np.ndarray
    k_boundary_predicted: float | None


def _pauli_features_from_hamiltonian(
    K: np.ndarray,
    omega: np.ndarray,
    max_weight: int = 2,
) -> np.ndarray:
    """Extract Pauli expectation features from the ground state of H(K).

    Uses exact diagonalization (no circuit evolution needed for ground state).
    """
    n = len(omega)
    H_op = knm_to_hamiltonian(K, omega)
    H_mat = H_op.to_matrix()
    eigenvalues, eigenvectors = np.linalg.eigh(H_mat)
    psi0 = np.ascontiguousarray(eigenvectors[:, 0])
    sv = Statevector(psi0)

    # Generate Pauli labels up to given weight
    from itertools import product as iterproduct

    paulis = ["I", "X", "Y", "Z"]
    labels = []
    for combo in iterproduct(paulis, repeat=n):
        weight = sum(1 for p in combo if p != "I")
        if 0 < weight <= max_weight:
            labels.append("".join(combo))

    features: np.ndarray = np.zeros(len(labels))
    for i, label in enumerate(labels):
        op = SparsePauliOp(label)
        features[i] = float(sv.expectation_value(op).real)

    return features


def generate_training_data(
    omega: np.ndarray,
    K_topology: np.ndarray,
    k_range: np.ndarray,
    k_threshold: float,
    max_weight: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate labeled feature matrix for QRC phase classification.

    K_topology: normalized coupling matrix (max=1).
    k_threshold: K_base value separating sync from desync labels.

    Returns (X, y) where X is (n_samples, n_features) and y is (n_samples,).
    """
    features_list = []
    labels = []

    for kb in k_range:
        K = float(kb) * K_topology
        feat = _pauli_features_from_hamiltonian(K, omega, max_weight)
        features_list.append(feat)
        labels.append(1.0 if kb >= k_threshold else 0.0)

    X: np.ndarray = np.array(features_list)
    y: np.ndarray = np.array(labels)
    return X, y


def train_linear_readout(
    X_train: np.ndarray,
    y_train: np.ndarray,
    alpha: float = 0.1,
) -> np.ndarray:
    """Ridge regression readout: W = (X^T X + αI)^{-1} X^T y."""
    n_feat = X_train.shape[1]
    W: np.ndarray = np.linalg.solve(
        X_train.T @ X_train + alpha * np.eye(n_feat),
        X_train.T @ y_train,
    )
    return W


def classify(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Binary classification: threshold at 0.5."""
    preds_raw = X @ W
    result: np.ndarray = (preds_raw > 0.5).astype(float)
    return result


def qrc_phase_detection(
    omega: np.ndarray,
    K_topology: np.ndarray,
    k_train: np.ndarray,
    k_test: np.ndarray,
    k_threshold: float,
    alpha: float = 0.1,
    max_weight: int = 2,
) -> QRCPhaseResult:
    """Full QRC pipeline: train on k_train, test on k_test.

    Returns accuracy and the trained readout weights.
    """
    X_train, y_train = generate_training_data(omega, K_topology, k_train, k_threshold, max_weight)
    X_test, y_test = generate_training_data(omega, K_topology, k_test, k_threshold, max_weight)

    W = train_linear_readout(X_train, y_train, alpha)
    y_pred = classify(X_test, W)
    accuracy = float(np.mean(y_pred == y_test))

    # Find predicted boundary: K where raw prediction crosses 0.5
    k_boundary = None
    if len(k_test) > 1:
        raw_preds = X_test @ W
        for i in range(len(k_test) - 1):
            if (raw_preds[i] - 0.5) * (raw_preds[i + 1] - 0.5) < 0:
                # Linear interpolation
                frac = (0.5 - raw_preds[i]) / (raw_preds[i + 1] - raw_preds[i])
                k_boundary = float(k_test[i] + frac * (k_test[i + 1] - k_test[i]))
                break

    return QRCPhaseResult(
        accuracy=accuracy,
        n_train=len(k_train),
        n_test=len(k_test),
        n_features=X_train.shape[1],
        weights=W,
        k_boundary_predicted=k_boundary,
    )
