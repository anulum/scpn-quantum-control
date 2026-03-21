# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
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
from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.synthesis import LieTrotter

from ..bridge.knm_hamiltonian import knm_to_hamiltonian


@dataclass
class ReservoirResult:
    """Quantum reservoir computation result."""

    features: np.ndarray  # (n_features,) Pauli expectation values
    n_qubits: int
    n_features: int
    feature_labels: list[str]


def _pauli_feature_set(n: int, max_weight: int = 2) -> list[str]:
    """Generate Pauli feature labels up to given weight."""
    labels: list[str] = []
    paulis = ["I", "X", "Y", "Z"]
    for combo in product(paulis, repeat=n):
        weight = sum(1 for p in combo if p != "I")
        if 0 < weight <= max_weight:
            labels.append("".join(combo))
    return labels


def reservoir_features(
    x: np.ndarray,
    K: np.ndarray,
    omega: np.ndarray | None = None,
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
    n = K.shape[0]
    if omega is None:
        omega = np.zeros(n)

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
    X: np.ndarray,
    K: np.ndarray,
    omega: np.ndarray | None = None,
    t: float = 1.0,
    max_weight: int = 2,
) -> np.ndarray:
    """Compute reservoir features for multiple inputs.

    Args:
        X: (n_samples, n_input_features) matrix
        K: coupling matrix
        omega: natural frequencies
        t: reservoir evolution time
        max_weight: Pauli weight limit

    Returns:
        (n_samples, n_reservoir_features) feature matrix
    """
    n_samples = X.shape[0]
    first = reservoir_features(X[0], K, omega, t, max_weight)
    n_feat = first.n_features

    F = np.zeros((n_samples, n_feat))
    F[0] = first.features

    for i in range(1, n_samples):
        r = reservoir_features(X[i], K, omega, t, max_weight)
        F[i] = r.features

    result: np.ndarray = F
    return result


def reservoir_ridge_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    K: np.ndarray,
    omega: np.ndarray | None = None,
    alpha: float = 1.0,
    max_weight: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Train quantum reservoir with ridge regression readout.

    Returns (weights, predictions_on_train).
    """
    F = reservoir_feature_matrix(X_train, K, omega, max_weight=max_weight)
    # Ridge: W = (F^T F + αI)^{-1} F^T y
    n_feat = F.shape[1]
    W = np.linalg.solve(F.T @ F + alpha * np.eye(n_feat), F.T @ y_train)
    preds: np.ndarray = F @ W
    return W, preds
