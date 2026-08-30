# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Quantum Kernel
"""Finite simulator fidelity kernels informed by oscillator coupling.

A quantum kernel K(x, x') = |<φ(x)|φ(x')>|² maps classical feature
vectors into quantum Hilbert space via a parameterised encoding circuit.

For the Kuramoto-XY system, the encoding uses the coupling topology:
    |φ(x)> = U_K(x) |0>

where U_K encodes features x into the XY Hamiltonian evolution:
    U_K(x) = exp(-i Σ_ij x_k K_ij (X_i X_j + Y_i Y_j) t)

The legacy encoder also applies local feature rotations. The edge-aligned
encoder added for topology-kernel instead assigns one feature to every canonical
undirected qubit pair and modulates only that coupling entry. Both paths use
exact local statevectors. They are feature-map implementations, not evidence
of quantum advantage, domain fitness, provider execution, or hardware results.

Reference: Havlíček et al., Nature 567, 209 (2019), DOI
10.1038/s41586-019-0980-2.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import Statevector
from qiskit.synthesis import LieTrotter

from ..bridge.knm_hamiltonian import knm_to_hamiltonian

Edge = tuple[int, int]


@dataclass
class QuantumKernelResult:
    """Quantum kernel computation result."""

    kernel_matrix: NDArray[np.float64]  # K(x_i, x_j) for all pairs
    n_samples: int
    n_qubits: int
    feature_dim: int


def _validated_coupling_matrix(K: NDArray[np.float64], n_qubits: int) -> NDArray[np.float64]:
    """Return a finite square coupling matrix matching the qubit count."""
    K_array = np.asarray(K, dtype=float)
    if K_array.ndim != 2 or K_array.shape[0] != K_array.shape[1]:
        raise ValueError("K must be a square 2-D coupling matrix.")
    if n_qubits <= 0:
        raise ValueError("n_qubits must be positive.")
    if K_array.shape[0] != n_qubits:
        raise ValueError("n_qubits must match the dimension of K.")
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


def canonical_edge_pairs(n_qubits: int) -> tuple[Edge, ...]:
    """Return canonical undirected pairs in upper-triangular order.

    Parameters
    ----------
    n_qubits:
        Positive qubit count.

    Returns
    -------
    tuple[tuple[int, int], ...]
        Pairs ``(i, j)`` with ``0 <= i < j < n_qubits``.

    Raises
    ------
    ValueError
        If ``n_qubits`` is not a positive integer.

    """
    if isinstance(n_qubits, bool) or not isinstance(n_qubits, int) or n_qubits < 1:
        raise ValueError("n_qubits must be a positive integer.")
    return tuple((i, j) for i in range(n_qubits) for j in range(i + 1, n_qubits))


def encode_topology_edge_features(
    x: NDArray[np.float64],
    K: NDArray[np.float64],
    n_qubits: int,
    *,
    t: float = 0.8,
    reps: int = 2,
    max_qubits: int = 8,
) -> Statevector:
    """Encode canonical edge features through a coupling-modulated XY circuit.

    Feature ``x[k]`` multiplies the coupling for the ``k``-th pair returned by
    :func:`canonical_edge_pairs`. The circuit prepares ``|+>**n`` and applies
    one Trotter-synthesised evolution of the resulting XY Hamiltonian. No local
    feature rotations or provider calls are performed.

    Parameters
    ----------
    x:
        Finite vector of length ``n_qubits * (n_qubits - 1) // 2``.
    K:
        Finite symmetric ``(n_qubits, n_qubits)`` coupling matrix with zero
        diagonal. Zero entries are topology masks and ignore their features.
    n_qubits:
        Qubit count, bounded by ``max_qubits`` before dense simulation.
    t:
        Positive finite evolution time.
    reps:
        Positive integer Lie-Trotter repetition count, at most 16.
    max_qubits:
        Positive dense-state allocation budget, at most 20.

    Returns
    -------
    qiskit.quantum_info.Statevector
        Exact local statevector of dimension ``2**n_qubits``.

    Raises
    ------
    ValueError
        If shapes, symmetry, diagonal, finiteness, evolution settings, or
        resource budgets violate the contract.

    """
    if isinstance(max_qubits, bool) or not isinstance(max_qubits, int):
        raise ValueError("max_qubits must be an integer.")
    if max_qubits < 1 or max_qubits > 20:
        raise ValueError("max_qubits must lie in [1, 20].")
    if n_qubits > max_qubits:
        raise ValueError("n_qubits exceeds the configured dense-state budget.")
    coupling = _validated_coupling_matrix(K, n_qubits)
    if not np.allclose(coupling, coupling.T, rtol=0.0, atol=1.0e-12):
        raise ValueError("K must be symmetric for edge-aligned encoding.")
    if not np.allclose(np.diag(coupling), 0.0, rtol=0.0, atol=1.0e-12):
        raise ValueError("K diagonal must be zero for edge-aligned encoding.")
    features = _validated_feature_vector(x)
    pairs = canonical_edge_pairs(n_qubits)
    if features.shape != (len(pairs),):
        raise ValueError(f"x must have shape ({len(pairs)},) for canonical edge features.")
    if not np.isfinite(t) or t <= 0.0:
        raise ValueError("t must be finite and positive.")
    if isinstance(reps, bool) or not isinstance(reps, int) or not 1 <= reps <= 16:
        raise ValueError("reps must be an integer in [1, 16].")

    modulated = np.zeros_like(coupling)
    for feature, (i, j) in zip(features, pairs, strict=True):
        value = float(feature * coupling[i, j])
        modulated[i, j] = value
        modulated[j, i] = value
    hamiltonian = knm_to_hamiltonian(modulated, np.zeros(n_qubits))
    circuit = QuantumCircuit(n_qubits)
    circuit.h(range(n_qubits))
    circuit.append(
        PauliEvolutionGate(
            hamiltonian,
            time=float(t),
            synthesis=LieTrotter(reps=reps),
        ),
        range(n_qubits),
    )
    return Statevector.from_instruction(circuit)


def _encode_features(
    x: NDArray[np.float64],
    K: NDArray[np.float64],
    n_qubits: int,
    t: float = 1.0,
    reps: int = 2,
) -> Statevector:
    """Encode feature vector x into quantum state via K_nm-informed circuit.

    x modulates the coupling strengths: K_eff_ij = x[k] × K_ij.
    """
    K = _validated_coupling_matrix(K, n_qubits)
    x = _validated_feature_vector(x)

    # Map features to coupling modulation
    n_features = len(x)

    # Distribute features across coupling pairs (cyclic if fewer features)
    modulated_K = K.copy()
    pair_idx = 0
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            feat_idx = pair_idx % n_features
            modulated_K[i, j] *= x[feat_idx]
            modulated_K[j, i] *= x[feat_idx]
            pair_idx += 1

    omega = np.zeros(n_qubits)
    H = knm_to_hamiltonian(modulated_K, omega)

    qc = QuantumCircuit(n_qubits)
    # Initial layer: Hadamard for superposition
    for q in range(n_qubits):
        qc.h(q)

    # Feature encoding layers
    synth = LieTrotter(reps=1)
    for _r in range(reps):
        evo = PauliEvolutionGate(H, time=t, synthesis=synth)
        qc.append(evo, range(n_qubits))
        # Interleave with feature-dependent single-qubit rotations
        for q in range(min(n_qubits, n_features)):
            qc.rz(float(x[q]), q)

    return Statevector.from_instruction(qc)


def quantum_kernel_entry(
    x1: NDArray[np.float64],
    x2: NDArray[np.float64],
    K: NDArray[np.float64],
    n_qubits: int,
) -> float:
    """Compute single kernel entry K(x1, x2) = |<φ(x1)|φ(x2)>|²."""
    x1 = _validated_feature_vector(x1, name="x1")
    x2 = _validated_feature_vector(x2, name="x2")
    if x1.shape != x2.shape:
        raise ValueError("x1 and x2 must have the same feature dimension.")
    K = _validated_coupling_matrix(K, n_qubits)
    sv1 = _encode_features(x1, K, n_qubits)
    sv2 = _encode_features(x2, K, n_qubits)
    overlap = abs(np.dot(sv1.data.conj(), sv2.data)) ** 2
    return float(overlap)


def compute_kernel_matrix(
    X: NDArray[np.float64],
    K: NDArray[np.float64],
    n_qubits: int,
) -> QuantumKernelResult:
    """Compute the full kernel matrix for a set of feature vectors.

    Args:
        X: (n_samples, n_features) feature matrix
        K: coupling matrix
        n_qubits: number of qubits for encoding

    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a 2-D feature matrix.")
    if X.shape[0] == 0:
        raise ValueError("X must contain at least one sample.")
    if X.shape[1] == 0:
        raise ValueError("X must contain at least one feature.")
    if not np.all(np.isfinite(X)):
        raise ValueError("X must contain only finite values.")
    K = _validated_coupling_matrix(K, n_qubits)

    n_samples, n_features = X.shape
    kernel = np.zeros((n_samples, n_samples))

    # Precompute all statevectors
    states: list[Statevector] = []
    for i in range(n_samples):
        states.append(_encode_features(X[i], K, n_qubits))

    for i in range(n_samples):
        for j in range(i, n_samples):
            overlap = abs(np.dot(states[i].data.conj(), states[j].data)) ** 2
            kernel[i, j] = overlap
            kernel[j, i] = overlap

    return QuantumKernelResult(
        kernel_matrix=kernel,
        n_samples=n_samples,
        n_qubits=n_qubits,
        feature_dim=n_features,
    )
