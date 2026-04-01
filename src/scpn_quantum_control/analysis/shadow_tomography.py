# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Shadow Tomography
"""Classical shadow tomography for efficient state characterisation.

Classical shadows (Huang, Kueng, Preskill, Nature Physics 16, 2020)
enable estimation of many observables from few measurements:

    M observables from O(log(M) × max_weight × 3^max_weight) shots

vs full tomography requiring O(4^n) shots.

Protocol:
    1. Apply random single-qubit Clifford U_i to each qubit
    2. Measure in computational basis → bitstring b
    3. Classical post-processing: σ_snapshot = ⊗_i (3|b_i><b_i| - I)
    4. Average over shots: ρ_shadow = (1/T) Σ σ_t

For Pauli observable O = ⊗ P_i:
    <O> = (1/T) Σ_t Π_i f(P_i, U_i, b_i)

where f is the single-qubit median-of-means estimator.

For the Kuramoto-XY system, shadows efficiently estimate:
    - All K_nm coupling correlators <X_i X_j + Y_i Y_j>
    - Kuramoto order parameter R from single-qubit X,Y expectations
    - Entanglement witnesses
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _build_clifford_group() -> list[np.ndarray]:
    """Generate all 24 single-qubit Clifford unitaries via BFS on generators H, S.

    Two matrices are considered the same element if they agree up to global phase
    (Nebe, Rains, Sloane, 2001: the single-qubit Clifford group has order 24).
    """
    H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    S = np.array([[1, 0], [0, 1j]], dtype=complex)
    generators = [H, S]
    group: list[np.ndarray] = [np.eye(2, dtype=complex)]

    def _same_up_to_phase(A: np.ndarray, B: np.ndarray) -> bool:
        tr = np.trace(B.conj().T @ A)
        if abs(tr) < 1e-10:
            return False
        phase = tr / abs(tr)
        return bool(np.max(np.abs(A - phase * B)) < 1e-10)

    queue = [np.eye(2, dtype=complex)]
    while queue:
        current = queue.pop(0)
        for g in generators:
            candidate = g @ current
            if not any(_same_up_to_phase(candidate, existing) for existing in group):
                group.append(candidate)
                queue.append(candidate)

    return group


# Single-qubit Clifford group: all 24 elements generated from H and S.
_CLIFFORD_GATES: list[np.ndarray] = _build_clifford_group()


@dataclass
class ShadowResult:
    """Classical shadow estimation result."""

    n_qubits: int
    n_shots: int
    estimated_observables: dict[str, float]
    shadow_norm_bound: float  # statistical error bound


def _random_clifford_layer(n: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Sample random single-qubit Clifford for each qubit."""
    return [_CLIFFORD_GATES[rng.integers(len(_CLIFFORD_GATES))] for _ in range(n)]


def _apply_clifford_and_measure(
    psi: np.ndarray,
    cliffords: list[np.ndarray],
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply single-qubit Cliffords and simulate measurement.

    Returns bitstring as array of 0/1.
    """
    # Build full unitary: U = ⊗_i U_i
    U = cliffords[0]
    for i in range(1, n):
        U = np.kron(U, cliffords[i])

    # Rotated state
    psi_rot = U @ psi
    probs = np.abs(psi_rot) ** 2
    probs = probs / probs.sum()  # normalise for numerical safety

    # Sample outcome
    outcome = rng.choice(len(probs), p=probs)
    bits: np.ndarray = np.array([(outcome >> (n - 1 - i)) & 1 for i in range(n)])
    return bits


def _snapshot_operator(
    bits: np.ndarray,
    cliffords: list[np.ndarray],
    n: int,
) -> np.ndarray:
    """Build classical shadow snapshot: ⊗_i (3 U_i†|b_i><b_i|U_i - I)."""
    result = np.array([[1.0]], dtype=complex)
    for i in range(n):
        b_vec = np.array([1 - bits[i], bits[i]], dtype=complex)
        proj = np.outer(b_vec, b_vec.conj())
        U_i = cliffords[i]
        rotated_proj = U_i.conj().T @ proj @ U_i
        snapshot_i = 3.0 * rotated_proj - np.eye(2, dtype=complex)
        result = np.kron(result, snapshot_i)
    out: np.ndarray = result
    return out


def estimate_pauli_expectation(
    psi: np.ndarray,
    n: int,
    pauli_label: str,
    n_shots: int = 500,
    seed: int = 42,
) -> float:
    """Estimate <pauli_label> from classical shadows.

    pauli_label: string like "XXIY" (n characters, I/X/Y/Z).
    """
    rng = np.random.default_rng(seed)
    estimates: list[float] = []

    for _ in range(n_shots):
        cliffords = _random_clifford_layer(n, rng)
        bits = _apply_clifford_and_measure(psi, cliffords, n, rng)
        snapshot = _snapshot_operator(bits, cliffords, n)

        # Build Pauli matrix
        pauli_mat = _pauli_from_label(pauli_label, n)
        val = float(np.real(np.trace(snapshot @ pauli_mat)))
        estimates.append(val)

    return float(np.median(estimates))


def _pauli_from_label(label: str, n: int) -> np.ndarray:
    """Build n-qubit Pauli matrix from label string."""
    paulis = {
        "I": np.eye(2, dtype=complex),
        "X": np.array([[0, 1], [1, 0]], dtype=complex),
        "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
        "Z": np.array([[1, 0], [0, -1]], dtype=complex),
    }
    mat = paulis[label[0]]
    for i in range(1, n):
        mat = np.kron(mat, paulis[label[i]])
    out: np.ndarray = mat
    return out


def classical_shadow_estimation(
    psi: np.ndarray,
    n: int,
    observables: dict[str, str],
    n_shots: int = 500,
    seed: int = 42,
) -> ShadowResult:
    """Estimate multiple Pauli observables from classical shadows.

    Args:
        psi: statevector
        n: number of qubits
        observables: dict of {name: pauli_label}
        n_shots: number of shadow shots
        seed: random seed
    """
    estimated: dict[str, float] = {}
    for name, label in observables.items():
        estimated[name] = estimate_pauli_expectation(psi, n, label, n_shots, seed)

    # Shadow norm bound: O(sqrt(3^k / T)) for k-local observables
    max_weight = max(sum(1 for c in label if c != "I") for label in observables.values())
    bound = float(np.sqrt(3**max_weight / n_shots))

    return ShadowResult(
        n_qubits=n,
        n_shots=n_shots,
        estimated_observables=estimated,
        shadow_norm_bound=bound,
    )
