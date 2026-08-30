# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Knm Key
"""K_nm coupling matrix to key material pipeline.

Prepares the VQE ground state of H(K_nm), measures in agreed basis,
extracts correlated bit string. Security: K_nm has 120 independent
continuous parameters — eavesdropper must reconstruct all 120 to
reproduce ground-state correlations.

Key material (raw and extracted bits) lives in ordinary NumPy arrays;
CPython provides no reliable memory zeroisation, so key bits must be
assumed to persist in process memory until interpreter exit.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from qiskit.quantum_info import Statevector

from .._constants import QBER_SECURITY_THRESHOLD
from ..phase.phase_vqe import PhaseVQE


def prepare_key_state(
    K: NDArray[np.float64],
    omega: NDArray[np.float64],
    ansatz_reps: int = 2,
    maxiter: int = 200,
) -> dict[str, Any]:
    """Build the VQE-optimized circuit encoding the K_nm ground state.

    Parameters
    ----------
    K
        Coupling matrix encoded by the phase Hamiltonian.
    omega
        Intrinsic angular frequency of every qubit/layer.
    ansatz_reps
        Number of repetitions in the VQE ansatz.
    maxiter
        Maximum optimizer iterations.

    Returns
    -------
    dict[str, Any]
        Bound circuit, ground energy, statevector, and qubit count.

    """
    vqe = PhaseVQE(K, omega, ansatz_reps=ansatz_reps)
    result = vqe.solve(maxiter=maxiter)
    bound_circuit = vqe.ansatz.assign_parameters(result["optimal_params"])
    sv = Statevector.from_instruction(bound_circuit)
    return {
        "circuit": bound_circuit,
        "energy": result["ground_energy"],
        "statevector": sv,
        "n_qubits": K.shape[0],
    }


def extract_raw_key(
    counts: dict[str, int],
    basis: str,
    keep_qubits: list[int] | None = None,
) -> NDArray[np.uint8]:
    """Sift measurement results into raw key bits.

    Parameters
    ----------
    counts
        Measurement outcome counts keyed by Qiskit bitstrings.
    basis
        Declared measurement-basis label. The current majority-vote extraction
        is basis independent and assumes counts are already in that basis.
    keep_qubits
        Qubit indices to extract; ``None`` selects every qubit.

    Returns
    -------
    numpy.ndarray
        One-dimensional ``uint8`` array of majority-vote bits.

    """
    n_qubits = len(next(iter(counts)))
    if keep_qubits is None:
        keep_qubits = list(range(n_qubits))

    # Majority vote per qubit across all shots
    bits: NDArray[np.uint8] = np.zeros(len(keep_qubits), dtype=np.uint8)
    for q_idx, q in enumerate(keep_qubits):
        ones = sum(c for bitstring, c in counts.items() if bitstring[-(q + 1)] == "1")
        zeros = sum(c for bitstring, c in counts.items() if bitstring[-(q + 1)] == "0")
        bits[q_idx] = 1 if ones > zeros else 0
    return bits


def estimate_qber(alice_bits: NDArray[np.uint8], bob_bits: NDArray[np.uint8]) -> float:
    """Estimate quantum bit error rate from a shared verification subset.

    Parameters
    ----------
    alice_bits
        Alice's verification bits.
    bob_bits
        Bob's verification bits.

    Returns
    -------
    float
        Fraction of disagreements, or ``1.0`` for an empty subset. The
        BB84-family policy threshold is handled by privacy amplification.

    """
    if len(alice_bits) == 0:
        return 1.0
    return float(np.mean(alice_bits != bob_bits))


def privacy_amplification(
    raw_key: NDArray[np.uint8], qber: float, *, seed: int
) -> NDArray[np.uint8]:
    """Toeplitz-hash privacy amplification (Universal₂, leftover-hash lemma).

    The extractor is a random binary Toeplitz matrix ``T`` of shape
    ``(n_secure_bits, len(raw_key))`` whose diagonals are drawn from the
    ``seed``-keyed PRNG. Binary Toeplitz matrices form a Universal₂ hash
    family, so the leftover-hash lemma bounds the adversary's information
    on the output ``T @ raw_key mod 2``. The output length is the
    asymptotic BB84 secret fraction ``1 - 2*h2(QBER)`` of the input length
    (Shor & Preskill, PRL 85 441); finite-key corrections are out of scope
    for this simulation-only module.

    Parameters
    ----------
    raw_key
        Sifted ``uint8`` key bits in ``{0, 1}``.
    qber
        Estimated quantum bit error rate.
    seed
        PRNG seed selecting the Toeplitz family member. A real deployment must
        use fresh public randomness agreed after the raw key exists; the
        parameter is required so no entropy is silently fabricated.

    Returns
    -------
    numpy.ndarray
        Extracted ``uint8`` bits of length ``n_secure_bits``; empty above the
        QBER security threshold or when the secret fraction rounds to zero.

    """
    if qber >= QBER_SECURITY_THRESHOLD:
        return np.zeros(0, dtype=np.uint8)

    h2_qber = -qber * np.log2(qber + 1e-15) - (1 - qber) * np.log2(1 - qber + 1e-15)
    secret_fraction = max(0.0, 1.0 - 2.0 * h2_qber)
    n_bits = int(raw_key.size)
    n_secure_bits = int(n_bits * secret_fraction)
    if n_secure_bits == 0:
        return np.zeros(0, dtype=np.uint8)

    rng = np.random.default_rng(seed)
    diagonals = rng.integers(0, 2, size=n_secure_bits + n_bits - 1, dtype=np.uint8)
    # T[i, j] = diagonals[i - j + n_bits - 1] — constant along each diagonal.
    rows = np.arange(n_secure_bits)[:, np.newaxis]
    cols = np.arange(n_bits)[np.newaxis, :]
    toeplitz = diagonals[rows - cols + (n_bits - 1)]
    product: NDArray[np.uint8] = (
        (toeplitz.astype(np.uint64) @ raw_key.astype(np.uint64)) % 2
    ).astype(np.uint8)
    return product
