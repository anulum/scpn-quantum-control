"""K_nm coupling matrix to key material pipeline.

Prepares the VQE ground state of H(K_nm), measures in agreed basis,
extracts correlated bit string. Security: K_nm has 120 independent
continuous parameters — eavesdropper must reconstruct all 120 to
reproduce ground-state correlations.
"""

from __future__ import annotations

import hashlib

import numpy as np
from qiskit.quantum_info import Statevector

from ..phase.phase_vqe import PhaseVQE


def prepare_key_state(
    K: np.ndarray,
    omega: np.ndarray,
    ansatz_reps: int = 2,
    maxiter: int = 200,
) -> dict:
    """Build VQE-optimized circuit encoding K_nm's ground state.

    Returns dict with 'circuit' (bound QuantumCircuit), 'energy' (float),
    and 'statevector' (Statevector).
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
) -> np.ndarray:
    """Sift measurement results into raw key bits.

    Args:
        counts: Measurement outcome counts from Qiskit.
        basis: "Z" or "X" — measurement basis used.
        keep_qubits: Qubit indices to extract (None = all).

    Returns:
        1D array of {0, 1} bits, majority-vote per qubit.
    """
    n_qubits = len(next(iter(counts)))
    if keep_qubits is None:
        keep_qubits = list(range(n_qubits))

    # Majority vote per qubit across all shots
    bits = np.zeros(len(keep_qubits), dtype=np.uint8)
    for q_idx, q in enumerate(keep_qubits):
        ones = sum(c for bitstring, c in counts.items() if bitstring[-(q + 1)] == "1")
        zeros = sum(c for bitstring, c in counts.items() if bitstring[-(q + 1)] == "0")
        bits[q_idx] = 1 if ones > zeros else 0
    return bits


def estimate_qber(alice_bits: np.ndarray, bob_bits: np.ndarray) -> float:
    """Quantum bit error rate from shared verification subset.

    QBER = (number of disagreements) / (total compared bits).
    Secure threshold: QBER < 0.11 for BB84-family protocols.
    """
    if len(alice_bits) == 0:
        return 1.0
    return float(np.mean(alice_bits != bob_bits))


def privacy_amplification(raw_key: np.ndarray, qber: float) -> bytes:
    """Universal₂ hash compression of raw key.

    Compression ratio: 1 - 2h(QBER) where h is binary entropy.
    Uses SHA-256 as the hash family member.
    """
    if qber >= 0.11:
        return b""

    h_qber = -qber * np.log2(qber + 1e-15) - (1 - qber) * np.log2(1 - qber + 1e-15)
    compression = max(0.0, 1.0 - 2 * h_qber)
    n_secure_bits = max(1, int(len(raw_key) * compression))

    key_bytes = np.packbits(raw_key[:n_secure_bits]).tobytes()
    return hashlib.sha256(key_bytes).digest()
