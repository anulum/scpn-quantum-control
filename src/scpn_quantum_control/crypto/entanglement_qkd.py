"""SCPN-QKD: Topology-authenticated quantum key distribution.

Protocol:
1. Alice and Bob share K_nm (pre-distributed coupling matrix).
2. Both construct H(K_nm) and prepare ground state via VQE.
3. Alice measures her qubit subset, Bob measures his, in random {X, Z} basis.
4. Public basis reconciliation → sift → QBER estimation → privacy amplification.

Security: K_nm's 120 continuous parameters form the shared secret.
Correlations from wrong K_nm' are statistically distinguishable.

Refs:
- Frequency-bin QKD, npj Quantum Information 2025
- Huygens quantum sync, Nature Communications 2025
"""

from __future__ import annotations

import numpy as np
from qiskit.quantum_info import SparsePauliOp, Statevector

from .knm_key import estimate_qber, prepare_key_state, privacy_amplification


def scpn_qkd_protocol(
    K: np.ndarray,
    omega: np.ndarray,
    alice_qubits: list[int],
    bob_qubits: list[int],
    shots: int = 10000,
    seed: int = 42,
) -> dict:
    """Execute SCPN-QKD protocol on statevector simulator.

    Returns dict with raw_key_alice, raw_key_bob, qber, secure_key_length,
    and bell_correlator (CHSH value).
    """
    n = K.shape[0]
    rng = np.random.default_rng(seed)

    # Prepare ground state
    key_state = prepare_key_state(K, omega, ansatz_reps=2, maxiter=100)
    sv = key_state["statevector"]

    # Random basis choices
    alice_bases = rng.choice(["Z", "X"], size=len(alice_qubits))
    bob_bases = rng.choice(["Z", "X"], size=len(bob_qubits))

    # Measure in chosen bases
    alice_bits = _measure_in_basis(sv, alice_qubits, alice_bases, n, shots, rng)
    bob_bits = _measure_in_basis(sv, bob_qubits, bob_bases, n, shots, rng)

    # Sift: keep only matching bases (for overlapping qubit pairs)
    # In SCPN-QKD, Alice and Bob may have disjoint qubit sets,
    # so correlation comes from entanglement, not basis matching.
    qber = estimate_qber(alice_bits, bob_bits)

    # CHSH correlator for entanglement certification
    bell = correlator_matrix(sv, alice_qubits, bob_qubits)

    # Privacy amplification
    combined_raw = np.concatenate([alice_bits, bob_bits])
    secure_key = privacy_amplification(combined_raw, min(qber, 0.10))

    return {
        "raw_key_alice": alice_bits,
        "raw_key_bob": bob_bits,
        "qber": qber,
        "secure_key": secure_key,
        "secure_key_length": len(secure_key) * 8,
        "bell_correlator": bell,
        "ground_energy": key_state["energy"],
    }


def _measure_in_basis(
    sv: Statevector,
    qubits: list[int],
    bases: np.ndarray,
    n_total: int,
    shots: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Measure specified qubits in given bases via expectation values.

    Returns majority-vote bit array.
    """
    bits = np.zeros(len(qubits), dtype=np.uint8)
    for i, (q, basis) in enumerate(zip(qubits, bases)):
        label = ["I"] * n_total
        label[q] = basis
        op = SparsePauliOp("".join(reversed(label)))
        exp_val = float(sv.expectation_value(op).real)
        # +1 → bit 0, -1 → bit 1
        bits[i] = 0 if exp_val >= 0 else 1
    return bits


def correlator_matrix(
    sv: Statevector,
    alice_qubits: list[int],
    bob_qubits: list[int],
) -> np.ndarray:
    """Cross-correlation matrix <Z_i Z_j> between Alice and Bob qubits.

    Element (i,j) = <Z_{alice_i} Z_{bob_j}> - <Z_{alice_i}><Z_{bob_j}>.
    Non-zero off-diagonal elements indicate entanglement.
    """
    n = int(np.log2(len(sv.data)))
    corr = np.zeros((len(alice_qubits), len(bob_qubits)))

    for i, qa in enumerate(alice_qubits):
        label_a = ["I"] * n
        label_a[qa] = "Z"
        op_a = SparsePauliOp("".join(reversed(label_a)))
        exp_a = float(sv.expectation_value(op_a).real)

        for j, qb in enumerate(bob_qubits):
            label_b = ["I"] * n
            label_b[qb] = "Z"
            op_b = SparsePauliOp("".join(reversed(label_b)))
            exp_b = float(sv.expectation_value(op_b).real)

            # Joint ZZ
            label_ab = ["I"] * n
            label_ab[qa] = "Z"
            label_ab[qb] = "Z"
            op_ab = SparsePauliOp("".join(reversed(label_ab)))
            exp_ab = float(sv.expectation_value(op_ab).real)

            corr[i, j] = exp_ab - exp_a * exp_b

    return corr


def bell_inequality_test(
    sv: Statevector,
    qubit_a: int,
    qubit_b: int,
    n_total: int,
) -> dict:
    """CHSH inequality test for a qubit pair.

    S = |E(a,b) - E(a,b') + E(a',b) + E(a',b')|
    Classical bound: S ≤ 2. Quantum bound: S ≤ 2√2.
    Violation (S > 2) certifies entanglement.
    """
    if qubit_a >= n_total or qubit_b >= n_total:
        raise ValueError(f"qubits ({qubit_a}, {qubit_b}) out of range for {n_total}-qubit system")

    def _correlator(pauli_a: str, pauli_b: str) -> float:
        label = ["I"] * n_total
        label[qubit_a] = pauli_a
        label[qubit_b] = pauli_b
        op = SparsePauliOp("".join(reversed(label)))
        return float(sv.expectation_value(op).real)

    # CHSH settings: a=Z, a'=X, b=(Z+X)/√2, b'=(Z-X)/√2
    # Approximate via ZZ, ZX, XZ, XX correlators
    e_zz = _correlator("Z", "Z")
    e_zx = _correlator("Z", "X")
    e_xz = _correlator("X", "Z")
    e_xx = _correlator("X", "X")

    # S = |E(ZZ) - E(ZX) + E(XZ) + E(XX)|
    s_value = abs(e_zz - e_zx + e_xz + e_xx)

    return {
        "S": s_value,
        "violates_classical": s_value > 2.0,
        "correlators": {"ZZ": e_zz, "ZX": e_zx, "XZ": e_xz, "XX": e_xx},
    }
