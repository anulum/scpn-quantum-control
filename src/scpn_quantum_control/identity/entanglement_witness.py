"""Entanglement witness for disposition pairs via CHSH inequality.

Measures the CHSH S-parameter for qubit pairs in a coupled identity
state. S > 2 proves non-classical correlation between the corresponding
dispositions — they cannot be described as independent states.
"""

from __future__ import annotations

import numpy as np
from qiskit.quantum_info import Statevector


def chsh_from_statevector(
    sv: Statevector,
    qubit_a: int,
    qubit_b: int,
) -> float:
    """Compute CHSH S-parameter for a qubit pair from a multi-qubit statevector.

    Measures correlations E(a,b), E(a,b'), E(a',b), E(a',b') with
    optimal angles a=0, a'=pi/2, b=pi/4, b'=-pi/4 (Tsirelson bound: 2*sqrt(2)).

    Returns S in [0, 2*sqrt(2)]. S > 2 certifies entanglement.
    """
    n_qubits = sv.num_qubits
    if not (0 <= qubit_a < n_qubits and 0 <= qubit_b < n_qubits):
        raise ValueError(
            f"qubit indices ({qubit_a}, {qubit_b}) out of range for {n_qubits}-qubit state"
        )
    if qubit_a == qubit_b:
        raise ValueError("qubit_a and qubit_b must differ")

    angles_a = [0.0, np.pi / 2]
    angles_b = [np.pi / 4, 3 * np.pi / 4]

    correlators = np.zeros((2, 2))
    for i, theta_a in enumerate(angles_a):
        for j, theta_b in enumerate(angles_b):
            correlators[i, j] = _two_qubit_correlator(sv, qubit_a, qubit_b, theta_a, theta_b)

    S = abs(correlators[0, 0] - correlators[0, 1] + correlators[1, 0] + correlators[1, 1])
    return float(S)


def _two_qubit_correlator(
    sv: Statevector,
    qa: int,
    qb: int,
    theta_a: float,
    theta_b: float,
) -> float:
    """Expectation of (cos(θ_a)Z_a + sin(θ_a)X_a) ⊗ (cos(θ_b)Z_b + sin(θ_b)X_b).

    Qiskit Pauli labels use reversed qubit ordering: label[0] is the
    highest-index qubit, label[n-1] is qubit 0.
    """
    from qiskit.quantum_info import SparsePauliOp

    n = sv.num_qubits
    ca, sa = np.cos(theta_a), np.sin(theta_a)
    cb, sb = np.cos(theta_b), np.sin(theta_b)

    # 4 Pauli product terms: ZZ, ZX, XZ, XX on qubits (qa, qb)
    pauli_pairs = [
        ("Z", "Z", ca * cb),
        ("Z", "X", ca * sb),
        ("X", "Z", sa * cb),
        ("X", "X", sa * sb),
    ]

    terms = []
    for p_a, p_b, coeff in pauli_pairs:
        label = ["I"] * n
        label[n - 1 - qa] = p_a
        label[n - 1 - qb] = p_b
        terms.append(("".join(label), coeff))

    op = SparsePauliOp.from_list(terms).simplify()
    return float(sv.expectation_value(op).real)


def disposition_entanglement_map(
    sv: Statevector,
    disposition_labels: list[str] | None = None,
) -> dict:
    """Compute CHSH S-parameter for all qubit pairs.

    Returns dict with 'pairs' (list of {qa, qb, S, entangled}),
    'max_S', 'n_entangled', 'integration_metric' (mean S / Tsirelson bound).
    """
    n = sv.num_qubits
    if disposition_labels is not None and len(disposition_labels) != n:
        raise ValueError(f"Expected {n} labels, got {len(disposition_labels)}")

    tsirelson = 2 * np.sqrt(2)
    pairs = []

    for i in range(n):
        for j in range(i + 1, n):
            S = chsh_from_statevector(sv, i, j)
            label_i = disposition_labels[i] if disposition_labels else f"q{i}"
            label_j = disposition_labels[j] if disposition_labels else f"q{j}"
            pairs.append(
                {
                    "qa": i,
                    "qb": j,
                    "label_a": label_i,
                    "label_b": label_j,
                    "S": S,
                    "entangled": S > 2.0,
                }
            )

    s_values = [p["S"] for p in pairs]
    n_entangled = sum(1 for p in pairs if p["entangled"])
    mean_s = float(np.mean(s_values)) if s_values else 0.0

    return {
        "pairs": pairs,
        "max_S": float(max(s_values)) if s_values else 0.0,
        "n_entangled": n_entangled,
        "n_pairs": len(pairs),
        "integration_metric": mean_s / tsirelson,
    }
