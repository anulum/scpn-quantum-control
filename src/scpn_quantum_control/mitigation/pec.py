# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Pec
"""Probabilistic Error Cancellation for single-qubit depolarizing channels.

Temme et al., PRL 119, 180509 (2017).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


@dataclass
class PECResult:
    """PEC estimation output."""

    mitigated_value: float
    overhead: float
    n_samples: int
    sign_distribution: list[float] = field(default_factory=list)


def pauli_twirl_decompose(gate_error_rate: float, n_qubits: int = 1) -> np.ndarray:
    """Quasi-probability coefficients for depolarizing channel inverse.

    Single-qubit: q_I = 1 + 3p/(4-4p), q_{X,Y,Z} = -p/(4-4p).
    Temme et al., PRL 119, 180509 (2017), Eq. 4.
    """
    if not 0.0 <= gate_error_rate < 1.0:
        raise ValueError(f"gate_error_rate must be in [0, 1), got {gate_error_rate}")
    if n_qubits != 1:
        raise NotImplementedError("Only single-qubit PEC implemented")

    p = gate_error_rate
    denom = 4.0 - 4.0 * p
    q_i = 1.0 + 3.0 * p / denom
    q_xyz = -p / denom
    coeffs: np.ndarray = np.array([q_i, q_xyz, q_xyz, q_xyz])
    return coeffs


def pec_sample(
    circuit: QuantumCircuit,
    gate_error_rate: float,
    n_samples: int,
    observable_qubit: int = 0,
    rng: np.random.Generator | None = None,
) -> PECResult:
    """Monte Carlo PEC: sample Paulis from quasi-probability distribution.

    Estimates <Z> on observable_qubit by inserting Pauli corrections after
    each gate and accumulating signed expectations.
    """
    if rng is None:
        rng = np.random.default_rng()

    coeffs = pauli_twirl_decompose(gate_error_rate)
    abs_coeffs = np.abs(coeffs)
    gamma_single = float(np.sum(abs_coeffs))
    probs = abs_coeffs / gamma_single
    signs = np.sign(coeffs)

    n_gates = circuit.size()
    gamma_total = gamma_single**n_gates

    acc = 0.0
    sign_list: list[float] = []

    for _ in range(n_samples):
        qc = circuit.copy()
        total_sign = 1.0

        for _ in range(n_gates):
            idx = int(rng.choice(4, p=probs))
            total_sign *= signs[idx]
            if idx == 1:
                qc.x(observable_qubit)
            elif idx == 2:
                qc.y(observable_qubit)
            elif idx == 3:
                qc.z(observable_qubit)

        sv = Statevector.from_instruction(qc)
        p0 = float(sv.probabilities([observable_qubit])[0])
        exp_z = 2.0 * p0 - 1.0

        acc += gamma_total * total_sign * exp_z
        sign_list.append(total_sign)

    return PECResult(
        mitigated_value=acc / n_samples,
        overhead=gamma_total,
        n_samples=n_samples,
        sign_distribution=sign_list,
    )
