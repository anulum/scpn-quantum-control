# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""SSGF geometry W adaptation from quantum R_global.

The existing SSGF adapter (ssgf_adapter.py) reads W and theta from
the SSGF engine, evolves quantum state, writes theta back — but
never updates W. This module closes the loop: quantum R_global
feeds back to modify the geometry matrix W.

Update rule:
    W_new_ij = W_old_ij + η × ΔR × C_ij

where:
    η = learning rate
    ΔR = R_quantum - R_target (synchronisation error)
    C_ij = <X_i X_j + Y_i Y_j> (quantum correlator, measures coupling demand)

Positive correlator + positive ΔR → strengthen coupling.
Negative correlator → weaken coupling (anti-correlated pair).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qiskit.quantum_info import SparsePauliOp, Statevector

from .ssgf_adapter import quantum_to_ssgf_state, ssgf_state_to_quantum, ssgf_w_to_hamiltonian


@dataclass
class WAdaptResult:
    """Result of W adaptation step."""

    W_updated: np.ndarray
    r_global: float
    delta_r: float
    correlators: np.ndarray
    max_update: float


def _measure_correlators(sv: Statevector, n: int) -> np.ndarray:
    """Measure <XX + YY> for all pairs from statevector."""
    C = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            xx = ["I"] * n
            xx[i] = "X"
            xx[j] = "X"
            yy = ["I"] * n
            yy[i] = "Y"
            yy[j] = "Y"
            op = SparsePauliOp(
                ["".join(reversed(xx)), "".join(reversed(yy))],
                coeffs=[1.0, 1.0],
            )
            val = float(sv.expectation_value(op).real)
            C[i, j] = C[j, i] = val
    result: np.ndarray = C
    return result


def adapt_w_from_quantum(
    W: np.ndarray,
    theta: np.ndarray,
    r_target: float = 0.9,
    learning_rate: float = 0.01,
    omega: np.ndarray | None = None,
    dt: float = 0.1,
    trotter_reps: int = 3,
    min_coupling: float = 0.0,
) -> WAdaptResult:
    """One step of W adaptation from quantum feedback.

    Args:
        W: current geometry matrix
        theta: current oscillator phases
        r_target: target synchronisation level
        learning_rate: update step size
        omega: natural frequencies (default: zeros)
        dt: Trotter evolution time
        trotter_reps: Trotter repetitions
        min_coupling: minimum coupling value (enforce non-negative)
    """
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import PauliEvolutionGate
    from qiskit.synthesis import LieTrotter

    n = W.shape[0]
    if omega is None:
        omega = np.zeros(n)

    H = ssgf_w_to_hamiltonian(W, omega)
    qc_init = ssgf_state_to_quantum({"theta": theta})
    sv = Statevector.from_instruction(qc_init)

    synth = LieTrotter(reps=trotter_reps)
    evo_gate = PauliEvolutionGate(H, time=dt, synthesis=synth)
    step_qc = QuantumCircuit(n)
    step_qc.append(evo_gate, range(n))
    sv = sv.evolve(step_qc)

    state = quantum_to_ssgf_state(sv, n)
    r_global = state["R_global"]
    delta_r = r_global - r_target

    correlators = _measure_correlators(sv, n)

    # Update W: strengthen coupling where correlator is positive and R is below target
    # W_new = W + η × (-ΔR) × C  (negative ΔR means below target → increase)
    update = -learning_rate * delta_r * correlators
    W_new = W + update
    np.fill_diagonal(W_new, 0.0)
    W_new = np.maximum(W_new, min_coupling)

    max_upd = float(np.max(np.abs(update)))

    return WAdaptResult(
        W_updated=W_new,
        r_global=r_global,
        delta_r=delta_r,
        correlators=correlators,
        max_update=max_upd,
    )
