# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Quantum cost terms for SSGF integration.

The SSGF engine uses 8 cost terms to optimise geometry. Three of these
have natural quantum analogs computed from the quantum-evolved state:

    C_micro: microscale coherence — Kuramoto order parameter R
        Quantum: R = |mean(exp(i θ_i))| from qubit phases
        C_micro = 1 - R

    C4_tcbo: topological coherence via persistent homology
        Quantum: entanglement entropy S(n/2) as proxy for topological order
        C4_tcbo = 1 - S/S_max (normalised, high entropy = high topological content)

    C_pgbo: phase-geometry tensor minimum
        Quantum: variance of coupling correlators <XX+YY>_ij
        C_pgbo = σ²(<XX+YY>) / max(σ², 1) (geometry-phase mismatch)

These costs feed into the quantum outer cycle as additional terms
beyond the basic C_quantum = 1 - R_global.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qiskit.quantum_info import SparsePauliOp, Statevector

from ..bridge.ssgf_adapter import (
    quantum_to_ssgf_state,
    ssgf_state_to_quantum,
    ssgf_w_to_hamiltonian,
)


@dataclass
class QuantumCosts:
    """All three quantum cost terms."""

    c_micro: float  # 1 - R_global
    c4_tcbo: float  # normalised entanglement entropy proxy
    c_pgbo: float  # correlator variance (geometry-phase mismatch)
    r_global: float
    half_chain_entropy: float
    correlator_variance: float


def _evolve_state(
    W: np.ndarray,
    theta_init: np.ndarray,
    omega: np.ndarray | None = None,
    dt: float = 0.1,
    trotter_reps: int = 3,
) -> Statevector:
    """Trotter-evolve initial state under W Hamiltonian."""
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import PauliEvolutionGate
    from qiskit.synthesis import LieTrotter

    n = W.shape[0]
    if omega is None:
        omega = np.zeros(n)

    H = ssgf_w_to_hamiltonian(W, omega)
    qc_init = ssgf_state_to_quantum({"theta": theta_init})
    sv = Statevector.from_instruction(qc_init)

    synth = LieTrotter(reps=trotter_reps)
    evo_gate = PauliEvolutionGate(H, time=dt, synthesis=synth)
    step_qc = QuantumCircuit(n)
    step_qc.append(evo_gate, range(n))
    return sv.evolve(step_qc)


def compute_c_micro(sv: Statevector, n: int) -> tuple[float, float]:
    """C_micro = 1 - R_global from qubit phases."""
    result = quantum_to_ssgf_state(sv, n)
    r = result["R_global"]
    return 1.0 - r, r


def compute_c4_tcbo(sv: Statevector, n: int) -> tuple[float, float]:
    """C4_tcbo proxy: normalised half-chain entanglement entropy.

    High entropy → rich topological content → low cost.
    S_max = n/2 (maximally entangled).
    """
    from ..analysis.quantum_phi import partial_trace, von_neumann_entropy

    rho = np.outer(sv.data, sv.data.conj())
    half = n // 2
    if half < 1:
        return 1.0, 0.0
    rho_a = partial_trace(rho, list(range(half)), n)
    s = von_neumann_entropy(rho_a)
    s_max = float(half)  # log2(2^(n/2)) = n/2
    s_norm = s / max(s_max, 1e-10)
    return 1.0 - s_norm, s


def compute_c_pgbo(sv: Statevector, n: int) -> tuple[float, float]:
    """C_pgbo: variance of XY coupling correlators.

    Low variance = uniform coupling response = good geometry-phase match.
    """
    correlators: list[float] = []
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
            correlators.append(val)

    if not correlators:
        return 1.0, 0.0

    variance = float(np.var(correlators))
    return min(variance, 1.0), variance


def compute_quantum_costs(
    W: np.ndarray,
    theta_init: np.ndarray,
    omega: np.ndarray | None = None,
    dt: float = 0.1,
    trotter_reps: int = 3,
) -> QuantumCosts:
    """Compute all three quantum cost terms after Trotter evolution."""
    n = W.shape[0]
    sv = _evolve_state(W, theta_init, omega, dt, trotter_reps)

    c_micro, r_global = compute_c_micro(sv, n)
    c4_tcbo, s_half = compute_c4_tcbo(sv, n)
    c_pgbo, corr_var = compute_c_pgbo(sv, n)

    return QuantumCosts(
        c_micro=c_micro,
        c4_tcbo=c4_tcbo,
        c_pgbo=c_pgbo,
        r_global=r_global,
        half_chain_entropy=s_half,
        correlator_variance=corr_var,
    )
