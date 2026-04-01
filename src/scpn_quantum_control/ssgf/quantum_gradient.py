# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Quantum Gradient
"""SSGF quantum gradient: dC_quantum/dz via parameter-shift rule.

The SSGF outer cycle optimises a latent vector z that parameterises
the geometry matrix W(z). The quantum cost function adds a term that
measures how well the quantum-evolved state preserves synchronisation:

    C_quantum(z) = 1 - R_global(z)

where R_global = |mean(exp(i θ_i))| is the Kuramoto order parameter
computed from the quantum-evolved oscillator phases.

The gradient dC_quantum/dz is computed via finite differences on z:

    dC/dz_k ≈ [C(z + ε e_k) - C(z - ε e_k)] / (2ε)

For each z perturbation:
    1. Build W(z ± ε e_k) from the SSGF geometry map
    2. Compile W → Hamiltonian
    3. Trotter-evolve initial state
    4. Extract R_global
    5. Compute finite difference

This is the missing quantum-in-the-loop gradient that connects
quantum evolution back to SSGF geometry learning.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qiskit.quantum_info import Statevector

from ..bridge.ssgf_adapter import (
    quantum_to_ssgf_state,
    ssgf_state_to_quantum,
    ssgf_w_to_hamiltonian,
)


@dataclass
class QuantumGradientResult:
    """Result of quantum gradient computation."""

    cost: float  # C_quantum at z
    gradient: np.ndarray  # dC/dz vector
    r_global: float  # order parameter at z
    n_evaluations: int  # total quantum evaluations


def _w_from_z(z: np.ndarray, n_osc: int) -> np.ndarray:
    """Map latent z to geometry matrix W.

    Simple parameterisation: z is the upper triangle of W,
    reshaped and symmetrised. W_ij = softplus(z_k) ensures
    non-negative coupling.
    """
    n_upper = n_osc * (n_osc - 1) // 2
    z_trunc = z[:n_upper]

    # Softplus: log(1 + exp(x)) for non-negative coupling
    w_values = np.log1p(np.exp(z_trunc))

    W = np.zeros((n_osc, n_osc))
    idx = 0
    for i in range(n_osc):
        for j in range(i + 1, n_osc):
            W[i, j] = W[j, i] = w_values[idx]
            idx += 1
    result: np.ndarray = W
    return result


def quantum_cost(
    W: np.ndarray,
    theta_init: np.ndarray,
    omega: np.ndarray | None = None,
    dt: float = 0.1,
    trotter_reps: int = 3,
) -> float:
    """Quantum cost function: C = 1 - R_global after Trotter evolution.

    Low cost = high synchronisation = good geometry.
    """
    from qiskit.circuit.library import PauliEvolutionGate
    from qiskit.synthesis import LieTrotter

    n = W.shape[0]
    if omega is None:
        omega = np.zeros(n)

    H = ssgf_w_to_hamiltonian(W, omega)
    qc_init = ssgf_state_to_quantum({"theta": theta_init})
    sv = Statevector.from_instruction(qc_init)

    from qiskit import QuantumCircuit

    synth = LieTrotter(reps=trotter_reps)
    evo_gate = PauliEvolutionGate(H, time=dt, synthesis=synth)
    step_qc = QuantumCircuit(n)
    step_qc.append(evo_gate, range(n))
    sv = sv.evolve(step_qc)

    result = quantum_to_ssgf_state(sv, n)
    return float(1.0 - result["R_global"])


def compute_quantum_gradient(
    z: np.ndarray,
    n_osc: int,
    theta_init: np.ndarray | None = None,
    omega: np.ndarray | None = None,
    epsilon: float = 0.01,
    dt: float = 0.1,
    trotter_reps: int = 3,
) -> QuantumGradientResult:
    """Compute dC_quantum/dz via central finite differences.

    Args:
        z: latent vector parameterising W
        n_osc: number of oscillators
        theta_init: initial oscillator phases (default: uniform on [0, 2π))
        omega: natural frequencies (default: zeros)
        epsilon: finite difference step size
        dt: Trotter evolution time
        trotter_reps: Trotter repetitions
    """
    if theta_init is None:
        theta_init = np.linspace(0, 2 * np.pi * (1 - 1 / n_osc), n_osc)
    if omega is None:
        omega = np.zeros(n_osc)

    W_center = _w_from_z(z, n_osc)
    cost_center = quantum_cost(W_center, theta_init, omega, dt, trotter_reps)
    r_global = 1.0 - cost_center

    n_upper = n_osc * (n_osc - 1) // 2
    gradient = np.zeros(len(z))
    n_evals = 1  # center evaluation

    for k in range(min(len(z), n_upper)):
        z_plus = z.copy()
        z_plus[k] += epsilon
        z_minus = z.copy()
        z_minus[k] -= epsilon

        W_plus = _w_from_z(z_plus, n_osc)
        W_minus = _w_from_z(z_minus, n_osc)

        c_plus = quantum_cost(W_plus, theta_init, omega, dt, trotter_reps)
        c_minus = quantum_cost(W_minus, theta_init, omega, dt, trotter_reps)

        gradient[k] = (c_plus - c_minus) / (2.0 * epsilon)
        n_evals += 2

    return QuantumGradientResult(
        cost=cost_center,
        gradient=gradient,
        r_global=r_global,
        n_evaluations=n_evals,
    )
