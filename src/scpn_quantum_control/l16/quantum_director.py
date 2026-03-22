# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Quantum L16 Director: Lyapunov monitoring for cybernetic closure.

The L16 layer in the SCPN framework provides cybernetic closure by
monitoring system stability via Lyapunov exponents. The quantum L16
extracts stability indicators from the quantum-evolved state:

    1. Loschmidt echo: L(t) = |<ψ(0)|ψ(t)>|² — state return probability.
       Decay rate = quantum Lyapunov exponent bound.

    2. Fidelity susceptibility: χ_F = -∂²F/∂ε² at ε=0
       where F = |<ψ(K)|ψ(K+ε)>|². Peaks at quantum phase transitions.

    3. Energy variance: ΔE² = <H²> - <H>² — stability indicator.
       Low variance = near eigenstate = stable. High = unstable.

    4. Order parameter rate: dR/dt estimated from consecutive snapshots.
       Positive = approaching synchronisation. Negative = destabilising.

The L16 director uses these to decide:
    - Continue evolution (stable)
    - Adjust coupling (drifting)
    - Halt and reset (unstable)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..bridge.knm_hamiltonian import knm_to_hamiltonian
from ..bridge.ssgf_adapter import quantum_to_ssgf_state
from ..hardware.classical import classical_exact_diag
from ..hardware.gpu_accel import expm


@dataclass
class L16Result:
    """L16 Lyapunov monitoring result."""

    loschmidt_echo: float  # |<ψ(0)|ψ(t)>|²
    energy_variance: float  # <H²> - <H>²
    fidelity_susceptibility: float  # sensitivity to parameter perturbation
    order_parameter: float  # R_global
    stability_score: float  # composite: 0 = unstable, 1 = stable
    action: str  # "continue", "adjust", "halt"


def _evolve_exact(psi: np.ndarray, H_mat: np.ndarray, t: float) -> np.ndarray:
    """Exact time evolution via matrix exponential."""
    U = expm(-1j * H_mat * t)
    result: np.ndarray = U @ psi
    return result


def loschmidt_echo(
    K: np.ndarray,
    omega: np.ndarray,
    t: float = 0.5,
) -> float:
    """Loschmidt echo L(t) = |<ψ(0)|ψ(t)>|²."""
    n = K.shape[0]
    exact = classical_exact_diag(n, K=K, omega=omega)
    psi_0 = np.ascontiguousarray(exact["ground_state"])

    H_op = knm_to_hamiltonian(K, omega)
    H_raw = H_op.to_matrix()
    H_mat = H_raw.toarray() if hasattr(H_raw, "toarray") else np.array(H_raw)

    psi_t = _evolve_exact(psi_0, H_mat, t)
    return float(abs(np.dot(psi_0.conj(), psi_t)) ** 2)


def energy_variance(
    K: np.ndarray,
    omega: np.ndarray,
) -> float:
    """Energy variance ΔE² = <H²> - <H>² for the ground state."""
    n = K.shape[0]
    exact = classical_exact_diag(n, K=K, omega=omega)
    psi = np.ascontiguousarray(exact["ground_state"])

    H_op = knm_to_hamiltonian(K, omega)
    H_raw = H_op.to_matrix()
    H_mat = H_raw.toarray() if hasattr(H_raw, "toarray") else np.array(H_raw)

    e_mean = float(np.real(psi.conj() @ H_mat @ psi))
    e2_mean = float(np.real(psi.conj() @ H_mat @ H_mat @ psi))
    return max(e2_mean - e_mean**2, 0.0)


def fidelity_susceptibility(
    K: np.ndarray,
    omega: np.ndarray,
    epsilon: float = 0.005,
) -> float:
    """Fidelity susceptibility: -d²F/dε² at ε=0.

    Measures sensitivity to uniform coupling perturbation K → K + εI_off.
    """
    n = K.shape[0]
    psi_0 = np.ascontiguousarray(classical_exact_diag(n, K=K, omega=omega)["ground_state"])

    K_plus = K.copy()
    K_plus[np.triu_indices(n, k=1)] += epsilon
    K_plus[np.tril_indices(n, k=-1)] += epsilon

    psi_plus = np.ascontiguousarray(classical_exact_diag(n, K=K_plus, omega=omega)["ground_state"])
    psi_plus *= np.exp(-1j * np.angle(np.dot(psi_0.conj(), psi_plus)))

    fidelity = abs(np.dot(psi_0.conj(), psi_plus)) ** 2
    # χ_F ≈ 2(1 - F) / ε²
    return float(2.0 * (1.0 - fidelity) / (epsilon**2))


def compute_l16_lyapunov(
    K: np.ndarray,
    omega: np.ndarray,
    t: float = 0.5,
) -> L16Result:
    """Full L16 Lyapunov monitoring assessment.

    Combines all four indicators into a stability score and action decision.
    """
    n = K.shape[0]

    le = loschmidt_echo(K, omega, t)
    ev = energy_variance(K, omega)
    fs = fidelity_susceptibility(K, omega)

    # Order parameter from ground state
    exact = classical_exact_diag(n, K=K, omega=omega)
    psi = np.ascontiguousarray(exact["ground_state"])

    from qiskit.quantum_info import Statevector

    sv = Statevector(psi)
    state_dict = quantum_to_ssgf_state(sv, n)
    r_global = state_dict["R_global"]

    # Stability score: high echo + low variance + moderate susceptibility + high R
    score = (
        0.25 * le + 0.25 * max(1.0 - ev, 0.0) + 0.25 * min(1.0, 1.0 / (1.0 + fs)) + 0.25 * r_global
    )

    if score > 0.7:
        action = "continue"
    elif score > 0.4:
        action = "adjust"
    else:
        action = "halt"

    return L16Result(
        loschmidt_echo=le,
        energy_variance=ev,
        fidelity_susceptibility=fs,
        order_parameter=r_global,
        stability_score=score,
        action=action,
    )
