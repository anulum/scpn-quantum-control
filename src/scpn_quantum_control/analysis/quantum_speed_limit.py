# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Quantum Speed Limit
"""Quantum speed limit for the synchronization phase transition.

The Mandelstam-Tamm bound gives the minimum time to evolve from
an initial state |ψ_0⟩ to a target state |ψ_T⟩:

    τ_MT ≥ arccos(|⟨ψ_T|ψ_0⟩|) / ΔE

where ΔE = √(⟨H²⟩ - ⟨H⟩²) is the energy variance.

The Margolus-Levitin bound gives an alternative:

    τ_ML ≥ π / (2(⟨H⟩ - E_0))

where E_0 is the ground state energy.

For the BKT synchronization transition, the correlation length
diverges as ξ ~ exp(b/√(K-K_c)), an essential singularity. This
should produce qualitatively different QSL scaling near K_c compared
to second-order transitions (where ξ ~ |K-K_c|^{-ν} gives power-law
QSL divergence).

Nobody has computed QSL at a BKT transition. The prediction:
τ_sync diverges LOGARITHMICALLY at K_c (from the essential singularity),
not as a power law. This is a qualitative signature of BKT physics
in the quantum speed limit.

References:
    Mandelstam & Tamm (1945): original QSL.
    Mukherjee et al., PRA 110 (2024): QSL as QPT probe.
    Impens & Guery-Odelin, arXiv:2210.05848: shortcut to synchronization.
    Wei et al., Sci. Rep. (2016): QSL anomalies at criticality.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm

from ..bridge.knm_hamiltonian import knm_to_hamiltonian
from ..hardware.classical import classical_exact_diag


@dataclass
class QSLResult:
    """Quantum speed limit computation result."""

    tau_MT: float
    tau_ML: float
    tau_actual: float
    tightness_MT: float  # τ_actual / τ_MT (≥ 1, closer to 1 = tighter)
    tightness_ML: float
    overlap: float  # |⟨ψ_T|ψ_0⟩|
    delta_E: float  # energy variance
    mean_E: float
    ground_E: float
    n_qubits: int


def compute_qsl(
    K: np.ndarray,
    omega: np.ndarray,
    t_target: float = 2.0,
    dt: float = 0.01,
    R_threshold: float = 0.5,
) -> QSLResult:
    """Compute quantum speed limits for reaching synchronization.

    Evolves |0...0⟩ under H_XY and finds the first time R > R_threshold.
    Computes both Mandelstam-Tamm and Margolus-Levitin bounds.
    """
    n = K.shape[0]
    H_op = knm_to_hamiltonian(K, omega)
    H_raw = H_op.to_matrix()
    H_mat = H_raw.toarray() if hasattr(H_raw, "toarray") else np.array(H_raw)

    # Initial state: |0...0⟩
    psi_0 = np.zeros(2**n, dtype=complex)
    psi_0[0] = 1.0

    # Ground state energy
    exact = classical_exact_diag(n, K=K, omega=omega)
    E_0 = exact["ground_energy"]

    # Energy statistics in initial state
    mean_E = float(np.real(psi_0.conj() @ H_mat @ psi_0))
    mean_E2 = float(np.real(psi_0.conj() @ H_mat @ H_mat @ psi_0))
    delta_E = np.sqrt(max(mean_E2 - mean_E**2, 0.0))

    # Time evolution: find when R first exceeds threshold
    n_steps = int(t_target / dt)
    U_dt = expm(-1j * H_mat * dt)

    psi = psi_0.copy()
    tau_actual = t_target  # default: didn't reach threshold

    from .entanglement_enhanced_sync import _state_order_parameter

    for step in range(1, n_steps + 1):
        psi = U_dt @ psi
        R = _state_order_parameter(psi, n)
        if R_threshold <= R:
            tau_actual = step * dt
            break

    # Target state = state at time tau_actual
    psi_target = psi

    # Overlap
    overlap = float(np.abs(psi_0.conj() @ psi_target))

    # Mandelstam-Tamm bound
    if overlap > 1.0 - 1e-15:
        tau_MT = 0.0
    else:
        tau_MT = np.arccos(min(overlap, 1.0)) / max(delta_E, 1e-15)

    # Margolus-Levitin bound
    E_diff = mean_E - E_0
    if E_diff < 1e-15:
        tau_ML = 0.0
    else:
        tau_ML = np.pi / (2 * E_diff)

    # Tightness
    tight_MT = tau_actual / max(tau_MT, 1e-15) if tau_MT > 0 else float("inf")
    tight_ML = tau_actual / max(tau_ML, 1e-15) if tau_ML > 0 else float("inf")

    return QSLResult(
        tau_MT=tau_MT,
        tau_ML=tau_ML,
        tau_actual=tau_actual,
        tightness_MT=tight_MT,
        tightness_ML=tight_ML,
        overlap=overlap,
        delta_E=delta_E,
        mean_E=mean_E,
        ground_E=E_0,
        n_qubits=n,
    )


def qsl_vs_coupling(
    K: np.ndarray,
    omega: np.ndarray,
    K_base_range: np.ndarray | None = None,
    n_K_values: int = 15,
    t_target: float = 5.0,
    R_threshold: float = 0.5,
) -> dict:
    """Scan QSL across coupling strengths to reveal BKT singularity.

    At K_c, the QSL should show anomalous behavior:
    - Second-order QPT: τ ~ |K-K_c|^{-zν} (power law divergence)
    - BKT: τ ~ exp(b/√(K-K_c)) (essential singularity → log divergence)

    The signature of BKT in the QSL is qualitatively different from
    standard QPTs.
    """
    if K_base_range is None:
        K_base_range = np.linspace(0.01, 3.0, n_K_values)

    tau_MT_vals = []
    tau_ML_vals = []
    tau_actual_vals = []
    delta_E_vals = []
    R_final_vals = []

    for k_base in K_base_range:
        K_scaled = K * k_base
        result = compute_qsl(K_scaled, omega, t_target, R_threshold=R_threshold)
        tau_MT_vals.append(result.tau_MT)
        tau_ML_vals.append(result.tau_ML)
        tau_actual_vals.append(result.tau_actual)
        delta_E_vals.append(result.delta_E)

        # Also compute final R
        from .entanglement_enhanced_sync import InitialState, simulate_sync_trajectory

        traj = simulate_sync_trajectory(
            K_scaled, omega, InitialState.PRODUCT, t_max=t_target, n_steps=20
        )
        R_final_vals.append(traj.final_R)

    return {
        "K_base": list(K_base_range),
        "tau_MT": tau_MT_vals,
        "tau_ML": tau_ML_vals,
        "tau_actual": tau_actual_vals,
        "delta_E": delta_E_vals,
        "R_final": R_final_vals,
    }
