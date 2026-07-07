# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Quantum Speed Limit
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

References
----------
    Mandelstam & Tamm (1945): original QSL.
    Mukherjee et al., PRA 110 (2024): QSL as QPT probe.
    Impens & Guery-Odelin, arXiv:2210.05848: shortcut to synchronization.
    Wei et al., Sci. Rep. (2016): QSL anomalies at criticality.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import expm

from ..bridge.knm_hamiltonian import knm_to_dense_matrix, knm_to_hamiltonian
from ..dense_budget import require_dense_allocation
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


def _validate_qsl_parameters(t_target: float, dt: float, R_threshold: float) -> None:
    if not np.isfinite(t_target):
        raise ValueError("t_target must be finite")
    if t_target < 0.0:
        raise ValueError("t_target must be non-negative")
    if not np.isfinite(dt):
        raise ValueError("dt must be finite")
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if not np.isfinite(R_threshold):
        raise ValueError("R_threshold must be finite")
    if not 0.0 <= R_threshold <= 1.0:
        raise ValueError("R_threshold must be in [0, 1]")


def compute_qsl(
    K: NDArray[np.float64],
    omega: NDArray[np.float64],
    t_target: float = 2.0,
    dt: float = 0.01,
    R_threshold: float = 0.5,
    *,
    max_dense_gib: float | None = None,
) -> QSLResult:
    """Compute quantum speed limits for reaching synchronization.

    Evolves |0...0⟩ under H_XY and finds the first time R > R_threshold.
    Computes both Mandelstam-Tamm and Margolus-Levitin bounds.
    """
    _validate_qsl_parameters(t_target, dt, R_threshold)
    n = K.shape[0]
    require_dense_allocation(
        n,
        rank=2,
        object_count=4,
        max_gib=max_dense_gib,
        label="QSL dense evolution workspace",
    )
    require_dense_allocation(
        n,
        rank=1,
        object_count=3,
        max_gib=max_dense_gib,
        label="QSL dense state workspace",
    )
    knm_to_hamiltonian(K, omega)
    H_raw = knm_to_dense_matrix(K, omega, max_dense_gib=max_dense_gib)
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

    # Time evolution: find when R first exceeds threshold.  The final partial
    # step is evolved exactly so the target state and reported time agree even
    # when t_target is not an integer multiple of dt.
    U_dt = expm(-1j * H_mat * dt)

    psi = psi_0.copy()
    tau_actual = t_target  # default: didn't reach threshold
    current_time = 0.0

    from .entanglement_enhanced_sync import _state_order_parameter

    while current_time < t_target - 1e-15:
        step_dt = min(dt, t_target - current_time)
        U_step = U_dt if abs(step_dt - dt) <= 1e-15 else expm(-1j * H_mat * step_dt)
        psi = U_step @ psi
        current_time += step_dt
        R = _state_order_parameter(psi, n)
        if R_threshold <= R:
            tau_actual = current_time
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
    K: NDArray[np.float64],
    omega: NDArray[np.float64],
    K_base_range: NDArray[np.float64] | None = None,
    n_K_values: int = 15,
    t_target: float = 5.0,
    R_threshold: float = 0.5,
    *,
    max_dense_gib: float | None = None,
) -> dict[str, Any]:
    """Scan QSL across coupling strengths to reveal BKT singularity.

    At K_c, the QSL should show anomalous behaviour:
    - Second-order QPT: τ ~ |K-K_c|^{-zν} (power law divergence)
    - BKT: τ ~ exp(b/√(K-K_c)) (essential singularity → log divergence)

    The signature of BKT in the QSL is qualitatively different from
    standard QPTs.
    """
    if K_base_range is None:
        K_base_range = np.linspace(0.01, 3.0, n_K_values, dtype=np.float64)

    tau_MT_vals = []
    tau_ML_vals = []
    tau_actual_vals = []
    delta_E_vals = []
    R_final_vals = []

    for k_base in K_base_range:
        K_scaled = K * k_base
        result = compute_qsl(
            K_scaled,
            omega,
            t_target,
            R_threshold=R_threshold,
            max_dense_gib=max_dense_gib,
        )
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
