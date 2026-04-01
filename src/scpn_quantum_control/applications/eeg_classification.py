# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

"""EEG state classification via structured VQE and quantum kernels.

Maps EEG functional connectivity matrices (Phase Locking Value - PLV)
to physical quantum Hamiltonians.

Leverages the K_nm-informed Structured Ansatz to build hardware-efficient
classification circuits, bypassing deep standard ansatze and focusing
entanglement exclusively on biologically coupled channels.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qiskit.quantum_info import Statevector

from scpn_quantum_control.bridge.knm_hamiltonian import knm_to_dense_matrix
from scpn_quantum_control.phase.phase_vqe import PhaseVQE
from scpn_quantum_control.phase.structured_ansatz import build_structured_ansatz


@dataclass
class EEGVQEResult:
    n_channels: int
    optimal_energy: float
    statevector: np.ndarray
    ansatz_depth: int
    n_params: int
    success: bool


def eeg_plv_to_vqe(
    plv_matrix: np.ndarray,
    natural_frequencies: np.ndarray,
    reps: int = 2,
    threshold: float = 0.1,
) -> EEGVQEResult:
    """Map an EEG PLV matrix to a VQE ground state using a Structured Ansatz.

    Args:
        plv_matrix: N x N Phase Locking Value matrix.
        natural_frequencies: Array of N peak frequencies per channel.
        reps: Depth of the structured ansatz.
        threshold: Minimum PLV required to insert an entangling gate.

    Returns:
        EEGVQEResult containing the optimized energy and statevector.
    """
    n = plv_matrix.shape[0]

    # 1. Build the physical Hamiltonian representation of the brain state
    # We use the fast Rust-accelerated dense matrix builder
    knm_to_dense_matrix(plv_matrix, natural_frequencies)

    # 2. Build the topology-informed ansatz tailored to this specific brain state
    ansatz = build_structured_ansatz(plv_matrix, reps=reps, threshold=threshold)
    n_params = ansatz.num_parameters

    # 3. Run VQE (using scipy.optimize internally via PhaseVQE logic)
    # We inject the structured ansatz into the VQE solver
    vqe = PhaseVQE(plv_matrix, natural_frequencies, ansatz_reps=reps)
    vqe.ansatz = ansatz  # Override with our threshold-filtered ansatz
    vqe.n_params = n_params

    sol = vqe.solve(maxiter=200, seed=42)

    # Extract optimized state
    opt_params = sol.get("optimal_params", np.zeros(n_params))
    bound = ansatz.assign_parameters(opt_params)
    sv = Statevector.from_instruction(bound).data

    return EEGVQEResult(
        n_channels=n,
        optimal_energy=sol.get("vqe_energy", sol.get("ground_energy", 0.0)),
        statevector=sv,
        ansatz_depth=ansatz.depth(),
        n_params=n_params,
        success=sol.get("converged", False),
    )


def eeg_quantum_kernel(state_a: np.ndarray, state_b: np.ndarray) -> float:
    """Compute the quantum kernel fidelity |<A|B>|^2 between two EEG VQE states."""
    overlap = np.vdot(state_a, state_b)
    return float(np.abs(overlap) ** 2)
