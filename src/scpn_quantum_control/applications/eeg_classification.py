# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — EEG Classification
"""EEG state classification via structured VQE and quantum kernels.

Maps EEG functional connectivity matrices (Phase Locking Value - PLV)
to physical quantum Hamiltonians.

Leverages the K_nm-informed Structured Ansatz to build hardware-efficient
classification circuits, bypassing deep standard ansatze and focusing
entanglement exclusively on biologically coupled channels.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import numpy as np
from qiskit.quantum_info import Statevector

from scpn_quantum_control.bridge.knm_hamiltonian import knm_to_dense_matrix
from scpn_quantum_control.phase.phase_vqe import PhaseVQE
from scpn_quantum_control.phase.structured_ansatz import build_structured_ansatz


@dataclass
class EEGVQEResult:
    """Result bundle for EEG PLV VQE classification."""

    n_channels: int
    optimal_energy: float
    statevector: np.ndarray
    ansatz_depth: int
    n_params: int
    success: bool


def _validated_plv_matrix(plv_matrix: np.ndarray) -> np.ndarray:
    plv = np.asarray(plv_matrix, dtype=float)
    if plv.ndim != 2 or plv.shape[0] != plv.shape[1]:
        raise ValueError("plv_matrix must be a square 2-D matrix.")
    if plv.shape[0] == 0:
        raise ValueError("plv_matrix must contain at least one channel.")
    if not np.all(np.isfinite(plv)):
        raise ValueError("plv_matrix must contain only finite values.")
    if np.any((plv < 0.0) | (plv > 1.0)):
        raise ValueError("PLV values must be in [0, 1].")
    if not np.allclose(plv, plv.T, atol=1e-12):
        raise ValueError("plv_matrix must be symmetric.")
    if not np.allclose(np.diag(plv), 0.0, atol=1e-12):
        raise ValueError("plv_matrix diagonal must be zero.")
    return plv


def _validated_natural_frequencies(
    natural_frequencies: np.ndarray,
    n_channels: int,
) -> np.ndarray:
    frequencies = np.asarray(natural_frequencies, dtype=float)
    if frequencies.ndim != 1 or frequencies.shape != (n_channels,):
        raise ValueError("natural_frequencies must match plv_matrix channel count as a 1-D array.")
    if not np.all(np.isfinite(frequencies)):
        raise ValueError("natural_frequencies must contain only finite values.")
    return frequencies


def _validated_reps(reps: int) -> int:
    if isinstance(reps, bool) or not isinstance(reps, int):
        raise ValueError("reps must be a positive integer.")
    if reps <= 0:
        raise ValueError("reps must be positive.")
    return reps


def _validated_threshold(threshold: float) -> float:
    threshold_value = float(threshold)
    if not np.isfinite(threshold_value) or not 0.0 <= threshold_value <= 1.0:
        raise ValueError("threshold must be in [0, 1].")
    return threshold_value


def _validated_statevector(state: np.ndarray, name: str) -> np.ndarray:
    vector = np.asarray(state, dtype=complex)
    if vector.ndim != 1:
        raise ValueError(f"{name} must be a 1-D statevector.")
    if vector.size == 0:
        raise ValueError(f"{name} must contain at least one amplitude.")
    if not np.all(np.isfinite(vector.real)) or not np.all(np.isfinite(vector.imag)):
        raise ValueError(f"{name} must contain only finite amplitudes.")
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ValueError(f"{name} must have non-zero norm.")
    return np.asarray(vector / norm, dtype=complex)


def eeg_plv_to_vqe(
    plv_matrix: np.ndarray,
    natural_frequencies: np.ndarray,
    reps: int = 2,
    threshold: float = 0.1,
    *,
    max_dense_gib: float | None = None,
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
    plv_matrix = _validated_plv_matrix(plv_matrix)
    n = plv_matrix.shape[0]
    natural_frequencies = _validated_natural_frequencies(natural_frequencies, n)
    reps = _validated_reps(reps)
    threshold = _validated_threshold(threshold)

    # Use fast Rust-accelerated dense matrix builder
    knm_to_dense_matrix(plv_matrix, natural_frequencies, max_dense_gib=max_dense_gib)

    ansatz = build_structured_ansatz(plv_matrix, reps=reps, threshold=threshold)
    n_params = ansatz.num_parameters

    # Inject the structured ansatz into the VQE solver
    vqe = PhaseVQE(plv_matrix, natural_frequencies, ansatz_reps=reps)
    vqe.ansatz = ansatz  # Override with our threshold-filtered ansatz
    vqe.n_params = n_params

    sol = vqe.solve(maxiter=200, seed=42)

    # Extract optimized state
    opt_params = np.asarray(sol.get("optimal_params", np.zeros(n_params)), dtype=float)
    bound = ansatz.assign_parameters(opt_params)
    sv = Statevector.from_instruction(bound).data
    energy_value = sol.get("vqe_energy", sol.get("ground_energy", 0.0))
    if not isinstance(energy_value, Real):
        raise TypeError("PhaseVQE returned a non-numeric energy")
    converged_value = sol.get("converged", False)
    if not isinstance(converged_value, (bool, np.bool_)):
        raise TypeError("PhaseVQE returned a non-boolean convergence flag")

    return EEGVQEResult(
        n_channels=n,
        optimal_energy=float(energy_value),
        statevector=sv,
        ansatz_depth=ansatz.depth(),
        n_params=n_params,
        success=bool(converged_value),
    )


def eeg_quantum_kernel(state_a: np.ndarray, state_b: np.ndarray) -> float:
    """Compute the quantum kernel fidelity |<A|B>|^2 between two EEG VQE states."""
    state_a = _validated_statevector(state_a, "state_a")
    state_b = _validated_statevector(state_b, "state_b")
    if state_a.shape != state_b.shape:
        raise ValueError("state_a and state_b must have the same shape.")
    overlap = np.vdot(state_a, state_b)
    return float(np.abs(overlap) ** 2)
