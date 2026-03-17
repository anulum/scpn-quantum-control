# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Identity attractor basin via VQE ground state analysis.

Computes the ground state of an identity coupling Hamiltonian H(K_nm)
and extracts the energy gap as a robustness metric. A large gap means
the identity resists perturbation; a small gap means fragile coupling.
"""

from __future__ import annotations

import numpy as np

from ..bridge.orchestrator_adapter import PhaseOrchestratorAdapter
from ..hardware.classical import classical_exact_diag
from ..phase.phase_vqe import PhaseVQE


class IdentityAttractor:
    """Characterize the attractor basin of an identity coupling topology.

    Wraps PhaseVQE with identity-specific interpretation: the ground state
    is the natural resting configuration, and the energy gap to the first
    excited state quantifies robustness against perturbation.
    """

    def __init__(
        self,
        K: np.ndarray,
        omega: np.ndarray,
        ansatz_reps: int = 2,
    ):
        if K.shape[0] != K.shape[1]:
            raise ValueError(f"K must be square, got {K.shape}")
        if K.shape[0] != len(omega):
            raise ValueError(f"K size {K.shape[0]} != omega length {len(omega)}")
        self.K = K
        self.omega = omega
        self._vqe = PhaseVQE(K, omega, ansatz_reps=ansatz_reps)
        self._result: dict | None = None

    @classmethod
    def from_binding_spec(
        cls,
        binding_spec: dict,
        ansatz_reps: int = 2,
    ) -> IdentityAttractor:
        """Build from an scpn-phase-orchestrator binding spec."""
        K = PhaseOrchestratorAdapter.build_knm_from_binding_spec(
            binding_spec,
            zero_diagonal=True,
        )
        omega = PhaseOrchestratorAdapter.build_omega_from_binding_spec(binding_spec)
        return cls(K, omega, ansatz_reps=ansatz_reps)

    def solve(self, maxiter: int = 200, seed: int | None = None) -> dict:
        """Find the ground state and compute robustness metrics.

        Returns dict with ground_energy, exact_energy, energy_gap,
        relative_error_pct, robustness_gap, n_dispositions.
        """
        self._result = self._vqe.solve(maxiter=maxiter, seed=seed)

        n = len(self.omega)
        exact = classical_exact_diag(n, K=self.K, omega=self.omega)
        eigenvalues = exact["eigenvalues"]

        # Robustness gap: E_1 - E_0 (first excited - ground)
        robustness_gap = 0.0
        if len(eigenvalues) >= 2:
            robustness_gap = float(eigenvalues[1] - eigenvalues[0])

        self._result["robustness_gap"] = robustness_gap
        self._result["n_dispositions"] = n
        self._result["eigenvalues"] = eigenvalues[: min(4, len(eigenvalues))].tolist()

        return self._result

    def robustness_gap(self) -> float:
        """Energy gap E_1 - E_0. Call solve() first."""
        if self._result is None:
            raise RuntimeError("Call solve() before robustness_gap()")
        return self._result["robustness_gap"]

    def ground_state(self):
        """Return the VQE-optimized ground state vector."""
        return self._vqe.ground_state()
