# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

"""Topological Quantum Reinforcement Learning / Optimizer.

This module provides an optimizer that iteratively rewires the physical
coupling matrix K_nm to minimize the topological defects (persistent
1-cycles, p_h1) of the resulting quantum state.

Instead of traditional energy minimization (VQE), this optimizes for
macroscopic coherence and the absence of phase vortices, bridging
Topological Data Analysis (TDA) and Quantum Optimal Control.

It simulates a biological system dynamically evolving its synaptic
topology to achieve a globally synchronized, vortex-free 'conscious' state.
"""

from __future__ import annotations

import numpy as np

from scpn_quantum_control.analysis.quantum_persistent_homology import (
    _RIPSER_AVAILABLE,
    quantum_persistent_homology,
)
from scpn_quantum_control.hardware.fast_classical import fast_sparse_evolution


class TopologicalCouplingOptimizer:
    """Optimizes a coupling graph to minimize quantum persistent homology."""

    def __init__(
        self,
        n_qubits: int,
        initial_K: np.ndarray,
        omega: np.ndarray,
        learning_rate: float = 0.05,
        dt: float = 1.0,
    ):
        self.n = n_qubits
        self.K = np.array(initial_K, dtype=float)
        self.omega = np.array(omega, dtype=float)
        self.lr = learning_rate
        self.dt = dt

        self.K = (self.K + self.K.T) / 2.0
        np.fill_diagonal(self.K, 0.0)

    def _simulate_measurement_counts(
        self, psi: np.ndarray, shots: int = 5000
    ) -> tuple[dict, dict]:
        """Simulate X and Y basis measurements from the statevector."""
        import qiskit.quantum_info as qi
        from qiskit.quantum_info import Statevector

        sv = Statevector(psi)

        # Measure in X basis (apply H to all)
        sv_x = sv.copy()
        for q in range(self.n):
            sv_x = sv_x.evolve(qi.Operator.from_label("H"), [q])
        x_counts = sv_x.sample_counts(shots)

        # Measure in Y basis (apply Sdg then H to all)
        sv_y = sv.copy()
        for q in range(self.n):
            sv_y = sv_y.evolve(qi.Operator.from_label("Sdg"), [q])
            sv_y = sv_y.evolve(qi.Operator.from_label("H"), [q])
        y_counts = sv_y.sample_counts(shots)

        return x_counts, y_counts

    def step(self, n_samples: int = 5) -> dict:
        """Perform one optimization step using finite-difference gradients on p_h1."""
        if not _RIPSER_AVAILABLE:
            raise ImportError("ripser not installed: pip install ripser")

        # 1. Baseline p_h1
        res_base = fast_sparse_evolution(self.K, self.omega, t_total=self.dt, n_steps=1)
        x_c, y_c = self._simulate_measurement_counts(res_base["final_state"])
        ph_base = quantum_persistent_homology(x_c, y_c, self.n, persistence_threshold=0.1).p_h1

        # 2. Gradient estimation via random perturbations (SPSA-like)
        grad_K = np.zeros_like(self.K)
        perturbation_scale = 0.05

        # We sample a few random symmetric perturbation matrices
        for _ in range(n_samples):
            delta_K = np.random.normal(0, perturbation_scale, size=(self.n, self.n))
            delta_K = (delta_K + delta_K.T) / 2.0
            np.fill_diagonal(delta_K, 0.0)

            # + Perturbation
            K_plus = np.maximum(self.K + delta_K, 0.0)
            res_p = fast_sparse_evolution(K_plus, self.omega, t_total=self.dt, n_steps=1)
            x_p, y_p = self._simulate_measurement_counts(res_p["final_state"])
            ph_plus = quantum_persistent_homology(x_p, y_p, self.n, persistence_threshold=0.1).p_h1

            # - Perturbation
            K_minus = np.maximum(self.K - delta_K, 0.0)
            res_m = fast_sparse_evolution(K_minus, self.omega, t_total=self.dt, n_steps=1)
            x_m, y_m = self._simulate_measurement_counts(res_m["final_state"])
            ph_minus = quantum_persistent_homology(
                x_m, y_m, self.n, persistence_threshold=0.1
            ).p_h1

            # Finite difference gradient contribution
            # We want to MINIMIZE p_h1. So if ph_plus > ph_minus, the gradient is positive in direction of delta_K.
            # To minimize, we step in the NEGATIVE gradient direction.
            diff = (ph_plus - ph_minus) / (2.0 * perturbation_scale)
            grad_K += diff * delta_K

        grad_K /= n_samples

        # 3. Update Coupling Matrix
        self.K = self.K - self.lr * grad_K

        # Enforce physical constraints
        self.K = (self.K + self.K.T) / 2.0
        np.fill_diagonal(self.K, 0.0)
        self.K = np.maximum(self.K, 0.0)

        return {
            "K_updated": self.K.copy(),
            "p_h1_current": ph_base,
            "gradient_norm": float(np.linalg.norm(grad_K)),
        }

    def optimize(self, steps: int = 10, n_samples: int = 3) -> list[dict]:
        """Run the topological optimization loop for a given number of steps."""
        history = []
        for _ in range(steps):
            history.append(self.step(n_samples))
        return history
