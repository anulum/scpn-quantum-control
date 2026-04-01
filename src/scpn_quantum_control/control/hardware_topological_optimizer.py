# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Hardware Topological Optimizer
"""Hardware-in-the-Loop Topological Feedback.

Uses IBM Quantum hardware (or noisy simulators) to perform topological
reinforcement learning. The optimizer measures p_h1 directly from the
live quantum processor and adjusts the biological coupling matrix to
minimize topological vortices in the real hardware environment.
"""

from __future__ import annotations

import numpy as np

from scpn_quantum_control.control.topological_optimizer import TopologicalCouplingOptimizer
from scpn_quantum_control.hardware.experiments import _build_evo_base, _build_xyz_circuits
from scpn_quantum_control.hardware.runner import HardwareRunner


class HardwareTopologicalOptimizer(TopologicalCouplingOptimizer):
    """Hardware-in-the-loop optimizer for coupling graphs."""

    def __init__(
        self,
        runner: HardwareRunner,
        n_qubits: int,
        initial_K: np.ndarray,
        omega: np.ndarray,
        learning_rate: float = 0.05,
        dt: float = 1.0,
    ):
        super().__init__(n_qubits, initial_K, omega, learning_rate, dt)
        self.runner = runner

    def _simulate_measurement_counts(
        self, psi: np.ndarray, shots: int = 5000
    ) -> tuple[dict, dict]:
        """Override to use the HardwareRunner for evolution and measurement."""
        # Build Trotter circuit for current K and omega
        base_qc = _build_evo_base(self.n, self.K, self.omega, t=self.dt, trotter_reps=1)

        # Build X and Y measurement circuits
        qc_z, qc_x, qc_y = _build_xyz_circuits(base_qc, self.n)

        # Run on hardware (or noisy simulator)
        job_results = self.runner.run_sampler([qc_x, qc_y], shots=shots, name="topo_opt")
        x_counts = job_results[0].counts or {}
        y_counts = job_results[1].counts or {}

        return x_counts, y_counts
