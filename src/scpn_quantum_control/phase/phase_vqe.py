# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Phase Vqe
"""VQE for Kuramoto/XY Hamiltonian ground state.

Finds the maximum synchronization configuration of the coupled oscillator
system using a variational quantum eigensolver with physics-informed ansatz
where entanglement topology matches Knm sparsity.
"""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray
from qiskit.quantum_info import Statevector
from scipy.optimize import minimize

from ..bridge.knm_hamiltonian import knm_to_ansatz, knm_to_hamiltonian
from ..differentiable import (
    GradientResult,
    parameter_shift_gradient,
    value_and_parameter_shift_grad,
)
from ..hardware.classical import classical_exact_diag

FloatArray: TypeAlias = NDArray[np.float64]
PhaseVQEResult: TypeAlias = dict[str, object]


class PhaseVQE:
    """VQE solver for the XY Kuramoto Hamiltonian ground state.

    The ground state corresponds to maximum phase synchronization
    (minimum energy = strongest coupling alignment).
    """

    def __init__(
        self,
        K: FloatArray,
        omega: FloatArray,
        ansatz_reps: int = 2,
        threshold: float = 0.01,
    ):
        """Build Hamiltonian and K_nm-informed ansatz from coupling parameters."""
        self.K: FloatArray = np.asarray(K, dtype=np.float64)
        self.omega: FloatArray = np.asarray(omega, dtype=np.float64)
        self.hamiltonian = knm_to_hamiltonian(self.K, self.omega)
        self.ansatz = knm_to_ansatz(self.K, reps=ansatz_reps, threshold=threshold)
        self.n_params = self.ansatz.num_parameters
        self._optimal_params: FloatArray | None = None
        self._ground_energy: float | None = None

    def _validate_params(self, params: FloatArray) -> FloatArray:
        """Return finite one-dimensional VQE parameters with the expected width."""
        values: FloatArray = np.asarray(params, dtype=np.float64)
        if values.shape != (self.n_params,):
            raise ValueError(f"params must have shape ({self.n_params},), got {values.shape}")
        if not np.all(np.isfinite(values)):
            raise ValueError("params must contain only finite values")
        return values

    def _cost(self, params: FloatArray) -> float:
        values = self._validate_params(params)
        bound = self.ansatz.assign_parameters(values)
        sv = Statevector.from_instruction(bound)
        return float(sv.expectation_value(self.hamiltonian).real)

    def parameter_shift_gradient(self, params: FloatArray) -> FloatArray:
        """Return analytic parameter-shift gradients for the current ansatz."""
        values = self._validate_params(params)
        return parameter_shift_gradient(self._cost, values)

    def value_and_parameter_shift_gradient(self, params: FloatArray) -> GradientResult:
        """Return the VQE energy and structured parameter-shift gradient metadata."""
        values = self._validate_params(params)
        return value_and_parameter_shift_grad(self._cost, values)

    def solve(
        self,
        optimizer: str = "COBYLA",
        maxiter: int = 200,
        seed: int | None = None,
        gradient_method: str | None = None,
    ) -> PhaseVQEResult:
        """Run VQE optimisation.

        Returns dict with ground_energy, optimal_params, n_evals, and gradient metadata.
        """
        gradient_mode = "none" if gradient_method is None else gradient_method.strip().lower()
        gradient_mode = gradient_mode.replace("-", "_")
        if gradient_mode not in {"none", "parameter_shift"}:
            raise ValueError("gradient_method must be one of: None, 'none', 'parameter_shift'")

        x0 = np.random.default_rng(seed).uniform(-np.pi, np.pi, self.n_params)

        effective_maxiter = max(maxiter, self.n_params + 10)
        effective_optimizer = optimizer
        n_grad_evals = 0
        jac = None

        if gradient_mode == "parameter_shift":
            if optimizer.upper() in {"COBYLA", "NELDER-MEAD", "POWELL"}:
                effective_optimizer = "L-BFGS-B"

            def jac(params: FloatArray) -> FloatArray:
                nonlocal n_grad_evals
                n_grad_evals += 1
                return self.parameter_shift_gradient(params)

        result = minimize(
            self._cost,
            x0,
            method=effective_optimizer,
            jac=jac,
            options={"maxiter": effective_maxiter},
        )

        self._optimal_params = result.x
        self._ground_energy = float(result.fun)

        n = len(self.omega)
        exact = classical_exact_diag(n, K=self.K, omega=self.omega)
        exact_e = exact["ground_energy"]
        gap = abs(self._ground_energy - exact_e)

        return {
            "ground_energy": self._ground_energy,
            "vqe_energy": self._ground_energy,  # alias for backward compatibility
            "exact_energy": exact_e,
            "energy_gap": gap,
            "relative_error_pct": gap / abs(exact_e) * 100
            if abs(exact_e) > 1e-15
            else float("nan"),
            "optimal_params": self._optimal_params,
            "n_evals": result.nfev,
            "n_grad_evals": n_grad_evals,
            "n_params": self.n_params,
            "optimizer": effective_optimizer,
            "gradient_method": gradient_mode,
            "gradient_norm": float(np.linalg.norm(self.parameter_shift_gradient(result.x)))
            if gradient_mode == "parameter_shift"
            else float("nan"),
            "converged": result.success,
        }

    def ground_state(self) -> Statevector | None:
        """Return the optimized ground state vector (call solve first)."""
        if self._optimal_params is None:
            return None
        bound = self.ansatz.assign_parameters(self._optimal_params)
        return Statevector.from_instruction(bound)
