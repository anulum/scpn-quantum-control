# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Vqls Gs
"""VQLS for Grad-Shafranov equilibrium.

Discretized Laplacian del^2(Psi) = source on N grid points -> sparse linear
system Ax=b.  The solver evaluates the VQLS ansatz with the variational cost
C = 1 - |<b|A|x>|^2 / <x|A^dag A|x>, then returns only a residual-certified
finite-difference solution.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from qiskit import QuantumCircuit
from qiskit.circuit.library import n_local
from qiskit.quantum_info import Statevector
from scipy.optimize import minimize

from .._constants import VQLS_DENOMINATOR_EPS


@dataclass(frozen=True)
class VQLSGradShafranovResult:
    """Residual certificate for a Grad-Shafranov linear solve."""

    solution: NDArray[np.float64]
    relative_residual: float
    residual_tolerance: float
    converged: bool
    variational_solution: NDArray[np.float64]
    variational_relative_residual: float
    variational_converged: bool
    reference_solution: NDArray[np.float64] | None
    reference_relative_error: float
    method: str
    cost: float
    n_restarts: int
    optimizer_success: bool
    optimizer_message: str
    condition_number: float


class VQLS_GradShafranov:
    """Variational Quantum Linear Solver for 1D Grad-Shafranov.

    Grid: N = 2^n_qubits points, uniform spacing on [0, 1].
    Source: peaked Gaussian J(x) = exp(-(x - 0.5)^2 / 0.05).
    """

    def __init__(self, n_qubits: int = 4, source_width: float = 0.05, imag_tol: float = 0.1):
        """Grid size = 2^n_qubits. source_width controls Gaussian J(x) width."""
        self.n_qubits = n_qubits
        self.grid_size = 2**n_qubits
        self.source_width = source_width
        self.imag_tol = imag_tol
        self._A: NDArray[np.float64] | None = None
        self._b: NDArray[np.float64] | None = None
        self._optimal_params: NDArray[np.float64] | None = None
        self.last_result: VQLSGradShafranovResult | None = None

    def discretize(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Build finite-difference Laplacian and source vector."""
        N = self.grid_size
        dx = 1.0 / (N + 1)
        x = np.linspace(dx, 1.0 - dx, N)

        # 1D Laplacian: [-1, 2, -1] / dx^2
        A = np.zeros((N, N))
        for i in range(N):
            A[i, i] = 2.0
            if i > 0:
                A[i, i - 1] = -1.0
            if i < N - 1:
                A[i, i + 1] = -1.0
        A /= dx**2

        # Gaussian source profile
        b = np.exp(-((x - 0.5) ** 2) / self.source_width)
        b /= np.linalg.norm(b)

        self._A = A
        self._b = b
        return A, b

    def build_ansatz(self, reps: int = 2) -> QuantumCircuit:
        """Ry + CZ linear-entanglement ansatz for the variational state."""
        return n_local(
            self.n_qubits,
            rotation_blocks=["ry"],
            entanglement_blocks="cz",
            reps=reps,
            entanglement="linear",
        )

    def solve(
        self,
        reps: int = 2,
        maxiter: int = 200,
        seed: int | None = None,
        *,
        n_restarts: int = 1,
        residual_tol: float = 1e-10,
        allow_classical_refinement: bool = True,
    ) -> NDArray[np.float64]:
        """Return a residual-certified Grad-Shafranov flux profile.

        Cost: C = 1 - |<b|A|x>|^2 / <x|A^dag A|x>
        """
        result = self.solve_with_diagnostics(
            reps=reps,
            maxiter=maxiter,
            seed=seed,
            n_restarts=n_restarts,
            residual_tol=residual_tol,
            allow_classical_refinement=allow_classical_refinement,
        )
        if not result.converged:
            raise RuntimeError(
                "VQLS Grad-Shafranov solve exceeded residual tolerance "
                f"{result.residual_tolerance:.2e}: "
                f"relative residual {result.relative_residual:.2e}"
            )
        return result.solution

    def solve_with_diagnostics(
        self,
        reps: int = 2,
        maxiter: int = 200,
        seed: int | None = None,
        *,
        n_restarts: int = 1,
        residual_tol: float = 1e-10,
        allow_classical_refinement: bool = True,
    ) -> VQLSGradShafranovResult:
        """Run VQLS and return residual/convergence diagnostics.

        When the variational ansatz does not meet ``residual_tol`` for the
        SPD finite-difference system, the default path repairs the result with
        the direct linear solve and records ``method="direct_spd_residual_repair"``.
        """
        if self._A is None:
            self.discretize()
        if self._A is None or self._b is None:
            raise RuntimeError("call discretize() before solve()")
        if n_restarts < 1:
            raise ValueError("n_restarts must be at least 1")
        if maxiter < 1:
            raise ValueError("maxiter must be at least 1")
        if residual_tol <= 0:
            raise ValueError("residual_tol must be positive")

        ansatz = self.build_ansatz(reps)
        A = self._A
        b = self._b
        AtA = A.T @ A

        def cost_fn(params: NDArray[np.float64]) -> float:
            bound = ansatz.assign_parameters(params)
            sv = Statevector.from_instruction(bound)
            x_vec = np.array(sv)

            Ax = A @ x_vec
            bAx = np.vdot(b, Ax)
            xAtAx = np.vdot(x_vec, AtA @ x_vec).real

            if xAtAx < VQLS_DENOMINATOR_EPS:
                return 1.0
            return float(1.0 - abs(bAx) ** 2 / xAtAx)

        n_params = ansatz.num_parameters
        rng = np.random.default_rng(seed)
        best_result = None
        best_cost = float("inf")
        for _ in range(n_restarts):
            x0 = rng.uniform(-np.pi, np.pi, n_params)
            opt_result = minimize(cost_fn, x0, method="COBYLA", options={"maxiter": maxiter})
            opt_cost = float(cost_fn(opt_result.x))
            if opt_cost < best_cost:
                best_cost = opt_cost
                best_result = opt_result

        if best_result is None:
            raise RuntimeError("VQLS optimizer did not produce a candidate")
        self._optimal_params = np.asarray(best_result.x, dtype=float)

        bound = ansatz.assign_parameters(self._optimal_params)
        sv = Statevector.from_instruction(bound)
        sv_arr = np.array(sv)

        # Ψ (flux function) is real-valued; verify the ansatz converged to a real state
        imag_norm = float(np.linalg.norm(sv_arr.imag))
        if imag_norm >= self.imag_tol:
            raise ValueError(
                f"solution imaginary norm {imag_norm:.2e} exceeds tolerance {self.imag_tol:.2e}"
            )
        psi = sv_arr.real

        # L2 projection rescaling: scale = <b|b> / <b|A|psi>
        # Bravo-Prieto et al., arXiv:1909.05820 (2019), post-processing step
        A_psi = A @ psi
        scale = np.vdot(b, b) / np.vdot(b, A_psi) if abs(np.vdot(b, A_psi)) > 1e-15 else 1.0
        variational_solution: NDArray[np.float64] = (psi * float(np.real(scale))).astype(
            np.float64
        )
        variational_relative_residual = self._relative_residual(variational_solution)

        reference_solution = self._direct_solution_or_none()
        reference_relative_error = self._reference_relative_error(
            variational_solution, reference_solution
        )
        condition_number = float(np.linalg.cond(A))
        variational_converged = variational_relative_residual <= residual_tol

        solution = variational_solution
        relative_residual = variational_relative_residual
        converged = variational_converged
        method = "variational_vqls"

        if not converged and allow_classical_refinement and reference_solution is not None:
            solution = reference_solution
            relative_residual = self._relative_residual(solution)
            reference_relative_error = self._reference_relative_error(solution, reference_solution)
            converged = relative_residual <= residual_tol
            method = "direct_spd_residual_repair"

        optimizer_success = bool(getattr(best_result, "success", False))
        optimizer_message = str(getattr(best_result, "message", ""))
        result = VQLSGradShafranovResult(
            solution=solution,
            relative_residual=float(relative_residual),
            residual_tolerance=float(residual_tol),
            converged=bool(converged),
            variational_solution=variational_solution,
            variational_relative_residual=float(variational_relative_residual),
            variational_converged=bool(variational_converged),
            reference_solution=reference_solution,
            reference_relative_error=float(reference_relative_error),
            method=method,
            cost=float(best_cost),
            n_restarts=n_restarts,
            optimizer_success=optimizer_success,
            optimizer_message=optimizer_message,
            condition_number=condition_number,
        )
        self.last_result = result
        return result

    def _relative_residual(self, solution: NDArray[np.float64]) -> float:
        if self._A is None or self._b is None:
            raise RuntimeError("call discretize() before residual evaluation")
        denominator = float(np.linalg.norm(self._b))
        residual = float(np.linalg.norm(self._A @ solution - self._b))
        if denominator < VQLS_DENOMINATOR_EPS:
            return residual
        return residual / denominator

    def _direct_solution_or_none(self) -> NDArray[np.float64] | None:
        if self._A is None or self._b is None:
            raise RuntimeError("call discretize() before direct solve")
        try:
            return np.linalg.solve(self._A, self._b).astype(np.float64)
        except np.linalg.LinAlgError:
            return None

    @staticmethod
    def _reference_relative_error(
        solution: NDArray[np.float64], reference_solution: NDArray[np.float64] | None
    ) -> float:
        if reference_solution is None:
            return float("nan")
        denominator = float(np.linalg.norm(reference_solution))
        error = float(np.linalg.norm(solution - reference_solution))
        if denominator < VQLS_DENOMINATOR_EPS:
            return error
        return error / denominator
