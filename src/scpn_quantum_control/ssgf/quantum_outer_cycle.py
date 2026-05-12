# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Quantum Outer Cycle
"""Quantum SSGF outer cycle: variational geometry descent via quantum cost.

The SSGF outer cycle optimises a latent vector z that parameterises
the geometry matrix W(z). In the classical SSGF, the cost function
includes C_micro (microscale coherence), C4_tcbo (topological
observer), and C_pgbo (geometric tensor minimum).

The quantum outer cycle adds a quantum cost term:
    C_quantum(z) = 1 - R_global(z)

where R_global is the Kuramoto order parameter after quantum
Trotter evolution of the state encoded from W(z).

Full quantum-enhanced SSGF cost:
    C_total(z) = α × C_quantum(z) + (1-α) × C_classical(z)

The outer cycle runs gradient descent on z using dC_total/dz,
where dC_quantum/dz comes from the quantum_gradient module.

This is the core missing feature that closes the quantum-in-the-loop:
quantum evolution → synchronisation measurement → geometry update.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .quantum_gradient import _w_from_z, compute_quantum_gradient


@dataclass
class OuterCycleResult:
    """Result of quantum SSGF outer cycle optimisation."""

    z_optimised: np.ndarray
    W_optimised: np.ndarray
    cost_history: list[float]
    r_global_history: list[float]
    n_iterations: int
    final_cost: float
    final_r_global: float
    converged: bool


def classical_cost(W: np.ndarray, *, allow_surrogate: bool = False) -> float:
    """Opt-in legacy surrogate for the classical SSGF cost.

    The full SSGF objective is not determined by ``W`` alone. Production
    mixed/classical optimisation must pass ``classical_cost_fn`` into
    ``quantum_outer_cycle``. This helper remains available only for explicitly
    labelled exploratory comparisons.
    """
    if not allow_surrogate:
        raise ValueError(
            "classical_cost is a surrogate and is disabled by default. Pass a real "
            "classical_cost_fn to quantum_outer_cycle, or set allow_surrogate=True "
            "only for labelled exploratory runs."
        )
    n = W.shape[0]
    off_diag = W[np.triu_indices(n, k=1)]
    if len(off_diag) == 0:
        return 1.0
    mean_coupling = float(np.mean(off_diag))
    std_coupling = float(np.std(off_diag))
    # Penalise weak coupling and imbalance
    return 1.0 - mean_coupling / (mean_coupling + 1.0) + std_coupling / (mean_coupling + 1.0)


def quantum_outer_cycle(
    n_osc: int,
    z_init: np.ndarray | None = None,
    alpha: float = 0.5,
    classical_cost_fn: Callable[[np.ndarray], float] | None = None,
    allow_classical_surrogate: bool = False,
    learning_rate: float = 0.1,
    max_iterations: int = 30,
    convergence_threshold: float = 1e-4,
    dt: float = 0.1,
    trotter_reps: int = 3,
    seed: int | None = None,
) -> OuterCycleResult:
    """Run the quantum SSGF outer cycle.

    Args:
        n_osc: number of oscillators
        z_init: initial latent vector (default: zeros)
        alpha: quantum cost weight (0 = pure classical, 1 = pure quantum)
        classical_cost_fn: production classical SSGF cost function for alpha < 1
        allow_classical_surrogate: opt into the legacy coupling-balance surrogate
        learning_rate: gradient descent step size
        max_iterations: maximum optimisation steps
        convergence_threshold: stop if |ΔC| < threshold
        dt: Trotter evolution time
        trotter_reps: Trotter repetitions
        seed: random seed
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if alpha < 1.0 and classical_cost_fn is None and not allow_classical_surrogate:
        raise ValueError(
            "quantum_outer_cycle with alpha < 1 requires classical_cost_fn. The legacy "
            "classical cost surrogate is disabled unless allow_classical_surrogate=True."
        )

    n_upper = n_osc * (n_osc - 1) // 2
    if z_init is None:
        rng = np.random.default_rng(seed)
        z_init = rng.normal(0, 0.5, size=n_upper)

    z = z_init.copy()
    cost_history: list[float] = []
    r_history: list[float] = []
    converged = False

    for iteration in range(max_iterations):
        # Quantum gradient
        qg = compute_quantum_gradient(
            z,
            n_osc,
            dt=dt,
            trotter_reps=trotter_reps,
        )

        W = _w_from_z(z, n_osc)
        c_class_grad = np.zeros_like(z)
        c_class = 0.0
        if alpha < 1.0:
            cost_fn = classical_cost_fn
            if cost_fn is None:

                def surrogate_cost(matrix: np.ndarray) -> float:
                    return classical_cost(matrix, allow_surrogate=True)

                cost_fn = surrogate_cost
            c_class = float(cost_fn(W))
            if not np.isfinite(c_class):
                raise ValueError("classical_cost_fn returned a non-finite value.")

            eps = 0.01
            for k in range(len(z)):
                z_plus = z.copy()
                z_plus[k] += eps
                z_minus = z.copy()
                z_minus[k] -= eps
                c_plus = float(cost_fn(_w_from_z(z_plus, n_osc)))
                c_minus = float(cost_fn(_w_from_z(z_minus, n_osc)))
                if not np.isfinite(c_plus) or not np.isfinite(c_minus):
                    raise ValueError(
                        "classical_cost_fn returned a non-finite finite-difference value."
                    )
                c_class_grad[k] = (c_plus - c_minus) / (2 * eps)

        # Combined cost and gradient
        c_total = alpha * qg.cost + (1.0 - alpha) * c_class
        grad_total = alpha * qg.gradient + (1.0 - alpha) * c_class_grad

        cost_history.append(c_total)
        r_history.append(qg.r_global)

        # Convergence check
        if iteration > 0 and abs(cost_history[-1] - cost_history[-2]) < convergence_threshold:
            converged = True
            break

        # Gradient descent step
        z = z - learning_rate * grad_total

    W_final = _w_from_z(z, n_osc)

    return OuterCycleResult(
        z_optimised=z,
        W_optimised=W_final,
        cost_history=cost_history,
        r_global_history=r_history,
        n_iterations=len(cost_history),
        final_cost=cost_history[-1] if cost_history else 1.0,
        final_r_global=r_history[-1] if r_history else 0.0,
        converged=converged,
    )
