# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — FRC pulsed-shot QAOA scheduling
"""QAOA and classical optimisation of the FRC pulsed-shot scheduling cost.

The scheduler chooses a binary capacitor-bank firing sequence
``u in {0,1}^horizon``. The cost
(:func:`~scpn_quantum_control.control.qaoa_pulsed_cost.frc_pulsed_shot_cost`) is
a general (non-quadratic) function of the bitstring, so the QAOA cost layer is
the exact diagonal phase separator ``exp(-i gamma diag(cost))`` rather than an
Ising reduction. Reference: Farhi, Goldstone, Gutmann, *A quantum approximate
optimization algorithm*, arXiv:1411.4028 (2014).

Three optimisers share the same cost table:

- :func:`optimal_schedule` — exact brute-force minimum (the reference baseline).
- :func:`classical_sqp_schedule` — SciPy SLSQP relaxation plus rounding, the
  classical NMPC-style baseline.
- :func:`solve_frc_pulsed_qaoa` — p-layer QAOA over the diagonal cost.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from .qaoa_pulsed_cost import (
    FRCPlasmaSurrogate,
    FRCQAOAObjective,
    frc_pulsed_shot_cost,
)

CostFunction = Callable[[NDArray[np.float64]], float]


@dataclass(frozen=True)
class FRCScheduleResult:
    """Outcome of an FRC pulsed-shot scheduling optimisation."""

    schedule: tuple[int, ...]
    cost: float
    method: str
    evaluations: int


def _bind_cost(
    target_b_profile: NDArray[np.floating],
    available_capacitor_energy_J: float,
    objective: FRCQAOAObjective,
    surrogate: FRCPlasmaSurrogate | None,
    delta_field_T: float,
    energy_per_bank_J: float,
    dt_s: float,
) -> CostFunction:
    def cost(schedule: NDArray[np.float64]) -> float:
        value = frc_pulsed_shot_cost(
            schedule,
            target_b_profile,
            available_capacitor_energy_J,
            objective,
            surrogate=surrogate,
            delta_field_T=delta_field_T,
            energy_per_bank_J=energy_per_bank_J,
            dt_s=dt_s,
        )
        return value[0] if isinstance(value, tuple) else float(value)

    return cost


def enumerate_costs(horizon: int, cost: CostFunction) -> NDArray[np.float64]:
    """Cost of every binary schedule, indexed by integer bitstring value."""
    if not isinstance(horizon, int) or not 1 <= horizon <= 16:
        raise ValueError("horizon must be an integer in [1, 16]")
    table = np.empty(1 << horizon, dtype=np.float64)
    for code in range(1 << horizon):
        schedule = np.array(
            [(code >> (horizon - 1 - t)) & 1 for t in range(horizon)], dtype=np.float64
        )
        table[code] = cost(schedule)
    return table


def _code_to_schedule(code: int, horizon: int) -> tuple[int, ...]:
    return tuple((code >> (horizon - 1 - t)) & 1 for t in range(horizon))


def optimal_schedule(
    target_b_profile: NDArray[np.floating],
    available_capacitor_energy_J: float,
    objective: FRCQAOAObjective,
    *,
    surrogate: FRCPlasmaSurrogate | None = None,
    delta_field_T: float = 0.5,
    energy_per_bank_J: float = 1.0e5,
    dt_s: float = 1.0e-6,
) -> FRCScheduleResult:
    """Exact brute-force minimum-cost schedule (reference baseline)."""
    horizon = int(np.asarray(target_b_profile).size)
    cost = _bind_cost(
        target_b_profile,
        available_capacitor_energy_J,
        objective,
        surrogate,
        delta_field_T,
        energy_per_bank_J,
        dt_s,
    )
    table = enumerate_costs(horizon, cost)
    best = int(np.argmin(table))
    return FRCScheduleResult(
        schedule=_code_to_schedule(best, horizon),
        cost=float(table[best]),
        method="bruteforce_optimal",
        evaluations=table.size,
    )


def classical_sqp_schedule(
    target_b_profile: NDArray[np.floating],
    available_capacitor_energy_J: float,
    objective: FRCQAOAObjective,
    *,
    surrogate: FRCPlasmaSurrogate | None = None,
    delta_field_T: float = 0.5,
    energy_per_bank_J: float = 1.0e5,
    dt_s: float = 1.0e-6,
    restarts: int = 8,
    seed: int | None = None,
) -> FRCScheduleResult:
    """Classical NMPC-style baseline: SLSQP over the relaxed schedule, then round."""
    horizon = int(np.asarray(target_b_profile).size)
    cost = _bind_cost(
        target_b_profile,
        available_capacitor_energy_J,
        objective,
        surrogate,
        delta_field_T,
        energy_per_bank_J,
        dt_s,
    )
    rng = np.random.default_rng(seed)
    bounds = [(0.0, 1.0)] * horizon
    best_schedule = (0,) * horizon
    best_cost = np.inf
    evaluations = 0
    for _ in range(restarts):
        x0 = rng.uniform(0.0, 1.0, horizon)
        result = minimize(cost, x0, method="SLSQP", bounds=bounds, options={"maxiter": 200})
        evaluations += int(result.nfev)
        rounded = (np.asarray(result.x) > 0.5).astype(np.float64)
        rounded_cost = cost(rounded)
        if rounded_cost < best_cost:
            best_cost = rounded_cost
            best_schedule = tuple(int(v) for v in rounded)
    return FRCScheduleResult(
        schedule=best_schedule,
        cost=float(best_cost),
        method="classical_sqp",
        evaluations=evaluations,
    )


def _apply_transverse_mixer(state: NDArray[np.complex128], beta: float, horizon: int) -> None:
    """In-place exp(-i beta sum_j X_j) on a 2^horizon statevector."""
    cos_b = np.cos(beta)
    sin_b = -1j * np.sin(beta)
    work = state.reshape((2,) * horizon)
    for axis in range(horizon):
        moved = np.moveaxis(work, axis, 0)
        a0 = moved[0].copy()
        a1 = moved[1].copy()
        moved[0] = cos_b * a0 + sin_b * a1
        moved[1] = sin_b * a0 + cos_b * a1
    state[:] = work.reshape(-1)


def _qaoa_state(
    params: NDArray[np.float64],
    diag_norm: NDArray[np.float64],
    p_layers: int,
    horizon: int,
) -> NDArray[np.complex128]:
    dim = 1 << horizon
    state = np.full(dim, 1.0 / np.sqrt(dim), dtype=np.complex128)
    gamma = params[:p_layers]
    beta = params[p_layers:]
    for layer in range(p_layers):
        state *= np.exp(-1j * gamma[layer] * diag_norm)
        _apply_transverse_mixer(state, float(beta[layer]), horizon)
    return state


def solve_frc_pulsed_qaoa(
    target_b_profile: NDArray[np.floating],
    available_capacitor_energy_J: float,
    objective: FRCQAOAObjective,
    *,
    surrogate: FRCPlasmaSurrogate | None = None,
    p_layers: int = 4,
    delta_field_T: float = 0.5,
    energy_per_bank_J: float = 1.0e5,
    dt_s: float = 1.0e-6,
    max_iter: int = 200,
    restarts: int = 8,
    n_samples: int = 16,
    seed: int | None = None,
) -> FRCScheduleResult:
    """p-layer QAOA over the exact diagonal FRC cost (statevector).

    Runs ``restarts`` COBYLA outer-loop restarts and returns the lowest-cost
    bitstring among the ``n_samples`` highest-probability outcomes of the best
    optimised state (QAOA used as a sampler).
    """
    horizon = int(np.asarray(target_b_profile).size)
    if not 1 <= p_layers <= 8:
        raise ValueError("p_layers must lie in [1, 8]")
    if restarts < 1:
        raise ValueError("restarts must be a positive integer")
    cost = _bind_cost(
        target_b_profile,
        available_capacitor_energy_J,
        objective,
        surrogate,
        delta_field_T,
        energy_per_bank_J,
        dt_s,
    )
    diag = enumerate_costs(horizon, cost)
    scale = float(np.max(np.abs(diag))) or 1.0
    diag_norm = diag / scale  # keep cost-layer phases well-conditioned

    def expectation(params: NDArray[np.float64]) -> float:
        state = _qaoa_state(params, diag_norm, p_layers, horizon)
        return float(np.dot(np.abs(state) ** 2, diag))

    rng = np.random.default_rng(seed)
    candidates: set[int] = set()
    evaluations = 0
    for _ in range(restarts):
        x0 = rng.uniform(0.0, np.pi, 2 * p_layers)
        result = minimize(expectation, x0, method="COBYLA", options={"maxiter": max_iter})
        evaluations += int(result.nfev)
        probs = np.abs(_qaoa_state(result.x, diag_norm, p_layers, horizon)) ** 2
        top = np.argsort(probs)[::-1][: min(n_samples, probs.size)]
        candidates.update(int(code) for code in top)

    best_code = min(candidates, key=lambda code: diag[code])
    return FRCScheduleResult(
        schedule=_code_to_schedule(best_code, horizon),
        cost=float(diag[best_code]),
        method=f"qaoa_p{p_layers}",
        evaluations=evaluations,
    )


__all__ = [
    "FRCScheduleResult",
    "classical_sqp_schedule",
    "enumerate_costs",
    "optimal_schedule",
    "solve_frc_pulsed_qaoa",
]
