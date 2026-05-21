# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Topology Control Optimizers
"""Projected optimisers for non-smooth persistent-H1 objectives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from .objectives import CouplingTopologyObjective, ObjectiveBreakdown

FloatArray: TypeAlias = NDArray[np.float64]


@dataclass(frozen=True)
class TopologyOptimisationStep:
    """One projected optimisation step."""

    index: int
    matrix: FloatArray
    objective: ObjectiveBreakdown
    gradient_norm: float
    step_size: float
    perturbation: float


@dataclass(frozen=True)
class TopologyOptimisationTrace:
    """Full optimisation trace and final projected matrix."""

    initial_matrix: FloatArray
    final_matrix: FloatArray
    steps: list[TopologyOptimisationStep]
    seed: int
    objective_schema: str = "topological_control_objective_v1"

    @property
    def final_objective(self) -> ObjectiveBreakdown:
        """Objective breakdown at the final matrix."""

        if not self.steps:
            raise ValueError("optimisation trace has no steps")
        return self.steps[-1].objective


class ProjectedSPSAOptimizer:
    """Deterministic projected SPSA optimiser for noisy PH objectives."""

    def __init__(
        self,
        *,
        seed: int = 0,
        max_steps: int = 32,
        step_size: float = 0.05,
        perturbation: float = 0.02,
    ) -> None:
        if max_steps < 0:
            raise ValueError("max_steps must be non-negative")
        if step_size <= 0.0:
            raise ValueError("step_size must be positive")
        if perturbation <= 0.0:
            raise ValueError("perturbation must be positive")
        self.seed = int(seed)
        self.max_steps = int(max_steps)
        self.step_size = float(step_size)
        self.perturbation = float(perturbation)

    def optimise(
        self,
        initial_matrix: np.ndarray,
        objective: CouplingTopologyObjective,
    ) -> TopologyOptimisationTrace:
        """Run projected SPSA and return an auditable trace."""

        rng = np.random.default_rng(self.seed)
        K = objective.ledger.project(initial_matrix)
        steps: list[TopologyOptimisationStep] = []

        for idx in range(self.max_steps):
            delta = rng.choice(np.array([-1.0, 1.0], dtype=np.float64), size=K.shape)
            delta = np.triu(delta, 1)
            delta = delta + delta.T
            np.fill_diagonal(delta, 0.0)

            plus = objective.ledger.project(K + self.perturbation * delta)
            minus = objective.ledger.project(K - self.perturbation * delta)
            plus_score = objective.evaluate(plus).total
            minus_score = objective.evaluate(minus).total
            gradient = ((plus_score - minus_score) / (2.0 * self.perturbation)) * delta
            gradient = (gradient + gradient.T) / 2.0
            np.fill_diagonal(gradient, 0.0)

            candidate = objective.ledger.project(K - self.step_size * gradient)
            K = candidate
            breakdown = objective.evaluate(K)
            steps.append(
                TopologyOptimisationStep(
                    index=idx,
                    matrix=cast(FloatArray, K.copy()),
                    objective=breakdown,
                    gradient_norm=float(np.linalg.norm(gradient)),
                    step_size=self.step_size,
                    perturbation=self.perturbation,
                )
            )

        if not steps:
            breakdown = objective.evaluate(K)
            steps.append(
                TopologyOptimisationStep(
                    index=0,
                    matrix=cast(FloatArray, K.copy()),
                    objective=breakdown,
                    gradient_norm=0.0,
                    step_size=0.0,
                    perturbation=0.0,
                )
            )

        return TopologyOptimisationTrace(
            initial_matrix=cast(FloatArray, objective.ledger.project(initial_matrix)),
            final_matrix=cast(FloatArray, K.copy()),
            steps=steps,
            seed=self.seed,
        )


class ProjectedScipyOptimizer:
    """Projected COBYLA wrapper with auditable per-iterate trace."""

    def __init__(self, *, maxiter: int = 100) -> None:
        if maxiter <= 0:
            raise ValueError("maxiter must be positive")
        self.maxiter = int(maxiter)

    def optimise(
        self,
        initial_matrix: np.ndarray,
        objective: CouplingTopologyObjective,
    ) -> TopologyOptimisationTrace:
        """Optimise using scipy when available."""

        from scipy.optimize import minimize

        initial = objective.ledger.project(initial_matrix)
        n = initial.shape[0]
        upper = [(i, j) for i in range(n) for j in range(i + 1, n)]

        def unpack(values: np.ndarray) -> FloatArray:
            K = np.zeros_like(initial)
            for value, (i, j) in zip(values, upper):
                K[i, j] = float(value)
                K[j, i] = float(value)
            return cast(FloatArray, objective.ledger.project(K))

        def score(values: np.ndarray) -> float:
            return objective.evaluate(unpack(values)).total

        x0 = np.array([initial[i, j] for i, j in upper], dtype=np.float64)
        steps: list[TopologyOptimisationStep] = []

        def on_step(values: np.ndarray) -> None:
            matrix = unpack(np.asarray(values, dtype=np.float64))
            steps.append(
                TopologyOptimisationStep(
                    index=len(steps),
                    matrix=matrix.copy(),
                    objective=objective.evaluate(matrix),
                    gradient_norm=0.0,
                    step_size=0.0,
                    perturbation=0.0,
                )
            )

        result = minimize(
            score,
            x0,
            method="COBYLA",
            callback=on_step,
            options={"maxiter": self.maxiter},
        )
        final = unpack(np.asarray(result.x, dtype=np.float64))
        if not steps:
            steps.append(
                TopologyOptimisationStep(
                    index=0,
                    matrix=final.copy(),
                    objective=objective.evaluate(final),
                    gradient_norm=0.0,
                    step_size=0.0,
                    perturbation=0.0,
                )
            )
        elif not np.allclose(steps[-1].matrix, final):
            steps.append(
                TopologyOptimisationStep(
                    index=len(steps),
                    matrix=final.copy(),
                    objective=objective.evaluate(final),
                    gradient_norm=0.0,
                    step_size=0.0,
                    perturbation=0.0,
                )
            )
        return TopologyOptimisationTrace(
            initial_matrix=initial,
            final_matrix=final,
            steps=steps,
            seed=0,
        )


__all__ = [
    "FloatArray",
    "ProjectedSPSAOptimizer",
    "ProjectedScipyOptimizer",
    "TopologyOptimisationStep",
    "TopologyOptimisationTrace",
]
