# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Parity-projected gradient optimiser
"""Deterministic projected-gradient loop for a synthetic parity-sector task."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from .objectives import ParityProtectedQuadraticObjective
from .schema import DLA_TOPOLOGY_CLAIM_BOUNDARY

ComplexArray: TypeAlias = NDArray[np.complex128]


def _read_only_complex(value: NDArray[np.complex128]) -> ComplexArray:
    result = np.array(value, dtype=np.complex128, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class ProjectedGradientConfig:
    """Backtracking policy for parity-projected gradient descent.

    Parameters
    ----------
    max_steps:
        Maximum number of accepted or terminal proposal records.
    initial_step_size:
        Positive step size tried first at every iteration.
    contraction:
        Multiplicative factor in ``(0, 1)`` for each backtrack.
    max_backtracks:
        Maximum contractions before a proposal is rejected.
    gradient_tolerance:
        Non-negative norm threshold for convergence.
    minimum_step_size:
        Smallest positive step size eligible for evaluation.

    """

    max_steps: int = 32
    initial_step_size: float = 0.5
    contraction: float = 0.5
    max_backtracks: int = 12
    gradient_tolerance: float = 1.0e-12
    minimum_step_size: float = 1.0e-12

    def __post_init__(self) -> None:
        """Validate step, contraction, backtracking, and tolerance limits."""
        if isinstance(self.max_steps, bool) or not isinstance(self.max_steps, int):
            raise ValueError("max_steps must be an integer")
        if self.max_steps < 1:
            raise ValueError("max_steps must be positive")
        if isinstance(self.max_backtracks, bool) or not isinstance(self.max_backtracks, int):
            raise ValueError("max_backtracks must be an integer")
        if self.max_backtracks < 0:
            raise ValueError("max_backtracks must be non-negative")
        if not np.isfinite(self.initial_step_size) or self.initial_step_size <= 0.0:
            raise ValueError("initial_step_size must be finite and positive")
        if not np.isfinite(self.contraction) or not 0.0 < self.contraction < 1.0:
            raise ValueError("contraction must lie strictly between zero and one")
        if not np.isfinite(self.gradient_tolerance) or self.gradient_tolerance < 0.0:
            raise ValueError("gradient_tolerance must be finite and non-negative")
        if not np.isfinite(self.minimum_step_size) or self.minimum_step_size <= 0.0:
            raise ValueError("minimum_step_size must be finite and positive")
        if self.minimum_step_size > self.initial_step_size:
            raise ValueError("minimum_step_size must not exceed initial_step_size")


@dataclass(frozen=True, slots=True)
class ProjectedGradientStep:
    """One accepted or fail-closed projected-gradient proposal.

    The record binds proposal index, backtracking count, objective/leakage
    before and after projection, gradient norm, and the immutable resulting
    state. Rejected records use zero step size and preserve the prior value.
    """

    index: int
    accepted: bool
    backtracks: int
    step_size: float
    original_value: float
    proposed_value: float
    leakage_before: float
    leakage_after: float
    gradient_norm: float
    state: ComplexArray

    def __post_init__(self) -> None:
        """Validate proposal scalars and retain an immutable state copy."""
        if isinstance(self.index, bool) or not isinstance(self.index, int) or self.index < 0:
            raise ValueError("index must be a non-negative integer")
        if (
            isinstance(self.backtracks, bool)
            or not isinstance(self.backtracks, int)
            or self.backtracks < 0
        ):
            raise ValueError("backtracks must be a non-negative integer")
        for name in (
            "step_size",
            "original_value",
            "proposed_value",
            "leakage_before",
            "leakage_after",
            "gradient_norm",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)
        if self.accepted:
            if self.step_size <= 0.0 or not self.proposed_value < self.original_value:
                raise ValueError("accepted steps require positive size and strict decrease")
        elif self.proposed_value != self.original_value or self.step_size != 0.0:
            raise ValueError("rejected steps must preserve value and use zero step size")
        state = np.asarray(self.state, dtype=np.complex128)
        if state.ndim != 1 or state.size == 0 or not np.all(np.isfinite(state)):
            raise ValueError("state must be a finite non-empty vector")
        object.__setattr__(self, "state", _read_only_complex(state))


@dataclass(frozen=True, slots=True)
class ParityProjectedOptimisationTrace:
    """Immutable initial/final states and ordered projected-gradient steps.

    Parameters
    ----------
    initial_state:
        Validated pre-optimisation state, which may contain leakage.
    final_state:
        Last accepted parity-projected state.
    steps:
        Ordered accepted steps and, if backtracking fails, one terminal
        rejected step.
    content_digest:
        SHA-256 binding the initial/final arrays and every scalar/state record.
    claim_boundary:
        Explicit finite synthetic and no-actuation boundary.

    """

    initial_state: ComplexArray
    final_state: ComplexArray
    steps: tuple[ProjectedGradientStep, ...]
    content_digest: str
    claim_boundary: str = DLA_TOPOLOGY_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate trace alignment, custody digest, and claim metadata."""
        initial = np.asarray(self.initial_state, dtype=np.complex128)
        final = np.asarray(self.final_state, dtype=np.complex128)
        if initial.ndim != 1 or initial.size == 0 or not np.all(np.isfinite(initial)):
            raise ValueError("initial_state must be a finite non-empty vector")
        if final.shape != initial.shape or not np.all(np.isfinite(final)):
            raise ValueError("final_state must be finite and match initial_state")
        if any(step.state.shape != initial.shape for step in self.steps):
            raise ValueError("every step state must match trace state shape")
        if len(self.content_digest) != 64 or any(
            char not in "0123456789abcdef" for char in self.content_digest
        ):
            raise ValueError("content_digest must be a lowercase SHA-256 digest")
        if not isinstance(self.claim_boundary, str) or not self.claim_boundary.strip():
            raise ValueError("claim_boundary must be a non-empty string")
        object.__setattr__(self, "initial_state", _read_only_complex(initial))
        object.__setattr__(self, "final_state", _read_only_complex(final))
        object.__setattr__(self, "claim_boundary", self.claim_boundary.strip())

    @property
    def accepted_steps(self) -> int:
        """Return the number of strict-decrease proposals accepted."""
        return sum(step.accepted for step in self.steps)


def _trace_digest(
    initial: ComplexArray,
    final: ComplexArray,
    steps: tuple[ProjectedGradientStep, ...],
) -> str:
    digest = hashlib.sha256(initial.tobytes() + final.tobytes())
    for step in steps:
        digest.update(
            repr(
                (
                    step.index,
                    step.accepted,
                    step.backtracks,
                    step.step_size,
                    step.original_value,
                    step.proposed_value,
                    step.leakage_before,
                    step.leakage_after,
                    step.gradient_norm,
                )
            ).encode()
        )
        digest.update(step.state.tobytes())
    return digest.hexdigest()


def optimise_parity_protected_state(
    initial_state: NDArray[np.complex128],
    objective: ParityProtectedQuadraticObjective,
    config: ProjectedGradientConfig | None = None,
) -> ParityProjectedOptimisationTrace:
    """Run local projected gradient descent with strict-decrease backtracking.

    Projection occurs inside every proposal. The returned state is a local
    numerical value and is never applied to a circuit, provider, or device.

    Parameters
    ----------
    initial_state:
        Finite dense complex state matching ``objective.projector``.
    objective:
        Validated parity-protected quadratic objective.
    config:
        Optional backtracking policy; defaults to ``ProjectedGradientConfig``.

    Returns
    -------
    ParityProjectedOptimisationTrace
        Immutable initial/final states, ordered proposals, and custody digest.

    Raises
    ------
    ValueError
        If the objective, config, or initial state violates its public contract.

    """
    if not isinstance(objective, ParityProtectedQuadraticObjective):
        raise ValueError("objective must be a ParityProtectedQuadraticObjective")
    settings = ProjectedGradientConfig() if config is None else config
    if not isinstance(settings, ProjectedGradientConfig):
        raise ValueError("config must be a ProjectedGradientConfig")
    initial = objective.projector.as_state(initial_state, name="initial_state")
    state = np.array(initial, dtype=np.complex128, copy=True)
    steps: list[ProjectedGradientStep] = []

    for index in range(settings.max_steps):
        current = objective.evaluate(state)
        gradient_norm = float(np.linalg.norm(current.gradient))
        if gradient_norm <= settings.gradient_tolerance:
            break
        step_size = settings.initial_step_size
        accepted = False
        accepted_state = np.array(state, dtype=np.complex128, copy=True)
        accepted_evaluation = current
        backtracks = 0
        for attempt in range(settings.max_backtracks + 1):
            backtracks = attempt
            raw_candidate = state - step_size * current.gradient
            candidate = objective.projector.project(raw_candidate)
            candidate_evaluation = objective.evaluate(candidate)
            if candidate_evaluation.value < current.value:
                accepted = True
                accepted_state = np.array(candidate, dtype=np.complex128, copy=True)
                accepted_evaluation = candidate_evaluation
                break
            step_size *= settings.contraction
            if step_size < settings.minimum_step_size:
                break

        if not accepted:
            steps.append(
                ProjectedGradientStep(
                    index=index,
                    accepted=False,
                    backtracks=backtracks,
                    step_size=0.0,
                    original_value=current.value,
                    proposed_value=current.value,
                    leakage_before=current.leakage_mass,
                    leakage_after=current.leakage_mass,
                    gradient_norm=gradient_norm,
                    state=state,
                )
            )
            break
        state = accepted_state
        steps.append(
            ProjectedGradientStep(
                index=index,
                accepted=True,
                backtracks=backtracks,
                step_size=step_size,
                original_value=current.value,
                proposed_value=accepted_evaluation.value,
                leakage_before=current.leakage_mass,
                leakage_after=accepted_evaluation.leakage_mass,
                gradient_norm=gradient_norm,
                state=state,
            )
        )

    final = _read_only_complex(state)
    ordered = tuple(steps)
    return ParityProjectedOptimisationTrace(
        initial_state=initial,
        final_state=final,
        steps=ordered,
        content_digest=_trace_digest(initial, final, ordered),
    )


__all__ = [
    "ComplexArray",
    "ParityProjectedOptimisationTrace",
    "ProjectedGradientConfig",
    "ProjectedGradientStep",
    "optimise_parity_protected_state",
]
