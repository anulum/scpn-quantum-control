# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Hierarchical chimera objectives
"""Differentiable hierarchy targets composed from existing analytic terms."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from scpn_quantum_control.phase.objectives import ComposedPhaseObjective
from scpn_quantum_control.phase.synchronisation_objectives import (
    cluster_synchronisation_target_term,
)

from .schema import (
    CHIMERA_CONTROL_CLAIM_BOUNDARY,
    ChimeraControlSpecification,
    FloatArray,
)


@dataclass(frozen=True, slots=True)
class PhaseControlProposal:
    """One unapplied analytic-gradient phase proposal.

    Attributes
    ----------
    original_value, proposed_value
        Objective values before and after the accepted backtracking step.
    step_size
        Accepted scalar gradient step; zero if no strict decrease was found.
    backtracks
        Number of halvings attempted.
    accepted
        Whether the proposal strictly reduced the objective.
    phase_delta, proposed_phases
        Read-only vectors. No external or persistent system is mutated.
    claim_boundary
        Explicit synthetic and non-actuating interpretation boundary.

    """

    original_value: float
    proposed_value: float
    step_size: float
    backtracks: int
    accepted: bool
    phase_delta: FloatArray
    proposed_phases: FloatArray
    claim_boundary: str = CHIMERA_CONTROL_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate proposal scalars, vectors, and acceptance consistency."""
        for name, value in (
            ("original_value", self.original_value),
            ("proposed_value", self.proposed_value),
            ("step_size", self.step_size),
        ):
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be a finite non-negative value")
        if (
            isinstance(self.backtracks, bool)
            or not isinstance(self.backtracks, int)
            or self.backtracks < 0
        ):
            raise ValueError("backtracks must be a non-negative integer")
        delta = np.array(self.phase_delta, dtype=np.float64, copy=True)
        proposed = np.array(self.proposed_phases, dtype=np.float64, copy=True)
        if delta.ndim != 1 or proposed.shape != delta.shape or delta.size == 0:
            raise ValueError("phase_delta and proposed_phases must be equal non-empty vectors")
        if not np.all(np.isfinite(delta)) or not np.all(np.isfinite(proposed)):
            raise ValueError("proposal vectors must contain only finite values")
        if not self.claim_boundary.strip():
            raise ValueError("claim_boundary must be non-empty")
        if self.accepted and not (
            self.step_size > 0.0 and self.proposed_value < self.original_value
        ):
            raise ValueError("accepted proposals must use a positive step and strict decrease")
        if not self.accepted and not (
            self.step_size == 0.0
            and self.proposed_value == self.original_value
            and np.all(delta == 0.0)
        ):
            raise ValueError("rejected proposals must preserve value with a zero step and delta")
        delta.setflags(write=False)
        proposed.setflags(write=False)
        object.__setattr__(self, "phase_delta", delta)
        object.__setattr__(self, "proposed_phases", proposed)


def build_chimera_control_objective(
    specification: ChimeraControlSpecification,
    *,
    min_order_parameter: float = 1.0e-12,
) -> ComposedPhaseObjective:
    """Build one analytic cluster-order term per non-zero hierarchy target.

    Parameters
    ----------
    specification
        Validated nested hierarchy and per-community target rows.
    min_order_parameter
        Positive singularity guard forwarded to the existing analytic
        cluster-order gradient.

    Returns
    -------
    ComposedPhaseObjective
        Weighted objective whose term names are
        ``chimera_<level_name>_target`` and whose gradients come from
        :func:`cluster_synchronisation_target_term`.

    Raises
    ------
    ValueError
        If the threshold is not finite and positive or every target weight is
        zero.

    """
    threshold = float(min_order_parameter)
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("min_order_parameter must be a finite positive value")
    terms = []
    for target in specification.targets:
        if target.weight == 0.0:
            continue
        level = specification.hierarchy.level(target.level_name)
        terms.append(
            cluster_synchronisation_target_term(
                specification.hierarchy.node_count,
                level.communities,
                targets=target.order_parameters,
                min_order_parameter=threshold,
                term_weight=target.weight,
                name=f"chimera_{target.level_name}_target",
            )
        )
    if not terms:
        raise ValueError("at least one hierarchy target must have positive weight")
    return ComposedPhaseObjective(
        terms=tuple(terms),
        name="chimera_multiscale_control",
        claim_boundary=specification.claim_boundary,
    )


def propose_phase_control_step(
    objective: ComposedPhaseObjective,
    phases: ArrayLike,
    *,
    initial_step_size: float = 0.25,
    max_backtracks: int = 16,
) -> PhaseControlProposal:
    """Propose an unapplied backtracking analytic-gradient phase step.

    The routine evaluates the supplied objective at a single phase vector,
    halves ``initial_step_size`` until a strict finite decrease is found, and
    returns the candidate without mutating ``phases`` or any external system.
    If the gradient is zero or no strict decrease is found, ``accepted`` is
    false and the returned candidate equals the input.

    Parameters
    ----------
    objective
        Composed phase objective, normally from
        :func:`build_chimera_control_objective`.
    phases
        Finite one-dimensional phase vector matching the objective width.
    initial_step_size
        Finite positive first backtracking step.
    max_backtracks
        Number of candidate evaluations after the initial point; at least one.

    """
    step = float(initial_step_size)
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("initial_step_size must be a finite positive value")
    if (
        isinstance(max_backtracks, bool)
        or not isinstance(max_backtracks, int)
        or max_backtracks < 1
    ):
        raise ValueError("max_backtracks must be a positive integer")
    vector = np.array(phases, dtype=np.float64, copy=True)
    if vector.ndim != 1 or vector.size == 0 or not np.all(np.isfinite(vector)):
        raise ValueError("phases must be a non-empty finite one-dimensional vector")
    evaluation = objective.evaluate(vector)
    gradient = evaluation.gradient
    if float(np.linalg.norm(gradient)) == 0.0:
        zero = np.zeros_like(vector)
        return PhaseControlProposal(
            original_value=evaluation.value,
            proposed_value=evaluation.value,
            step_size=0.0,
            backtracks=0,
            accepted=False,
            phase_delta=zero,
            proposed_phases=vector,
            claim_boundary=objective.claim_boundary,
        )
    for backtracks in range(max_backtracks):
        delta = -step * gradient
        candidate = vector + delta
        proposed_value = objective(candidate)
        if np.isfinite(proposed_value) and proposed_value < evaluation.value:
            return PhaseControlProposal(
                original_value=evaluation.value,
                proposed_value=proposed_value,
                step_size=step,
                backtracks=backtracks,
                accepted=True,
                phase_delta=delta,
                proposed_phases=candidate,
                claim_boundary=objective.claim_boundary,
            )
        step *= 0.5
    zero = np.zeros_like(vector)
    return PhaseControlProposal(
        original_value=evaluation.value,
        proposed_value=evaluation.value,
        step_size=0.0,
        backtracks=max_backtracks,
        accepted=False,
        phase_delta=zero,
        proposed_phases=vector,
        claim_boundary=objective.claim_boundary,
    )


__all__ = [
    "PhaseControlProposal",
    "build_chimera_control_objective",
    "propose_phase_control_step",
]
