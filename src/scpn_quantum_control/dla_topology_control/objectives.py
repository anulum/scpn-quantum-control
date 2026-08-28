# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Parity-protected objectives
"""Analytic synthetic objective inside a fixed DLA-parity sector."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from .parity import ParitySectorProjector
from .schema import DLA_TOPOLOGY_CLAIM_BOUNDARY

ComplexArray: TypeAlias = NDArray[np.complex128]


def _read_only_complex(value: NDArray[np.complex128]) -> ComplexArray:
    result = np.array(value, dtype=np.complex128, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class ParityProtectedObjectiveEvaluation:
    """Decomposed objective value and analytic gradient for one dense state.

    Parameters
    ----------
    value:
        Non-negative total objective.
    target_distance:
        Half squared Euclidean distance from the configured target.
    leakage_mass:
        Absolute outside-sector squared norm before weighting.
    state:
        Read-only complex state copy used for this evaluation.
    gradient:
        Read-only exact Euclidean complex gradient.
    claim_boundary:
        Limit on physical and operational interpretation.

    """

    value: float
    target_distance: float
    leakage_mass: float
    state: ComplexArray
    gradient: ComplexArray
    claim_boundary: str = DLA_TOPOLOGY_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Normalize and validate objective values, arrays, and claim custody."""
        for name in ("value", "target_distance", "leakage_mass"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)
        state = np.asarray(self.state, dtype=np.complex128)
        gradient = np.asarray(self.gradient, dtype=np.complex128)
        if state.ndim != 1 or state.size == 0 or not np.all(np.isfinite(state)):
            raise ValueError("state must be a finite non-empty vector")
        if gradient.shape != state.shape or not np.all(np.isfinite(gradient)):
            raise ValueError("gradient must be finite and match state shape")
        object.__setattr__(self, "state", _read_only_complex(state))
        object.__setattr__(self, "gradient", _read_only_complex(gradient))
        if not isinstance(self.claim_boundary, str) or not self.claim_boundary.strip():
            raise ValueError("claim_boundary must be a non-empty string")
        object.__setattr__(self, "claim_boundary", self.claim_boundary.strip())


@dataclass(frozen=True, slots=True)
class ParityProtectedQuadraticObjective:
    """Target-distance objective with an analytic outside-sector penalty.

    Parameters
    ----------
    projector:
        Fixed parity-sector projector defining the protected subspace.
    target_state:
        Finite non-zero vector lying entirely in the selected sector.
    leakage_weight:
        Non-negative coefficient multiplying absolute outside-sector mass.

    Notes
    -----
    For state ``psi`` and target ``tau``, the objective is
    ``0.5 * ||psi - tau||^2 + leakage_weight * ||Q psi||^2`` where
    ``Q = I - P``. This is a synthetic differentiable task, not a physical
    control Hamiltonian or a controllability certificate.

    """

    projector: ParitySectorProjector
    target_state: ComplexArray
    leakage_weight: float = 1.0
    claim_boundary: str = DLA_TOPOLOGY_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Normalize and validate the parity-protected quadratic objective."""
        if not isinstance(self.projector, ParitySectorProjector):
            raise ValueError("projector must be a ParitySectorProjector")
        target = self.projector.as_state(self.target_state, name="target_state")
        if float(np.vdot(target, target).real) <= 0.0:
            raise ValueError("target_state must have positive norm")
        projected = self.projector.project(target)
        if not np.allclose(projected, target, rtol=0.0, atol=1.0e-12):
            raise ValueError("target_state must lie entirely in the selected parity sector")
        if not np.isfinite(self.leakage_weight) or self.leakage_weight < 0.0:
            raise ValueError("leakage_weight must be finite and non-negative")
        if not isinstance(self.claim_boundary, str) or not self.claim_boundary.strip():
            raise ValueError("claim_boundary must be a non-empty string")
        object.__setattr__(self, "target_state", target)
        object.__setattr__(self, "leakage_weight", float(self.leakage_weight))
        object.__setattr__(self, "claim_boundary", self.claim_boundary.strip())

    def evaluate(self, state: NDArray[np.complex128]) -> ParityProtectedObjectiveEvaluation:
        """Evaluate the objective and exact Euclidean complex gradient.

        Parameters
        ----------
        state:
            Finite dense complex vector matching the configured projector.

        Returns
        -------
        ParityProtectedObjectiveEvaluation
            Total value, target-distance term, unweighted leakage mass, input
            custody, and analytic gradient.

        Raises
        ------
        ValueError
            If ``state`` has the wrong shape or contains non-finite values.

        """
        value = self.projector.as_state(state)
        difference = value - self.target_state
        target_distance = 0.5 * float(np.vdot(difference, difference).real)
        leakage = self.projector.leakage_value_and_gradient(value)
        gradient = difference + self.leakage_weight * leakage.gradient
        total = target_distance + self.leakage_weight * leakage.value
        return ParityProtectedObjectiveEvaluation(
            value=float(total),
            target_distance=float(target_distance),
            leakage_mass=float(leakage.value),
            state=value,
            gradient=np.asarray(gradient, dtype=np.complex128),
            claim_boundary=self.claim_boundary,
        )

    def __call__(self, state: NDArray[np.complex128]) -> float:
        """Return the scalar objective value."""
        return self.evaluate(state).value


__all__ = [
    "ComplexArray",
    "ParityProtectedObjectiveEvaluation",
    "ParityProtectedQuadraticObjective",
]
