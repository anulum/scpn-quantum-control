# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Exact-validated surrogate proposal port
"""co-design proposal composition with mandatory exact local validation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray

from ..codesign.contracts import ControllerProposal
from .fidelity import SurrogateFidelityCertificate
from .models import CLASSICAL_SURROGATE_CLAIM_BOUNDARY, GaussianRBFSurrogate

FloatArray = NDArray[np.float64]

HYBRID_SURROGATE_CLAIM_BOUNDARY = (
    CLASSICAL_SURROGATE_CLAIM_BOUNDARY
    + " The co-design ControllerProposal remains unapplied; exact local validation "
    "is an acceptance observation, not a safety decision or actuator command."
)


@dataclass(frozen=True, slots=True)
class ExactValidatedSurrogateProposal:
    """Unapplied surrogate proposal with exact local objective observations."""

    proposal: ControllerProposal
    surrogate_current: float
    surrogate_candidate: float
    exact_current: float
    exact_candidate: float
    surrogate_predicted_improvement: float
    exact_observed_improvement: float
    accepted_by_exact_objective: bool
    reason: str
    hardware_execution: bool = False
    applied: bool = False
    claim_boundary: str = HYBRID_SURROGATE_CLAIM_BOUNDARY

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready validation mapping."""
        payload = asdict(self)
        payload["proposal"] = self.proposal.to_dict()
        return payload


def propose_and_validate_surrogate_step(
    model: GaussianRBFSurrogate,
    current_parameters: FloatArray,
    exact_objective: Callable[[FloatArray], float],
    fidelity: SurrogateFidelityCertificate,
    *,
    learning_rate: float,
    max_step_norm: float,
) -> ExactValidatedSurrogateProposal:
    """Create a bounded co-design proposal and query the exact local objective.

    The surrogate may suggest the candidate, but it cannot accept itself. A
    passing disjoint value-fidelity certificate is required first, and the
    returned acceptance flag is based only on the exact local objective. The
    proposal is never applied by this function.
    """
    if not fidelity.passed or fidelity.training_overlap_count != 0:
        raise ValueError("a passing disjoint fidelity certificate is required.")
    current = np.asarray(current_parameters, dtype=np.float64)
    if current.ndim != 1 or current.shape != (model.n_parameters,):
        raise ValueError("current_parameters must match the surrogate parameter dimension.")
    if not np.all(np.isfinite(current)):
        raise ValueError("current_parameters must contain only finite values.")
    rate = float(learning_rate)
    limit = float(max_step_norm)
    if not np.isfinite(rate) or rate <= 0.0:
        raise ValueError("learning_rate must be finite and positive.")
    if not np.isfinite(limit) or limit <= 0.0:
        raise ValueError("max_step_norm must be finite and positive.")

    raw_update = -rate * model.gradient(current)
    raw_norm = float(np.linalg.norm(raw_update))
    update = raw_update if raw_norm <= limit else raw_update * (limit / raw_norm)
    candidate = current + update
    proposal = ControllerProposal(
        parameters=tuple(float(value) for value in candidate),
        update=tuple(float(value) for value in update),
        gain_scale=1.0,
    )

    surrogate_current = model.value(current)
    surrogate_candidate = model.value(candidate)
    exact_current = float(exact_objective(current))
    exact_candidate = float(exact_objective(candidate))
    if not np.isfinite(exact_current) or not np.isfinite(exact_candidate):
        raise ValueError("exact_objective must return finite values.")
    exact_improvement = exact_current - exact_candidate
    accepted = exact_improvement > 0.0
    reason = "exact_local_improvement" if accepted else "exact_local_non_improvement"
    return ExactValidatedSurrogateProposal(
        proposal=proposal,
        surrogate_current=surrogate_current,
        surrogate_candidate=surrogate_candidate,
        exact_current=exact_current,
        exact_candidate=exact_candidate,
        surrogate_predicted_improvement=surrogate_current - surrogate_candidate,
        exact_observed_improvement=exact_improvement,
        accepted_by_exact_objective=accepted,
        reason=reason,
    )


__all__ = [
    "ExactValidatedSurrogateProposal",
    "HYBRID_SURROGATE_CLAIM_BOUNDARY",
    "propose_and_validate_surrogate_step",
]
