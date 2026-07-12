# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — symmetry sector replay module
# scpn-quantum-control -- symmetry-sector raw-count replay adapter
"""Offline raw-count replay for symmetry-sector mitigation plans.

The adapter consumes a validated :class:`SymmetrySectorProblem`, raw
computational-basis counts, and the planner output. It applies only primitives
that are already backed by raw-count operations: parity postselection and
symmetry expansion. GUESS symmetry-decay correction is reported as deferred
until a calibrated noise-scaled observable table is wired into the execution
contract.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

from .symmetry_sector_compiler import (
    MitigationPrimitive,
    SymmetrySectorPlan,
    SymmetrySectorProblem,
    plan_symmetry_sector_mitigation,
)
from .symmetry_verification import parity_postselect, symmetry_expand

ReplayStatus = Literal["applied", "blocked"]


@dataclass(frozen=True, slots=True)
class SymmetrySectorReplayResult:
    """Result of applying an eligible symmetry-sector plan to raw counts."""

    status: ReplayStatus
    plan: SymmetrySectorPlan
    raw_counts: dict[str, int]
    postselected_counts: dict[str, int]
    expanded_counts: dict[str, int]
    rejected_counts: dict[str, int]
    raw_shots: int
    postselected_shots: int
    expanded_shots: int
    rejection_rate: float
    applied_primitives: tuple[MitigationPrimitive, ...]
    deferred_primitives: tuple[MitigationPrimitive, ...]
    blockers: tuple[str, ...]
    claim_boundary: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable replay result."""
        return asdict(self)


def _normalise_counts(counts: dict[str, int], *, n_qubits: int) -> dict[str, int]:
    """Validate and normalise computational-basis counts."""
    if not counts:
        raise ValueError("raw counts must not be empty")
    normalised: dict[str, int] = {}
    for bitstring, count in counts.items():
        clean = bitstring.replace(" ", "")
        if len(clean) != n_qubits or any(bit not in {"0", "1"} for bit in clean):
            raise ValueError(
                f"raw count key must be a {n_qubits}-bit computational-basis string: {bitstring!r}"
            )
        if not isinstance(count, int) or count < 0:
            raise ValueError(f"raw count value must be a non-negative integer: {bitstring!r}")
        normalised[clean] = normalised.get(clean, 0) + count
    if sum(normalised.values()) <= 0:
        raise ValueError("raw counts must contain at least one shot")
    return normalised


def replay_symmetry_sector_counts(
    problem: SymmetrySectorProblem,
    counts: dict[str, int],
) -> SymmetrySectorReplayResult:
    """Apply eligible raw-count symmetry-sector primitives.

    The function is fail-closed. Blocked planner outputs raise ``ValueError``
    and invalid counts never reach mitigation primitives. The returned result
    reports GUESS as deferred rather than silently approximating it from counts.
    """
    plan = plan_symmetry_sector_mitigation(problem)
    if plan.status != "eligible" or plan.expected_parity is None:
        raise ValueError(f"symmetry-sector plan is blocked: {plan.blockers}")

    raw_counts = _normalise_counts(counts, n_qubits=problem.n_qubits)
    postselection = parity_postselect(raw_counts, plan.expected_parity)
    expanded_counts = symmetry_expand(raw_counts, plan.expected_parity)

    applied: list[MitigationPrimitive] = []
    deferred: list[MitigationPrimitive] = []
    for primitive in plan.primitives:
        if primitive == "parity_postselection" or primitive == "symmetry_expansion":
            applied.append(primitive)
        elif primitive == "guess_symmetry_decay":
            deferred.append(primitive)

    blockers: list[str] = []
    if deferred:
        blockers.append("GUESS replay requires calibrated noise-scaled symmetry observable rows")

    return SymmetrySectorReplayResult(
        status="applied",
        plan=plan,
        raw_counts=raw_counts,
        postselected_counts=postselection.verified_counts,
        expanded_counts=expanded_counts,
        rejected_counts=postselection.rejected_counts,
        raw_shots=postselection.raw_shots,
        postselected_shots=postselection.verified_shots,
        expanded_shots=sum(expanded_counts.values()),
        rejection_rate=postselection.rejection_rate,
        applied_primitives=tuple(applied),
        deferred_primitives=tuple(deferred),
        blockers=tuple(blockers),
        claim_boundary=(
            "Raw-count replay applies parity postselection and symmetry expansion only. "
            "It does not mutate circuits, submit hardware jobs, or estimate GUESS "
            "without calibrated noise-scaled symmetry observables."
        ),
    )
