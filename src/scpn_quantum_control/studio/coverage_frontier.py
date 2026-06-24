# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — WS-6 coverage frontier for the differentiable claim ledger
"""WS-6 coverage-frontier report for QUANTUM's differentiable claim ledger.

The honesty floor without coverage is theatre: a ledger that answers nothing is
perfectly honest and perfectly useless. WS-6 (FLUCTARA's contract, type-A
abstaining gate) requires every gate to publish an *answer rate* beside its
integrity claim and to ship — and honestly flag — its **frontier operating point**:
the most coverage it can give subject to the integrity floor, never the
safest-looking point.

The differentiable claim ledger
(:mod:`scpn_quantum_control.differentiable_claim_ledger`) carries a maturity axis,
``promotion_status``. This module maps that maturity axis onto the federation's
honesty axis — the platform :class:`~scpn_studio_platform.evidence.ClaimStatus` — and
measures the resulting answer rate.

The mapping is **conservative by CEO LOCK-4**: promotion is *maturity*, not
*validation*, and must never launder into ``reference-validated``. A promoted claim
rests at :data:`~scpn_studio_platform.evidence.ClaimStatus.BOUNDED_MODEL` and reaches
``reference-validated`` only when reference-validation evidence is attached to its
bundle (the WS-3 dependency) — never by relabelling. So on a ledger of candidates
the honest answer rate is zero, and the gate reports itself as knowingly
*off-frontier* (over-conservative): the frontier advances by attaching evidence, not
by moving labels.

This module reads the ledger and the federation vocabulary; it lives on the studio
federation surface and so depends on the platform SDK (the ``studio`` extra).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from scpn_studio_platform.evidence import ClaimStatus

from ..differentiable_claim_ledger import (
    ClaimLedger,
    ClaimLedgerRow,
    PromotionStatus,
)

ANSWERED_STATUSES: frozenset[ClaimStatus] = frozenset(
    {ClaimStatus.REFERENCE_VALIDATED, ClaimStatus.REFUTED}
)
"""The claim statuses that count as a *confident answer* (WS-6 type-A).

A claim is answered only when it is established against a reference
(``reference-validated``) or promoted as a negative finding (``refuted``). Every
other status — a bounded model, a validation gap, an external-dependency block — is
an honest *abstention*: the ledger declines to answer rather than over-claim.
"""

#: The maturity status a claim must hold before reference-validation evidence can
#: elevate it to ``reference-validated`` (LOCK-4: promotion precedes validation).
_REFERENCE_VALIDATION_ELIGIBLE: frozenset[PromotionStatus] = frozenset({"promoted"})


def map_claim_status(
    promotion_status: PromotionStatus, *, reference_validated: bool = False
) -> ClaimStatus:
    """Map a ledger ``promotion_status`` onto a federation :class:`ClaimStatus`.

    The mapping is conservative by LOCK-4 — a maturity label never launders into a
    validation grade:

    - ``promoted`` → ``bounded-model``, or ``reference-validated`` *only* when
      ``reference_validated`` certifies that reference-validation evidence is
      attached to the claim's bundle (the WS-3 dependency).
    - ``SOTA-candidate`` → ``bounded-model`` (a credible, within-bounds candidate;
      ``bounded-support`` is reserved for domain fail-closed states, which a
      candidacy is not).
    - ``hard_gap`` → ``validation-gap``.
    - ``blocked`` → ``external-dependency-blocked``.

    Parameters
    ----------
    promotion_status
        The ledger maturity status.
    reference_validated
        Whether reference-validation evidence is attached to this claim's bundle.
        Honoured only for ``promoted`` claims; supplying it for any other status is a
        contradiction (an unpromoted claim cannot be reference-validated) and fails
        closed.

    Returns
    -------
    ClaimStatus
        The federation honesty status.

    Raises
    ------
    ValueError
        If ``promotion_status`` is unknown, or ``reference_validated`` is set for a
        status that is not promotion-eligible.
    """
    if reference_validated and promotion_status not in _REFERENCE_VALIDATION_ELIGIBLE:
        raise ValueError(
            f"reference_validated is only admissible for a promoted claim, "
            f"not {promotion_status!r} (LOCK-4: promotion precedes validation)"
        )
    if promotion_status == "promoted":
        return (
            ClaimStatus.REFERENCE_VALIDATED if reference_validated else ClaimStatus.BOUNDED_MODEL
        )
    if promotion_status == "SOTA-candidate":
        return ClaimStatus.BOUNDED_MODEL
    if promotion_status == "hard_gap":
        return ClaimStatus.VALIDATION_GAP
    if promotion_status == "blocked":
        return ClaimStatus.EXTERNAL_DEPENDENCY_BLOCKED
    raise ValueError(f"unknown promotion_status: {promotion_status!r}")


@dataclass(frozen=True)
class CoverageFrontierReport:
    """A WS-6 type-A coverage report for the differentiable claim ledger.

    Parameters
    ----------
    total
        Number of claims measured.
    answered_confident
        Claims resting at an answered status (``reference-validated`` ∪ ``refuted``).
    answer_rate
        ``answered_confident / total`` (``0.0`` for an empty ledger).
    grade_distribution
        Count of claims per resulting :class:`ClaimStatus` (string-keyed for JSON).
    claim_status_by_id
        Per-claim resulting status, for transparency and audit.
    off_frontier
        Whether the gate is knowingly off the frontier — here, over-conservative:
        candidate claims rest unanswered that could be answered by attaching
        reference-validation evidence (WS-3), not by relabelling.
    off_frontier_reason
        A one-line justification when ``off_frontier`` is set, else ``None``.
    calibration_status
        The state of the answered verdicts' false-positive rate. This number is
        owned by WS-3 (validate-the-grader); until that lands it is reported as
        ``"pending-ws3"`` rather than fabricated.
    """

    total: int
    answered_confident: int
    answer_rate: float
    grade_distribution: dict[str, int]
    claim_status_by_id: dict[str, str]
    off_frontier: bool
    off_frontier_reason: str | None
    calibration_status: str

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-serialisable report."""
        return {
            "total": self.total,
            "answered_confident": self.answered_confident,
            "answer_rate": self.answer_rate,
            "grade_distribution": dict(self.grade_distribution),
            "claim_status_by_id": dict(self.claim_status_by_id),
            "off_frontier": self.off_frontier,
            "off_frontier_reason": self.off_frontier_reason,
            "calibration_status": self.calibration_status,
        }


def measure_coverage_frontier(
    rows_or_ledger: ClaimLedger | Iterable[ClaimLedgerRow],
    *,
    reference_validated_claim_ids: Iterable[str] = (),
) -> CoverageFrontierReport:
    """Measure the WS-6 coverage frontier of the differentiable claim ledger.

    Parameters
    ----------
    rows_or_ledger
        The loaded ledger or its rows.
    reference_validated_claim_ids
        Claim IDs an external reference-validation process (WS-3) has certified as
        carrying reference-validation evidence. Empty by default — so a bare ledger
        of candidates reports an honest ``0.0`` answer rate, and nothing is laundered
        upward. A non-eligible claim listed here fails closed (see
        :func:`map_claim_status`).

    Returns
    -------
    CoverageFrontierReport
        The type-A report with the chosen operating point and the off-frontier flag.
    """
    rows = tuple(
        rows_or_ledger.rows if isinstance(rows_or_ledger, ClaimLedger) else rows_or_ledger
    )
    certified = frozenset(reference_validated_claim_ids)

    status_by_id: dict[str, str] = {}
    distribution: dict[str, int] = {}
    answered = 0
    improvable = 0
    for row in rows:
        status = map_claim_status(
            row.promotion_status, reference_validated=row.claim_id in certified
        )
        status_by_id[row.claim_id] = status.value
        distribution[status.value] = distribution.get(status.value, 0) + 1
        if status in ANSWERED_STATUSES:
            answered += 1
        elif status is ClaimStatus.BOUNDED_MODEL:
            # A within-bounds candidate: unanswered, but answerable by attaching
            # reference-validation evidence (WS-3). Validation gaps and external
            # blocks are honestly unanswerable now and are NOT counted as improvable.
            improvable += 1

    total = len(rows)
    answer_rate = answered / total if total else 0.0
    off_frontier = improvable > 0
    reason: str | None = None
    if off_frontier:
        reason = (
            f"{improvable} candidate claim(s) rest at bounded-model awaiting attached "
            "reference-validation evidence (WS-3); the frontier advances by attaching "
            "that evidence, never by relabelling (LOCK-4)."
        )

    return CoverageFrontierReport(
        total=total,
        answered_confident=answered,
        answer_rate=answer_rate,
        grade_distribution=distribution,
        claim_status_by_id=status_by_id,
        off_frontier=off_frontier,
        off_frontier_reason=reason,
        calibration_status="pending-ws3",
    )


def render_coverage_frontier_markdown(report: CoverageFrontierReport) -> str:
    """Render the coverage-frontier report as the WS-6 release artefact.

    Parameters
    ----------
    report
        The measured report.

    Returns
    -------
    str
        A reviewer-facing Markdown summary carrying the real numbers, the chosen
        operating point, and the off-frontier flag.
    """
    lines = [
        "<!--",
        "SPDX-License-Identifier: AGPL-3.0-or-later",
        "Commercial license available",
        "© Concepts 1996–2026 Miroslav Šotek. All rights reserved.",
        "© Code 2020–2026 Miroslav Šotek. All rights reserved.",
        "ORCID: 0009-0009-3560-0851",
        "Contact: www.anulum.li | protoscience@anulum.li",
        "SCPN Quantum Control — WS-6 Differentiable Coverage Frontier",
        "-->",
        "",
        "# WS-6 Coverage Frontier — Differentiable Claim Ledger",
        "",
        f"- **Answer rate:** {report.answer_rate:.3f} "
        f"({report.answered_confident}/{report.total} confident)",
        f"- **Calibration (answered FPR):** {report.calibration_status}",
        f"- **Operating point:** {'OFF-frontier (over-conservative)' if report.off_frontier else 'on-frontier'}",
    ]
    if report.off_frontier_reason is not None:
        lines.append(f"- **Off-frontier reason:** {report.off_frontier_reason}")
    lines.extend(
        [
            "",
            "## Grade distribution",
            "",
            "| Claim status | Count |",
            "|---|---|",
        ]
    )
    for status_value in sorted(report.grade_distribution):
        lines.append(f"| {status_value} | {report.grade_distribution[status_value]} |")
    lines.extend(
        [
            "",
            "Answered = reference-validated ∪ refuted. Promotion is maturity, not "
            "validation (LOCK-4): a candidate is answered only when reference-validation "
            "evidence is attached, never by relabelling.",
            "",
        ]
    )
    return "\n".join(lines)


__all__ = [
    "ANSWERED_STATUSES",
    "CoverageFrontierReport",
    "map_claim_status",
    "measure_coverage_frontier",
    "render_coverage_frontier_markdown",
]
