# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Tests for the WS-6 differentiable coverage frontier
"""Tests for studio/coverage_frontier.py — the WS-6 type-A coverage report.

Covers the LOCK-4 conservative mapping, the answer-rate measurement, the
off-frontier flag (over-conservative when candidates await reference-validation
evidence), the markdown artefact, and the real committed ledger.
"""

from __future__ import annotations

import pytest

pytest.importorskip("scpn_studio_platform.evidence", reason="studio extra not installed")

from scpn_studio_platform.evidence import ClaimStatus  # noqa: E402

from scpn_quantum_control.differentiable_claim_ledger import (  # noqa: E402
    ClaimLedgerRow,
    PromotionStatus,
    load_differentiable_claim_ledger,
)
from scpn_quantum_control.studio.coverage_frontier import (  # noqa: E402
    ANSWERED_STATUSES,
    CoverageFrontierReport,
    map_claim_status,
    measure_coverage_frontier,
    measure_coverage_frontier_from_certifications,
    render_coverage_frontier_markdown,
)
from scpn_quantum_control.studio.reference_validation import (  # noqa: E402
    ReferenceValidationCertification,
    ReferenceValidationRegistry,
)


def _row(claim_id: str, promotion_status: PromotionStatus) -> ClaimLedgerRow:
    """Build a minimally-valid ledger row with a chosen maturity status."""
    return ClaimLedgerRow(
        claim_id=claim_id,
        claim_text=f"bounded claim {claim_id}",
        implementation_surface=("src/scpn_quantum_control/differentiable.py",),
        test_surface=("tests/test_coverage_frontier.py",),
        docs_surface=("docs/internal/TODO.md",),
        evidence_artifact_ids=("artefact-1",),
        benchmark_artifact_ids=("benchmark-1",),
        known_gaps=("isolated benchmark evidence pending",),
        promotion_status=promotion_status,
        claim_boundary="bounded differentiable claim",
    )


# ── the LOCK-4 conservative mapping ────────────────────────────────────


def test_promoted_maps_to_bounded_model_without_evidence() -> None:
    """A promoted claim is maturity, not validation — it rests at bounded-model."""
    assert map_claim_status("promoted") is ClaimStatus.BOUNDED_MODEL


def test_promoted_with_evidence_maps_to_reference_validated() -> None:
    """Reference-validation evidence elevates a promoted claim, never a label change."""
    assert (
        map_claim_status("promoted", reference_validated=True) is ClaimStatus.REFERENCE_VALIDATED
    )


def test_sota_candidate_and_gaps_map_conservatively() -> None:
    """Candidate, gap, and block statuses each map to their honest non-answered status."""
    assert map_claim_status("SOTA-candidate") is ClaimStatus.BOUNDED_MODEL
    assert map_claim_status("hard_gap") is ClaimStatus.VALIDATION_GAP
    assert map_claim_status("blocked") is ClaimStatus.EXTERNAL_DEPENDENCY_BLOCKED


def test_reference_validated_on_non_promoted_fails_closed() -> None:
    """An unpromoted claim cannot be reference-validated (LOCK-4 ordering)."""
    with pytest.raises(ValueError, match="only admissible for a promoted claim"):
        map_claim_status("SOTA-candidate", reference_validated=True)


def test_unknown_promotion_status_is_rejected() -> None:
    """An out-of-vocabulary status is rejected, not silently graded."""
    with pytest.raises(ValueError, match="unknown promotion_status"):
        map_claim_status("invented-status")  # type: ignore[arg-type]


def test_answered_statuses_are_exactly_validated_and_refuted() -> None:
    """A confident answer is reference-validated or refuted; nothing else."""
    assert frozenset({ClaimStatus.REFERENCE_VALIDATED, ClaimStatus.REFUTED}) == ANSWERED_STATUSES


# ── measurement ────────────────────────────────────────────────────────


def test_candidate_ledger_answers_nothing_and_is_off_frontier() -> None:
    """A ledger of candidates honestly answers 0% and flags itself over-conservative."""
    rows = [_row(f"c{i}", "SOTA-candidate") for i in range(4)]
    report = measure_coverage_frontier(rows)
    assert report.total == 4
    assert report.answered_confident == 0
    assert report.answer_rate == 0.0
    assert report.grade_distribution == {"bounded-model": 4}
    assert report.off_frontier is True
    assert report.off_frontier_reason is not None
    assert "reference-validation evidence" in report.off_frontier_reason
    assert report.calibration_status == "pending-ws3"


def test_attaching_evidence_advances_the_frontier() -> None:
    """Certifying reference-validation evidence on a promoted claim raises the rate."""
    rows = [_row("p1", "promoted"), _row("c1", "SOTA-candidate")]
    report = measure_coverage_frontier(rows, reference_validated_claim_ids=["p1"])
    assert report.answered_confident == 1
    assert report.answer_rate == 0.5
    assert report.claim_status_by_id == {
        "p1": "reference-validated",
        "c1": "bounded-model",
    }
    # One candidate still rests at bounded-model, so the gate remains off-frontier.
    assert report.off_frontier is True


def test_certification_registry_feeds_reference_validated_claim_ids() -> None:
    """WS-3 per-claim certifications are the approved feed into the WS-6 frontier."""
    registry = ReferenceValidationRegistry(
        schema="studio.reference-validation-certifications.v1",
        certifications=(
            ReferenceValidationCertification(
                claim_id="p1",
                certificate_ref="studio.ws3/p1",
                reference_artifact_digest="sha256:" + "a" * 64,
                adjudicated_at="2026-06-26T00:00:00Z",
            ),
        ),
    )
    report = measure_coverage_frontier_from_certifications(
        [_row("p1", "promoted"), _row("c1", "SOTA-candidate")],
        registry=registry,
    )

    assert report.claim_status_by_id["p1"] == "reference-validated"
    assert report.answer_rate == 0.5


def test_fully_validated_ledger_is_on_frontier() -> None:
    """When every claim is answered the gate is on the frontier (answer_rate 1.0)."""
    rows = [_row("p1", "promoted"), _row("p2", "promoted")]
    report = measure_coverage_frontier(rows, reference_validated_claim_ids=["p1", "p2"])
    assert report.answer_rate == 1.0
    assert report.off_frontier is False
    assert report.off_frontier_reason is None


def test_gaps_and_blocks_are_not_improvable() -> None:
    """Validation gaps and external blocks are honestly unanswerable, not off-frontier."""
    rows = [_row("g1", "hard_gap"), _row("b1", "blocked")]
    report = measure_coverage_frontier(rows)
    assert report.answer_rate == 0.0
    assert report.grade_distribution == {
        "validation-gap": 1,
        "external-dependency-blocked": 1,
    }
    # Nothing rests at bounded-model, so nothing is improvable by attaching evidence.
    assert report.off_frontier is False
    assert report.off_frontier_reason is None


def test_empty_ledger_is_zero_and_on_frontier() -> None:
    """An empty ledger answers 0.0 without dividing by zero and is not off-frontier."""
    report = measure_coverage_frontier([])
    assert report.total == 0
    assert report.answer_rate == 0.0
    assert report.off_frontier is False


def test_non_eligible_certified_claim_fails_closed() -> None:
    """Listing a non-promoted claim as reference-validated fails closed in measurement."""
    rows = [_row("c1", "SOTA-candidate")]
    with pytest.raises(ValueError, match="only admissible for a promoted claim"):
        measure_coverage_frontier(rows, reference_validated_claim_ids=["c1"])


def test_to_dict_round_trips_the_report() -> None:
    """The report serialises every field for the release artefact."""
    report = measure_coverage_frontier([_row("c1", "SOTA-candidate")])
    payload = report.to_dict()
    assert payload["total"] == 1
    assert payload["answer_rate"] == 0.0
    assert payload["grade_distribution"] == {"bounded-model": 1}
    assert payload["claim_status_by_id"] == {"c1": "bounded-model"}
    assert payload["off_frontier"] is True
    assert payload["calibration_status"] == "pending-ws3"


# ── the markdown artefact ──────────────────────────────────────────────


def test_markdown_reports_off_frontier_with_reason() -> None:
    """The off-frontier artefact carries the answer rate, the flag, and the reason."""
    report = measure_coverage_frontier([_row("c1", "SOTA-candidate")])
    markdown = render_coverage_frontier_markdown(report)
    assert "WS-6 Coverage Frontier" in markdown
    assert "Answer rate:** 0.000 (0/1 confident)" in markdown
    assert "OFF-frontier (over-conservative)" in markdown
    assert "Off-frontier reason:" in markdown
    assert "| bounded-model | 1 |" in markdown
    assert "pending-ws3" in markdown


def test_markdown_on_frontier_omits_the_reason_line() -> None:
    """An on-frontier report renders no off-frontier reason line."""
    report = measure_coverage_frontier(
        [_row("p1", "promoted")], reference_validated_claim_ids=["p1"]
    )
    markdown = render_coverage_frontier_markdown(report)
    assert "on-frontier" in markdown
    assert "Off-frontier reason:" not in markdown
    assert "| reference-validated | 1 |" in markdown


# ── the real committed ledger (real numbers, no fabrication) ────────────


def test_real_committed_ledger_is_thirteen_candidates_at_zero() -> None:
    """The shipped ledger is 13 candidates: honest answer rate 0.0, off-frontier."""
    ledger = load_differentiable_claim_ledger()
    report = measure_coverage_frontier(ledger)
    assert report.total == 13
    assert report.answer_rate == 0.0
    assert report.grade_distribution == {"bounded-model": 13}
    assert report.off_frontier is True
    assert isinstance(report, CoverageFrontierReport)
