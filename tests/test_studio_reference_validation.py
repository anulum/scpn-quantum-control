# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — reference-validation registry tests
"""Tests for Studio reference-validation certification wiring."""

from __future__ import annotations

from dataclasses import replace
from typing import cast

import pytest

pytest.importorskip("scpn_studio_platform.evidence", reason="studio extra not installed")

from scpn_quantum_control.differentiable_claim_ledger import (  # noqa: E402
    ClaimLedgerRow,
    load_differentiable_claim_ledger,
)
from scpn_quantum_control.studio.coverage_frontier import (  # noqa: E402
    measure_coverage_frontier_from_certifications,
)
from scpn_quantum_control.studio.reference_validation import (  # noqa: E402
    DEFAULT_REFERENCE_VALIDATION_PATH,
    REFERENCE_VALIDATION_SCHEMA,
    ReferenceValidationCertification,
    ReferenceValidationRegistry,
    ReferenceValidationStatus,
    load_reference_validation_registry,
)

SHA_A = "sha256:" + "a" * 64
SHA_B = "sha256:" + "b" * 64


def _row(claim_id: str, status: str) -> ClaimLedgerRow:
    """Build a minimally valid differentiable claim row for registry tests."""
    return ClaimLedgerRow(
        claim_id=claim_id,
        claim_text=f"{claim_id} claim",
        implementation_surface=("src/scpn_quantum_control/studio/reference_validation.py",),
        test_surface=("tests/test_studio_reference_validation.py",),
        docs_surface=("docs/studio_federation.md",),
        evidence_artifact_ids=("artifact",),
        benchmark_artifact_ids=("artifact",),
        known_gaps=("none",),
        promotion_status=status,  # type: ignore[arg-type]
        claim_boundary="bounded test claim",
    )


def _cert(claim_id: str, *, digest: str = SHA_A) -> ReferenceValidationCertification:
    """Build a valid reference-validation certification for a claim ID."""
    return ReferenceValidationCertification(
        claim_id=claim_id,
        certificate_ref=f"studio.reference-validation/{claim_id}",
        reference_artifact_digest=digest,
        grader_report_digest=SHA_B,
        adjudicated_at="2026-06-26T00:00:00Z",
        validity_domain="synthetic reference-validation test domain",
    )


def _registry(*certifications: ReferenceValidationCertification) -> ReferenceValidationRegistry:
    """Return a registry with the supplied certifications."""
    return ReferenceValidationRegistry(
        schema=REFERENCE_VALIDATION_SCHEMA,
        generated_from="unit test",
        certifications=tuple(certifications),
    )


def test_committed_empty_registry_keeps_real_ledger_unanswered() -> None:
    """No committed certifications means the real ledger stays unanswered."""
    registry = load_reference_validation_registry()
    ledger = load_differentiable_claim_ledger()

    validation = registry.validate_against(ledger)
    report = measure_coverage_frontier_from_certifications(ledger, registry=registry)

    assert DEFAULT_REFERENCE_VALIDATION_PATH.exists()
    assert validation.passed
    assert validation.certificate_count == 0
    assert validation.reference_validated_claim_ids == ()
    assert report.total == 16
    assert report.answer_rate == 0.0
    assert report.grade_distribution == {"bounded-model": 16}


def test_promoted_claim_certificate_advances_frontier() -> None:
    """A promoted certified claim becomes reference-validated."""
    rows = [_row("p1", "promoted"), _row("c1", "bounded_candidate")]
    registry = _registry(_cert("p1"))

    validation = registry.validate_against(rows)
    report = measure_coverage_frontier_from_certifications(rows, registry=registry)

    assert validation.passed
    assert validation.reference_validated_claim_ids == ("p1",)
    assert report.answer_rate == 0.5
    assert report.claim_status_by_id == {
        "p1": "reference-validated",
        "c1": "bounded-model",
    }


def test_candidate_certificate_fails_closed_before_measurement() -> None:
    """A candidate cannot be reference-validated through the registry."""
    rows = [_row("c1", "bounded_candidate")]
    registry = _registry(_cert("c1"))

    validation = registry.validate_against(rows)

    assert not validation.passed
    assert "requires promotion_status='promoted'" in validation.errors[0]
    with pytest.raises(ValueError, match="requires promotion_status='promoted'"):
        measure_coverage_frontier_from_certifications(rows, registry=registry)


def test_unknown_claim_certificate_fails_closed() -> None:
    """A registry cannot certify claim IDs absent from the ledger."""
    registry = _registry(_cert("missing"))
    validation = registry.validate_against([_row("p1", "promoted")])

    assert not validation.passed
    assert validation.errors == ("missing: certified claim is absent from ledger",)


def test_duplicate_claim_certificate_fails_closed() -> None:
    """A claim may have at most one active certification row."""
    registry = _registry(_cert("p1", digest=SHA_A), _cert("p1", digest=SHA_B))
    validation = registry.validate_against([_row("p1", "promoted")])

    assert not validation.passed
    assert validation.errors == ("p1: duplicate reference-validation certification",)


def test_certification_rejects_bad_digest() -> None:
    """Digest fields must be strict sha256-prefixed lowercase hex values."""
    with pytest.raises(ValueError, match="reference_artifact_digest"):
        _cert("p1", digest="sha256:not-a-digest")


def test_certification_rejects_invalid_required_metadata() -> None:
    """Every required certification field fails closed at construction."""
    certification = _cert("p1")
    with pytest.raises(ValueError, match="claim_id"):
        replace(certification, claim_id="")
    with pytest.raises(ValueError, match="status"):
        replace(certification, status=cast(ReferenceValidationStatus, "candidate"))
    with pytest.raises(ValueError, match="certificate_ref"):
        replace(certification, certificate_ref="")
    with pytest.raises(ValueError, match="grader_report_digest"):
        replace(certification, grader_report_digest="sha256:not-a-digest")
    with pytest.raises(ValueError, match="adjudicated_at"):
        replace(certification, adjudicated_at="")


def test_certification_optional_fields_round_trip_without_aliases() -> None:
    """Omitted and blank optional JSON values normalise to absent fields."""
    payload = {
        "claim_id": "p1",
        "certificate_ref": "studio.reference-validation/p1",
        "reference_artifact_digest": SHA_A,
        "adjudicated_at": "2026-06-26T00:00:00Z",
        "validity_domain": "",
    }
    certification = ReferenceValidationCertification.from_dict(payload)

    assert certification.grader_report_digest is None
    assert certification.validity_domain is None
    assert certification.to_dict() == {
        "claim_id": "p1",
        "status": "reference-validated",
        "certificate_ref": "studio.reference-validation/p1",
        "reference_artifact_digest": SHA_A,
        "adjudicated_at": "2026-06-26T00:00:00Z",
    }


def test_registry_round_trips_to_json_ready_dict() -> None:
    """The registry exposes a deterministic JSON-ready representation."""
    registry = _registry(_cert("p1"))
    payload = registry.to_dict()
    rebuilt = ReferenceValidationRegistry.from_dict(payload)

    assert rebuilt == registry
    assert payload["schema"] == REFERENCE_VALIDATION_SCHEMA
    assert payload["certifications"][0]["claim_id"] == "p1"
    assert rebuilt.validate_against([_row("p1", "promoted")]).to_dict() == {
        "passed": True,
        "errors": [],
        "reference_validated_claim_ids": ["p1"],
        "certificate_count": 1,
    }


def test_registry_rejects_unknown_schema() -> None:
    """Only the locked reference-validation schema is accepted."""
    with pytest.raises(ValueError, match="unsupported reference-validation schema"):
        ReferenceValidationRegistry(schema="bad.v1", certifications=())


def test_registry_rejects_malformed_certification_collections() -> None:
    """The public JSON loader refuses missing lists and non-object rows."""
    with pytest.raises(ValueError, match="certifications list"):
        ReferenceValidationRegistry.from_dict(
            {"schema": REFERENCE_VALIDATION_SCHEMA, "certifications": "invalid"}
        )
    with pytest.raises(ValueError, match="entries must be objects"):
        ReferenceValidationRegistry.from_dict(
            {"schema": REFERENCE_VALIDATION_SCHEMA, "certifications": [3]}
        )


def test_registry_without_generation_note_serialises_minimally() -> None:
    """An absent generation note stays absent from the public JSON mapping."""
    registry = ReferenceValidationRegistry(
        schema=REFERENCE_VALIDATION_SCHEMA,
        certifications=(),
    )

    assert registry.to_dict() == {
        "schema": REFERENCE_VALIDATION_SCHEMA,
        "certifications": [],
    }
