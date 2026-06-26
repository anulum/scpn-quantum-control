# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — WS-3 reference-validation registry
"""WS-3 per-claim reference-validation certifications for the Studio frontier.

The differentiable claim ledger carries research maturity. It is not itself a
validated-grade source. This module is the missing WS-3 bridge: it loads a small,
committed registry of per-claim reference-validation certifications and validates
that every certified claim is known, unique, and already ``promoted`` before its ID
is fed into :func:`scpn_quantum_control.studio.coverage_frontier.measure_coverage_frontier`.

An empty registry is valid and means the honest answer rate remains zero. A bad
registry fails closed rather than allowing a candidate to be laundered into
``reference-validated``.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from ..differentiable_claim_ledger import REPO_ROOT, ClaimLedger, ClaimLedgerRow

REFERENCE_VALIDATION_SCHEMA = "studio.reference-validation-certifications.v1"
"""Wire schema for committed WS-3 per-claim reference-validation certifications."""

DEFAULT_REFERENCE_VALIDATION_PATH = (
    REPO_ROOT / "data" / "differentiable_phase_qnode" / "reference_validation_certifications.json"
)
"""Default committed WS-3 certification registry path."""

ReferenceValidationStatus = Literal["reference-validated"]
"""Allowed status for a per-claim WS-3 certification."""

_SHA256_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class ReferenceValidationCertification:
    """One WS-3 certification that a promoted claim has reference evidence attached."""

    claim_id: str
    certificate_ref: str
    reference_artifact_digest: str
    adjudicated_at: str
    status: ReferenceValidationStatus = "reference-validated"
    grader_report_digest: str | None = None
    validity_domain: str | None = None

    def __post_init__(self) -> None:
        """Validate the per-certificate wire fields."""
        if not self.claim_id:
            raise ValueError("claim_id must be non-empty")
        if self.status != "reference-validated":
            raise ValueError("reference-validation status must be 'reference-validated'")
        if not self.certificate_ref:
            raise ValueError("certificate_ref must be non-empty")
        if not _SHA256_PATTERN.fullmatch(self.reference_artifact_digest):
            raise ValueError("reference_artifact_digest must be sha256:<64 lowercase hex>")
        if self.grader_report_digest is not None and not _SHA256_PATTERN.fullmatch(
            self.grader_report_digest
        ):
            raise ValueError("grader_report_digest must be sha256:<64 lowercase hex>")
        if not self.adjudicated_at:
            raise ValueError("adjudicated_at must be non-empty")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ReferenceValidationCertification:
        """Build a certification from a JSON mapping."""
        return cls(
            claim_id=str(payload["claim_id"]),
            certificate_ref=str(payload["certificate_ref"]),
            reference_artifact_digest=str(payload["reference_artifact_digest"]),
            adjudicated_at=str(payload["adjudicated_at"]),
            status=str(payload.get("status", "reference-validated")),  # type: ignore[arg-type]
            grader_report_digest=_optional_str(payload.get("grader_report_digest")),
            validity_domain=_optional_str(payload.get("validity_domain")),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready certification mapping."""
        payload: dict[str, Any] = {
            "claim_id": self.claim_id,
            "status": self.status,
            "certificate_ref": self.certificate_ref,
            "reference_artifact_digest": self.reference_artifact_digest,
            "adjudicated_at": self.adjudicated_at,
        }
        if self.grader_report_digest is not None:
            payload["grader_report_digest"] = self.grader_report_digest
        if self.validity_domain is not None:
            payload["validity_domain"] = self.validity_domain
        return payload


@dataclass(frozen=True, slots=True)
class ReferenceValidationRegistryValidation:
    """Validation result for a WS-3 certification registry against a claim ledger."""

    passed: bool
    errors: tuple[str, ...]
    reference_validated_claim_ids: tuple[str, ...]
    certificate_count: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready registry validation summary."""
        return {
            "passed": self.passed,
            "errors": list(self.errors),
            "reference_validated_claim_ids": list(self.reference_validated_claim_ids),
            "certificate_count": self.certificate_count,
        }


@dataclass(frozen=True, slots=True)
class ReferenceValidationRegistry:
    """Committed WS-3 registry of per-claim reference-validation certifications."""

    schema: str
    certifications: tuple[ReferenceValidationCertification, ...]
    generated_from: str | None = None

    def __post_init__(self) -> None:
        """Validate the top-level registry schema."""
        if self.schema != REFERENCE_VALIDATION_SCHEMA:
            raise ValueError(
                f"unsupported reference-validation schema: {self.schema!r}; "
                f"expected {REFERENCE_VALIDATION_SCHEMA!r}"
            )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ReferenceValidationRegistry:
        """Build a registry from a JSON mapping."""
        raw_certifications = payload.get("certifications")
        if not isinstance(raw_certifications, list):
            raise ValueError("reference-validation registry must contain a certifications list")
        certifications: list[ReferenceValidationCertification] = []
        for certification in raw_certifications:
            if not isinstance(certification, Mapping):
                raise ValueError("reference-validation certification entries must be objects")
            certifications.append(ReferenceValidationCertification.from_dict(certification))
        return cls(
            schema=str(payload["schema"]),
            generated_from=_optional_str(payload.get("generated_from")),
            certifications=tuple(certifications),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready registry mapping."""
        payload: dict[str, Any] = {
            "schema": self.schema,
            "certifications": [certification.to_dict() for certification in self.certifications],
        }
        if self.generated_from is not None:
            payload["generated_from"] = self.generated_from
        return payload

    def validate_against(
        self, rows_or_ledger: ClaimLedger | Iterable[ClaimLedgerRow]
    ) -> ReferenceValidationRegistryValidation:
        """Validate this registry against a claim ledger without raising.

        Parameters
        ----------
        rows_or_ledger
            The ledger whose claims may be reference-validated.

        Returns
        -------
        ReferenceValidationRegistryValidation
            Result carrying all errors and the certified IDs that are safe to pass
            into ``measure_coverage_frontier`` when validation succeeds.
        """
        rows = _rows_tuple(rows_or_ledger)
        rows_by_id = {row.claim_id: row for row in rows}
        errors: list[str] = []
        seen: set[str] = set()
        certified: list[str] = []
        for certification in self.certifications:
            if certification.claim_id in seen:
                errors.append(f"{certification.claim_id}: duplicate WS-3 certification")
                continue
            seen.add(certification.claim_id)
            row = rows_by_id.get(certification.claim_id)
            if row is None:
                errors.append(f"{certification.claim_id}: certified claim is absent from ledger")
                continue
            if row.promotion_status != "promoted":
                errors.append(
                    f"{certification.claim_id}: WS-3 reference validation requires "
                    f"promotion_status='promoted', got {row.promotion_status!r}"
                )
                continue
            certified.append(certification.claim_id)
        return ReferenceValidationRegistryValidation(
            passed=not errors,
            errors=tuple(errors),
            reference_validated_claim_ids=tuple(certified),
            certificate_count=len(self.certifications),
        )

    def reference_validated_claim_ids(
        self, rows_or_ledger: ClaimLedger | Iterable[ClaimLedgerRow]
    ) -> tuple[str, ...]:
        """Return certified claim IDs, raising if the registry is invalid."""
        validation = self.validate_against(rows_or_ledger)
        if not validation.passed:
            raise ValueError("; ".join(validation.errors))
        return validation.reference_validated_claim_ids


def load_reference_validation_registry(
    path: Path = DEFAULT_REFERENCE_VALIDATION_PATH,
) -> ReferenceValidationRegistry:
    """Load the committed WS-3 reference-validation registry."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return ReferenceValidationRegistry.from_dict(payload)


def _rows_tuple(
    rows_or_ledger: ClaimLedger | Iterable[ClaimLedgerRow],
) -> tuple[ClaimLedgerRow, ...]:
    """Return ledger rows as a tuple without duplicating call-site logic."""
    return (
        rows_or_ledger.rows if isinstance(rows_or_ledger, ClaimLedger) else tuple(rows_or_ledger)
    )


def _optional_str(value: Any) -> str | None:
    """Return a string for non-empty optional JSON values."""
    if value is None:
        return None
    text = str(value)
    return text or None


__all__ = [
    "DEFAULT_REFERENCE_VALIDATION_PATH",
    "REFERENCE_VALIDATION_SCHEMA",
    "ReferenceValidationCertification",
    "ReferenceValidationRegistry",
    "ReferenceValidationRegistryValidation",
    "ReferenceValidationStatus",
    "load_reference_validation_registry",
]
