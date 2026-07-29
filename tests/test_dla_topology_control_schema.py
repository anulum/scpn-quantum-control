# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — BL-54 support-schema tests
"""Tests for immutable derivative-support and parity-sector contracts."""

from __future__ import annotations

from typing import cast

import pytest

from scpn_quantum_control.dla_topology_control.schema import (
    DLA_TOPOLOGY_CLAIM_BOUNDARY,
    ConstraintSupportRow,
    DifferentiabilityKind,
    DifferentiabilityReport,
    ParitySector,
    SupportStatus,
    UnsupportedDifferentiableConstraintError,
)


def _row(capability: str = "linear_projection") -> ConstraintSupportRow:
    return ConstraintSupportRow(
        capability,
        "supported",
        DifferentiabilityKind.LINEAR,
        "exact self-adjoint projector",
        "local derivative only",
    )


def test_support_row_normalises_text_and_serialises_enum_values() -> None:
    """Normalise row text and expose deterministic primitive fields."""
    row = ConstraintSupportRow(
        "  fixed mask  ",
        "supported",
        DifferentiabilityKind.AFFINE,
        "  zero tangent on masked entries  ",
        "  edge identities are fixed  ",
    )
    assert row.capability == "fixed mask"
    assert row.to_dict() == {
        "capability": "fixed mask",
        "status": "supported",
        "differentiability": "affine",
        "evidence": "zero tangent on masked entries",
        "boundary": "edge identities are fixed",
    }


@pytest.mark.parametrize("field", ["capability", "evidence", "boundary"])
def test_support_row_rejects_blank_or_non_string_text(field: str) -> None:
    """Reject blank and non-string support-row descriptions."""
    values: dict[str, object] = {
        "capability": "cap",
        "status": "supported",
        "differentiability": DifferentiabilityKind.LINEAR,
        "evidence": "proof",
        "boundary": "scope",
    }
    for invalid in (" ", 4):
        values[field] = invalid
        with pytest.raises(ValueError, match=field):
            ConstraintSupportRow(**values)


def test_support_row_rejects_invalid_status_and_derivative_kind() -> None:
    """Reject status and derivative labels outside the closed enums."""
    with pytest.raises(ValueError, match="status"):
        ConstraintSupportRow(
            "cap",
            cast(SupportStatus, "maybe"),
            DifferentiabilityKind.LINEAR,
            "proof",
            "scope",
        )
    with pytest.raises(ValueError, match="DifferentiabilityKind"):
        ConstraintSupportRow(
            "cap",
            "supported",
            cast(DifferentiabilityKind, "linear"),
            "proof",
            "scope",
        )


def test_report_digest_and_supported_properties_are_stable() -> None:
    """Bind ordered supported rows and claim boundary to a stable digest."""
    report = DifferentiabilityReport((_row(), _row("affine_mask")))
    assert report.derivative_supported
    assert report.blocking_capabilities == ()
    assert len(report.content_digest) == 64
    assert report.content_digest == DifferentiabilityReport(report.rows).content_digest
    assert report.claim_boundary == DLA_TOPOLOGY_CLAIM_BOUNDARY
    report.require_supported()


def test_report_refuses_unsupported_and_descoped_rows() -> None:
    """Raise with every blocker when any row lacks derivative support."""
    report = DifferentiabilityReport(
        (
            _row(),
            ConstraintSupportRow(
                "clip_kink",
                "unsupported",
                DifferentiabilityKind.NON_SMOOTH,
                "active set changes",
                "no invented derivative",
            ),
            ConstraintSupportRow(
                "qgnn_wiring",
                "descoped",
                DifferentiabilityKind.NOT_APPLICABLE,
                "no typed consumer",
                "separate objects",
            ),
        )
    )
    assert not report.derivative_supported
    assert report.blocking_capabilities == ("clip_kink", "qgnn_wiring")
    with pytest.raises(
        UnsupportedDifferentiableConstraintError,
        match="clip_kink, qgnn_wiring",
    ):
        report.require_supported()


def test_report_rejects_empty_duplicate_or_blank_contracts() -> None:
    """Require non-empty uniquely named rows and a non-blank boundary."""
    with pytest.raises(ValueError, match="at least one"):
        DifferentiabilityReport(())
    with pytest.raises(ValueError, match="unique"):
        DifferentiabilityReport((_row(), _row()))
    with pytest.raises(ValueError, match="claim_boundary"):
        DifferentiabilityReport((_row(),), claim_boundary=" ")


def test_parity_sector_values_match_hamming_weight_convention() -> None:
    """Keep public even/odd enum values aligned with the existing projector."""
    assert ParitySector.EVEN.value == 0
    assert ParitySector.ODD.value == 1
