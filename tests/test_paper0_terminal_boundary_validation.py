# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 terminal boundary fixtures
"""Tests for Paper 0 EBS and terminal boundary fixtures."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.terminal_boundary_validation import (
    EnhancedBoundarySet,
    TerminalBoundaryConfig,
    bind_enhanced_boundary_set,
    terminal_catalogue,
    validate_terminal_boundary_fixture,
)


def test_terminal_catalogue_preserves_t1_to_t7_categories() -> None:
    catalogue = terminal_catalogue()

    assert tuple(item.terminal_id for item in catalogue) == (
        "T1",
        "T2",
        "T3",
        "T4",
        "T5",
        "T6",
        "T7",
    )
    assert tuple(item.category for item in catalogue) == (
        "bio-measurement",
        "body-side actuation",
        "cognitive/linguistic input",
        "environmental and planetary context",
        "cosmic geometry",
        "noospheric information",
        "simulation control",
    )
    assert all(item.boundary_role for item in catalogue)


def test_ebs_binding_requires_active_terminal_subset_and_hash() -> None:
    ebs = EnhancedBoundarySet(
        ebs_id="EBS-2026-05-15-001",
        local_bio_geometry={"posture": "seated"},
        environmental_fields={"magnetic_field_uT": 47.0},
        cosmic_geometry_pack={"sidereal_phase": 0.25},
        operator_state={"blind_condition": "observer"},
    )

    binding = bind_enhanced_boundary_set(ebs, active_terminals=("T1", "T5", "T7"))

    assert binding.ebs_id == "EBS-2026-05-15-001"
    assert binding.active_terminals == ("T1", "T5", "T7")
    assert len(binding.ebs_hash) == 64
    assert binding.reproducible_boundary_conditions is True

    with pytest.raises(ValueError, match="unknown terminal ids"):
        bind_enhanced_boundary_set(ebs, active_terminals=("T8",))
    with pytest.raises(ValueError, match="at least one active terminal"):
        bind_enhanced_boundary_set(ebs, active_terminals=())


def test_terminal_boundary_fixture_preserves_claim_traceability_boundary() -> None:
    result = validate_terminal_boundary_fixture()

    assert result.hardware_status == "boundary_protocol_no_device_execution"
    assert result.source_ledger_span == ("P0R07073", "P0R07080")
    assert result.terminal_count == 7
    assert result.spec_count == 4
    assert result.null_controls["unbound_claim_rejection_label"] == 1.0
    assert result.null_controls["missing_ebs_hash_rejection_label"] == 1.0
    assert result.null_controls["unknown_terminal_rejection_label"] == 1.0
    assert "no unbound empirical claim" in result.claim_boundary

    with pytest.raises(ValueError, match="expected_terminal_count must be at least 1"):
        TerminalBoundaryConfig(expected_terminal_count=0)
