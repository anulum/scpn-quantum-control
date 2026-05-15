# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Logos recursive closure fixtures
"""Tests for Paper 0 Logos recursive-closure fixtures."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.logos_recursive_closure_validation import (
    LogosRecursiveClosureConfig,
    classify_axiom_status,
    classify_hint_axiom_role,
    recursive_closure_labels,
    validate_logos_recursive_closure_fixture,
)


def test_axiom_status_classifier_preserves_three_statuses() -> None:
    assert classify_axiom_status("axiom_1") == "metaphysical_postulate"
    assert classify_axiom_status("axiom_2") == "falsifiable_physical_hypothesis"
    assert classify_axiom_status("axiom_3") == "normative_teleological_postulate"

    with pytest.raises(ValueError, match="unknown Logos axiom"):
        classify_axiom_status("axiom_4")


def test_hint_axiom_role_classifier_preserves_interaction_terms() -> None:
    assert classify_hint_axiom_role("axiom_1") == "defines_psi_s_ground"
    assert classify_hint_axiom_role("axiom_2") == "defines_lambda_sigma_information_geometry"
    assert classify_hint_axiom_role("axiom_3") == "defines_sec_directional_bias"

    with pytest.raises(ValueError, match="unknown H_int axiom role"):
        classify_hint_axiom_role("axiom_0")


def test_recursive_closure_labels_preserve_anulum_boundary() -> None:
    labels = recursive_closure_labels()

    assert labels["hierarchy"] == "15-layer hierarchy"
    assert labels["closure"] == "recursive closure"
    assert labels["figure"] == "SCPN Hierarchy & Recursive Closure"


def test_logos_recursive_closure_fixture_preserves_scope_counts_and_boundaries() -> None:
    result = validate_logos_recursive_closure_fixture()

    assert result.hardware_status == "source_methodology_no_experiment"
    assert result.source_ledger_span == ("P0R00545", "P0R00577")
    assert result.blank_separator_count == 3
    assert result.axiom_count == 3
    assert result.hint_role_count == 3
    assert result.next_source_boundary == "P0R00578"
    assert result.null_controls["axioms_are_not_established_truths"] == 1.0
    assert result.null_controls["figure_caption_is_not_validation_evidence"] == 1.0

    with pytest.raises(ValueError, match="expected_blank_separator_count must equal 3"):
        LogosRecursiveClosureConfig(expected_blank_separator_count=2)
