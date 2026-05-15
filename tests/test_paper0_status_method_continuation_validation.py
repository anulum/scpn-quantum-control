# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Status and Method continuation fixtures
"""Tests for Paper 0 Status and Method continuation fixtures."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.status_method_continuation_validation import (
    StatusMethodContinuationConfig,
    classify_disagreement_move,
    classify_not_boundary,
    operational_commitment_labels,
    validate_status_method_continuation_fixture,
)


def test_not_boundary_classifier_rejects_literalisation_and_bypass() -> None:
    assert classify_not_boundary("absolute_truths") == "reject_doctrine_status"
    assert classify_not_boundary("metaphor_literalisation") == "reject_ontological_load"
    assert classify_not_boundary("empirical_bypass") == "reject_method_bypass"

    with pytest.raises(ValueError, match="unknown Status and Method boundary"):
        classify_not_boundary("hidden_premise")


def test_disagreement_classifier_preserves_productive_protocol() -> None:
    assert classify_disagreement_move("prediction_baseline") == "run_comparison"
    assert classify_disagreement_move("analogy_handle") == "supply_or_refute_empirical_handle"
    assert classify_disagreement_move("replacement_model") == "same_slot_stricter_fit"

    with pytest.raises(ValueError, match="unknown disagreement move"):
        classify_disagreement_move("dismiss_without_test")


def test_operational_commitment_labels_preserve_four_commitments() -> None:
    labels = operational_commitment_labels()

    assert labels == (
        "falsifiability_first",
        "hypothesis_registry",
        "tiered_status",
        "versioning_and_correction",
    )


def test_status_method_continuation_fixture_preserves_scope_and_counts() -> None:
    result = validate_status_method_continuation_fixture()

    assert result.hardware_status == "source_methodology_no_experiment"
    assert result.source_ledger_span == ("P0R00391", "P0R00400")
    assert result.blank_separator_count == 1
    assert result.operational_commitment_count == 4
    assert result.scp_mandate_boundary == "P0R00401"
    assert result.null_controls["literalised_metaphor_rejection_label"] == 1.0
    assert result.null_controls["analogy_without_handle_rejection_label"] == 1.0

    with pytest.raises(ValueError, match="expected_blank_separator_count must equal 1"):
        StatusMethodContinuationConfig(expected_blank_separator_count=2)
