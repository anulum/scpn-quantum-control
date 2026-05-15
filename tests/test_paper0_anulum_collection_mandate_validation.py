# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Anulum Collection mandate fixtures
"""Tests for Paper 0 Anulum Collection mandate fixtures."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.anulum_collection_mandate_validation import (
    AnulumCollectionMandateConfig,
    classify_book_role,
    classify_meta_framework_integration,
    master_publication_entries,
    validate_anulum_collection_mandate_fixture,
)


def test_book_role_classifier_preserves_five_book_programme_map() -> None:
    assert classify_book_role("book_i") == "foundational_physics"
    assert classify_book_role("book_ii") == "engineering_and_architecture"
    assert classify_book_role("book_iii") == "philosophical_interpretation"
    assert classify_book_role("book_iv") == "boundary_probe"
    assert classify_book_role("book_v") == "practitioner_tooling"

    with pytest.raises(ValueError, match="unknown Anulum Collection book"):
        classify_book_role("book_vi")


def test_meta_framework_classifier_preserves_hpc_and_field_coupling_roles() -> None:
    assert classify_meta_framework_integration("predictive_coding") == "research_process_hpc"
    assert classify_meta_framework_integration("paper0_deep_priors") == "slow_prior_source"
    assert (
        classify_meta_framework_integration("papers_1_16_generative_cascade")
        == "layer_hypothesis_cascade"
    )
    assert (
        classify_meta_framework_integration("part_iii_prediction_error")
        == "validation_error_minimisation"
    )
    assert (
        classify_meta_framework_integration("psi_sigma_coupling")
        == "layer_sigma_lambda_measurement_plan"
    )

    with pytest.raises(ValueError, match="unknown meta-framework integration"):
        classify_meta_framework_integration("unlabelled_bridge")


def test_master_publication_entries_preserve_book_and_paper0_location() -> None:
    entries = master_publication_entries()

    assert sum(1 for key in entries if key.startswith("book_")) == 5
    assert entries["book_ii"] == "The Sentient-Consciousness Projection Network"
    assert entries["paper0_location"] == "Paper 0: The Foundational Framework - You are Here"


def test_anulum_collection_mandate_fixture_preserves_scope_counts_and_boundaries() -> None:
    result = validate_anulum_collection_mandate_fixture()

    assert result.hardware_status == "source_methodology_no_experiment"
    assert result.source_ledger_span == ("P0R00401", "P0R00435")
    assert result.blank_separator_count == 2
    assert result.book_count == 5
    assert result.meta_framework_count == 5
    assert result.validation_suite_range == ("Papers 17", "Papers 20")
    assert result.coupling_equation == "H_int = -lambda * Psi_s * sigma"
    assert result.next_source_boundary == "P0R00436"
    assert result.null_controls["programme_map_is_not_empirical_validation"] == 1.0
    assert result.null_controls["unmeasured_lambda_rejection_label"] == 1.0

    with pytest.raises(ValueError, match="expected_blank_separator_count must equal 2"):
        AnulumCollectionMandateConfig(expected_blank_separator_count=1)
