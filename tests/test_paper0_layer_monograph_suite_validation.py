# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 layer monograph suite fixtures
"""Tests for Paper 0 layer monograph and validation-suite fixtures."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.layer_monograph_suite_validation import (
    LayerMonographSuiteConfig,
    classify_layer_domain,
    classify_validation_suite_paper,
    layer_publication_catalogue,
    validate_layer_monograph_suite_fixture,
)


def test_layer_domain_classifier_preserves_series_boundaries() -> None:
    assert classify_layer_domain(1) == "domain_i_biological_substrate"
    assert classify_layer_domain(4) == "domain_i_biological_substrate"
    assert classify_layer_domain(8) == "domain_ii_organismal_planetary"
    assert classify_layer_domain(12) == "domain_iii_iv_memory_control_collective"
    assert classify_layer_domain(15) == "domain_v_meta_universal"
    assert classify_layer_domain(16) == "domain_vi_cybernetic_closure"

    with pytest.raises(ValueError, match="layer must be in the closed interval 1..16"):
        classify_layer_domain(17)


def test_validation_suite_classifier_preserves_part_iii_roles() -> None:
    assert classify_validation_suite_paper(17) == "methodological_experimental_blueprint"
    assert classify_validation_suite_paper(18) == "unified_simulation_architecture"
    assert classify_validation_suite_paper(19) == "critical_dialogue_falsifiability_roadmap"
    assert classify_validation_suite_paper(20) == "philosophical_capstone"

    with pytest.raises(
        ValueError, match="validation-suite paper must be in the closed interval 17..20"
    ):
        classify_validation_suite_paper(16)


def test_layer_publication_catalogue_preserves_all_sixteen_layers() -> None:
    catalogue = layer_publication_catalogue()

    assert len(catalogue) == 16
    assert catalogue[1] == "Quantum Biological"
    assert catalogue[11] == "Noospheric-Cultural-Informational"
    assert catalogue[16] == "Meta-Layer 16"


def test_layer_monograph_suite_fixture_preserves_counts_and_boundaries() -> None:
    result = validate_layer_monograph_suite_fixture()

    assert result.hardware_status == "source_methodology_no_experiment"
    assert result.source_ledger_span == ("P0R00436", "P0R00463")
    assert result.blank_separator_count == 1
    assert result.layer_monograph_count == 16
    assert result.domain_series_count == 5
    assert result.validation_suite_paper_count == 4
    assert result.next_source_boundary == "P0R00464"
    assert result.null_controls["publication_map_is_not_validation_evidence"] == 1.0
    assert result.null_controls["unmapped_layer_rejection_label"] == 1.0

    with pytest.raises(ValueError, match="expected_blank_separator_count must equal 1"):
        LayerMonographSuiteConfig(expected_blank_separator_count=2)
