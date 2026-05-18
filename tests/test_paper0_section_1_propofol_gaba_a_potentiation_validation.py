# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 1. Propofol (GABA-A Potentiation): validation tests
"""Tests for Paper 0 1. Propofol (GABA-A Potentiation): source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.section_1_propofol_gaba_a_potentiation_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    Section1PropofolGabaAPotentiationConfig,
    classify_section_1_propofol_gaba_a_potentiation_component,
    section_1_propofol_gaba_a_potentiation_labels,
    validate_section_1_propofol_gaba_a_potentiation_fixture,
)


def test_section_1_propofol_gaba_a_potentiation_fixture_preserves_source_boundary() -> None:
    result = validate_section_1_propofol_gaba_a_potentiation_fixture()
    assert result.source_ledger_span == ("P0R05102", "P0R05112")
    assert result.source_record_count == 11
    assert result.component_count == 4
    assert result.next_source_boundary == "P0R05113"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_section_1_propofol_gaba_a_potentiation_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R05102"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R05112"


def test_section_1_propofol_gaba_a_potentiation_classification_and_labels_are_explicit() -> None:
    for component in (
        "1_propofol_gaba_a_potentiation",
        "2_ketamine_nmda_antagonism",
        "3_nsaids_acetaminophen",
        "vi_synthesis_and_ethical_considerations_l15",
    ):
        assert (
            classify_section_1_propofol_gaba_a_potentiation_component(component)
            == f"{component}_source_boundary"
        )
    labels = section_1_propofol_gaba_a_potentiation_labels()
    assert labels["section"] == "1. Propofol (GABA-A Potentiation):"
    assert labels["next_boundary"] == "P0R05113"


def test_section_1_propofol_gaba_a_potentiation_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 11"):
        Section1PropofolGabaAPotentiationConfig(expected_source_record_count=10)
    with pytest.raises(ValueError, match="expected_component_count must equal 4"):
        Section1PropofolGabaAPotentiationConfig(expected_component_count=5)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R05113"):
        Section1PropofolGabaAPotentiationConfig(next_source_boundary="P0R05112")
    with pytest.raises(
        ValueError, match="unknown section_1_propofol_gaba_a_potentiation component"
    ):
        classify_section_1_propofol_gaba_a_potentiation_component("empirical_validation_claim")
