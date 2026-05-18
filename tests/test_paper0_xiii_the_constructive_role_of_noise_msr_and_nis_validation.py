# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 XIII. The Constructive Role of Noise (MSR and NIS) validation tests
"""Tests for Paper 0 XIII. The Constructive Role of Noise (MSR and NIS) source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.xiii_the_constructive_role_of_noise_msr_and_nis_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    XiiiTheConstructiveRoleOfNoiseMsrAndNisConfig,
    classify_xiii_the_constructive_role_of_noise_msr_and_nis_component,
    validate_xiii_the_constructive_role_of_noise_msr_and_nis_fixture,
    xiii_the_constructive_role_of_noise_msr_and_nis_labels,
)


def test_xiii_the_constructive_role_of_noise_msr_and_nis_fixture_preserves_source_boundary() -> (
    None
):
    result = validate_xiii_the_constructive_role_of_noise_msr_and_nis_fixture()
    assert result.source_ledger_span == ("P0R06066", "P0R06087")
    assert result.source_record_count == 22
    assert result.component_count == 3
    assert result.next_source_boundary == "P0R06088"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_xiii_the_constructive_role_of_noise_msr_and_nis_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R06066"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R06087"


def test_xiii_the_constructive_role_of_noise_msr_and_nis_classification_and_labels_are_explicit() -> (
    None
):
    for component in (
        "xiii_the_constructive_role_of_noise_msr_and_nis",
        "xiv_the_physics_of_information_energy_transduction_iet",
        "resolving_the_first_law_paradox_the_psi_field_as_an_information_catalyst",
    ):
        assert (
            classify_xiii_the_constructive_role_of_noise_msr_and_nis_component(component)
            == f"{component}_source_boundary"
        )
    labels = xiii_the_constructive_role_of_noise_msr_and_nis_labels()
    assert labels["section"] == "XIII. The Constructive Role of Noise (MSR and NIS)"
    assert labels["next_boundary"] == "P0R06088"


def test_xiii_the_constructive_role_of_noise_msr_and_nis_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 22"):
        XiiiTheConstructiveRoleOfNoiseMsrAndNisConfig(expected_source_record_count=21)
    with pytest.raises(ValueError, match="expected_component_count must equal 3"):
        XiiiTheConstructiveRoleOfNoiseMsrAndNisConfig(expected_component_count=4)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R06088"):
        XiiiTheConstructiveRoleOfNoiseMsrAndNisConfig(next_source_boundary="P0R06087")
    with pytest.raises(
        ValueError, match="unknown xiii_the_constructive_role_of_noise_msr_and_nis component"
    ):
        classify_xiii_the_constructive_role_of_noise_msr_and_nis_component(
            "empirical_validation_claim"
        )
