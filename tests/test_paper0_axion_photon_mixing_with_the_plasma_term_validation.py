# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Axionphoton mixing with the plasma term. validation tests
"""Tests for Paper 0 Axionphoton mixing with the plasma term. source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.axion_photon_mixing_with_the_plasma_term_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    AxionPhotonMixingWithThePlasmaTermConfig,
    axion_photon_mixing_with_the_plasma_term_labels,
    classify_axion_photon_mixing_with_the_plasma_term_component,
    validate_axion_photon_mixing_with_the_plasma_term_fixture,
)


def test_axion_photon_mixing_with_the_plasma_term_fixture_preserves_source_boundary() -> None:
    result = validate_axion_photon_mixing_with_the_plasma_term_fixture()
    assert result.source_ledger_span == ("P0R04348", "P0R04358")
    assert result.source_record_count == 11
    assert result.component_count == 1
    assert result.next_source_boundary == "P0R04359"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_axion_photon_mixing_with_the_plasma_term_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R04348"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R04358"


def test_axion_photon_mixing_with_the_plasma_term_classification_and_labels_are_explicit() -> None:
    for component in ("axionphoton_mixing_with_the_plasma_term",):
        assert (
            classify_axion_photon_mixing_with_the_plasma_term_component(component)
            == f"{component}_source_boundary"
        )
    labels = axion_photon_mixing_with_the_plasma_term_labels()
    assert labels["section"] == "Axionphoton mixing with the plasma term."
    assert labels["next_boundary"] == "P0R04359"


def test_axion_photon_mixing_with_the_plasma_term_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 11"):
        AxionPhotonMixingWithThePlasmaTermConfig(expected_source_record_count=10)
    with pytest.raises(ValueError, match="expected_component_count must equal 1"):
        AxionPhotonMixingWithThePlasmaTermConfig(expected_component_count=2)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R04359"):
        AxionPhotonMixingWithThePlasmaTermConfig(next_source_boundary="P0R04358")
    with pytest.raises(
        ValueError, match="unknown axion_photon_mixing_with_the_plasma_term component"
    ):
        classify_axion_photon_mixing_with_the_plasma_term_component("empirical_validation_claim")
