# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Resolving the Epigenetic Time-Scale Disconnect: Conformational Spin-Locking validation tests
"""Tests for Paper 0 Resolving the Epigenetic Time-Scale Disconnect: Conformational Spin-Locking source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.resolving_the_epigenetic_time_scale_disconnect_conformational_spin_locki_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    ResolvingTheEpigeneticTimeScaleDisconnectConformationalSpinLockiConfig,
    classify_resolving_the_epigenetic_time_scale_disconnect_conformational_spin_locki_component,
    resolving_the_epigenetic_time_scale_disconnect_conformational_spin_locki_labels,
    validate_resolving_the_epigenetic_time_scale_disconnect_conformational_spin_locki_fixture,
)


def test_resolving_the_epigenetic_time_scale_disconnect_conformational_spin_locki_fixture_preserves_source_boundary() -> (
    None
):
    result = (
        validate_resolving_the_epigenetic_time_scale_disconnect_conformational_spin_locki_fixture()
    )
    assert result.source_ledger_span == ("P0R02128", "P0R02176")
    assert result.source_record_count == 49
    assert result.component_count == 1
    assert result.next_source_boundary == "P0R02177"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_resolving_the_epigenetic_time_scale_disconnect_conformational_spin_locki_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R02128"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R02176"


def test_resolving_the_epigenetic_time_scale_disconnect_conformational_spin_locki_classification_and_labels_are_explicit() -> (
    None
):
    for component in ("resolving_the_epigenetic_time_scale_disconnect_conformational_spin_locki",):
        assert (
            classify_resolving_the_epigenetic_time_scale_disconnect_conformational_spin_locki_component(
                component
            )
            == f"{component}_source_boundary"
        )
    labels = resolving_the_epigenetic_time_scale_disconnect_conformational_spin_locki_labels()
    assert (
        labels["section"]
        == "Resolving the Epigenetic Time-Scale Disconnect: Conformational Spin-Locking"
    )
    assert labels["next_boundary"] == "P0R02177"


def test_resolving_the_epigenetic_time_scale_disconnect_conformational_spin_locki_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 49"):
        ResolvingTheEpigeneticTimeScaleDisconnectConformationalSpinLockiConfig(
            expected_source_record_count=48
        )
    with pytest.raises(ValueError, match="expected_component_count must equal 1"):
        ResolvingTheEpigeneticTimeScaleDisconnectConformationalSpinLockiConfig(
            expected_component_count=2
        )
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R02177"):
        ResolvingTheEpigeneticTimeScaleDisconnectConformationalSpinLockiConfig(
            next_source_boundary="P0R02176"
        )
    with pytest.raises(
        ValueError,
        match="unknown resolving_the_epigenetic_time_scale_disconnect_conformational_spin_locki component",
    ):
        classify_resolving_the_epigenetic_time_scale_disconnect_conformational_spin_locki_component(
            "empirical_validation_claim"
        )
