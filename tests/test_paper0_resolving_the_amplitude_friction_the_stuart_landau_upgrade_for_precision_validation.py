# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Resolving the Amplitude Friction: The Stuart-Landau Upgrade for Precision Weighting validation tests
"""Tests for Paper 0 Resolving the Amplitude Friction: The Stuart-Landau Upgrade for Precision Weighting source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.resolving_the_amplitude_friction_the_stuart_landau_upgrade_for_precision_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    ResolvingTheAmplitudeFrictionTheStuartLandauUpgradeForPrecisionConfig,
    classify_resolving_the_amplitude_friction_the_stuart_landau_upgrade_for_precision_component,
    resolving_the_amplitude_friction_the_stuart_landau_upgrade_for_precision_labels,
    validate_resolving_the_amplitude_friction_the_stuart_landau_upgrade_for_precision_fixture,
)


def test_resolving_the_amplitude_friction_the_stuart_landau_upgrade_for_precision_fixture_preserves_source_boundary() -> (
    None
):
    result = (
        validate_resolving_the_amplitude_friction_the_stuart_landau_upgrade_for_precision_fixture()
    )
    assert result.source_ledger_span == ("P0R06179", "P0R06196")
    assert result.source_record_count == 18
    assert result.component_count == 1
    assert result.next_source_boundary == "P0R06197"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_resolving_the_amplitude_friction_the_stuart_landau_upgrade_for_precision_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R06179"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R06196"


def test_resolving_the_amplitude_friction_the_stuart_landau_upgrade_for_precision_classification_and_labels_are_explicit() -> (
    None
):
    for component in ("resolving_the_amplitude_friction_the_stuart_landau_upgrade_for_precision",):
        assert (
            classify_resolving_the_amplitude_friction_the_stuart_landau_upgrade_for_precision_component(
                component
            )
            == f"{component}_source_boundary"
        )
    labels = resolving_the_amplitude_friction_the_stuart_landau_upgrade_for_precision_labels()
    assert (
        labels["section"]
        == "Resolving the Amplitude Friction: The Stuart-Landau Upgrade for Precision Weighting"
    )
    assert labels["next_boundary"] == "P0R06197"


def test_resolving_the_amplitude_friction_the_stuart_landau_upgrade_for_precision_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 18"):
        ResolvingTheAmplitudeFrictionTheStuartLandauUpgradeForPrecisionConfig(
            expected_source_record_count=17
        )
    with pytest.raises(ValueError, match="expected_component_count must equal 1"):
        ResolvingTheAmplitudeFrictionTheStuartLandauUpgradeForPrecisionConfig(
            expected_component_count=2
        )
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R06197"):
        ResolvingTheAmplitudeFrictionTheStuartLandauUpgradeForPrecisionConfig(
            next_source_boundary="P0R06196"
        )
    with pytest.raises(
        ValueError,
        match="unknown resolving_the_amplitude_friction_the_stuart_landau_upgrade_for_precision component",
    ):
        classify_resolving_the_amplitude_friction_the_stuart_landau_upgrade_for_precision_component(
            "empirical_validation_claim"
        )
