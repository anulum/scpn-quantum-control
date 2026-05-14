# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Layer 5 Triple Network fixture tests
"""Tests for Paper 0 Layer 5 Triple Network fixtures."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.paper0.l5_triple_network_validation import (
    L5TripleNetworkConfig,
    anatomical_mapping_score,
    anti_correlation_index,
    interoceptive_salience,
    salience_switch_state,
    validate_l5_triple_network_fixture,
)


def test_anatomical_mapping_score_requires_all_three_networks() -> None:
    complete = anatomical_mapping_score(
        dmn_mapping=0.88,
        cen_mapping=0.84,
        sn_mapping=0.86,
    )
    missing_sn = anatomical_mapping_score(
        dmn_mapping=0.88,
        cen_mapping=0.84,
        sn_mapping=0.0,
    )

    assert complete > missing_sn
    assert complete > L5TripleNetworkConfig().mapping_threshold


def test_anti_correlation_index_detects_dmn_cen_opposition() -> None:
    dmn = np.array([0.8, 0.7, 0.2, 0.1], dtype=np.float64)
    cen = np.array([0.1, 0.2, 0.7, 0.8], dtype=np.float64)

    assert anti_correlation_index(dmn_activity=dmn, cen_activity=cen) > 0.95


def test_interoceptive_salience_and_switch_state_follow_source_formula() -> None:
    salience = interoceptive_salience(
        precision=np.array([0.5, 1.5, 2.0], dtype=np.float64),
        prediction_error=np.array([0.1, -0.4, 0.6], dtype=np.float64),
    )

    np.testing.assert_allclose(salience, np.array([0.05, 0.6, 1.2]))
    assert salience_switch_state(salience=salience, threshold=0.7) == "CEN_engagement"
    assert (
        salience_switch_state(
            salience=np.array([0.05, 0.2, 0.3], dtype=np.float64),
            threshold=0.7,
        )
        == "DMN_dominance"
    )


def test_l5_triple_network_fixture_preserves_boundaries_and_controls() -> None:
    with pytest.raises(ValueError, match="mapping_threshold must be finite and positive"):
        L5TripleNetworkConfig(mapping_threshold=0.0)
    with pytest.raises(ValueError, match="activity vectors must have the same shape"):
        anti_correlation_index(
            dmn_activity=np.array([0.1, 0.2]),
            cen_activity=np.array([0.1, 0.2, 0.3]),
        )
    with pytest.raises(ValueError, match="precision must be non-negative"):
        interoceptive_salience(
            precision=np.array([1.0, -1.0]),
            prediction_error=np.array([0.2, 0.3]),
        )

    result = validate_l5_triple_network_fixture()

    assert result.spec_keys == (
        "l5_triple_network.anatomical_mapping",
        "l5_triple_network.salience_switching",
        "l5_triple_network.interoceptive_inference",
        "l5_triple_network.homeostatic_qualia_boundary",
    )
    assert result.hardware_status == "simulator_only_no_provider_submission"
    assert result.source_ledger_span == ("P0R06485", "P0R06503")
    assert result.mapping_score > result.config_thresholds["mapping_threshold"]
    assert result.dmn_cen_anti_correlation > result.config_thresholds["anti_correlation_threshold"]
    assert result.switch_state == "CEN_engagement"
    assert result.null_controls["missing_salience_network_rejection_label"] == 1.0
    assert result.null_controls["shape_mismatch_rejection_label"] == 1.0
    assert result.null_controls["unsupported_empirical_mapping_rejection_label"] == 1.0
    assert "not empirical evidence" in result.claim_boundary
