# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 pathology/criticality validation tests
"""Executable fixture tests for Paper 0 pathology/criticality records."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.pathology_criticality_validation import (
    PathologyCriticalityConfig,
    classify_criticality,
    pathology_index,
    restore_toward_quasicriticality,
    validate_coherence_breakdown_index_fixture,
    validate_criticality_deviation_classifier_fixture,
    validate_therapeutic_restoration_targets_fixture,
)


def test_coherence_breakdown_index_increases_with_modelled_dysregulation() -> None:
    baseline = PathologyCriticalityConfig(
        free_energy=0.2,
        sigma=1.0,
        order_parameter=0.9,
        qec_success_probability=0.95,
    )
    dysregulated = PathologyCriticalityConfig(
        free_energy=1.1,
        sigma=1.35,
        order_parameter=0.45,
        qec_success_probability=0.55,
    )

    baseline_index = pathology_index(baseline)
    dysregulated_index = pathology_index(dysregulated)
    result = validate_coherence_breakdown_index_fixture(dysregulated)

    assert dysregulated_index > baseline_index
    assert result.spec_key == "applied.pathology.coherence_breakdown_index"
    assert result.claim_boundary.startswith("simulator-only")
    assert result.index_delta_vs_baseline > 0.0
    assert result.null_controls["negative_probability_rejection_label"] == pytest.approx(1.0)


def test_criticality_deviation_classifier_respects_sigma_tolerance() -> None:
    assert classify_criticality(0.72, tolerance=0.05) == "subcritical"
    assert classify_criticality(1.31, tolerance=0.05) == "supercritical"
    assert classify_criticality(1.02, tolerance=0.05) == "quasicritical"

    result = validate_criticality_deviation_classifier_fixture(
        PathologyCriticalityConfig(sigma=1.31)
    )

    assert result.spec_key == "applied.pathology.criticality_deviation_classifier"
    assert result.sigma_label == "supercritical"
    assert result.null_controls["sigma_neutral_label_is_quasicritical"] == pytest.approx(1.0)


def test_therapeutic_restoration_targets_reduce_index_without_clinical_promotion() -> None:
    config = PathologyCriticalityConfig(
        free_energy=1.3,
        sigma=1.4,
        order_parameter=0.4,
        qec_success_probability=0.65,
        free_energy_step=0.25,
        sigma_step=0.2,
        synchronisation_step=0.15,
        qec_step=0.1,
    )

    restored = restore_toward_quasicriticality(config)
    result = validate_therapeutic_restoration_targets_fixture(config)

    assert pathology_index(restored) < pathology_index(config)
    assert restored.sigma < config.sigma
    assert restored.order_parameter > config.order_parameter
    assert result.restoration_index_delta < 0.0
    assert "not clinical evidence" in result.claim_boundary
    assert result.null_controls["zero_update_index_delta_abs"] == pytest.approx(0.0)
    assert result.null_controls["wrong_direction_index_delta"] > 0.0


def test_pathology_criticality_fixtures_reject_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        PathologyCriticalityConfig(free_energy=-0.1)

    with pytest.raises(ValueError, match="positive"):
        PathologyCriticalityConfig(sigma=0.0)

    with pytest.raises(ValueError, match="unit interval"):
        PathologyCriticalityConfig(order_parameter=1.2)

    with pytest.raises(ValueError, match="tolerance"):
        classify_criticality(1.0, tolerance=-0.1)
