# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 anomalous-boundary validation tests
"""Executable fixture tests for Paper 0 anomalous-boundary records."""

from __future__ import annotations

import math

import numpy as np
import pytest

from scpn_quantum_control.paper0.anomalous_boundary_validation import (
    AnomalousBoundaryConfig,
    bell_chsh_value,
    bounded_weak_measurement_bias,
    validate_anomalous_boundary_fixture,
    validate_entanglement_correlation_boundary_fixture,
    validate_tsvf_precognition_boundary_fixture,
    validate_weak_measurement_bias_boundary_fixture,
)
from scpn_quantum_control.paper0.computational_unifier_validation import abl_probabilities


def test_tsvf_boundary_is_abl_conditioning_not_retrocausal_evidence() -> None:
    config = AnomalousBoundaryConfig()
    probabilities = abl_probabilities(config.pre_state, config.post_state, config.projectors)

    shifted = abl_probabilities(
        config.pre_state,
        np.array([math.sqrt(0.2), math.sqrt(0.8)], dtype=np.complex128),
        config.projectors,
    )
    result = validate_tsvf_precognition_boundary_fixture(config)

    assert sum(probabilities) == pytest.approx(1.0)
    assert probabilities != pytest.approx(shifted)
    assert result.spec_key == "applied.anomalous_boundary.tsvf_precognition_boundary"
    assert result.probability_normalisation_error == pytest.approx(0.0)
    assert "not anomalous evidence" in result.claim_boundary
    assert result.null_controls["zero_denominator_rejection_label"] == pytest.approx(1.0)
    assert result.null_controls["retrocausal_signalling_supported_label"] == pytest.approx(0.0)


def test_entanglement_boundary_checks_chsh_without_signalling() -> None:
    config = AnomalousBoundaryConfig()
    result = validate_entanglement_correlation_boundary_fixture(config)

    assert bell_chsh_value(config.bell_state, config.chsh_angles) == pytest.approx(
        2.0 * math.sqrt(2.0)
    )
    assert result.chsh_value > 2.0
    assert result.no_signalling_residual == pytest.approx(0.0, abs=1.0e-12)
    assert result.null_controls["product_state_chsh_value"] <= 2.0
    assert result.null_controls["signalling_rejection_label"] == pytest.approx(1.0)
    assert "not anomalous evidence" in result.claim_boundary


def test_weak_measurement_bias_is_bounded_monotone_and_null_controlled() -> None:
    config = AnomalousBoundaryConfig(
        prior_probability=0.42,
        intent_bias=0.35,
        measurement_strength=0.7,
    )

    biased = bounded_weak_measurement_bias(
        config.prior_probability,
        config.intent_bias,
        config.measurement_strength,
    )
    unbiased = bounded_weak_measurement_bias(config.prior_probability, 0.0, 0.7)
    result = validate_weak_measurement_bias_boundary_fixture(config)

    assert config.prior_probability < biased < 1.0
    assert unbiased == pytest.approx(config.prior_probability)
    assert result.biased_probability == pytest.approx(biased)
    assert result.null_controls["zero_intent_probability_shift_abs"] == pytest.approx(0.0)
    assert result.null_controls["out_of_range_bias_rejection_label"] == pytest.approx(1.0)
    assert "not anomalous evidence" in result.claim_boundary


def test_anomalous_boundary_fixture_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="unit interval"):
        AnomalousBoundaryConfig(prior_probability=1.0)

    with pytest.raises(ValueError, match="intent_bias"):
        AnomalousBoundaryConfig(intent_bias=1.2)

    with pytest.raises(ValueError, match="finite"):
        AnomalousBoundaryConfig(measurement_strength=float("nan"))

    with pytest.raises(ValueError, match="normalised"):
        AnomalousBoundaryConfig(bell_state=np.array([1.0, 1.0], dtype=np.complex128))


def test_anomalous_boundary_default_fixture_wires_all_boundaries() -> None:
    result = validate_anomalous_boundary_fixture()

    assert result.hardware_status == "simulator_only_no_provider_submission"
    assert result.spec_keys == (
        "applied.anomalous_boundary.tsvf_precognition_boundary",
        "applied.anomalous_boundary.entanglement_correlation_boundary",
        "applied.anomalous_boundary.weak_measurement_bias_boundary",
    )
    assert result.tsvf.probability_normalisation_error == pytest.approx(0.0)
    assert result.entanglement.chsh_value > 2.0
    assert 0.0 < result.weak_measurement.biased_probability < 1.0
    assert "not anomalous evidence" in result.claim_boundary
