# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 L11 interface validation tests
"""Executable fixture tests for Paper 0 Noosphere-Technosphere L11 records."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.paper0.l11_interface_validation import (
    L11InterfaceConfig,
    effective_sigma,
    frustration_index,
    hybrid_coupling_matrix,
    validate_accelerated_supercriticality_boundary_fixture,
    validate_fragmentation_spin_glass_risk_fixture,
    validate_hybrid_collective_coupling_fixture,
    validate_l11_interface_fixture,
)


def test_hybrid_collective_coupling_adds_cross_layer_edges() -> None:
    config = L11InterfaceConfig(technosphere_coupling_gain=0.35)

    baseline = hybrid_coupling_matrix(config, include_technosphere=False)
    hybrid = hybrid_coupling_matrix(config, include_technosphere=True)
    result = validate_hybrid_collective_coupling_fixture(config)

    assert np.sum(hybrid) > np.sum(baseline)
    assert result.hybrid_coupling_gain > 0.0
    assert result.null_controls["zero_gain_delta_abs"] == pytest.approx(0.0)
    assert "not societal evidence" in result.claim_boundary


def test_accelerated_supercriticality_boundary_tracks_sigma_and_temperature() -> None:
    config = L11InterfaceConfig(
        technosphere_coupling_gain=0.45,
        effective_temperature_gain=0.4,
        sigma_baseline=0.9,
    )

    baseline_sigma = effective_sigma(
        config.sigma_baseline,
        coupling_gain=0.0,
        temperature_gain=0.0,
    )
    accelerated_sigma = effective_sigma(
        config.sigma_baseline,
        coupling_gain=config.technosphere_coupling_gain,
        temperature_gain=config.effective_temperature_gain,
    )
    result = validate_accelerated_supercriticality_boundary_fixture(config)

    assert accelerated_sigma > baseline_sigma
    assert result.accelerated_sigma > 1.0
    assert result.null_controls["baseline_supercritical_label"] == pytest.approx(0.0)
    assert result.null_controls["zero_gain_sigma_delta_abs"] == pytest.approx(0.0)


def test_fragmentation_spin_glass_risk_tracks_frustration() -> None:
    config = L11InterfaceConfig()

    frustrated = frustration_index(config.fragmented_coupling_matrix)
    coherent = frustration_index(config.coherent_coupling_matrix)
    result = validate_fragmentation_spin_glass_risk_fixture(config)

    assert frustrated > coherent
    assert result.frustration_delta > 0.0
    assert result.spin_glass_risk_label is True
    assert result.null_controls["coherent_spin_glass_risk_label"] == pytest.approx(0.0)
    assert "not societal evidence" in result.claim_boundary


def test_l11_interface_fixture_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        L11InterfaceConfig(technosphere_coupling_gain=-0.1)

    with pytest.raises(ValueError, match="positive"):
        L11InterfaceConfig(sigma_baseline=0.0)

    with pytest.raises(ValueError, match="symmetric"):
        L11InterfaceConfig(fragmented_coupling_matrix=np.array([[0.0, 1.0], [0.2, 0.0]]))

    with pytest.raises(ValueError, match="same shape"):
        L11InterfaceConfig(
            coherent_coupling_matrix=np.zeros((2, 2)),
            fragmented_coupling_matrix=np.zeros((3, 3)),
        )


def test_l11_interface_default_fixture_wires_all_boundaries() -> None:
    result = validate_l11_interface_fixture()

    assert result.hardware_status == "simulator_only_no_provider_submission"
    assert result.spec_keys == (
        "applied.l11_interface.hybrid_collective_coupling",
        "applied.l11_interface.accelerated_supercriticality_boundary",
        "applied.l11_interface.fragmentation_spin_glass_risk",
    )
    assert result.hybrid.hybrid_coupling_gain > 0.0
    assert result.supercriticality.accelerated_sigma > result.supercriticality.baseline_sigma
    assert result.fragmentation.frustration_delta > 0.0
    assert "not societal evidence" in result.claim_boundary
