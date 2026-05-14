# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 HPC/UPDE bridge validation tests
"""Executable fixture tests for Paper 0 HPC/UPDE bridge records."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.paper0.hpc_upde_bridge_validation import (
    HpcHierarchyConfig,
    PhasePredictionErrorConfig,
    UpdeGradientBridgeConfig,
    hpc_prediction_errors,
    kuramoto_gradient_bridge_terms,
    phase_prediction_error_terms,
    validate_hpc_bidirectional_flow_fixture,
    validate_upde_free_energy_gradient_bridge_fixture,
    validate_upde_phase_prediction_error_fixture,
)


def test_hpc_bidirectional_flow_fixture_checks_error_only_upward_flow() -> None:
    config = HpcHierarchyConfig(
        lower_state=np.array([0.1, 0.4, 0.9], dtype=np.float64),
        higher_state=np.array([0.2, 0.6], dtype=np.float64),
        generative_weights=np.array([[0.5, 0.0], [0.0, 0.5], [0.5, 0.5]], dtype=np.float64),
        upward_error_weights=np.array([[1.0, 0.0, 0.0], [0.0, 0.5, 0.5]], dtype=np.float64),
    )

    terms = hpc_prediction_errors(config)
    result = validate_hpc_bidirectional_flow_fixture(config)

    assert result.spec_key == "computational.hpc_bidirectional_flow"
    assert result.hardware_status == "simulator_only_no_provider_submission"
    assert np.linalg.norm(terms.prediction_error) > 0.0
    assert result.upward_state_copy_residual > 0.0
    assert result.null_controls["disconnected_hierarchy_residual_norm"] == pytest.approx(0.0)
    assert result.null_controls["layer_order_reversal_rejection_label"] == pytest.approx(1.0)


def test_phase_prediction_error_fixture_checks_precision_weighted_dissipation() -> None:
    config = PhasePredictionErrorConfig(
        theta_lower=np.array([0.2, -0.1, 0.45], dtype=np.float64),
        theta_upper=np.array([0.5, 0.25, 0.1], dtype=np.float64),
        coupling=np.array([0.7, 0.4, 0.8], dtype=np.float64),
        precision=np.array([1.5, 0.6, 2.0], dtype=np.float64),
        step_size=0.05,
    )

    terms = phase_prediction_error_terms(config)
    result = validate_upde_phase_prediction_error_fixture(config)

    assert result.spec_key == "computational.upde_phase_prediction_error"
    assert np.allclose(terms.phase_residual, np.sin(config.theta_upper - config.theta_lower))
    assert result.weighted_residual_norm > 0.0
    assert result.squared_error_delta < 0.0
    assert result.null_controls["zero_coupling_error_delta_abs"] == pytest.approx(0.0)


def test_gradient_bridge_fixture_checks_xy_potential_and_finite_difference_gradient() -> None:
    config = UpdeGradientBridgeConfig(
        theta=np.array([0.1, 0.35, -0.2], dtype=np.float64),
        coupling=np.array([[0.0, 0.4, 0.2], [0.4, 0.0, 0.3], [0.2, 0.3, 0.0]], dtype=np.float64),
        omega=np.array([0.03, -0.02, 0.01], dtype=np.float64),
        eta=np.array([0.001, -0.002, 0.0], dtype=np.float64),
    )

    terms = kuramoto_gradient_bridge_terms(config)
    result = validate_upde_free_energy_gradient_bridge_fixture(config)

    assert result.spec_key == "computational.upde_free_energy_gradient_bridge"
    assert terms.aligned_potential < terms.initial_potential
    assert result.max_gradient_residual < 1.0e-6
    assert result.max_drift_residual < 1.0e-12
    assert result.null_controls["wrong_sign_potential_delta"] > 0.0


def test_hpc_upde_bridge_fixtures_reject_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="finite"):
        PhasePredictionErrorConfig(
            theta_lower=np.array([0.0, np.nan]),
            theta_upper=np.array([0.0, 0.1]),
            coupling=np.array([1.0, 1.0]),
            precision=np.array([1.0, 1.0]),
        )

    with pytest.raises(ValueError, match="non-negative"):
        PhasePredictionErrorConfig(precision=np.array([1.0, -0.1, 1.0]))

    with pytest.raises(ValueError, match="square"):
        UpdeGradientBridgeConfig(coupling=np.ones((2, 3), dtype=np.float64))
