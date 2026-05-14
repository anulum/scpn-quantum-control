# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 ethical-gauge validation tests
"""Executable fixture tests for Paper 0 EQ0123-EQ0128 anchors."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.paper0.ethical_gauge_validation import (
    CausalEntropicForceConfig,
    EthicalConnectionBoundaryConfig,
    EthicalYangMillsActionConfig,
    causal_entropy,
    causal_entropy_force,
    ethical_yang_mills_action,
    validate_causal_entropic_force_fixture,
    validate_ethical_connection_boundary_fixture,
    validate_ethical_yang_mills_action_fixture,
)


def test_ethical_yang_mills_fixture_checks_action_and_gauge_invariance() -> None:
    config = EthicalYangMillsActionConfig()
    action = ethical_yang_mills_action(config.curvature, lambda_e=config.lambda_e)
    result = validate_ethical_yang_mills_action_fixture(config)

    assert action > 0.0
    assert result.spec_key == "computational.ethical_yang_mills_action"
    assert result.source_equation_ids == ("EQ0123", "EQ0124")
    assert result.hardware_status == "simulator_only_no_provider_submission"
    assert result.action_value > 0.0
    assert result.gauge_invariance_error < 1.0e-12
    assert result.stationary_residual < 1.0e-12
    assert result.null_controls["wrong_sign_metric_rejection_label"] == pytest.approx(1.0)
    assert result.null_controls["lambda_zero_action_abs"] == pytest.approx(0.0)


def test_ethical_connection_boundary_fixture_checks_residual_and_flux() -> None:
    result = validate_ethical_connection_boundary_fixture(EthicalConnectionBoundaryConfig())

    assert result.spec_key == "computational.ethical_connection_boundary"
    assert result.source_equation_ids == ("EQ0125", "EQ0126", "EQ0127")
    assert result.hardware_status == "simulator_only_no_provider_submission"
    assert result.euler_lagrange_residual < 1.0e-12
    assert result.orientation_reversal_error < 1.0e-12
    assert result.complexity_flux_margin > 0.0
    assert result.null_controls["wrong_sign_kappa_violation_label"] == pytest.approx(1.0)
    assert result.null_controls["zero_flux_boundary_drive_abs"] == pytest.approx(0.0)


def test_causal_entropic_force_fixture_checks_gradient_and_entropy_ascent() -> None:
    config = CausalEntropicForceConfig(
        position=np.array([0.2, -0.5, 0.7], dtype=np.float64),
        target=np.array([1.0, 0.0, -0.3], dtype=np.float64),
        causal_temperature=0.4,
    )

    force = causal_entropy_force(config)
    result = validate_causal_entropic_force_fixture(config)

    assert np.allclose(force, config.causal_temperature * (config.target - config.position))
    assert result.spec_key == "computational.causal_entropic_force"
    assert result.source_equation_ids == ("EQ0128",)
    assert result.hardware_status == "simulator_only_no_provider_submission"
    assert result.gradient_residual < 1.0e-8
    assert result.entropy_ascent_delta > 0.0
    assert result.null_controls["flat_entropy_force_norm"] == pytest.approx(0.0)
    assert result.null_controls["zero_temperature_force_norm"] == pytest.approx(0.0)


def test_ethical_gauge_fixtures_reject_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="curvature"):
        EthicalYangMillsActionConfig(curvature=np.array([[1.0, 2.0, 3.0]]))

    with pytest.raises(ValueError, match="kappa_eth"):
        EthicalConnectionBoundaryConfig(kappa_eth=-1.0)

    with pytest.raises(ValueError, match="causal_temperature"):
        CausalEntropicForceConfig(causal_temperature=-0.1)

    with pytest.raises(ValueError, match="position and target"):
        causal_entropy(
            np.array([0.0, 1.0], dtype=np.float64),
            np.array([0.0], dtype=np.float64),
        )
