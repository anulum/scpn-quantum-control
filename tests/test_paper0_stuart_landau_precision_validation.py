# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Stuart-Landau precision validation tests
"""Executable fixture tests for Paper 0 Stuart-Landau precision records."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.paper0.stuart_landau_precision_validation import (
    StuartLandauPrecisionConfig,
    complex_stuart_landau_derivative,
    polar_stuart_landau_rates,
    precision_weighted_phase_terms,
    validate_salience_radial_precision_control_fixture,
    validate_stuart_landau_precision_upgrade_fixture,
    validate_stuart_landau_precision_weighted_dynamics_fixture,
)


def test_stuart_landau_complex_derivative_matches_polar_projection() -> None:
    config = StuartLandauPrecisionConfig(
        radius=np.array([1.1, 0.7, 1.4], dtype=np.float64),
        theta=np.array([0.2, -0.3, 0.6], dtype=np.float64),
        rho=np.array([0.3, 0.2, 0.5], dtype=np.float64),
        omega=np.array([0.05, -0.02, 0.01], dtype=np.float64),
        coupling=np.array([[0.0, 0.2, 0.1], [0.2, 0.0, 0.3], [0.1, 0.3, 0.0]], dtype=np.float64),
    )

    complex_derivative = complex_stuart_landau_derivative(config)
    rates = polar_stuart_landau_rates(config)
    reconstructed = np.exp(1j * config.theta) * (
        rates.radius_dot + 1j * config.radius * rates.theta_dot
    )
    result = validate_stuart_landau_precision_upgrade_fixture(config)

    assert np.max(np.abs(complex_derivative - reconstructed)) < 1.0e-12
    assert result.spec_key == "computational.stuart_landau_precision_upgrade"
    assert result.max_complex_polar_residual < 1.0e-12
    assert result.null_controls["zero_radius_rejection_label"] == pytest.approx(1.0)


def test_precision_weighted_dynamics_fixture_checks_amplitude_ratio_and_uniform_limit() -> None:
    config = StuartLandauPrecisionConfig(
        radius=np.array([0.6, 1.8, 0.9], dtype=np.float64),
        theta=np.array([0.1, 0.7, -0.2], dtype=np.float64),
        coupling=np.array([[0.0, 0.4, 0.2], [0.4, 0.0, 0.3], [0.2, 0.3, 0.0]], dtype=np.float64),
    )

    terms = precision_weighted_phase_terms(config)
    result = validate_stuart_landau_precision_weighted_dynamics_fixture(config)

    assert result.spec_key == "computational.precision_weighted_phase_amplitude_dynamics"
    assert terms.max_amplitude_ratio_deviation > 0.0
    assert result.max_phase_ratio_residual < 1.0e-12
    assert result.null_controls["uniform_amplitude_ratio_deviation"] == pytest.approx(0.0)


def test_salience_radial_precision_control_fixture_checks_rho_and_amplitude_dominance() -> None:
    config = StuartLandauPrecisionConfig(
        radius=np.array([0.5, 2.5, 0.7], dtype=np.float64),
        theta=np.array([0.0, 0.6, -0.4], dtype=np.float64),
        rho=np.array([0.1, 0.4, 0.2], dtype=np.float64),
        coupling=np.array([[0.0, 0.5, 0.2], [0.5, 0.0, 0.1], [0.2, 0.1, 0.0]], dtype=np.float64),
    )

    result = validate_salience_radial_precision_control_fixture(config)

    assert result.spec_key == "computational.salience_radial_precision_control"
    assert result.rho_gain_radius_dot_delta > 0.0
    assert result.high_incoming_over_prior_phase_drive_ratio > 1.0
    assert result.null_controls["direct_phase_salience_delta_abs"] == pytest.approx(0.0)


def test_stuart_landau_precision_fixtures_reject_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        StuartLandauPrecisionConfig(radius=np.array([1.0, 0.0, 0.8]))

    with pytest.raises(ValueError, match="symmetric"):
        StuartLandauPrecisionConfig(
            coupling=np.array([[0.0, 0.2, 0.0], [0.1, 0.0, 0.3], [0.0, 0.3, 0.0]])
        )

    with pytest.raises(ValueError, match="finite"):
        StuartLandauPrecisionConfig(rho=np.array([0.1, np.inf, 0.2]))
