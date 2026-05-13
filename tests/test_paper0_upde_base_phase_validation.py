# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Paper 0 UPDE base-phase validation tests
"""Executable simulator fixtures for the Paper 0 UPDE base-phase equation."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.paper0.upde_validation import (
    BasePhaseValidationConfig,
    finite_difference_negative_gradient,
    kuramoto_phase_drift,
    negative_cosine_potential,
    validate_upde_base_phase_fixture,
)


def _fixture_problem() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    K_nm = np.array(
        [
            [0.0, 0.72, 0.18, 0.0],
            [0.72, 0.0, 0.35, 0.11],
            [0.18, 0.35, 0.0, 0.41],
            [0.0, 0.11, 0.41, 0.0],
        ],
        dtype=np.float64,
    )
    omega = np.array([0.14, -0.05, 0.09, -0.02], dtype=np.float64)
    theta = np.array([0.21, -0.37, 0.84, 1.38], dtype=np.float64)
    return K_nm, omega, theta


def test_base_phase_drift_is_negative_gradient_of_source_potential() -> None:
    K_nm, omega, theta = _fixture_problem()

    drift = kuramoto_phase_drift(theta, K_nm, omega)
    finite_difference_drift = finite_difference_negative_gradient(
        negative_cosine_potential,
        theta,
        K_nm,
        omega,
        step=2.5e-6,
    )

    np.testing.assert_allclose(drift, finite_difference_drift, atol=2e-8, rtol=2e-8)


def test_base_phase_fixture_consumes_paper0_spec_and_reports_controls() -> None:
    K_nm, omega, theta = _fixture_problem()

    result = validate_upde_base_phase_fixture(K_nm, omega, theta)

    assert result.spec_key == "upde.base_phase"
    assert result.validation_protocol == "paper0.upde.base_phase.xy_gradient_and_locking"
    assert result.hardware_status == "simulator_only_no_provider_submission"
    assert result.source_equation_ids == ("EQ0003", "EQ0032", "EQ0037", "EQ0039", "EQ0129")
    assert result.gradient_error_linf < 1e-7
    assert result.null_controls["zero_coupling_drift_linf"] < 1e-12
    assert result.null_controls["sign_flip_response_l2"] > 0.5
    assert result.null_controls["shuffled_topology_response_l2"] > 0.01
    assert result.null_controls["off_onset_order_parameter_delta"] > 1e-4


def test_base_phase_fixture_wires_through_kuramoto_problem_metadata() -> None:
    K_nm, omega, theta = _fixture_problem()

    result = validate_upde_base_phase_fixture(
        K_nm,
        omega,
        theta,
        config=BasePhaseValidationConfig(finite_difference_step=2.5e-6),
    )

    assert result.problem_metadata["paper0_spec_key"] == "upde.base_phase"
    assert result.problem_metadata["paper0_validation_protocol"] == (
        "paper0.upde.base_phase.xy_gradient_and_locking"
    )
    assert result.problem_metadata["hardware_status"] == "simulator_only_no_provider_submission"
    assert result.problem_metadata["n_oscillators"] == 4


def test_base_phase_fixture_rejects_invalid_inputs_before_simulation() -> None:
    K_nm, omega, theta = _fixture_problem()

    with pytest.raises(ValueError, match="symmetric"):
        validate_upde_base_phase_fixture(K_nm + np.triu(np.ones_like(K_nm), 1), omega, theta)

    bad_theta = theta.copy()
    bad_theta[1] = np.nan
    with pytest.raises(ValueError, match="theta must contain only finite values"):
        validate_upde_base_phase_fixture(K_nm, omega, bad_theta)

    with pytest.raises(ValueError, match="theta must have shape"):
        validate_upde_base_phase_fixture(K_nm, omega, theta[:-1])
