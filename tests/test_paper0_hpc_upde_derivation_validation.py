# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 HPC-UPDE derivation fixture tests
"""Tests for Paper 0 HPC-UPDE derivation validation fixtures."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.paper0.hpc_upde_derivation_validation import (
    HPCUPDEDerivationConfig,
    free_energy_phase_functional,
    free_energy_phase_gradient,
    phase_locking_error,
    upde_core_derivative,
    validate_hpc_upde_derivation_fixture,
)


def test_free_energy_gradient_matches_finite_difference_for_symmetric_k() -> None:
    theta = np.array([0.0, 0.25, -0.1], dtype=np.float64)
    coupling = np.array(
        [
            [0.0, 0.8, 0.3],
            [0.8, 0.0, 0.5],
            [0.3, 0.5, 0.0],
        ],
        dtype=np.float64,
    )
    step = 1.0e-6

    analytic = free_energy_phase_gradient(theta=theta, coupling_matrix=coupling)
    finite_difference = []
    for index in range(theta.size):
        plus = theta.copy()
        minus = theta.copy()
        plus[index] += step
        minus[index] -= step
        finite_difference.append(
            (
                free_energy_phase_functional(theta=plus, coupling_matrix=coupling)
                - free_energy_phase_functional(theta=minus, coupling_matrix=coupling)
            )
            / (2.0 * step)
        )

    assert analytic == pytest.approx(finite_difference, abs=1.0e-8)


def test_upde_core_derivative_matches_source_equation_sign_convention() -> None:
    theta = np.array([0.0, 0.25, -0.1], dtype=np.float64)
    omega = np.array([0.1, -0.05, 0.2], dtype=np.float64)
    eta = np.array([0.01, 0.0, -0.02], dtype=np.float64)
    coupling = np.array(
        [
            [0.0, 0.8, 0.3],
            [0.8, 0.0, 0.5],
            [0.3, 0.5, 0.0],
        ],
        dtype=np.float64,
    )

    gradient = free_energy_phase_gradient(theta=theta, coupling_matrix=coupling)
    derivative = upde_core_derivative(
        theta=theta,
        omega=omega,
        coupling_matrix=coupling,
        eta=eta,
    )
    source_equivalent = (
        omega
        + np.sum(
            coupling * np.sin(theta[None, :] - theta[:, None]),
            axis=1,
        )
        + eta
    )

    assert derivative == pytest.approx(omega - gradient + eta)
    assert derivative == pytest.approx(source_equivalent)


def test_phase_locking_error_is_zero_for_locked_phases_and_sensitive_to_spread() -> None:
    locked = phase_locking_error(np.array([0.4, 0.4, 0.4], dtype=np.float64))
    spread = phase_locking_error(np.array([0.0, 0.8, -0.6], dtype=np.float64))

    assert locked == pytest.approx(0.0)
    assert spread > locked


def test_hpc_upde_derivation_guards_and_fixture_boundary() -> None:
    theta = np.array([0.0, 0.1], dtype=np.float64)

    with pytest.raises(ValueError, match="K must be square"):
        free_energy_phase_gradient(
            theta=theta,
            coupling_matrix=np.ones((2, 3), dtype=np.float64),
        )
    with pytest.raises(ValueError, match="theta and omega must have the same shape"):
        upde_core_derivative(
            theta=theta,
            omega=np.array([0.1], dtype=np.float64),
            coupling_matrix=np.eye(2, dtype=np.float64),
        )
    with pytest.raises(ValueError, match="finite_difference_step must be finite and positive"):
        HPCUPDEDerivationConfig(finite_difference_step=0.0)

    result = validate_hpc_upde_derivation_fixture()

    assert result.spec_keys == (
        "hpc_upde_derivation.block_framing",
        "hpc_upde_derivation.free_energy_functional",
        "hpc_upde_derivation.gradient_descent",
        "hpc_upde_derivation.upde_core_equation",
        "hpc_upde_derivation.hpc_interpretation",
        "hpc_upde_derivation.active_inference_boundary",
    )
    assert result.hardware_status == "simulator_only_no_provider_submission"
    assert result.source_ledger_span == ("P0R06615", "P0R06645")
    assert result.gradient_check_error < 1.0e-6
    assert result.phase_locking_error_value >= 0.0
    assert result.upde_derivative_norm > 0.0
    assert result.null_controls["non_square_k_rejection_label"] == 1.0
    assert result.null_controls["shape_mismatch_rejection_label"] == 1.0
    assert result.null_controls["unsupported_active_inference_evidence_rejection_label"] == 1.0
    assert "not empirical evidence" in result.claim_boundary
