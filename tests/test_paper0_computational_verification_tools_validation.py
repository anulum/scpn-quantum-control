# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 computational verification fixtures
"""Tests for Paper 0 computational verification tool fixtures."""

from __future__ import annotations

import math

import numpy as np
import pytest

from scpn_quantum_control.paper0.computational_verification_tools_validation import (
    ComputationalVerificationToolsConfig,
    class_goldstone_density_pressure,
    class_goldstone_equation_of_state,
    lambda_eff,
    lambda_psi_rho,
    lattice_hmc_action,
    lattice_mass_targets,
    validate_computational_verification_tools_fixture,
    washout_regime,
)


def test_lattice_hmc_action_preserves_full_source_formula_and_quenched_boundary() -> None:
    config = ComputationalVerificationToolsConfig(lattice_size=16, lambda4=0.10, v=1.0)
    rho = np.zeros((2, 2, 2, 2), dtype=float)
    phi = np.zeros((2, 2, 2, 2), dtype=float)
    phi[1, 0, 0, 0] = 0.5

    result = lattice_hmc_action(rho, phi, config=config)
    targets = lattice_mass_targets(config)

    assert result.radial_gradient == 0.0
    assert result.goldstone_gradient > 0.0
    assert result.potential == 0.0
    assert result.fermion_boundary == "quenched_flat_line_test"
    assert result.total == pytest.approx(result.goldstone_gradient)
    assert targets.yukawa == pytest.approx(math.sqrt(config.lambda4 / 2.0))
    assert targets.pcac_mass == pytest.approx(targets.yukawa / math.sqrt(2.0) * config.v)
    assert targets.mass_ratio_target == pytest.approx(math.sqrt(2.0))


def test_class_goldstone_patch_preserves_washout_and_density_pressure_relations() -> None:
    assert class_goldstone_equation_of_state(
        1.0, eps=0.1, omega_log=3.0, phase=0.0
    ) == pytest.approx(-0.9)
    assert washout_regime(eps=1.0e-4, omega_log=500.0) is True
    assert washout_regime(eps=0.1, omega_log=100.0) is False

    density, pressure, w_value = class_goldstone_density_pressure(
        0.25,
        rho_phi0=1.0e-4,
        eps=0.0,
        omega_log=500.0,
        phase=0.0,
    )

    assert density == pytest.approx(1.0e-4 * 0.25**-3)
    assert w_value == pytest.approx(-1.0)
    assert pressure == pytest.approx(-density)


def test_lambda_eff_utility_preserves_dark_energy_matched_constants() -> None:
    psi_t = 2.0e-9

    rho = lambda_psi_rho(psi_t)
    effective = lambda_eff(psi_t)

    assert rho == pytest.approx(1.068935e-122 * psi_t**2 * (2.435e18**2))
    assert effective == pytest.approx(1.1056e-52 + rho)
    assert lambda_eff(0.0) == pytest.approx(1.1056e-52)


def test_computational_verification_fixture_preserves_execution_boundary() -> None:
    result = validate_computational_verification_tools_fixture()

    assert result.hardware_status == "computational_protocol_no_claimed_execution"
    assert result.source_ledger_span == ("P0R07006", "P0R07072")
    assert result.tool_count == 3
    assert result.spec_count == 5
    assert result.null_controls["hmc_execution_overclaim_rejection_label"] == 1.0
    assert result.null_controls["class_patch_without_parameter_audit_rejection_label"] == 1.0
    assert result.null_controls["lambda_units_mismatch_rejection_label"] == 1.0
    assert "not empirical execution evidence" in result.claim_boundary

    with pytest.raises(ValueError, match="lattice_size must be at least 1"):
        ComputationalVerificationToolsConfig(lattice_size=0)
    with pytest.raises(ValueError, match="scale factor a must be positive"):
        class_goldstone_equation_of_state(0.0, eps=0.1, omega_log=1.0, phase=0.0)
