# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 glial-control validation tests
"""Executable simulator fixture tests for Paper 0 EQ0105-EQ0112 anchors."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.paper0.glial_control_validation import (
    GlialSigmaControlConfig,
    QuantumImmuneInterfaceConfig,
    build_quantum_immune_hamiltonian,
    cytokine_modulated_lambda,
    integrate_glial_sigma_control,
    validate_glial_sigma_control_fixture,
    validate_quantum_immune_interface_fixture,
)


def test_quantum_immune_hamiltonian_is_hermitian_and_cytokine_sensitive() -> None:
    low = QuantumImmuneInterfaceConfig(
        qubits=3,
        lambda_base=0.08,
        psi_state=0.4,
        cytokine_state=0.1,
        cytokine_sensitivity=0.5,
    )
    high = QuantumImmuneInterfaceConfig(
        qubits=3,
        lambda_base=0.08,
        psi_state=0.4,
        cytokine_state=1.4,
        cytokine_sensitivity=0.5,
    )

    low_h = build_quantum_immune_hamiltonian(low)
    high_h = build_quantum_immune_hamiltonian(high)

    assert low_h.shape == (8, 8)
    assert np.allclose(low_h, low_h.conj().T)
    assert np.allclose(high_h, high_h.conj().T)
    assert np.linalg.norm(high_h, ord=2) > np.linalg.norm(low_h, ord=2)
    assert cytokine_modulated_lambda(high) > cytokine_modulated_lambda(low)


def test_quantum_immune_fixture_consumes_spec_and_records_controls() -> None:
    result = validate_quantum_immune_interface_fixture()

    assert result.spec_key == "embodied.quantum_immune_interface"
    assert result.validation_protocol == (
        "paper0.embodied.quantum_immune.hamiltonian_parameter_scan"
    )
    assert result.hardware_status == "simulator_only_no_provider_submission"
    assert result.source_equation_ids == ("EQ0105",)
    assert "P0R05361" in result.source_ledger_ids
    assert result.hermiticity_error < 1.0e-12
    assert result.cytokine_spectral_shift > 0.0
    assert result.null_controls["zero_lambda_operator_norm"] == pytest.approx(0.0)
    assert result.null_controls["non_hermitian_rejection_label"] == pytest.approx(1.0)


def test_glial_sigma_integration_tracks_slow_calcium_driven_set_point() -> None:
    config = GlialSigmaControlConfig(
        initial_sigma=1.0,
        initial_G=0.0,
        kappa=0.65,
        gamma=0.42,
        alpha=0.9,
        beta=0.55,
        duration=40.0,
        dt=0.02,
    )

    driven = integrate_glial_sigma_control(config)
    blocked = integrate_glial_sigma_control(config.with_updates(gamma=0.0))

    assert driven.time.shape == driven.sigma.shape == driven.G.shape
    assert np.all(driven.G >= -1.0e-12)
    assert driven.final_sigma_shift > 0.05
    assert abs(blocked.final_sigma_shift) < 0.01
    assert driven.integrated_calcium_drive > 0.0


def test_glial_sigma_fixture_consumes_spec_and_records_blockade_falsifier() -> None:
    result = validate_glial_sigma_control_fixture()

    assert result.spec_key == "embodied.glial_sigma_control"
    assert result.validation_protocol == "paper0.embodied.glial_sigma.two_timescale_control"
    assert result.hardware_status == "simulator_only_no_provider_submission"
    assert result.source_equation_ids == (
        "EQ0106",
        "EQ0107",
        "EQ0108",
        "EQ0109",
        "EQ0110",
        "EQ0111",
        "EQ0112",
    )
    assert "P0R05406" in result.source_ledger_ids
    assert result.final_sigma > 1.0
    assert result.null_controls["gamma_zero_blockade_attenuation"] > 0.05
    assert result.null_controls["zero_calcium_G_final_abs"] < 1.0e-9
    assert result.null_controls["baseline_sigma_relaxation_error"] < 0.02


def test_glial_control_fixtures_reject_invalid_inputs_before_simulation() -> None:
    with pytest.raises(ValueError, match="qubits must be between"):
        QuantumImmuneInterfaceConfig(qubits=0)

    with pytest.raises(ValueError, match="beta must be finite and positive"):
        GlialSigmaControlConfig(beta=-0.1)

    with pytest.raises(ValueError, match="calcium_drive must have shape"):
        integrate_glial_sigma_control(
            GlialSigmaControlConfig(duration=1.0, dt=0.1),
            calcium_drive=np.ones(3),
        )
