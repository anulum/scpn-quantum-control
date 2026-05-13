# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 neurovascular validation tests
"""Executable simulator fixture tests for the Paper 0 neurovascular anchor."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.paper0.neurovascular_validation import (
    NeurovascularValidationConfig,
    integrate_neurovascular_phase_coupling,
    mean_frequency_slip,
    phase_locking_value,
    validate_neurovascular_phase_coupling_fixture,
)


def test_phase_locking_observable_detects_aligned_and_dispersed_phases() -> None:
    aligned = np.full(128, 0.25, dtype=np.float64)
    dispersed = np.linspace(0.0, 2.0 * np.pi, 128, endpoint=False)

    assert phase_locking_value(aligned) == pytest.approx(1.0)
    assert phase_locking_value(dispersed) < 1.0e-12


def test_neurovascular_integration_phase_locks_neural_to_hemodynamic_drive() -> None:
    config = NeurovascularValidationConfig(
        omega_neural=0.97,
        omega_hemo=1.0,
        K_NH=0.42,
        duration=80.0,
        dt=0.02,
        transient_fraction=0.5,
    )

    trajectory = integrate_neurovascular_phase_coupling(config)
    phase_difference = trajectory.theta_hemo - trajectory.theta_neural

    assert trajectory.theta_neural.shape == trajectory.time.shape
    assert phase_locking_value(phase_difference[trajectory.analysis_start_index :]) > 0.98
    assert (
        abs(mean_frequency_slip(phase_difference, config.dt, trajectory.analysis_start_index))
        < 0.01
    )


def test_neurovascular_fixture_consumes_spec_and_records_controls() -> None:
    result = validate_neurovascular_phase_coupling_fixture()

    assert result.spec_key == "embodied.neurovascular_phase_coupling"
    assert result.validation_protocol == "paper0.embodied.neurovascular.phase_locking"
    assert result.hardware_status == "simulator_only_no_provider_submission"
    assert result.source_equation_ids == ("EQ0093",)
    assert "P0R04890" in result.source_ledger_ids
    assert result.phase_locking_value > 0.98
    assert abs(result.mean_frequency_slip) < 0.01
    assert result.null_controls["zero_K_NH_slip_abs"] > 0.02
    assert result.null_controls["detuned_phase_locking_drop"] > 0.1
    assert result.null_controls["shuffled_drive_phase_locking_drop"] > 0.1
    assert result.null_controls["impaired_cbf_boundary_label"] == pytest.approx(1.0)
    assert result.problem_metadata["analysis_start_index"] > 0


def test_neurovascular_fixture_rejects_invalid_inputs_before_simulation() -> None:
    with pytest.raises(ValueError, match="dt must be finite and positive"):
        NeurovascularValidationConfig(dt=0.0)

    with pytest.raises(ValueError, match="transient_fraction"):
        NeurovascularValidationConfig(transient_fraction=1.0)

    with pytest.raises(ValueError, match="K_NH must be finite"):
        validate_neurovascular_phase_coupling_fixture(
            config=NeurovascularValidationConfig(K_NH=float("nan"))
        )
