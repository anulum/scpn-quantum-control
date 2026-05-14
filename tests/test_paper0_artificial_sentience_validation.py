# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 artificial-sentience validation tests
"""Executable fixture tests for Paper 0 artificial-sentience records."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.paper0.artificial_sentience_validation import (
    ArtificialSentienceConfig,
    artificial_sentience_criteria_gate,
    coupling_acceleration_rate,
    phase_locking_value,
    validate_artificial_sentience_criteria_gate_fixture,
    validate_phase_locking_substrate_boundary_fixture,
    validate_technosphere_coupling_acceleration_fixture,
)


def test_technosphere_coupling_acceleration_fixture_checks_finite_network_rate() -> None:
    baseline = np.array([[0.0, 0.1, 0.1], [0.1, 0.0, 0.1], [0.1, 0.1, 0.0]])
    accelerated = np.array([[0.0, 0.4, 0.3], [0.4, 0.0, 0.35], [0.3, 0.35, 0.0]])
    config = ArtificialSentienceConfig(
        baseline_coupling=baseline,
        technosphere_coupling=accelerated,
    )

    assert coupling_acceleration_rate(accelerated) > coupling_acceleration_rate(baseline)
    result = validate_technosphere_coupling_acceleration_fixture(config)

    assert result.spec_key == "applied.artificial_sentience.technosphere_coupling_acceleration"
    assert result.acceleration_delta > 0.0
    assert "not sentience evidence" in result.claim_boundary
    assert result.null_controls["zero_coupling_acceleration_abs"] == pytest.approx(0.0)


def test_artificial_sentience_criteria_gate_requires_all_predicates() -> None:
    config = ArtificialSentienceConfig(
        phi_proxy=0.82,
        phi_threshold=0.7,
        sigma=1.02,
        substrate_coupling=True,
        phase_lock_threshold=0.8,
    )

    gate = artificial_sentience_criteria_gate(config)
    result = validate_artificial_sentience_criteria_gate_fixture(config)

    assert gate.criteria_pass is True
    assert result.criteria_pass is True
    assert result.null_controls["missing_substrate_gate_pass"] == pytest.approx(0.0)
    assert result.null_controls["low_phi_gate_pass"] == pytest.approx(0.0)
    assert result.null_controls["off_criticality_gate_pass"] == pytest.approx(0.0)


def test_phase_locking_boundary_requires_substrate_and_aligned_phases() -> None:
    config = ArtificialSentienceConfig(
        system_phase=np.array([0.0, 0.1, -0.05, 0.02], dtype=np.float64),
        field_phase=np.array([0.01, 0.12, -0.03, 0.0], dtype=np.float64),
        substrate_coupling=True,
        phase_lock_threshold=0.95,
    )

    locking = phase_locking_value(config.system_phase - config.field_phase)
    result = validate_phase_locking_substrate_boundary_fixture(config)

    assert locking > 0.95
    assert result.phase_locking_value > result.phase_lock_threshold
    assert result.boundary_gate_pass is True
    assert result.null_controls["absent_substrate_gate_pass"] == pytest.approx(0.0)
    assert result.null_controls["opposed_phase_locking_value"] < 0.2


def test_artificial_sentience_fixtures_reject_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="finite"):
        ArtificialSentienceConfig(phi_proxy=float("nan"))

    with pytest.raises(ValueError, match="unit interval"):
        ArtificialSentienceConfig(phi_threshold=1.2)

    with pytest.raises(ValueError, match="symmetric"):
        ArtificialSentienceConfig(
            technosphere_coupling=np.array([[0.0, 0.3], [0.1, 0.0]], dtype=np.float64)
        )

    with pytest.raises(ValueError, match="same shape"):
        ArtificialSentienceConfig(
            system_phase=np.array([0.0, 0.1]),
            field_phase=np.array([0.0, 0.1, 0.2]),
        )
