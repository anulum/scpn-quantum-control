# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — OpenPulse Control Tests
"""Tests for OpenPulse schedule and calibration workflow support."""

from __future__ import annotations

import numpy as np
import pytest

import scpn_quantum_control.hardware.openpulse_control as openpulse_module
from scpn_quantum_control.hardware.openpulse_control import (
    build_rabi_amplitude_calibration_workflow,
    compile_hypergeometric_openpulse_schedule,
    estimate_rabi_pi_amplitude,
    schedule_to_qiskit_pulse,
)
from scpn_quantum_control.phase.pulse_shaping import build_hypergeometric_pulse


def test_compile_hypergeometric_openpulse_schedule_payload() -> None:
    pulse = build_hypergeometric_pulse(t_total=1.0, omega_0=0.8, alpha=0.5, beta=0.5, n_points=64)
    schedule = compile_hypergeometric_openpulse_schedule(
        pulse,
        qubit=2,
        dt=2.22e-10,
        amp_limit=0.7,
        schedule_name="ibm_test_drive",
    )

    payload = schedule.to_payload()
    assert payload["schema"] == "openpulse_schedule_v1"
    assert payload["qubit"] == 2
    assert payload["dt"] == pytest.approx(2.22e-10)
    waveform = payload["waveforms"][0]
    assert isinstance(waveform, dict)
    assert np.max(np.abs(np.asarray(waveform["samples"], dtype=float))) <= 0.7000000001


def test_build_rabi_amplitude_calibration_workflow_payload() -> None:
    workflow = build_rabi_amplitude_calibration_workflow(
        backend_name="ibm_fez",
        qubit=1,
        amplitude_grid=np.linspace(0.05, 0.95, 9),
        shots=4096,
        dt=2.22e-10,
        sigma=80,
        duration=320,
    )
    payload = workflow.to_payload()
    assert payload["workflow_id"] == "openpulse_rabi_calibration_ibm_fez_q1"
    assert payload["hardware_submission"] is False
    assert len(payload["points"]) == 9
    assert payload["points"][0]["shots"] == 4096


def test_estimate_rabi_pi_amplitude_rust_python_parity(monkeypatch: pytest.MonkeyPatch) -> None:
    amplitudes = np.linspace(0.05, 0.95, 15)
    excited_population = np.sin(np.pi * amplitudes) ** 2

    rust_result = estimate_rabi_pi_amplitude(amplitudes, excited_population)

    original = openpulse_module._rabi_fit_rust
    monkeypatch.setattr(openpulse_module, "_rabi_fit_rust", None)
    python_result = estimate_rabi_pi_amplitude(amplitudes, excited_population)
    monkeypatch.setattr(openpulse_module, "_rabi_fit_rust", original)

    assert rust_result.pi_amplitude == pytest.approx(
        python_result.pi_amplitude, rel=1e-9, abs=1e-9
    )
    assert rust_result.peak_population == pytest.approx(
        python_result.peak_population, rel=1e-12, abs=1e-12
    )
    assert rust_result.confidence == pytest.approx(python_result.confidence, rel=1e-9, abs=1e-9)


def test_schedule_to_qiskit_pulse_or_explicit_missing_dependency() -> None:
    pulse = build_hypergeometric_pulse(t_total=1.0, omega_0=0.3, alpha=0.0, beta=0.0, n_points=16)
    schedule = compile_hypergeometric_openpulse_schedule(
        pulse,
        qubit=0,
        dt=2.22e-10,
        schedule_name="qiskit_bridge",
    )
    try:
        qiskit_schedule = schedule_to_qiskit_pulse(schedule)
    except RuntimeError as exc:
        assert "qiskit pulse module is required" in str(exc)
        return

    # If pulse module exists, the conversion must preserve schedule identity.
    assert getattr(qiskit_schedule, "name", "") == "qiskit_bridge"
