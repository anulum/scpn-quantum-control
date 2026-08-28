# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — OpenPulse Control Tests
"""Tests for OpenPulse schedule and calibration workflow support."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

import scpn_quantum_control.hardware.openpulse_control as openpulse_module
from scpn_quantum_control.hardware.openpulse_control import (
    OpenPulseCalibrationWorkflow,
    OpenPulseInstruction,
    OpenPulseSchedule,
    OpenPulseWaveform,
    RabiCalibrationPoint,
    build_rabi_amplitude_calibration_workflow,
    compile_hypergeometric_openpulse_schedule,
    estimate_rabi_pi_amplitude,
    schedule_to_qiskit_pulse,
)
from scpn_quantum_control.phase.pulse_shaping import (
    HypergeometricPulse,
    build_hypergeometric_pulse,
)


def test_compile_hypergeometric_openpulse_schedule_payload() -> None:
    """Compile, normalize, and serialize a provider-neutral pulse schedule."""
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
    """Build and serialize a no-submit Rabi calibration dossier."""
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
    """Keep the Rust and Python Rabi estimators numerically aligned."""
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
    """Convert with legacy Pulse support or refuse its missing dependency."""
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


@pytest.mark.parametrize(
    ("name", "samples", "match"),
    [
        ("", np.ones(2), "name must be non-empty"),
        ("wf", np.ones((2, 2)), "one-dimensional and non-empty"),
        ("wf", np.array([]), "one-dimensional and non-empty"),
        ("wf", np.array([np.nan]), "samples must be finite"),
    ],
)
def test_waveform_rejects_invalid_public_fields(
    name: str,
    samples: np.ndarray,
    match: str,
) -> None:
    """Reject invalid waveform identities and sample arrays."""
    with pytest.raises(ValueError, match=match):
        OpenPulseWaveform(name=name, samples=samples)


@pytest.mark.parametrize(
    ("name", "channel", "t0", "waveform", "match"),
    [
        ("", "d0", 0, "wf", "name must be non-empty"),
        ("play", "", 0, "wf", "channel must be non-empty"),
        ("play", "d0", -1, "wf", "t0 must be non-negative"),
        ("play", "d0", 0, "", "waveform must be non-empty"),
    ],
)
def test_instruction_rejects_invalid_public_fields(
    name: str,
    channel: str,
    t0: int,
    waveform: str,
    match: str,
) -> None:
    """Reject invalid instruction identities, channels, times, and waveforms."""
    with pytest.raises(ValueError, match=match):
        OpenPulseInstruction(name=name, channel=channel, t0=t0, waveform=waveform)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"name": ""}, "name must be non-empty"),
        ({"dt": 0.0}, "dt must be positive"),
        ({"qubit": -1}, "qubit index must be non-negative"),
        ({"waveforms": ()}, "at least one waveform"),
        ({"instructions": ()}, "at least one instruction"),
    ],
)
def test_schedule_rejects_invalid_public_fields(overrides: dict[str, object], match: str) -> None:
    """Reject invalid schedule identity, timing, target, and contents."""
    waveform = OpenPulseWaveform("wf", np.ones(2))
    instruction = OpenPulseInstruction("play", "d0", 0, "wf")
    values: dict[str, object] = {
        "name": "schedule",
        "dt": 1e-9,
        "qubit": 0,
        "waveforms": (waveform,),
        "instructions": (instruction,),
        "metadata": {},
    }
    values.update(overrides)
    with pytest.raises(ValueError, match=match):
        OpenPulseSchedule(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("amplitude", "shots", "match"),
    [
        (np.nan, 1, "amplitude must be finite"),
        (-0.1, 1, "amplitude must be in"),
        (1.1, 1, "amplitude must be in"),
        (0.5, 0, "shots must be positive"),
    ],
)
def test_calibration_point_rejects_invalid_fields(
    amplitude: float,
    shots: int,
    match: str,
) -> None:
    """Reject nonfinite or out-of-range amplitudes and nonpositive shots."""
    with pytest.raises(ValueError, match=match):
        RabiCalibrationPoint(amplitude, shots)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"workflow_id": ""}, "workflow_id must be non-empty"),
        ({"backend_name": ""}, "backend_name must be non-empty"),
        ({"qubit": -1}, "qubit index must be non-negative"),
        ({"dt": 0.0}, "dt must be positive"),
        ({"sigma": 0}, "sigma must be positive"),
        ({"duration": 0}, "duration must be positive"),
        ({"points": ()}, "at least one sweep point"),
    ],
)
def test_calibration_workflow_rejects_invalid_fields(
    overrides: dict[str, object], match: str
) -> None:
    """Reject invalid calibration workflow identity, timing, and contents."""
    values: dict[str, object] = {
        "workflow_id": "workflow",
        "backend_name": "backend",
        "qubit": 0,
        "dt": 1e-9,
        "sigma": 64,
        "duration": 256,
        "points": (RabiCalibrationPoint(0.5, 10),),
        "claim_boundary": "no submit",
    }
    values.update(overrides)
    with pytest.raises(ValueError, match=match):
        OpenPulseCalibrationWorkflow(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"qubit": -1}, "qubit must be non-negative"),
        ({"dt": 0.0}, "dt must be positive"),
        ({"amp_limit": 0.0}, "amp_limit must be positive"),
    ],
)
def test_compile_rejects_invalid_control_parameters(kwargs: dict[str, object], match: str) -> None:
    """Reject invalid target, timing, and amplitude-limit parameters."""
    pulse = build_hypergeometric_pulse(1.0, 0.5, n_points=8)
    values: dict[str, object] = {"qubit": 0, "dt": 1e-9, **kwargs}
    with pytest.raises(ValueError, match=match):
        compile_hypergeometric_openpulse_schedule(pulse, **values)  # type: ignore[arg-type]


@pytest.mark.parametrize("envelope", [np.array([]), np.ones((2, 2))])
def test_compile_rejects_invalid_pulse_envelope(envelope: np.ndarray) -> None:
    """Reject empty and non-vector pulse envelopes."""
    pulse = HypergeometricPulse(
        times=np.arange(envelope.size, dtype=float),
        envelope=envelope,
        alpha=0.5,
        beta=0.5,
        gamma_width=1.0,
        omega_0=0.5,
    )
    with pytest.raises(ValueError, match="one-dimensional and non-empty"):
        compile_hypergeometric_openpulse_schedule(pulse, qubit=0, dt=1e-9)


def test_compile_preserves_custom_channel_without_normalization() -> None:
    """Preserve a custom channel when samples already fit the amplitude limit."""
    pulse = build_hypergeometric_pulse(1.0, 0.2, n_points=8)
    schedule = compile_hypergeometric_openpulse_schedule(
        pulse, qubit=1, dt=1e-9, channel="d7", amp_limit=1.0
    )
    assert schedule.instructions[0].channel == "d7"
    assert schedule.metadata["normalisation_applied"] is False


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"shots": 0}, "shots must be positive"),
        ({"amplitude_grid": [0.1, 0.2]}, "at least three points"),
        ({"amplitude_grid": [[0.1, 0.2], [0.3, 0.4]]}, "at least three points"),
        ({"amplitude_grid": [0.1, np.nan, 0.3]}, "finite values"),
        ({"amplitude_grid": [0.1, 0.2, 1.1]}, "values must be in"),
    ],
)
def test_workflow_builder_rejects_invalid_sweep(kwargs: dict[str, object], match: str) -> None:
    """Reject nonpositive shots and malformed amplitude grids."""
    values: dict[str, object] = {
        "backend_name": "backend",
        "qubit": 0,
        "amplitude_grid": [0.1, 0.2, 0.3],
        "shots": 10,
        "dt": 1e-9,
        **kwargs,
    }
    with pytest.raises(ValueError, match=match):
        build_rabi_amplitude_calibration_workflow(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("amplitudes", "populations", "match"),
    [
        ([[0.1, 0.2], [0.3, 0.4]], [0.1, 0.2], "one-dimensional and equal length"),
        ([0.1, 0.2, 0.3], [0.1, 0.2], "one-dimensional and equal length"),
        ([0.1, 0.2], [0.1, 0.2], "at least three"),
        ([0.1, np.nan, 0.3], [0.1, 0.2, 0.3], "must be finite"),
        ([0.1, 0.2, 0.3], [0.1, 1.1, 0.3], "within"),
        ([0.1, 0.1, 0.3], [0.1, 0.2, 0.3], "strictly increasing"),
    ],
)
def test_rabi_estimator_rejects_invalid_sweeps(
    amplitudes: object,
    populations: object,
    match: str,
) -> None:
    """Reject malformed, nonfinite, nonphysical, and unordered Rabi sweeps."""
    with pytest.raises(ValueError, match=match):
        estimate_rabi_pi_amplitude(amplitudes, populations)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("populations", "expected_amplitude", "expected_confidence"),
    [
        ([0.0, 0.0, 0.0], 0.1, 0.0),
        ([1.0, 0.5, 0.0], 0.1, 0.5),
    ],
)
def test_python_rabi_estimator_handles_edge_and_zero_peaks(
    monkeypatch: pytest.MonkeyPatch,
    populations: list[float],
    expected_amplitude: float,
    expected_confidence: float,
) -> None:
    """Handle boundary peaks and zero-contrast sweeps without interpolation."""
    monkeypatch.setattr(openpulse_module, "_rabi_fit_rust", None)
    estimate = estimate_rabi_pi_amplitude([0.1, 0.2, 0.3], populations)
    assert estimate.pi_amplitude == expected_amplitude
    assert estimate.confidence == expected_confidence


def test_qiskit_adapter_constructs_and_validates_drive_channels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the public legacy-Pulse adapter contract without submission."""
    pulse_module = types.ModuleType("qiskit.pulse")

    class DriveChannel:
        def __init__(self, index: int) -> None:
            self.index = index

    class Waveform:
        def __init__(self, samples: np.ndarray, name: str) -> None:
            self.samples = samples
            self.name = name

    class Play:
        def __init__(self, waveform: Waveform, channel: DriveChannel) -> None:
            self.waveform = waveform
            self.channel = channel

    class Schedule:
        def __init__(self, name: str) -> None:
            self.name = name
            self.instructions: list[tuple[int, Play]] = []

        def insert(self, t0: int, play: Play) -> Schedule:
            self.instructions.append((t0, play))
            return self

    pulse_module.DriveChannel = DriveChannel  # type: ignore[attr-defined]
    pulse_module.Play = Play  # type: ignore[attr-defined]
    pulse_module.Schedule = Schedule  # type: ignore[attr-defined]
    pulse_module.Waveform = Waveform  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "qiskit.pulse", pulse_module)

    pulse = build_hypergeometric_pulse(1.0, 0.2, n_points=8)
    valid = compile_hypergeometric_openpulse_schedule(pulse, qubit=0, dt=1e-9)
    converted = schedule_to_qiskit_pulse(valid)
    assert converted.name == valid.name
    assert converted.instructions[0][1].channel.index == 0

    unsupported = compile_hypergeometric_openpulse_schedule(pulse, qubit=0, dt=1e-9, channel="m0")
    with pytest.raises(ValueError, match="unsupported channel"):
        schedule_to_qiskit_pulse(unsupported)

    unmatched = OpenPulseSchedule(
        name="unmatched",
        dt=1e-9,
        qubit=0,
        waveforms=(OpenPulseWaveform("wf", np.ones(2)),),
        instructions=(OpenPulseInstruction("play", "d0", 0, "other"),),
        metadata={},
    )
    assert schedule_to_qiskit_pulse(unmatched).instructions == []
