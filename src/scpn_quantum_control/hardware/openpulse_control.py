# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — OpenPulse Control and Calibration Workflows
"""OpenPulse schedule construction and calibration workflow primitives.

This module is intentionally provider-neutral and no-submit by default.
It builds OpenPulse-compatible payloads from SCPN pulse envelopes and
generates reproducible calibration workflows for IBM-style pulse lanes.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .._rust_accel import optional_rust_engine
from ..phase.pulse_shaping import HypergeometricPulse

_engine = optional_rust_engine()
_rabi_fit_rust = getattr(_engine, "rabi_pi_amplitude_fit", None) if _engine is not None else None


@dataclass(frozen=True)
class OpenPulseWaveform:
    """One waveform definition for an OpenPulse schedule."""

    name: str
    samples: NDArray[np.float64]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("waveform name must be non-empty")
        if self.samples.ndim != 1 or self.samples.size == 0:
            raise ValueError("waveform samples must be one-dimensional and non-empty")
        if not np.all(np.isfinite(self.samples)):
            raise ValueError("waveform samples must be finite")

    def to_payload(self) -> dict[str, object]:
        """Serialise waveform to OpenPulse JSON-compatible mapping."""
        return {
            "name": self.name,
            "samples": [float(value) for value in self.samples.tolist()],
        }


@dataclass(frozen=True)
class OpenPulseInstruction:
    """Single OpenPulse play instruction."""

    name: str
    channel: str
    t0: int
    waveform: str

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("instruction name must be non-empty")
        if not self.channel:
            raise ValueError("channel must be non-empty")
        if self.t0 < 0:
            raise ValueError("t0 must be non-negative")
        if not self.waveform:
            raise ValueError("waveform must be non-empty")

    def to_payload(self) -> dict[str, object]:
        """Serialise instruction to OpenPulse JSON-compatible mapping."""
        return {
            "name": self.name,
            "channel": self.channel,
            "t0": int(self.t0),
            "waveform": self.waveform,
        }


@dataclass(frozen=True)
class OpenPulseSchedule:
    """Provider-neutral OpenPulse schedule."""

    name: str
    dt: float
    qubit: int
    waveforms: tuple[OpenPulseWaveform, ...]
    instructions: tuple[OpenPulseInstruction, ...]
    metadata: dict[str, object]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("schedule name must be non-empty")
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        if self.qubit < 0:
            raise ValueError("qubit index must be non-negative")
        if not self.waveforms:
            raise ValueError("schedule must contain at least one waveform")
        if not self.instructions:
            raise ValueError("schedule must contain at least one instruction")

    def to_payload(self) -> dict[str, object]:
        """Serialise schedule to OpenPulse-like JSON mapping."""
        return {
            "schema": "openpulse_schedule_v1",
            "name": self.name,
            "dt": float(self.dt),
            "qubit": int(self.qubit),
            "waveforms": [waveform.to_payload() for waveform in self.waveforms],
            "instructions": [instruction.to_payload() for instruction in self.instructions],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class RabiCalibrationPoint:
    """Calibration point in an amplitude sweep."""

    amplitude: float
    shots: int

    def __post_init__(self) -> None:
        if not np.isfinite(self.amplitude):
            raise ValueError("amplitude must be finite")
        if not 0.0 <= self.amplitude <= 1.0:
            raise ValueError("amplitude must be in [0, 1]")
        if self.shots <= 0:
            raise ValueError("shots must be positive")


@dataclass(frozen=True)
class OpenPulseCalibrationWorkflow:
    """Calibration workflow artefact for pulse-level amplitude tuning."""

    workflow_id: str
    backend_name: str
    qubit: int
    dt: float
    sigma: int
    duration: int
    points: tuple[RabiCalibrationPoint, ...]
    claim_boundary: str

    def __post_init__(self) -> None:
        if not self.workflow_id:
            raise ValueError("workflow_id must be non-empty")
        if not self.backend_name:
            raise ValueError("backend_name must be non-empty")
        if self.qubit < 0:
            raise ValueError("qubit index must be non-negative")
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        if self.sigma <= 0:
            raise ValueError("sigma must be positive")
        if self.duration <= 0:
            raise ValueError("duration must be positive")
        if not self.points:
            raise ValueError("calibration workflow requires at least one sweep point")

    def to_payload(self) -> dict[str, object]:
        """Serialise calibration workflow dossier."""
        return {
            "workflow_id": self.workflow_id,
            "backend_name": self.backend_name,
            "qubit": int(self.qubit),
            "dt": float(self.dt),
            "sigma": int(self.sigma),
            "duration": int(self.duration),
            "points": [
                {"amplitude": float(point.amplitude), "shots": int(point.shots)}
                for point in self.points
            ],
            "claim_boundary": self.claim_boundary,
            "hardware_submission": False,
        }


@dataclass(frozen=True)
class RabiPiCalibrationEstimate:
    """Estimated π-pulse amplitude from a Rabi sweep."""

    pi_amplitude: float
    peak_population: float
    confidence: float
    method: str


def compile_hypergeometric_openpulse_schedule(
    pulse: HypergeometricPulse,
    *,
    qubit: int,
    dt: float,
    channel: str | None = None,
    amp_limit: float = 1.0,
    schedule_name: str = "scpn_hypergeometric_drive",
) -> OpenPulseSchedule:
    """Compile a hypergeometric pulse envelope into an OpenPulse schedule."""
    if qubit < 0:
        raise ValueError("qubit must be non-negative")
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if amp_limit <= 0.0:
        raise ValueError("amp_limit must be positive")
    if pulse.envelope.ndim != 1 or pulse.envelope.size == 0:
        raise ValueError("pulse envelope must be one-dimensional and non-empty")

    drive_channel = channel or f"d{qubit}"
    scaled_samples = np.asarray(pulse.envelope, dtype=np.float64) * float(pulse.omega_0)
    max_abs = float(np.max(np.abs(scaled_samples)))
    if max_abs > amp_limit:
        scaled_samples = scaled_samples * (amp_limit / max_abs)

    waveform_name = f"{schedule_name}_q{qubit}_wf0"
    waveform = OpenPulseWaveform(name=waveform_name, samples=scaled_samples)
    instruction = OpenPulseInstruction(
        name="play",
        channel=drive_channel,
        t0=0,
        waveform=waveform_name,
    )
    return OpenPulseSchedule(
        name=schedule_name,
        dt=float(dt),
        qubit=qubit,
        waveforms=(waveform,),
        instructions=(instruction,),
        metadata={
            "family": "hypergeometric",
            "alpha": float(pulse.alpha),
            "beta": float(pulse.beta),
            "gamma_width": float(pulse.gamma_width),
            "omega_0": float(pulse.omega_0),
            "normalisation_applied": bool(max_abs > amp_limit),
            "claim_boundary": "Schedule generation only; no backend calibration is implied.",
        },
    )


def build_rabi_amplitude_calibration_workflow(
    *,
    backend_name: str,
    qubit: int,
    amplitude_grid: Sequence[float],
    shots: int,
    dt: float,
    sigma: int = 64,
    duration: int = 256,
) -> OpenPulseCalibrationWorkflow:
    """Build a no-submit pulse-amplitude calibration workflow dossier."""
    if shots <= 0:
        raise ValueError("shots must be positive")
    grid = np.asarray(list(amplitude_grid), dtype=np.float64)
    if grid.ndim != 1 or grid.size < 3:
        raise ValueError("amplitude_grid must have at least three points")
    if not np.all(np.isfinite(grid)):
        raise ValueError("amplitude_grid must contain finite values")
    if np.any((grid < 0.0) | (grid > 1.0)):
        raise ValueError("amplitude_grid values must be in [0, 1]")

    points = tuple(RabiCalibrationPoint(amplitude=float(value), shots=shots) for value in grid)
    return OpenPulseCalibrationWorkflow(
        workflow_id=f"openpulse_rabi_calibration_{backend_name}_q{qubit}",
        backend_name=backend_name,
        qubit=qubit,
        dt=dt,
        sigma=sigma,
        duration=duration,
        points=points,
        claim_boundary=(
            "This workflow defines calibration sweep metadata only; it does not submit "
            "OpenPulse jobs, fit backend-specific drift, or claim pulse-level advantage."
        ),
    )


def estimate_rabi_pi_amplitude(
    amplitudes: Sequence[float],
    excited_population: Sequence[float],
) -> RabiPiCalibrationEstimate:
    """Estimate π-pulse amplitude from a Rabi calibration sweep."""
    amp = np.asarray(list(amplitudes), dtype=np.float64)
    pop = np.asarray(list(excited_population), dtype=np.float64)
    if amp.ndim != 1 or pop.ndim != 1 or amp.size != pop.size:
        raise ValueError(
            "amplitudes and excited_population must be one-dimensional and equal length"
        )
    if amp.size < 3:
        raise ValueError("at least three calibration points are required")
    if not np.all(np.isfinite(amp)) or not np.all(np.isfinite(pop)):
        raise ValueError("amplitudes and excited_population must be finite")
    if np.any((pop < 0.0) | (pop > 1.0)):
        raise ValueError("excited_population must be within [0, 1]")
    if np.any(np.diff(amp) <= 0.0):
        raise ValueError("amplitudes must be strictly increasing")

    if _rabi_fit_rust is not None:
        pi_amp, peak, confidence = _rabi_fit_rust(amp, pop)
        return RabiPiCalibrationEstimate(
            pi_amplitude=float(pi_amp),
            peak_population=float(peak),
            confidence=float(confidence),
            method="rust:rabi_pi_amplitude_fit",
        )

    peak_index = int(np.argmax(pop))
    pi_amp = float(amp[peak_index])
    peak = float(pop[peak_index])
    if 0 < peak_index < amp.size - 1:
        x0, x1, x2 = amp[peak_index - 1], amp[peak_index], amp[peak_index + 1]
        y0, y1, y2 = pop[peak_index - 1], pop[peak_index], pop[peak_index + 1]
        h0 = x0 - x1
        h2 = x2 - x1
        denom = (h0 * h2) * (h0 - h2)
        if abs(float(denom)) > 1e-15:
            a = (h2 * (y0 - y1) - h0 * (y2 - y1)) / denom
            b = (h2**2 * (y1 - y0) + h0**2 * (y2 - y1)) / denom
            if abs(float(a)) > 1e-15:
                vertex = x1 - b / (2.0 * a)
                if x0 <= vertex <= x2:
                    pi_amp = float(vertex)
    edge_mean = 0.5 * float(pop[0] + pop[-1])
    contrast = max(0.0, peak - edge_mean)
    confidence = min(1.0, contrast / peak) if peak > 1e-12 else 0.0
    return RabiPiCalibrationEstimate(
        pi_amplitude=pi_amp,
        peak_population=peak,
        confidence=float(confidence),
        method="python:parabolic_peak",
    )


def schedule_to_qiskit_pulse(schedule: OpenPulseSchedule) -> Any:
    """Convert a provider-neutral schedule to a Qiskit Pulse ``Schedule``."""
    try:
        from qiskit.pulse import DriveChannel, Play, Schedule, Waveform
    except Exception as exc:
        raise RuntimeError("qiskit pulse module is required for schedule conversion") from exc

    result = Schedule(name=schedule.name)
    for waveform in schedule.waveforms:
        qiskit_waveform = Waveform(
            np.asarray(waveform.samples, dtype=np.complex128), name=waveform.name
        )
        matching = [item for item in schedule.instructions if item.waveform == waveform.name]
        for item in matching:
            if not item.channel.startswith("d"):
                raise ValueError(
                    f"unsupported channel '{item.channel}', expected drive channel like d0"
                )
            channel_index = int(item.channel[1:])
            result = result.insert(item.t0, Play(qiskit_waveform, DriveChannel(channel_index)))
    return result


__all__ = [
    "OpenPulseWaveform",
    "OpenPulseInstruction",
    "OpenPulseSchedule",
    "OpenPulseCalibrationWorkflow",
    "RabiCalibrationPoint",
    "RabiPiCalibrationEstimate",
    "compile_hypergeometric_openpulse_schedule",
    "build_rabi_amplitude_calibration_workflow",
    "estimate_rabi_pi_amplitude",
    "schedule_to_qiskit_pulse",
]
