# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — S3 pulse feasibility probes
"""No-submit provider feasibility probes for S3 pulse schedules."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from scpn_quantum_control.phase.pulse_shaping import PulseSchedule

PulseFeasibilityStatus = Literal["ready", "blocked", "manual_review", "unknown"]


@dataclass(frozen=True)
class PulseProviderSnapshot:
    """Provider metadata needed for S3 pulse feasibility without submission."""

    provider: str
    backend_name: str
    n_qubits: int
    supports_pulse_control: bool
    supports_native_xy: bool
    min_time_step: float | None = None
    max_pulse_duration: float | None = None
    max_pulses: int | None = None
    supported_features: tuple[str, ...] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.provider:
            raise ValueError("provider must be non-empty")
        if not self.backend_name:
            raise ValueError("backend_name must be non-empty")
        if self.n_qubits < 1:
            raise ValueError("n_qubits must be positive")
        if self.min_time_step is not None and self.min_time_step <= 0.0:
            raise ValueError("min_time_step must be positive when provided")
        if self.max_pulse_duration is not None and self.max_pulse_duration <= 0.0:
            raise ValueError("max_pulse_duration must be positive when provided")
        if self.max_pulses is not None and self.max_pulses < 1:
            raise ValueError("max_pulses must be positive when provided")


@dataclass(frozen=True)
class PulseScheduleSummary:
    """Provider-independent summary of an S3 pulse schedule."""

    n_qubits: int
    pulse_count: int
    total_time: float
    max_points_per_pulse: int
    min_sample_spacing: float
    infidelity_bound: float

    def to_dict(self) -> dict[str, float | int]:
        """Serialise the pulse schedule summary."""
        return {
            "n_qubits": self.n_qubits,
            "pulse_count": self.pulse_count,
            "total_time": self.total_time,
            "max_points_per_pulse": self.max_points_per_pulse,
            "min_sample_spacing": self.min_sample_spacing,
            "infidelity_bound": self.infidelity_bound,
        }


@dataclass(frozen=True)
class PulseFeasibilityDecision:
    """No-submit pulse feasibility decision for one provider snapshot."""

    provider: PulseProviderSnapshot
    status: PulseFeasibilityStatus
    reasons: tuple[str, ...]
    schedule: PulseScheduleSummary
    hardware_submission: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialise the decision."""
        return {
            "provider": self.provider.provider,
            "backend_name": self.provider.backend_name,
            "status": self.status,
            "reasons": list(self.reasons),
            "schedule": self.schedule.to_dict(),
            "supports_pulse_control": self.provider.supports_pulse_control,
            "supports_native_xy": self.provider.supports_native_xy,
            "supported_features": list(self.provider.supported_features),
            "hardware_submission": self.hardware_submission,
        }


def summarise_pulse_schedule(schedule: PulseSchedule) -> PulseScheduleSummary:
    """Summarise a pulse schedule for provider feasibility checks."""
    if schedule.n_qubits < 1:
        raise ValueError("schedule n_qubits must be positive")
    max_points = max((len(pulse.times) for pulse in schedule.pulses), default=0)
    spacings = []
    for pulse in schedule.pulses:
        if len(pulse.times) >= 2:
            spacings.append(
                float(
                    min(
                        pulse.times[index + 1] - pulse.times[index]
                        for index in range(len(pulse.times) - 1)
                    )
                )
            )
    min_spacing = min(spacings) if spacings else schedule.total_time
    return PulseScheduleSummary(
        n_qubits=schedule.n_qubits,
        pulse_count=len(schedule.pulses),
        total_time=float(schedule.total_time),
        max_points_per_pulse=int(max_points),
        min_sample_spacing=float(min_spacing),
        infidelity_bound=float(schedule.infidelity_bound),
    )


def assess_pulse_provider_feasibility(
    provider: PulseProviderSnapshot,
    schedule: PulseSchedule,
) -> PulseFeasibilityDecision:
    """Assess one provider snapshot without submitting pulse jobs."""
    summary = summarise_pulse_schedule(schedule)
    reasons: list[str] = []
    if provider.n_qubits < summary.n_qubits:
        reasons.append(
            f"provider has {provider.n_qubits} qubits but schedule requires {summary.n_qubits}"
        )
    if not provider.supports_pulse_control and not provider.supports_native_xy:
        reasons.append("provider declares neither pulse control nor native XY execution")
    if provider.max_pulses is not None and provider.max_pulses < summary.pulse_count:
        reasons.append(
            f"provider max_pulses={provider.max_pulses} below required {summary.pulse_count}"
        )
    if (
        provider.max_pulse_duration is not None
        and provider.max_pulse_duration < summary.total_time
    ):
        reasons.append(
            f"provider max_pulse_duration={provider.max_pulse_duration} below required {summary.total_time}"
        )
    if provider.min_time_step is not None and provider.min_time_step > summary.min_sample_spacing:
        reasons.append(
            f"provider min_time_step={provider.min_time_step} exceeds schedule spacing {summary.min_sample_spacing}"
        )
    if not provider.supported_features:
        status: PulseFeasibilityStatus = "unknown"
        reasons.append("provider did not declare supported_features")
    elif reasons and provider.supports_native_xy:
        status = "manual_review"
    elif reasons:
        status = "blocked"
    else:
        status = "ready"
        reasons.append("provider metadata satisfies S3 pulse schedule requirements")
    return PulseFeasibilityDecision(
        provider=provider,
        status=status,
        reasons=tuple(reasons),
        schedule=summary,
    )


def assess_pulse_provider_fleet(
    providers: Sequence[PulseProviderSnapshot],
    schedule: PulseSchedule,
) -> tuple[PulseFeasibilityDecision, ...]:
    """Assess multiple provider snapshots for the same schedule."""
    return tuple(assess_pulse_provider_feasibility(provider, schedule) for provider in providers)


def pulse_snapshot_from_metadata(metadata: Mapping[str, Any]) -> PulseProviderSnapshot:
    """Build a pulse provider snapshot from provider-neutral metadata."""
    return PulseProviderSnapshot(
        provider=_required_text(metadata, "provider"),
        backend_name=_required_text(metadata, "backend_name"),
        n_qubits=_required_int(metadata, "n_qubits"),
        supports_pulse_control=_required_bool(metadata, "supports_pulse_control"),
        supports_native_xy=_required_bool(metadata, "supports_native_xy"),
        min_time_step=_optional_float(metadata.get("min_time_step"), "min_time_step"),
        max_pulse_duration=_optional_float(
            metadata.get("max_pulse_duration"), "max_pulse_duration"
        ),
        max_pulses=_optional_int(metadata.get("max_pulses"), "max_pulses"),
        supported_features=_string_tuple(metadata.get("supported_features", ())),
        metadata=dict(metadata.get("metadata", {}))
        if isinstance(metadata.get("metadata", {}), Mapping)
        else {},
    )


def _required_text(metadata: Mapping[str, Any], key: str) -> str:
    value = metadata.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be non-empty text")
    return value


def _required_int(metadata: Mapping[str, Any], key: str) -> int:
    value = metadata.get(key)
    if not isinstance(value, int) or value < 1:
        raise ValueError(f"{key} must be a positive integer")
    return value


def _required_bool(metadata: Mapping[str, Any], key: str) -> bool:
    value = metadata.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be boolean")
    return value


def _optional_float(value: Any, key: str) -> float | None:
    if value is None:
        return None
    if not isinstance(value, int | float) or value <= 0.0:
        raise ValueError(f"{key} must be positive when provided")
    return float(value)


def _optional_int(value: Any, key: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or value < 1:
        raise ValueError(f"{key} must be positive when provided")
    return value


def _string_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, tuple | list | set | frozenset):
        raise ValueError("expected a sequence of strings")
    result = tuple(str(item) for item in value)
    if any(not item for item in result):
        raise ValueError("string sequences must not contain empty entries")
    return result
