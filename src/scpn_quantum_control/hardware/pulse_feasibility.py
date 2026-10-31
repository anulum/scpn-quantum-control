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
from math import isfinite
from typing import Any, Literal

from scpn_quantum_control.phase.pulse_shaping import PulseSchedule

PulseFeasibilityStatus = Literal["ready", "blocked", "manual_review", "unknown"]
PulseLimitsSource = Literal["design", "provider_metadata"]

PULSE_DESIGN_CLAIM_BOUNDARY = (
    "design feasibility only: timing grid, amplitude, slew and connectivity were "
    "checked against the declared limits; this is not a device-executable or "
    "calibrated physical-control schedule and confers no submission authority"
)
"""Claim attached to every pulse-design assessment and export."""

_GRID_TOLERANCE = 1.0e-9
_AMPLITUDE_TOLERANCE = 1.0e-12


@dataclass(frozen=True)
class PulseProviderSnapshot:
    """Validated metadata for S3 feasibility, never submission authority.

    Capacities are positive integers excluding booleans; capability flags are
    exact booleans. Optional time limits are finite positive numbers in seconds.
    Invalid identity, capacity, capability or time fields raise ValueError.
    """

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
        """Reject snapshots with unusable identity, capacity, or limits."""
        fields = {
            "provider": self.provider,
            "backend_name": self.backend_name,
            "n_qubits": self.n_qubits,
            "supports_pulse_control": self.supports_pulse_control,
            "supports_native_xy": self.supports_native_xy,
        }
        _required_text(fields, "provider")
        _required_text(fields, "backend_name")
        _required_int(fields, "n_qubits")
        _required_bool(fields, "supports_pulse_control")
        _required_bool(fields, "supports_native_xy")
        _optional_float(self.min_time_step, "min_time_step")
        _optional_float(self.max_pulse_duration, "max_pulse_duration")
        _optional_int(self.max_pulses, "max_pulses")


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


@dataclass(frozen=True)
class PulseDeviceLimits:
    """Declared native limits for converting and checking design pulses.

    Attributes
    ----------
    dt:
        Native sample period in seconds.
    granularity:
        Every start time and duration must be a multiple of this many samples.
    rabi_rate_per_amplitude:
        Rabi rate in rad/s produced by native amplitude 1.0; this is the
        design-to-native unit conversion.
    max_amplitude:
        Largest allowed absolute native amplitude.
    n_qubits:
        Number of physical qubits.
    coupling_map:
        Undirected physical qubit pairs that two-qubit pulses may drive.
    max_slew:
        Largest allowed change of native amplitude between samples, including
        the ramps from and back to zero at the pulse edges; ``None`` disables
        the slew check.
    source:
        ``design`` for limits chosen by the designer, ``provider_metadata`` for
        limits copied from a provider. Neither confers device authority.

    """

    dt: float
    granularity: int
    rabi_rate_per_amplitude: float
    max_amplitude: float
    n_qubits: int
    coupling_map: tuple[tuple[int, int], ...] = ()
    max_slew: float | None = None
    source: PulseLimitsSource = "design"

    def __post_init__(self) -> None:
        """Reject non-physical periods, conversions, limits or couplings."""
        for name in ("dt", "rabi_rate_per_amplitude", "max_amplitude"):
            _required_positive_float(getattr(self, name), name)
        if self.max_slew is not None:
            _required_positive_float(self.max_slew, "max_slew")
        _required_int({"granularity": self.granularity}, "granularity")
        _required_int({"n_qubits": self.n_qubits}, "n_qubits")
        if self.source not in ("design", "provider_metadata"):
            raise ValueError("source must be 'design' or 'provider_metadata'")
        for pair in self.coupling_map:
            if (
                not isinstance(pair, tuple)
                or len(pair) != 2
                or any(isinstance(q, bool) or not isinstance(q, int) for q in pair)
                or not all(0 <= q < self.n_qubits for q in pair)
                or pair[0] == pair[1]
            ):
                raise ValueError("coupling_map pairs must be two distinct physical qubits")


@dataclass(frozen=True)
class DesignPulse:
    """One pulse in design units: seconds and Rabi rate in rad/s.

    Attributes
    ----------
    qubits:
        Logical qubits the pulse drives; one for a drive, two for a coupler.
    start:
        Start time in seconds.
    duration:
        Duration in seconds.
    rabi_rates:
        Piecewise-constant Rabi rate in rad/s over equal segments of the
        duration; a single value is a constant pulse.

    """

    qubits: tuple[int, ...]
    start: float
    duration: float
    rabi_rates: tuple[float, ...]


@dataclass(frozen=True)
class NativePulse:
    """One pulse converted to native samples on a physical channel.

    Attributes
    ----------
    channel:
        ``d<q>`` for a drive or ``u<a>_<b>`` for a coupler.
    physical_qubits:
        Physical qubits after placement.
    t0:
        Start sample.
    samples:
        Native amplitudes, one per sample.

    """

    channel: str
    physical_qubits: tuple[int, ...]
    t0: int
    samples: tuple[float, ...]


@dataclass(frozen=True)
class PulseDesignAssessment:
    """Feasibility of a design against declared native limits.

    Attributes
    ----------
    feasible:
        Whether no violation was found.
    violations:
        ``code: detail`` strings with codes ``off_grid``, ``non_positive_duration``,
        ``amplitude_over_limit``, ``slew_over_limit``, ``unplaced_qubit``,
        ``placement_out_of_range``, ``non_injective_placement``,
        ``disconnected_placement``, ``unsupported_arity``,
        ``segment_mismatch`` and ``channel_overlap``.
    native:
        Native pulses when feasible, otherwise empty.
    limits:
        Limits the design was checked against.
    device_authority:
        Always ``False``: design or copied limits never authorise a device.
    hardware_submission:
        Always ``False``: assessment never submits.
    claim_boundary:
        :data:`PULSE_DESIGN_CLAIM_BOUNDARY`.

    """

    feasible: bool
    violations: tuple[str, ...]
    native: tuple[NativePulse, ...]
    limits: PulseDeviceLimits
    device_authority: bool = False
    hardware_submission: bool = False
    claim_boundary: str = PULSE_DESIGN_CLAIM_BOUNDARY


def assess_pulse_design(
    pulses: Sequence[DesignPulse],
    limits: PulseDeviceLimits,
    placement: Mapping[int, int],
) -> PulseDesignAssessment:
    """Check timing grid, amplitude, slew and placement, then convert units.

    Parameters
    ----------
    pulses:
        Design pulses in seconds and rad/s.
    limits:
        Declared native limits and the design-to-native conversion.
    placement:
        Logical-to-physical qubit map.

    Returns
    -------
    PulseDesignAssessment
        Every violation found, and native pulses only when there is none.

    Raises
    ------
    ValueError
        If a pulse is not a :class:`DesignPulse` or has non-finite fields.

    """
    violations: list[str] = []
    native: list[NativePulse] = []
    physical_values = list(placement.values())
    if len(set(physical_values)) != len(physical_values):
        violations.append("non_injective_placement: two logical qubits share a physical qubit")
    edges = {frozenset(pair) for pair in limits.coupling_map}
    occupied: dict[str, list[tuple[int, int]]] = {}
    for index, pulse in enumerate(pulses):
        if not isinstance(pulse, DesignPulse):
            raise ValueError("pulses must be DesignPulse records")
        values = (pulse.start, pulse.duration, *pulse.rabi_rates)
        if not pulse.rabi_rates or any(
            isinstance(value, bool) or not isinstance(value, int | float) or not isfinite(value)
            for value in values
        ):
            raise ValueError(f"pulse {index} needs finite start, duration and rabi_rates")
        start_samples = _grid_samples(pulse.start, limits)
        duration_samples = _grid_samples(pulse.duration, limits)
        if pulse.duration <= 0.0:
            violations.append(f"non_positive_duration: pulse {index}")
            continue
        if start_samples is None or duration_samples is None:
            violations.append(
                f"off_grid: pulse {index} start {pulse.start / limits.dt:.6g} and duration "
                f"{pulse.duration / limits.dt:.6g} samples are not multiples of "
                f"{limits.granularity}"
            )
            continue
        if duration_samples % len(pulse.rabi_rates):
            violations.append(
                f"segment_mismatch: pulse {index} has {duration_samples} samples for "
                f"{len(pulse.rabi_rates)} segments"
            )
            continue
        physical = _place(index, pulse.qubits, placement, limits, edges, violations)
        repeat = duration_samples // len(pulse.rabi_rates)
        samples = tuple(
            rate / limits.rabi_rate_per_amplitude
            for rate in pulse.rabi_rates
            for _ in range(repeat)
        )
        peak = max(abs(sample) for sample in samples)
        if peak > limits.max_amplitude * (1.0 + _AMPLITUDE_TOLERANCE):
            violations.append(
                f"amplitude_over_limit: pulse {index} peak {peak:.12g} exceeds "
                f"{limits.max_amplitude:.12g}"
            )
        if limits.max_slew is not None:
            edged = (0.0, *samples, 0.0)
            slew = max(
                abs(after - before) for before, after in zip(edged[:-1], edged[1:], strict=True)
            )
            if slew > limits.max_slew * (1.0 + _AMPLITUDE_TOLERANCE):
                violations.append(
                    f"slew_over_limit: pulse {index} step {slew:.12g} exceeds "
                    f"{limits.max_slew:.12g}"
                )
        if physical is None:
            continue
        channel = f"d{physical[0]}" if len(physical) == 1 else f"u{min(physical)}_{max(physical)}"
        window = (start_samples, start_samples + duration_samples)
        for other in occupied.get(channel, []):
            if window[0] < other[1] and other[0] < window[1]:
                violations.append(f"channel_overlap: pulse {index} overlaps on {channel}")
        occupied.setdefault(channel, []).append(window)
        native.append(NativePulse(channel, physical, start_samples, samples))
    feasible = not violations
    return PulseDesignAssessment(
        feasible=feasible,
        violations=tuple(violations),
        native=tuple(native) if feasible else (),
        limits=limits,
    )


def export_native_pulse_program(assessment: PulseDesignAssessment) -> dict[str, Any]:
    """Serialise a feasible assessment without contacting any provider.

    Parameters
    ----------
    assessment:
        Feasible result of :func:`assess_pulse_design`.

    Returns
    -------
    dict
        Native program payload with ``submissions`` equal to zero, no device
        authority and the design claim boundary.

    Raises
    ------
    ValueError
        If the assessment is not feasible.

    """
    if not assessment.feasible:
        raise ValueError("only a feasible pulse design can be exported")
    return {
        "schema": "native_pulse_design_v1",
        "dt": assessment.limits.dt,
        "limits_source": assessment.limits.source,
        "pulses": [
            {
                "channel": pulse.channel,
                "physical_qubits": list(pulse.physical_qubits),
                "t0": pulse.t0,
                "samples": list(pulse.samples),
            }
            for pulse in assessment.native
        ],
        "device_authority": False,
        "hardware_submission": False,
        "submissions": 0,
        "claim_boundary": assessment.claim_boundary,
    }


def _grid_samples(seconds: float, limits: PulseDeviceLimits) -> int | None:
    """Return a whole number of samples on the granularity grid, else ``None``."""
    samples = seconds / limits.dt
    nearest = round(samples)
    if abs(samples - nearest) > _GRID_TOLERANCE * max(1.0, abs(samples)):
        return None
    if nearest % limits.granularity:
        return None
    return int(nearest)


def _place(
    index: int,
    qubits: tuple[int, ...],
    placement: Mapping[int, int],
    limits: PulseDeviceLimits,
    edges: set[frozenset[int]],
    violations: list[str],
) -> tuple[int, ...] | None:
    """Map logical qubits to physical ones and record placement violations."""
    if len(qubits) not in (1, 2):
        violations.append(f"unsupported_arity: pulse {index} drives {len(qubits)} qubits")
        return None
    missing = [qubit for qubit in qubits if qubit not in placement]
    if missing:
        violations.append(f"unplaced_qubit: pulse {index} logical {missing}")
        return None
    physical = tuple(placement[qubit] for qubit in qubits)
    if not all(
        not isinstance(qubit, bool) and isinstance(qubit, int) and 0 <= qubit < limits.n_qubits
        for qubit in physical
    ):
        violations.append(f"placement_out_of_range: pulse {index} physical {list(physical)}")
        return None
    if len(physical) == 2 and frozenset(physical) not in edges:
        violations.append(
            f"disconnected_placement: pulse {index} physical {list(physical)} is not coupled"
        )
        return None
    return physical


def _required_positive_float(value: Any, key: str) -> float:
    """Return a finite positive float or refuse."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{key} must be a finite positive number")
    numeric = float(value)
    if not isfinite(numeric) or numeric <= 0.0:
        raise ValueError(f"{key} must be a finite positive number")
    return numeric


def _required_text(metadata: Mapping[str, Any], key: str) -> str:
    value = metadata.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be non-empty text")
    return value


def _required_int(metadata: Mapping[str, Any], key: str) -> int:
    value = metadata.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
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
    if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0.0:
        raise ValueError(f"{key} must be positive when provided")
    try:
        numeric = float(value)
    except OverflowError as exc:
        raise ValueError(f"{key} must be finite when provided") from exc
    if not isfinite(numeric):
        raise ValueError(f"{key} must be finite when provided")
    return numeric


def _optional_int(value: Any, key: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{key} must be positive when provided")
    return value


def _string_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, tuple | list | set | frozenset):
        raise ValueError("expected a sequence of strings")
    if any(not isinstance(item, str) for item in value):
        raise ValueError("string sequences must contain only text entries")
    result = tuple(value)
    if any(not item for item in result):
        raise ValueError("string sequences must not contain empty entries")
    return result
