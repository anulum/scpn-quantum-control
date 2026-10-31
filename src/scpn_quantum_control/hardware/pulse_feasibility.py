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


_MAX_PULSE_SAMPLES = 1 << 24
"""Longest pulse, in samples, that is converted to native samples."""
_MAX_SAMPLE_INDEX = 1 << 36
"""Largest start or end sample; below it the grid tolerance stays far under half a sample."""
_GRID_RELATIVE_TOLERANCE = 1.0e-12
"""Rounding admitted per sample of distance from zero, for starts summed from many durations."""
_MAX_DESIGN_SAMPLES = 1 << 26
"""Most native samples one design may convert in total, across all of its pulses."""


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
        Rabi rate in rad/s produced by native amplitude 1.0 on a single-qubit
        drive channel; this is the design-to-native unit conversion.
    max_amplitude:
        Largest allowed absolute native amplitude.
    n_qubits:
        Number of physical qubits.
    coupling_map:
        Undirected physical qubit pairs that two-qubit pulses may drive.
    max_slew:
        Largest allowed change of native amplitude between consecutive samples
        of a channel's whole waveform. The channel is zero outside its pulses,
        so the step from and back to zero at a pulse edge counts, and two
        pulses that meet on a channel are checked across their junction.
        ``None`` disables the slew check.
    source:
        ``design`` for limits chosen by the designer, ``provider_metadata`` for
        limits copied from a provider snapshot. Neither confers device authority.
    coupler_rate_per_amplitude:
        Coupling rate in rad/s produced by native amplitude 1.0 on a two-qubit
        coupler channel. It only converts a coupler design rate into a native
        amplitude; no two-qubit Hamiltonian form is claimed. ``None`` means
        coupler pulses cannot be converted and are reported.
    snapshot:
        The provider snapshot ``provider_metadata`` limits were copied from.
        It is required for that source and refused for ``design``. Its
        ``min_time_step`` and qubit count must equal ``dt`` and ``n_qubits``,
        it must declare pulse control, and its pulse-count and pulse-duration
        limits are checked by :func:`assess_pulse_design`.

    """

    dt: float
    granularity: int
    rabi_rate_per_amplitude: float
    max_amplitude: float
    n_qubits: int
    coupling_map: tuple[tuple[int, int], ...] = ()
    max_slew: float | None = None
    source: PulseLimitsSource = "design"
    coupler_rate_per_amplitude: float | None = None
    snapshot: PulseProviderSnapshot | None = None

    def __post_init__(self) -> None:
        """Reject non-physical periods, conversions, limits, couplings or sources."""
        for name in ("dt", "rabi_rate_per_amplitude", "max_amplitude"):
            _required_positive_float(getattr(self, name), name)
        for name in ("max_slew", "coupler_rate_per_amplitude"):
            if getattr(self, name) is not None:
                _required_positive_float(getattr(self, name), name)
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
        if self.source == "design":
            if self.snapshot is not None:
                raise ValueError("design limits cannot carry a provider snapshot")
            return
        if not isinstance(self.snapshot, PulseProviderSnapshot):
            raise ValueError(
                "provider_metadata limits must carry the PulseProviderSnapshot they were copied from"
            )
        if not self.snapshot.supports_pulse_control:
            raise ValueError("the provider snapshot does not declare pulse control")
        if self.snapshot.min_time_step != self.dt:
            raise ValueError("dt must equal the provider snapshot's min_time_step")
        if self.snapshot.n_qubits != self.n_qubits:
            raise ValueError("n_qubits must equal the provider snapshot's qubit count")


def pulse_limits_from_snapshot(
    snapshot: PulseProviderSnapshot,
    *,
    granularity: int,
    rabi_rate_per_amplitude: float,
    max_amplitude: float,
    coupling_map: tuple[tuple[int, int], ...] = (),
    max_slew: float | None = None,
    coupler_rate_per_amplitude: float | None = None,
) -> PulseDeviceLimits:
    """Copy the sample period and qubit count of a provider snapshot into limits.

    Parameters
    ----------
    snapshot:
        Validated provider snapshot; it must declare pulse control and a
        ``min_time_step``, which becomes the native sample period.
    granularity, rabi_rate_per_amplitude, max_amplitude, coupling_map, max_slew, coupler_rate_per_amplitude:
        The remaining limits, as for :class:`PulseDeviceLimits`.

    Returns
    -------
    PulseDeviceLimits
        Limits with ``source="provider_metadata"`` bound to ``snapshot``.

    Raises
    ------
    ValueError
        If the snapshot declares no ``min_time_step`` or no pulse control, or a
        limit is invalid.

    """
    if snapshot.min_time_step is None:
        raise ValueError("the provider snapshot declares no min_time_step to use as dt")
    return PulseDeviceLimits(
        dt=snapshot.min_time_step,
        granularity=granularity,
        rabi_rate_per_amplitude=rabi_rate_per_amplitude,
        max_amplitude=max_amplitude,
        n_qubits=snapshot.n_qubits,
        coupling_map=coupling_map,
        max_slew=max_slew,
        source="provider_metadata",
        coupler_rate_per_amplitude=coupler_rate_per_amplitude,
        snapshot=snapshot,
    )


@dataclass(frozen=True)
class DesignPulse:
    """One pulse in design units: seconds and rates in rad/s.

    Attributes
    ----------
    qubits:
        Logical qubits the pulse drives; one for a drive, two for a coupler.
    start:
        Start time in seconds, at or after zero.
    duration:
        Duration in seconds.
    rabi_rates:
        Piecewise-constant rate in rad/s over equal segments of the duration:
        a Rabi rate on a drive, a coupling rate on a coupler. A single value is
        a constant pulse.

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
        ``code: detail`` strings with codes ``negative_start``, ``off_grid``,
        ``non_positive_duration``, ``segment_mismatch``, ``duration_over_limit``,
        ``amplitude_over_limit``, ``slew_over_limit``, ``unplaced_qubit``,
        ``repeated_qubit``, ``placement_out_of_range``,
        ``non_injective_placement``, ``disconnected_placement``,
        ``unsupported_arity``, ``undefined_coupler_conversion``,
        ``channel_overlap`` and ``too_many_pulses``.
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


@dataclass(frozen=True)
class _ConvertedPulse:
    """A placed, on-grid pulse and the amplitudes at its edges."""

    index: int
    native: NativePulse

    @property
    def end(self) -> int:
        """Return the first sample after the pulse."""
        return self.native.t0 + len(self.native.samples)


def assess_pulse_design(
    pulses: Sequence[DesignPulse],
    limits: PulseDeviceLimits,
    placement: Mapping[int, int],
) -> PulseDesignAssessment:
    """Check timing grid, amplitude, slew and placement, then convert units.

    Every pulse is checked for everything that does not depend on a failed
    check: an off-grid pulse is still checked for amplitude, slew between its
    segments and placement. Slew at pulse edges is checked on each channel's
    whole waveform, so pulses that meet on a channel are checked across the
    junction.

    Parameters
    ----------
    pulses:
        Design pulses in seconds and rad/s.
    limits:
        Declared native limits and the design-to-native conversions.
    placement:
        Logical-to-physical qubit map.

    Returns
    -------
    PulseDesignAssessment
        Every violation found, and native pulses only when there is none.

    Raises
    ------
    ValueError
        If a pulse is not a :class:`DesignPulse`, has non-finite fields or
        non-integer qubits, starts or ends beyond ``2**36`` samples, or is
        longer than ``2**24`` samples, or the pulses together are longer than
        ``2**26`` samples.

    """
    violations: list[str] = []
    physical_values = list(placement.values())
    if len(set(physical_values)) != len(physical_values):
        violations.append("non_injective_placement: two logical qubits share a physical qubit")
    edges = {frozenset(pair) for pair in limits.coupling_map}
    occupied: dict[str, list[tuple[int, int]]] = {}
    converted: list[_ConvertedPulse] = []
    total_samples = 0.0
    for index, pulse in enumerate(pulses):
        _validate_design_pulse(index, pulse, limits)
        total_samples += max(pulse.duration / limits.dt, 0.0)
        if total_samples > _MAX_DESIGN_SAMPLES:
            raise ValueError(f"the design converts more than {_MAX_DESIGN_SAMPLES} samples")
        result = _assess_one(index, pulse, limits, placement, edges, violations)
        if result is None:
            continue
        window = (result.native.t0, result.end)
        for other in occupied.get(result.native.channel, []):
            if window[0] < other[1] and other[0] < window[1]:
                violations.append(
                    f"channel_overlap: pulse {index} overlaps on {result.native.channel}"
                )
        occupied.setdefault(result.native.channel, []).append(window)
        converted.append(result)
    if limits.max_slew is not None:
        _check_channel_edges(converted, limits.max_slew, violations)
    snapshot = limits.snapshot
    if (
        snapshot is not None
        and snapshot.max_pulses is not None
        and len(pulses) > snapshot.max_pulses
    ):
        violations.append(
            f"too_many_pulses: {len(pulses)} pulses exceed the provider's {snapshot.max_pulses}"
        )
    feasible = not violations
    return PulseDesignAssessment(
        feasible=feasible,
        violations=tuple(violations),
        native=tuple(item.native for item in converted) if feasible else (),
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
        Native program payload with the limits and conversions needed to read
        its samples in physical units, the provider snapshot identity when the
        limits were copied, ``submissions`` equal to zero, no device authority
        and the design claim boundary.

    Raises
    ------
    ValueError
        If the assessment is not feasible.

    """
    if not assessment.feasible:
        raise ValueError("only a feasible pulse design can be exported")
    limits = assessment.limits
    snapshot = limits.snapshot
    return {
        "schema": "native_pulse_design_v1",
        "dt": limits.dt,
        "limits_source": limits.source,
        "limits": {
            "granularity": limits.granularity,
            "rabi_rate_per_amplitude": limits.rabi_rate_per_amplitude,
            "coupler_rate_per_amplitude": limits.coupler_rate_per_amplitude,
            "max_amplitude": limits.max_amplitude,
            "max_slew": limits.max_slew,
            "n_qubits": limits.n_qubits,
            "coupling_map": [list(pair) for pair in limits.coupling_map],
        },
        "provider_snapshot": None
        if snapshot is None
        else {"provider": snapshot.provider, "backend_name": snapshot.backend_name},
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


def _validate_design_pulse(index: int, pulse: DesignPulse, limits: PulseDeviceLimits) -> None:
    """Refuse a record that cannot be checked or would exceed the conversion budget."""
    if not isinstance(pulse, DesignPulse):
        raise ValueError("pulses must be DesignPulse records")
    if not isinstance(pulse.qubits, tuple) or any(
        isinstance(qubit, bool) or not isinstance(qubit, int) for qubit in pulse.qubits
    ):
        raise ValueError(f"pulse {index} qubits must be a tuple of integers")
    values = (pulse.start, pulse.duration, *pulse.rabi_rates)
    if not pulse.rabi_rates or any(
        isinstance(value, bool) or not isinstance(value, int | float) or not isfinite(value)
        for value in values
    ):
        raise ValueError(f"pulse {index} needs finite start, duration and rabi_rates")
    for name, seconds in (("start", pulse.start), ("end", pulse.start + pulse.duration)):
        if not abs(seconds / limits.dt) < _MAX_SAMPLE_INDEX:
            raise ValueError(f"pulse {index} {name} lies beyond {_MAX_SAMPLE_INDEX} samples")
    if pulse.duration / limits.dt > _MAX_PULSE_SAMPLES:
        raise ValueError(f"pulse {index} is longer than {_MAX_PULSE_SAMPLES} samples")


def _assess_one(
    index: int,
    pulse: DesignPulse,
    limits: PulseDeviceLimits,
    placement: Mapping[int, int],
    edges: set[frozenset[int]],
    violations: list[str],
) -> _ConvertedPulse | None:
    """Record one pulse's violations; return it converted when timing and placement hold."""
    convertible = True
    if pulse.start / limits.dt < -_GRID_TOLERANCE:
        violations.append(
            f"negative_start: pulse {index} starts at sample {pulse.start / limits.dt:.6g}"
        )
        convertible = False
    start_samples = _grid_samples(pulse.start, limits)
    duration_samples = _grid_samples(pulse.duration, limits)
    segments = len(pulse.rabi_rates)
    # A duration within the grid tolerance of zero samples is as empty as a zero one.
    if pulse.duration <= 0.0 or duration_samples == 0:
        violations.append(f"non_positive_duration: pulse {index}")
        convertible = False
    elif start_samples is None or duration_samples is None:
        violations.append(
            f"off_grid: pulse {index} start {pulse.start / limits.dt:.6g} and duration "
            f"{pulse.duration / limits.dt:.6g} samples are not multiples of "
            f"{limits.granularity}"
        )
        convertible = False
    elif duration_samples % segments:
        violations.append(
            f"segment_mismatch: pulse {index} has {duration_samples} samples for "
            f"{segments} segments"
        )
        convertible = False
    snapshot = limits.snapshot
    if (
        snapshot is not None
        and snapshot.max_pulse_duration is not None
        and pulse.duration > snapshot.max_pulse_duration * (1.0 + _AMPLITUDE_TOLERANCE)
    ):
        violations.append(
            f"duration_over_limit: pulse {index} lasts {pulse.duration:.12g} s, over the "
            f"provider's {snapshot.max_pulse_duration:.12g} s"
        )
    physical = _place(index, pulse.qubits, placement, limits, edges, violations)
    scale = _conversion(index, pulse.qubits, limits, violations)
    if scale is None:
        return None
    amplitudes = tuple(rate / scale for rate in pulse.rabi_rates)
    peak = max(abs(amplitude) for amplitude in amplitudes)
    if peak > limits.max_amplitude * (1.0 + _AMPLITUDE_TOLERANCE):
        violations.append(
            f"amplitude_over_limit: pulse {index} peak {peak:.12g} exceeds "
            f"{limits.max_amplitude:.12g}"
        )
    placed = convertible and physical is not None
    if limits.max_slew is not None:
        # Edges of a placed pulse are checked on its channel's whole waveform.
        steps = amplitudes if placed else (0.0, *amplitudes, 0.0)
        slew = max(
            (abs(after - before) for before, after in zip(steps[:-1], steps[1:], strict=True)),
            default=0.0,
        )
        if slew > limits.max_slew * (1.0 + _AMPLITUDE_TOLERANCE):
            violations.append(
                f"slew_over_limit: pulse {index} step {slew:.12g} exceeds {limits.max_slew:.12g}"
            )
    if not placed or physical is None or start_samples is None or duration_samples is None:
        return None
    repeat = duration_samples // segments
    channel = f"d{physical[0]}" if len(physical) == 1 else f"u{min(physical)}_{max(physical)}"
    samples = tuple(amplitude for amplitude in amplitudes for _ in range(repeat))
    return _ConvertedPulse(index, NativePulse(channel, physical, start_samples, samples))


def _conversion(
    index: int, qubits: tuple[int, ...], limits: PulseDeviceLimits, violations: list[str]
) -> float | None:
    """Return the rate of native amplitude 1.0 for the pulse's channel kind."""
    if len(qubits) == 1:
        return limits.rabi_rate_per_amplitude
    if len(qubits) != 2:
        return None
    if limits.coupler_rate_per_amplitude is None:
        violations.append(
            f"undefined_coupler_conversion: pulse {index} drives a coupler but the limits "
            "declare no coupler_rate_per_amplitude"
        )
    return limits.coupler_rate_per_amplitude


def _check_channel_edges(
    converted: list[_ConvertedPulse], max_slew: float, violations: list[str]
) -> None:
    """Check each channel's waveform where it leaves or returns to zero, or two pulses meet."""
    channels: dict[str, list[_ConvertedPulse]] = {}
    for item in converted:
        channels.setdefault(item.native.channel, []).append(item)
    for channel, members in channels.items():
        ordered = sorted(members, key=lambda item: (item.native.t0, item.index))
        for position, item in enumerate(ordered):
            previous = ordered[position - 1] if position else None
            following = ordered[position + 1] if position + 1 < len(ordered) else None
            before = (
                previous.native.samples[-1]
                if previous is not None and previous.end == item.native.t0
                else 0.0
            )
            checks = [(item.native.t0, abs(item.native.samples[0] - before))]
            if following is None or following.native.t0 != item.end:
                checks.append((item.end, abs(item.native.samples[-1])))
            for sample, step in checks:
                if step > max_slew * (1.0 + _AMPLITUDE_TOLERANCE):
                    violations.append(
                        f"slew_over_limit: {channel} steps {step:.12g} at sample {sample} "
                        f"(pulse {item.index}), exceeding {max_slew:.12g}"
                    )


def _grid_samples(seconds: float, limits: PulseDeviceLimits) -> int | None:
    """Return a whole number of samples on the granularity grid, else ``None``.

    The admitted rounding, ``1e-9 + 1e-12 * |samples|``, stays below a tenth of
    a sample for every start and end below ``2**36`` samples.
    """
    samples = seconds / limits.dt
    nearest = round(samples)
    if abs(samples - nearest) > _GRID_TOLERANCE + _GRID_RELATIVE_TOLERANCE * abs(samples):
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
    if len(set(qubits)) != len(qubits):
        violations.append(f"repeated_qubit: pulse {index} drives logical {list(qubits)}")
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
