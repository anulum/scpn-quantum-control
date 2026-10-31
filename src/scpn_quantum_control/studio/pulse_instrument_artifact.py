# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — committed data for the Studio pulse and analog feasibility instrument
"""Build, write and check the data the Studio pulse instrument shows.

Every case is a stated pulse design checked by :func:`assess_pulse_design`
against stated limits, either chosen by the designer or copied from a dated
provider capability snapshot with :func:`pulse_limits_from_snapshot`. Each
case keeps:

* the design pulses in seconds and rad/s, with their logical qubits and the
  logical-to-physical placement and coupling map;
* every violation as the owner wrote it, with its code and the pulse or
  channel it names, so the view can mark the source span;
* the native pulses in samples and native amplitude when the design is
  feasible;
* a no-submit feasibility dossier from :func:`export_native_pulse_program`,
  only when the design is feasible and its capability snapshot, if any, was
  valid at the assessment time; otherwise the reason it is refused.

Snapshot validity is decided here against the fixed assessment time, with the
same exact UTC-second parsing as the operator policy. Nothing is submitted,
and no case confers device authority. The browser selects among these values
and computes none of them.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from ..hardware.operator_policy_contracts import utc_second
from ..hardware.pulse_feasibility import (
    PULSE_DESIGN_CLAIM_BOUNDARY,
    DesignPulse,
    PulseDeviceLimits,
    assess_pulse_design,
    export_native_pulse_program,
    pulse_limits_from_snapshot,
    pulse_snapshot_from_metadata,
)
from .committed_data import differences, run_committed_data_cli, serialise

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[3]
PULSE_INSTRUMENT_SCHEMA: Final[str] = "scpn_qc_studio_pulse_instrument_v1"
PULSE_INSTRUMENT_ARTIFACT_ID: Final[str] = "studio-pulse-instrument-20261031"
DEFAULT_PULSE_INSTRUMENT_JSON_PATH: Final[Path] = Path(
    "data/studio/pulse_instrument_20261031.json"
)
ASSESSED_AT: Final[str] = "2026-10-31T12:00:00Z"
"""Fixed assessment time; a snapshot that expired before it refuses the dossier."""
PULSE_INSTRUMENT_CLAIM_BOUNDARY: Final[str] = (
    "Stated designs checked against stated or snapshot-copied limits; a feasible "
    "design is a no-submit dossier in design units, not a device-executable or "
    "calibrated schedule, and no case contacts or authorises a provider"
)

_VIOLATION = re.compile(
    r"^(?P<code>[a-z_]+): (?:pulse (?P<pulse>\d+)\b|(?P<channel>[du][0-9_]+) )?"
)


@dataclass(frozen=True)
class PulseCase:
    """One stated design and the limits it is checked against.

    Parameters
    ----------
    case_id
        Stable identifier.
    title
        What the case demonstrates.
    pulses
        Design pulses in seconds and rad/s.
    placement
        Logical-to-physical qubit map.
    limits
        Declared limits; ``None`` when ``snapshot`` supplies them.
    snapshot
        Provider-neutral capability metadata with ``observed_at`` and
        ``expires_at``, from which the sample period and qubit count are copied.
    snapshot_limits
        The remaining limits used with ``snapshot``.

    """

    case_id: str
    title: str
    pulses: tuple[DesignPulse, ...]
    placement: Mapping[int, int]
    limits: PulseDeviceLimits | None = None
    snapshot: Mapping[str, Any] | None = None
    snapshot_limits: Mapping[str, Any] | None = None


_DT: Final[float] = 2.0e-9
_RABI: Final[float] = 1.0e8
_DESIGN_LIMITS: Final[PulseDeviceLimits] = PulseDeviceLimits(
    dt=_DT,
    granularity=4,
    rabi_rate_per_amplitude=_RABI,
    max_amplitude=1.0,
    n_qubits=3,
    coupling_map=((0, 1), (1, 2)),
    max_slew=0.6,
    coupler_rate_per_amplitude=2.0e7,
)


def _snapshot(expires_at: str) -> dict[str, Any]:
    """Return dated provider-neutral capability metadata for three qubits."""
    return {
        "provider": "example-provider",
        "backend_name": "example-pulse-backend",
        "n_qubits": 3,
        "supports_pulse_control": True,
        "supports_native_xy": False,
        "min_time_step": _DT,
        "max_pulse_duration": 4.0e-6,
        "max_pulses": 8,
        "supported_features": ["pulse"],
        "metadata": {"observed_at": "2026-10-30T09:00:00Z", "expires_at": expires_at},
    }


_SNAPSHOT_LIMITS: Final[dict[str, Any]] = {
    "granularity": 4,
    "rabi_rate_per_amplitude": _RABI,
    "max_amplitude": 1.0,
    "coupling_map": ((0, 1), (1, 2)),
    "max_slew": 0.6,
    "coupler_rate_per_amplitude": 2.0e7,
}

_FEASIBLE_PULSES: Final[tuple[DesignPulse, ...]] = (
    DesignPulse((0,), 0.0, 64e-9, (2.0e7, 4.0e7, 4.0e7, 2.0e7)),
    DesignPulse((1,), 16e-9, 48e-9, (3.0e7,)),
    DesignPulse((0, 1), 96e-9, 32e-9, (1.0e7,)),
)

PULSE_CASES: Final[tuple[PulseCase, ...]] = (
    PulseCase(
        "feasible-design",
        "Two drives and a coupler on declared design limits export a no-submit dossier",
        _FEASIBLE_PULSES,
        {0: 0, 1: 1},
        limits=_DESIGN_LIMITS,
    ),
    PulseCase(
        "half-grid-and-over-amplitude",
        "A start half a grid step off and a segment above the amplitude limit are marked",
        (
            DesignPulse((0,), 4e-9, 64e-9, (2.0e7,)),
            DesignPulse((1,), 0.0, 48e-9, (2.0e7, 1.5e8, 2.0e7)),
        ),
        {0: 0, 1: 1},
        limits=_DESIGN_LIMITS,
    ),
    PulseCase(
        "disconnected-coupler",
        "A coupler between qubits that the coupling map does not join is refused",
        (DesignPulse((0, 1), 0.0, 32e-9, (1.0e7,)),),
        {0: 0, 1: 2},
        limits=_DESIGN_LIMITS,
    ),
    PulseCase(
        "current-snapshot",
        "Limits copied from a capability snapshot valid at the assessment time",
        _FEASIBLE_PULSES,
        {0: 0, 1: 1},
        snapshot=_snapshot("2026-11-30T09:00:00Z"),
        snapshot_limits=_SNAPSHOT_LIMITS,
    ),
    PulseCase(
        "expired-snapshot",
        "The same design against a snapshot that expired before the assessment",
        _FEASIBLE_PULSES,
        {0: 0, 1: 1},
        snapshot=_snapshot("2026-10-30T21:00:00Z"),
        snapshot_limits=_SNAPSHOT_LIMITS,
    ),
)
"""Instrument cases; each repeats exactly from its stated design and limits."""


def _violation(text: str) -> dict[str, object]:
    """Return a violation with its code and the pulse or channel it names."""
    match = _VIOLATION.match(text)
    if match is None:
        raise ValueError(f"violation does not start with a code: {text!r}")
    pulse = match.group("pulse")
    return {
        "code": match.group("code"),
        "pulse": None if pulse is None else int(pulse),
        "channel": match.group("channel"),
        "text": text,
    }


def _limits(limits: PulseDeviceLimits) -> dict[str, object]:
    """Return the limits in the units the view shows."""
    return {
        "source": limits.source,
        "dt_seconds": limits.dt,
        "granularity_samples": limits.granularity,
        "rabi_rate_per_amplitude": limits.rabi_rate_per_amplitude,
        "coupler_rate_per_amplitude": limits.coupler_rate_per_amplitude,
        "max_amplitude": limits.max_amplitude,
        "max_slew": limits.max_slew,
        "n_qubits": limits.n_qubits,
        "coupling_map": [list(pair) for pair in limits.coupling_map],
    }


def _case(case: PulseCase) -> dict[str, object]:
    """Assess one case and return everything the view shows."""
    snapshot_view: dict[str, object] | None = None
    dossier_refusal: str | None = None
    if case.snapshot is not None and case.snapshot_limits is not None:
        snapshot = pulse_snapshot_from_metadata(case.snapshot)
        limits = pulse_limits_from_snapshot(snapshot, **case.snapshot_limits)
        observed_at = str(snapshot.metadata["observed_at"])
        expires_at = str(snapshot.metadata["expires_at"])
        valid = utc_second(observed_at) <= utc_second(ASSESSED_AT) < utc_second(expires_at)
        snapshot_view = {
            "provider": snapshot.provider,
            "backend_name": snapshot.backend_name,
            "observed_at": observed_at,
            "expires_at": expires_at,
            "valid_at_assessment": valid,
            "max_pulses": snapshot.max_pulses,
            "max_pulse_duration_seconds": snapshot.max_pulse_duration,
        }
        if utc_second(ASSESSED_AT) < utc_second(observed_at):
            dossier_refusal = (
                f"the capability snapshot was observed at {observed_at}, after the assessment "
                f"at {ASSESSED_AT}"
            )
        elif not valid:
            dossier_refusal = (
                f"the capability snapshot expired at {expires_at}, before the assessment "
                f"at {ASSESSED_AT}"
            )
    elif case.limits is not None:
        limits = case.limits
    else:
        raise ValueError(f"case {case.case_id} has neither limits nor a snapshot")
    assessment = assess_pulse_design(case.pulses, limits, case.placement)
    if not assessment.feasible:
        dossier_refusal = "the design has violations; only a feasible design is exported"
    return {
        "case_id": case.case_id,
        "title": case.title,
        "placement": [
            {"logical": logical, "physical": physical}
            for logical, physical in sorted(case.placement.items())
        ],
        "limits": _limits(limits),
        "snapshot": snapshot_view,
        "design": [
            {
                "index": index,
                "qubits": list(pulse.qubits),
                "start_seconds": pulse.start,
                "duration_seconds": pulse.duration,
                "rates_rad_per_second": list(pulse.rabi_rates),
            }
            for index, pulse in enumerate(case.pulses)
        ],
        "feasible": assessment.feasible,
        "violations": [_violation(text) for text in assessment.violations],
        "native": [
            {
                "channel": pulse.channel,
                "physical_qubits": list(pulse.physical_qubits),
                "t0": pulse.t0,
                "samples": list(pulse.samples),
            }
            for pulse in assessment.native
        ],
        "device_authority": assessment.device_authority,
        "hardware_submission": assessment.hardware_submission,
        "dossier": None
        if dossier_refusal is not None
        else export_native_pulse_program(assessment),
        "dossier_refusal": dossier_refusal,
    }


def build_pulse_instrument_artifact() -> dict[str, object]:
    """Build the committed instrument data from its source owners.

    Returns
    -------
    dict[str, object]
        JSON-ready payload with one entry per case.

    """
    return {
        "schema": PULSE_INSTRUMENT_SCHEMA,
        "artifact_id": PULSE_INSTRUMENT_ARTIFACT_ID,
        "generated_by": "python -m scpn_quantum_control.studio.pulse_instrument_artifact --write",
        "claim_boundary": PULSE_INSTRUMENT_CLAIM_BOUNDARY,
        "design_claim_boundary": PULSE_DESIGN_CLAIM_BOUNDARY,
        "assessed_at": ASSESSED_AT,
        "cases": [_case(case) for case in PULSE_CASES],
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Print, write or check the committed instrument data.

    Parameters
    ----------
    argv
        Optional argument vector, defaulting to ``sys.argv[1:]``.

    Returns
    -------
    int
        ``0`` on success, ``1`` when the committed file differs from a fresh
        build, ``2`` when it cannot be read.

    """
    return run_committed_data_cli(
        argv,
        build=build_pulse_instrument_artifact,
        default_path=REPO_ROOT / DEFAULT_PULSE_INSTRUMENT_JSON_PATH,
        label="pulse instrument data",
        description=__doc__,
    )


__all__ = [
    "ASSESSED_AT",
    "DEFAULT_PULSE_INSTRUMENT_JSON_PATH",
    "PULSE_CASES",
    "PULSE_INSTRUMENT_ARTIFACT_ID",
    "PULSE_INSTRUMENT_CLAIM_BOUNDARY",
    "PULSE_INSTRUMENT_SCHEMA",
    "PulseCase",
    "build_pulse_instrument_artifact",
    "differences",
    "main",
    "serialise",
]


if __name__ == "__main__":
    raise SystemExit(main())
