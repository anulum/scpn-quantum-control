# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — host readiness gate for isolated_affinity benchmarks
"""Decide whether a host can produce ``isolated_affinity`` benchmark evidence.

The differentiable-benchmark isolation classifier promotes evidence to
``isolated_affinity`` only on a self-hosted isolated-benchmark runner with a
reserved CPU, a recorded governor/frequency, and low host load. Running the CI
job on a host that does not meet those conditions silently downgrades the
artefact to ``functional_non_isolated``. This module checks the physical host
preconditions up front so a benchmark run is not wasted, reusing the same load
threshold the classifier applies.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .differentiable_evidence import (
    capture_host_load,
    read_cpu_frequency_mhz,
    read_cpu_governor,
)

# The isolation classifier accepts a run as low-load when every load average is
# at or below this value (mirrors ``_host_load_is_low``).
MAX_ISOLATED_LOAD = 1.0
# A fixed frequency governor keeps benchmark timing stable.
_STABLE_GOVERNORS = frozenset({"performance"})


@dataclass(frozen=True)
class HostReadiness:
    """Verdict on whether a host is configured for isolated benchmarking."""

    ready: bool
    reserved_core: int
    governor: str | None
    governor_is_stable: bool
    frequency_mhz: float | None
    load_average: tuple[float, float, float] | None
    load_is_low: bool
    blockers: tuple[str, ...] = field(default_factory=tuple)


def assess_host_readiness(
    *,
    reserved_core: int,
    governor: str | None,
    frequency_mhz: float | None,
    load_average: tuple[float, float, float] | None,
) -> HostReadiness:
    """Assess host isolation readiness from already-captured metadata.

    Pure function over the captured host metadata so the gating logic is
    testable without touching the live host.
    """
    if reserved_core < 0:
        raise ValueError("reserved_core must be a non-negative integer")

    governor_is_stable = governor in _STABLE_GOVERNORS
    load_is_low = load_average is not None and max(load_average) <= MAX_ISOLATED_LOAD

    blockers: list[str] = []
    if governor is None:
        blockers.append(
            f"cpu{reserved_core} frequency governor is unreadable; the classifier "
            "requires governor metadata"
        )
    elif not governor_is_stable:
        blockers.append(
            f"cpu{reserved_core} governor is {governor!r}; set it to 'performance' "
            "for stable benchmark timing"
        )
    if frequency_mhz is None:
        blockers.append(f"cpu{reserved_core} frequency is unreadable")
    if load_average is None:
        blockers.append("host load average is unavailable")
    elif not load_is_low:
        blockers.append(
            f"host load {max(load_average):.2f} exceeds the isolated threshold "
            f"{MAX_ISOLATED_LOAD:.2f}; quiesce concurrent jobs"
        )

    return HostReadiness(
        ready=not blockers,
        reserved_core=reserved_core,
        governor=governor,
        governor_is_stable=governor_is_stable,
        frequency_mhz=frequency_mhz,
        load_average=load_average,
        load_is_low=load_is_low,
        blockers=tuple(blockers),
    )


def capture_host_readiness(reserved_core: int = 0) -> HostReadiness:
    """Capture live host metadata and assess isolated-benchmark readiness."""
    return assess_host_readiness(
        reserved_core=reserved_core,
        governor=read_cpu_governor(reserved_core),
        frequency_mhz=read_cpu_frequency_mhz(reserved_core),
        load_average=capture_host_load(),
    )
