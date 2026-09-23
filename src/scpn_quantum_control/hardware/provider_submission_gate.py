# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — submit-time provider capability gate
"""Bind no-submit provider metadata to a cloud adapter immediately before run."""

from __future__ import annotations

from datetime import datetime

from .hal import BackendProfile, QuantumWorkload
from .provider_capability_core import (
    ProviderCapabilityDecision,
    ProviderCapabilitySnapshot,
    assess_provider_capability_snapshot,
)


def require_submit_time_capability(
    snapshot: ProviderCapabilitySnapshot,
    *,
    profile: BackendProfile,
    target_name: str,
    workload: QuantumWorkload,
    max_calibration_age_seconds: float,
    checked_at: datetime,
) -> ProviderCapabilityDecision:
    """Require current target, resource, route and calibration agreement.

    Parameters
    ----------
    snapshot
        No-submit metadata captured by the configured probe at submission time.
    profile
        HAL route that will receive the workload.
    target_name
        Target identity read from the adapter's configured provider object.
    workload
        Validated submitted programme and resource request.
    max_calibration_age_seconds
        Finite nonnegative age limit for the calibration timestamp.
    checked_at
        Current timezone-aware UTC clock observation, taken inside submit.

    Returns
    -------
    ProviderCapabilityDecision
        Ready decision retaining the checked timestamp and source snapshot.

    Raises
    ------
    ValueError
        If metadata or the policy cannot qualify this exact submission.

    """
    if not isinstance(snapshot, ProviderCapabilitySnapshot):
        raise ValueError("capability probe must return ProviderCapabilitySnapshot")
    decision = assess_provider_capability_snapshot(
        snapshot,
        aggregator=profile.broker,
        provider=profile.provider,
        backend_id=profile.backend_id,
        required_ir_format=workload.ir_format,
        min_qubits=workload.n_qubits,
        max_calibration_age_seconds=max_calibration_age_seconds,
        as_of=checked_at,
    )
    blockers = list(decision.blockers)
    if snapshot.target_name != target_name:
        blockers.append("provider target does not match configured adapter target")
    if snapshot.simulator != ("simulator" in profile.modality):
        blockers.append("simulator classification does not match route")
    if snapshot.max_shots is None:
        blockers.append("provider shots limit is unknown")
    elif isinstance(snapshot.max_shots, bool) or workload.shots > snapshot.max_shots:
        blockers.append("provider shots limit is invalid or exceeded")
    if decision.status != "ready" or blockers:
        raise ValueError(
            "submit-time capability refused: " + "; ".join((*blockers, *decision.warnings))
        )
    return decision


__all__ = ["require_submit_time_capability"]
