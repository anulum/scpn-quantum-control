# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — resolved workspace operator policy projection
"""Bind original resolved settings to the core's no-submit policy boundary."""

from __future__ import annotations

from collections.abc import Mapping

from ..hardware.hal import HardwareAbstractionLayer, QuantumWorkload
from ..hardware.operator_policy import workload_fingerprint
from ..hardware.operator_policy_contracts import OperatorRequest, PricingEstimate
from .canonical import canonical_digest
from .contracts import ResolvedSettings
from .json_transport import write_json
from .settings import SettingsRefused, validate_setting_values

OPERATOR_DECISION_SCHEMA = "studio.operator-policy-decision.v1"
"""Source-owned verdict envelope; imported values never grant execute authority."""
MAX_OPERATOR_DECISION_BYTES = 1024 * 1024
"""Product UTF-8 envelope bound, not a measured latency or host-capacity claim."""


def operator_request_from_settings(
    workload: QuantumWorkload, settings: ResolvedSettings
) -> OperatorRequest:
    """Project literal effective settings into an exact original-workload request.

    Parameters
    ----------
    workload
        Original HAL programme and native semantics, never reconstructed by the view.
    settings
        Original immutable settings with all requested/effective values and origins.

    Returns
    -------
    OperatorRequest
        Exact backend, device, region, shots, planned concurrency, declared milliseconds
        and unattended intent; no missing field is filled from an implicit default.

    Raises
    ------
    SettingsRefused
        If fields are missing, invalid or differ from their original requested values.

    """
    effective = settings.body["effective"]
    requested = settings.body["requested"]
    assert isinstance(effective, Mapping) and isinstance(requested, Mapping)
    validate_setting_values(effective)
    validate_setting_values(requested)
    if requested != effective or settings.body["rejected_fields"]:
        raise SettingsRefused("Operator admission refuses substituted settings.")
    required = (
        "backend",
        "device",
        "region",
        "shots",
        "concurrency",
        "time_limit_ms",
        "unattended",
    )
    if not all(key in effective for key in required):
        raise SettingsRefused("Operator admission requires all explicit plan settings.")
    return OperatorRequest.from_dict(
        {
            "workload_sha256": workload_fingerprint(workload),
            "backend_id": effective["backend"],
            "target": effective["device"],
            "region": effective["region"],
            "shots": effective["shots"],
            "concurrency": effective["concurrency"],
            "time_limit_ms": effective["time_limit_ms"],
            "unattended": effective["unattended"],
        }
    )


def assess_workspace_operator_policy(
    hal: HardwareAbstractionLayer,
    workload: QuantumWorkload,
    settings: ResolvedSettings,
    *,
    estimate: PricingEstimate | None = None,
    now: str | None = None,
) -> dict[str, object]:
    """Export the core verdict with original complete settings and provenance.

    Parameters
    ----------
    hal, workload
        Existing policy-configured HAL and unchanged original programme.
    settings
        Complete original immutable effective/requested settings and source origins.
    estimate
        Dated exact complete-request estimate, or unknown; no provider is consulted.
    now
        Explicit UTC-second instant for offline assessment, or actual current UTC.

    Returns
    -------
    dict[str, object]
        Detached v1 envelope with exact settings, decision and typed canonical identity.
        A passing historical snapshot is read-only and cannot be reused by HAL submit.

    Raises
    ------
    SettingsRefused
        If original settings are substituted, incomplete or refer to another policy.
    ValueError
        If the original native request or export bound is invalid.

    """
    request = operator_request_from_settings(workload, settings)
    decision = hal.assess_operator_policy(
        request.backend_id, workload, request, estimate=estimate, now=now
    )
    reference = settings.body["policy_ref"]
    assert isinstance(reference, Mapping)
    if reference["schema"] != "operator_policy.v1" or reference["sha256"] != canonical_digest(
        "operator_policy.v1", decision.policy.to_dict()
    ):
        raise SettingsRefused("Operator settings refer to a different governing policy.")
    body: dict[str, object] = {
        "no_submit": True,
        "claim_boundary": "dated_plan_admission_only",
        "settings": settings.to_dict(),
        "settings_sha256": settings.digest,
        "decision": decision.to_dict(),
    }
    envelope: dict[str, object] = {
        "schema": OPERATOR_DECISION_SCHEMA,
        "body": body,
        "extensions": {},
    }
    envelope["sha256"] = canonical_digest(OPERATOR_DECISION_SCHEMA, envelope)
    if len(write_json(envelope).encode("utf-8")) > MAX_OPERATOR_DECISION_BYTES:
        raise ValueError("Operator policy decision exceeds the UTF-8 bound.")
    return envelope
