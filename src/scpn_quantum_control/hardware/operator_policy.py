# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — source-owned operator policy decisions
"""Admit declared operator plans against exact workloads before transport."""

from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING

from ..canonical_encoding import canonical_digest
from .operator_policy_contracts import (
    OperatorPolicy,
    OperatorPolicyDecision,
    OperatorRequest,
    PricingEstimate,
    utc_second,
)

if TYPE_CHECKING:
    from .hal import BackendProfile, QuantumWorkload


def workload_fingerprint(workload: QuantumWorkload) -> str:
    """Bind the original HAL request and complete native source semantics.

    Parameters
    ----------
    workload
        Original programme, annotations, shots and optional native semantics.

    Returns
    -------
    str
        Exact typed canonical SHA-256; computing it never loads a provider SDK.

    """
    return canonical_digest(
        "operator_workload.v1",
        {
            "source": workload.to_semantic_source(),
            "semantics": None if workload.semantics is None else asdict(workload.semantics),
        },
    )


def assess_operator_policy(
    profile: BackendProfile,
    workload: QuantumWorkload,
    request: OperatorRequest,
    policy: OperatorPolicy,
    *,
    estimate: PricingEstimate | None = None,
    now: str | None = None,
) -> OperatorPolicyDecision:
    """Resolve exact plan ceilings without clamping, substitution or dispatch.

    Parameters
    ----------
    profile, workload
        Governing original HAL declarations and workload to be dispatched.
    request, policy
        Immutable exact plan and trusted configured ceilings.
    estimate
        Supplied complete-plan price; unknown price cannot satisfy a cost ceiling.
    now
        Explicit UTC seconds for offline assessment, or current UTC for a fresh verdict.
        HAL submit always uses current UTC and accepts no caller-controlled clock.

    Returns
    -------
    OperatorPolicyDecision
        Ordered source refusals and untouched original values. Allowed means plan
        admission, not provider authentication, execution permission or actual spend.

    Raises
    ------
    TypeError
        If a caller supplies unvalidated input records.
    ValueError
        If the explicit no-submit clock is malformed.

    """
    if not isinstance(request, OperatorRequest) or not isinstance(policy, OperatorPolicy):
        raise TypeError("operator policy inputs must be immutable native contracts")
    if estimate is not None and not isinstance(estimate, PricingEstimate):
        raise TypeError("estimate must be an immutable PricingEstimate")
    assessed_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ") if now is None else now
    instant = utc_second(assessed_at)
    reasons: list[str] = []
    substitutions: list[str] = []
    if instant < utc_second(policy.valid_from):
        reasons.append("policy_future")
    if instant >= utc_second(policy.expires_at):
        reasons.append("policy_expired")
    if request.backend_id != policy.backend_id:
        reasons.append("backend_forbidden")
    bindings = [
        ("profile_backend_mismatch", request.backend_id == profile.backend_id),
        ("workload_source_mismatch", request.workload_sha256 == workload_fingerprint(workload)),
        ("workload_shots_mismatch", request.shots == workload.shots),
    ]
    native_target = None if workload.semantics is None else workload.semantics.requested_target
    if profile.is_cloud:
        if request.target is None or native_target is None:
            reasons.append("target_unbound")
        else:
            bindings.append(("native_target_mismatch", request.target == native_target))
        if request.region is None or profile.region is None:
            reasons.append("region_unknown")
        bindings.append(("profile_region_mismatch", request.region == profile.region))
    else:
        bindings.extend(
            [
                (
                    "native_target_mismatch",
                    request.target == (native_target or profile.backend_id),
                ),
                ("profile_region_mismatch", request.region == profile.region),
            ]
        )
    for reason, same in bindings:
        if not same:
            reasons.append(reason)
            substitutions.append(reason)
    if request.target not in policy.targets:
        reasons.append("target_forbidden")
    # A local profile may explicitly declare no geographical region.
    if request.region is not None and request.region not in policy.regions:
        reasons.append("region_forbidden")
    for field, ceiling in (
        ("shots", policy.max_shots),
        ("concurrency", policy.max_concurrency),
        ("time_limit_ms", policy.max_time_limit_ms),
    ):
        if getattr(request, field) > ceiling:
            reasons.append(field + "_ceiling")
    if estimate is None or estimate.amount is None:
        reasons.append("price_unknown")
    if estimate is not None:
        if estimate.request_sha256 != request.sha256:
            reasons.append("price_request_mismatch")
        if instant < utc_second(estimate.observed_at):
            reasons.append("price_future")
        if instant >= utc_second(estimate.expires_at):
            reasons.append("price_expired")
        if estimate.currency != policy.currency:
            reasons.append("price_currency_mismatch")
        if estimate.amount is not None and Decimal(estimate.amount) > Decimal(policy.max_cost):
            reasons.append("cost_ceiling")
    return OperatorPolicyDecision(
        request, policy, estimate, assessed_at, tuple(reasons), tuple(substitutions)
    )
