# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Studio plan source custody
"""Capture actual Studio previews without executing their planned actions."""

from __future__ import annotations

import json
from typing import Any

from scpn_quantum_control import stable_core_product as codec
from scpn_quantum_control.studio.executive import ExecutiveRequest, preview_action
from scpn_quantum_control.studio.executive_cli import build_default_registry


def studio_plan_sources() -> tuple[dict[str, Any], ...]:
    """Capture complete requests, resolved plans and approval boundaries.

    Returns
    -------
    tuple
        Default and requested-backend compile previews plus an unapproved
        execute preview. Only planning executes; no compilation, provider SDK,
        submission or hardware qualification is inferred. Studio is required.

    """
    network = {
        "K_nm": [[0.0, 0.4], [0.4, 0.0]],
        "omega": [-0.1, 0.1],
        "time": 0.1,
        "trotter_steps": 1,
        "trotter_order": 2,
    }
    requests = (
        (
            "studio_compile_default_preserves_plan",
            ExecutiveRequest("compile", "source-compile-default", network),
        ),
        (
            "studio_compile_requested_rust_preserves_plan",
            ExecutiveRequest("compile", "source-compile-rust", network, backend="rust"),
        ),
        (
            "studio_execute_preserves_approval_plan",
            ExecutiveRequest(
                "execute",
                "source-deploy-preview",
                {
                    "provider": "ibm-quantum",
                    "endpoint": "fixture-target-not-contacted",
                    "circuit_digest": "sha256:" + "0" * 64,
                    "circuit_ref": "fixture:unsubmitted-circuit",
                    "shots": 100,
                },
            ),
        ),
    )
    registry = build_default_registry()
    sources = []
    for case_id, request in requests:
        plan = preview_action(request, registry=registry)
        request_wire, plan_wire = request.to_dict(), plan.to_dict()
        source: dict[str, Any] = json.loads(
            codec.canonical_json_bytes(
                {
                    "case_id": case_id,
                    "producer": "scpn_quantum_control.studio.executive.preview_action",
                    "registry_producer": "scpn_quantum_control.studio.executive_cli.build_default_registry",
                    "request": request_wire,
                    "plan": plan_wire,
                    "request_type": f"{type(request).__module__}.{type(request).__qualname__}",
                    "plan_type": f"{type(plan).__module__}.{type(plan).__qualname__}",
                    "request_sha256": codec.digest_stable_core_payload(request_wire),
                    "plan_sha256": codec.digest_stable_core_payload(plan_wire),
                    "requires_approval": plan.requires_approval,
                    "stage": "planning",
                    "executed_action": False,
                    "unavailable": ["hardware_observation", "companion_validation"],
                }
            )
        )
        sources.append(source)
    return tuple(sources)
