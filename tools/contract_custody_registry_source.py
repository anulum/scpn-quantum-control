# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — registry source evidence capture
"""Capture declared registry support without promoting it to executable lowering."""

from __future__ import annotations

from typing import Any, Final

from scpn_quantum_control import stable_core_product as scp
from scpn_quantum_control.program_ad_registry import (
    CustomDerivativeRegistry,
    primitive_contract_for,
    program_ad_registry_dispatch_coverage_report,
)

PROBE_IDENTITY: Final = "scpn.program_ad.elementwise:sin@1"
"""Existing primitive whose metadata is complete but executable lowering absent."""


def registry_evidence_source(*, empty: bool = False) -> dict[str, Any]:
    """Capture real default or empty-registry reports and a representative contract.

    Parameters
    ----------
    empty
        Use a fresh empty registry to record unsupported coverage. The shared
        default registry is inspected only and is never changed.

    Returns
    -------
    dict
        Full public report, original digest and copied contract metadata.
        A missing contract is None, not fabricated support. Declared rules
        and metadata do not prove their numerical or compiled execution.

    """
    registry = CustomDerivativeRegistry() if empty else None
    report = program_ad_registry_dispatch_coverage_report(registry=registry).to_dict()
    contract = (
        registry.contract_for(PROBE_IDENTITY)
        if registry is not None
        else primitive_contract_for(PROBE_IDENTITY)
    )
    snapshot = (
        None
        if contract is None
        else {
            "producer_identity": f"{type(contract).__module__}.{type(contract).__qualname__}",
            "identity": contract.identity.key,
            "derivative_rule": contract.derivative_rule.name,
            "parameter_names": list(contract.derivative_rule.parameter_names),
            "trainable": list(contract.derivative_rule.trainable),
            "has_jvp_rule": contract.derivative_rule.jvp_rule is not None,
            "has_vjp_rule": contract.derivative_rule.vjp_rule is not None,
            "has_batching_rule": contract.batching_rule is not None,
            "has_lowering_rule": contract.lowering_rule is not None,
            "has_shape_rule": contract.shape_rule is not None,
            "has_dtype_rule": contract.dtype_rule is not None,
            "has_static_argument_rule": contract.static_argument_rule is not None,
            "lowering_metadata": dict(contract.lowering_metadata),
            "nondifferentiable_policy": contract.nondifferentiable_policy,
            "effect": contract.effect,
        }
    )
    return {
        "producer": "scpn_quantum_control.program_ad_registry.program_ad_registry_dispatch_coverage_report",
        "inputs": {"registry": "empty" if empty else "default", "probe_identity": PROBE_IDENTITY},
        "report": report,
        "report_sha256": scp.digest_stable_core_payload(report),
        "contract": snapshot,
        "claim_boundary": report["claim_boundary"],
    }
