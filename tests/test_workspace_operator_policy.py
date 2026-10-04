# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — complete workspace operator provenance
"""Qualify the public bridge without replacing original settings or policy owners."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import cast

import pytest

from scpn_quantum_control.hardware.hal import (
    BackendCapabilities,
    BackendProfile,
    HardwareAbstractionLayer,
    QuantumWorkload,
)
from scpn_quantum_control.hardware.operator_policy_contracts import OperatorPolicy, PricingEstimate
from scpn_quantum_control.hardware.provider_semantics import WorkloadSemantics
from scpn_quantum_control.studio import workspace
from scpn_quantum_control.studio_workspace.operator_policy import MAX_OPERATOR_DECISION_BYTES
from tools.export_operator_policy_decisions import build_operator_policy_example


def source_inputs() -> tuple[
    HardwareAbstractionLayer, QuantumWorkload, workspace.ResolvedSettings
]:
    """Restore source inputs from the real native example without any SDK import."""
    envelope = build_operator_policy_example()
    body = cast(Mapping[str, object], envelope["body"])
    decision = cast(Mapping[str, object], body["decision"])
    policy = OperatorPolicy.from_dict(cast(Mapping[str, object], decision["policy"]))
    profile = BackendProfile(
        "synthetic-policy",
        "synthetic",
        "direct",
        "gate_model",
        "none",
        ("openqasm3",),
        BackendCapabilities(True, True, False, False, False, False),
        is_cloud=True,
        submit_requires_approval=True,
        region="eu-north1",
    )
    source = "OPENQASM 3.0; qubit q; bit c; c = measure q;"
    semantics = WorkloadSemantics(
        hashlib.sha256(source.encode()).hexdigest(),
        1,
        1,
        ((0, 0),),
        (("c", (0,)),),
        requested_target="synthetic-device",
    )
    work = QuantumWorkload(
        "synthetic-operator-plan", "openqasm3", source, 1, 1024, semantics=semantics
    )
    settings = workspace.parse_resolved_settings(cast(Mapping[str, object], body["settings"]))
    return HardwareAbstractionLayer([profile], operator_policy=policy), work, settings


def test_operator_policy_decisions_04() -> None:
    """Source export preserves all values, origins, identities and the exact core verdict."""
    hal, work, settings = source_inputs()
    plan = workspace.operator_request_from_settings(work, settings)
    estimate = PricingEstimate(
        plan.sha256,
        None,
        "USD",
        "synthetic-conformance-only",
        "2026-10-03T00:00:00Z",
        "2026-10-05T00:00:00Z",
    )
    actual = workspace.assess_workspace_operator_policy(
        hal, work, settings, estimate=estimate, now="2026-10-04T00:00:00Z"
    )
    assert actual == build_operator_policy_example()
    body = cast(Mapping[str, object], actual["body"])
    assert body["settings"] == settings.to_dict()
    assert body["settings_sha256"] == settings.digest
    assert plan.shots == 1024
    assert cast(Mapping[str, object], settings.body["effective"])["seed"] == 9007199254740993


@pytest.mark.parametrize(
    "damage", ["missing", "substitution", "rejected", "policy", "unknown_field"]
)
def test_candidate_refusals_preserve_settings(damage: str) -> None:
    """Unbound or altered imported candidates refuse without modifying their original record."""
    hal, work, settings = source_inputs()
    snapshot = settings.to_dict()
    body = cast(dict[str, object], snapshot["body"])
    effective = cast(dict[str, object], body["effective"])
    if damage == "missing":
        del effective["unattended"]
        cast(dict[str, object], body["requested"]).pop("unattended")
        cast(dict[str, object], body["origins"]).pop("unattended")
    elif damage == "substitution":
        effective["shots"] = 17
    elif damage == "rejected":
        body["rejected_fields"] = ["shots"]
    elif damage == "policy":
        cast(dict[str, object], body["policy_ref"])["sha256"] = "f" * 64
    else:
        effective["token"] = "unsupported"
        cast(dict[str, object], body["requested"])["token"] = "unsupported"
        cast(dict[str, object], body["origins"])["token"] = "run"
    candidate = workspace.parse_resolved_settings(snapshot)
    before = candidate.digest
    with pytest.raises(workspace.SettingsRefused):
        workspace.assess_workspace_operator_policy(
            hal, work, candidate, now="2026-10-04T00:00:00Z"
        )
    assert candidate.digest == before
    assert settings.digest != candidate.digest


def test_export_bound_refuses_without_overwriting_source() -> None:
    """An oversized extension refuses projection while its source remains unchanged."""
    hal, work, settings = source_inputs()
    raw = settings.to_dict()
    raw["extensions"] = {"retained": "x" * MAX_OPERATOR_DECISION_BYTES}
    candidate = workspace.parse_resolved_settings(raw)
    before = candidate.digest
    with pytest.raises(ValueError, match="UTF-8 bound"):
        workspace.assess_workspace_operator_policy(
            hal, work, candidate, now="2026-10-04T00:00:00Z"
        )
    assert candidate.digest == before


def test_original_public_facades_resolve_the_same_owners() -> None:
    """The stable package and Studio facades expose original bridge functions lazily."""
    from scpn_quantum_control import studio, studio_workspace

    assert studio.assess_workspace_operator_policy is workspace.assess_workspace_operator_policy
    assert (
        studio_workspace.operator_request_from_settings is workspace.operator_request_from_settings
    )
