# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — immutable operator review dossier contracts
"""Exercise review binding through the original public no-submit handler."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from typing import Any

import pytest

pytest.importorskip("scpn_studio_platform", reason="studio extra not installed")

from scpn_quantum_control.canonical_encoding import canonical_digest  # noqa: E402
from scpn_quantum_control.hardware.hal import (  # noqa: E402
    BackendCapabilities,
    BackendProfile,
    HardwareAbstractionLayer,
    QuantumWorkload,
)
from scpn_quantum_control.hardware.operator_policy_contracts import (  # noqa: E402
    OperatorPolicy,
    PricingEstimate,
)
from scpn_quantum_control.hardware.provider_semantics import WorkloadSemantics  # noqa: E402
from scpn_quantum_control.studio.executive import ExecutiveRequest  # noqa: E402
from scpn_quantum_control.studio.executive_execute import ExecuteActionHandler  # noqa: E402
from scpn_quantum_control.studio.operator_review_dossier import (  # noqa: E402
    OperatorReviewDecision,
    OperatorReviewDossier,
)
from scpn_quantum_control.studio.workspace import (  # noqa: E402
    ResolvedSettings,
    operator_request_from_settings,
)


def review_inputs() -> tuple[ExecutiveRequest, dict[str, Any]]:
    """Return an independently labelled synthetic native plan and actual verdict.

    Returns
    -------
    tuple
        Existing execute request and immutable source inputs; no adapter or SDK.

    """
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
    policy = OperatorPolicy(
        "synthetic-only",
        profile.backend_id,
        ("device-A", "device-B"),
        ("eu-north1",),
        1024,
        2,
        60000,
        "12.50",
        "USD",
        "2026-10-03T00:00:00Z",
        "2026-10-05T00:00:00Z",
    )
    program = "OPENQASM 3.0; qubit q; bit c; c = measure q;"
    work = QuantumWorkload(
        "synthetic-review",
        "openqasm3",
        program,
        1,
        1024,
        semantics=WorkloadSemantics(
            hashlib.sha256(program.encode()).hexdigest(),
            1,
            1,
            ((0, 0),),
            (("c", (0,)),),
            requested_target="device-A",
        ),
    )
    values: dict[str, object] = {
        "backend": profile.backend_id,
        "device": "device-A",
        "region": "eu-north1",
        "shots": 1024,
        "concurrency": 2,
        "time_limit_ms": 60000,
        "unattended": True,
        "seed": 9007199254740993,
        "theme": "dark",
    }
    settings = ResolvedSettings(
        {
            "requested": values,
            "effective": values,
            "origins": {k: "run" for k in values},
            "policy_ref": {
                "schema": "operator_policy.v1",
                "sha256": canonical_digest("operator_policy.v1", policy.to_dict()),
                "media_type": "application/json",
            },
            "environment_ref": {
                "schema": "synthetic_environment.v1",
                "sha256": "0" * 64,
                "media_type": "application/json",
            },
            "rejected_fields": [],
        }
    )
    operational = operator_request_from_settings(work, settings)
    estimate = PricingEstimate(
        operational.sha256,
        "12.50",
        "USD",
        "synthetic-only",
        "2026-10-03T00:00:00Z",
        "2026-10-05T00:00:00Z",
    )
    decision = HardwareAbstractionLayer([profile], operator_policy=policy).assess_operator_policy(
        profile.backend_id,
        work,
        operational,
        estimate=estimate,
        now="2026-10-04T00:00:00Z",
    )
    payload = b"compiled synthetic payload: device-A\n"
    request = ExecutiveRequest(
        "execute",
        "synthetic-review",
        {
            "provider": profile.provider,
            "endpoint": "device-A",
            "shots": 1024,
            "circuit_digest": "sha256:" + hashlib.sha256(payload).hexdigest(),
            "circuit_ref": "synthetic/compiled-payload",
            "calibration_ref": "synthetic/calibration",
        },
        backend="provider-hal",
    )
    return request, {
        "profile": profile,
        "workload": work,
        "compiled_payload": payload,
        "settings": settings,
        "policy_decision": decision,
        "calibration": {
            "reference": "synthetic/calibration",
            "sha256": "c" * 64,
            "target": "device-A",
            "observed_at": "2026-10-03T00:00:00Z",
            "expires_at": "2026-10-05T00:00:00Z",
        },
        "created_at": "2026-10-04T00:00:00Z",
        "expires_at": "2026-10-05T00:00:00Z",
    }


def test_operator_review_dossiers_03() -> None:
    """Public dossier and executable review script bind original payload bytes."""
    request, inputs = review_inputs()
    dossier = ExecuteActionHandler().prepare_review(request, **inputs)
    wire = dossier.to_dict()
    assert wire["schema"] == "studio.operator-review-dossier.v1"
    assert wire["body"]["no_submit"] is True
    expected = hashlib.sha256(inputs["compiled_payload"]).hexdigest()
    assert wire["body"]["payload"]["sha256"] == expected
    assert expected in dossier.script.source
    assert dossier.review_status(None, now="2026-10-04T00:00:00Z") == "pending"


def update_settings(inputs: dict[str, Any], **changes: object) -> None:
    """Replace fixture settings through the actual immutable workspace constructor.

    Parameters
    ----------
    inputs
        Synthetic native input fixture owned by this test.
    changes
        Literal requested and effective values for a new candidate.

    """
    body = inputs["settings"].to_dict()["body"]
    for field in ("requested", "effective"):
        body[field].update(changes)
    for key in changes:
        body["origins"].setdefault(key, "run")
    inputs["settings"] = ResolvedSettings(body)


def reassess(inputs: dict[str, Any]) -> None:
    """Produce the actual native verdict for changed synthetic source inputs.

    Parameters
    ----------
    inputs
        Current native workload, settings, route and original estimate/policy.

    """
    original = inputs["policy_decision"]
    request = operator_request_from_settings(inputs["workload"], inputs["settings"])
    estimate = original.estimate
    if estimate is not None:
        estimate = replace(estimate, request_sha256=request.sha256)
    inputs["policy_decision"] = HardwareAbstractionLayer(
        [inputs["profile"]],
        operator_policy=original.policy,
    ).assess_operator_policy(
        inputs["profile"].backend_id,
        inputs["workload"],
        request,
        estimate=estimate,
        now=original.assessed_at,
    )


@pytest.mark.parametrize(
    "field",
    [
        "payload",
        "device",
        "shots",
        "calibration",
        "expiry",
        "action",
        "backend",
        "seed",
        "origin",
        "price",
        "policy",
        "profile",
        "shot_capacity",
        "program",
    ],
)
def test_operator_review_dossiers_01(field: str) -> None:
    """Every execution source change invalidates its prior human review.

    Parameters
    ----------
    field
        One independent execution source or provenance change.

    """
    request, inputs = review_inputs()
    original = ExecuteActionHandler().prepare_review(request, **inputs)
    review = original.record_review(
        "approved", reviewer_ref="operator-local", recorded_at="2026-10-04T00:00:00Z"
    )
    assert original.review_status(review, now="2026-10-04T00:00:00Z") == "approved"
    parameters = dict(request.parameters)
    if field == "payload":
        inputs["compiled_payload"] += b"changed\n"
        parameters["circuit_digest"] = (
            "sha256:" + hashlib.sha256(inputs["compiled_payload"]).hexdigest()
        )
    elif field == "device":
        parameters["endpoint"] = "device-B"
        update_settings(inputs, device="device-B")
        inputs["workload"] = replace(
            inputs["workload"],
            semantics=replace(inputs["workload"].semantics, requested_target="device-B"),
        )
        inputs["calibration"]["target"] = "device-B"
        reassess(inputs)
    elif field == "shots":
        parameters["shots"] = 512
        update_settings(inputs, shots=512)
        inputs["workload"] = replace(inputs["workload"], shots=512)
        reassess(inputs)
    elif field == "calibration":
        inputs["calibration"]["sha256"] = "d" * 64
    elif field == "expiry":
        inputs["expires_at"] = "2026-10-04T23:59:59Z"
    elif field == "action":
        request = replace(request, action_id="another-review")
    elif field == "backend":
        request = replace(request, backend="qiskit-runtime")
    elif field == "seed":
        update_settings(inputs, seed=9007199254740994)
    elif field == "origin":
        body = inputs["settings"].to_dict()["body"]
        body["origins"]["seed"] = "experiment"
        inputs["settings"] = ResolvedSettings(body)
    elif field == "price":
        inputs["policy_decision"] = replace(
            inputs["policy_decision"],
            estimate=replace(inputs["policy_decision"].estimate, amount="12.40"),
        )
        reassess(inputs)
    elif field == "policy":
        policy = replace(inputs["policy_decision"].policy, max_cost="25.00")
        body = inputs["settings"].to_dict()["body"]
        body["policy_ref"]["sha256"] = canonical_digest("operator_policy.v1", policy.to_dict())
        inputs["settings"] = ResolvedSettings(body)
        inputs["policy_decision"] = replace(inputs["policy_decision"], policy=policy)
        reassess(inputs)
    elif field == "profile":
        inputs["profile"] = replace(inputs["profile"], notes=("different route source",))
    elif field == "shot_capacity":
        inputs["profile"] = replace(
            inputs["profile"],
            capabilities=replace(inputs["profile"].capabilities, max_shots=2048),
        )
    else:
        inputs["workload"] = replace(inputs["workload"], workload_id="another-original-source")
        reassess(inputs)
    changed = ExecuteActionHandler().prepare_review(
        replace(request, parameters=parameters), **inputs
    )
    assert changed.execution_sha256 != original.execution_sha256
    assert changed.review_status(review, now="2026-10-04T00:00:00Z") == "invalidated"
    human_body = review.to_dict()["body"]
    assert isinstance(human_body, dict)
    assert human_body["dossier_sha256"] == original.sha256
    assert human_body["no_submit"] is True


def test_operator_review_dossiers_05() -> None:
    """Display-only changes preserve review while original source bytes stay immutable."""
    request, inputs = review_inputs()
    original = ExecuteActionHandler().prepare_review(request, **inputs)
    review = original.record_review(
        "approved", reviewer_ref="operator-local", recorded_at="2026-10-04T00:00:00Z"
    )
    update_settings(inputs, theme="light", notation="dirac", layout="wide", plot_rounding=2)
    changed = ExecuteActionHandler().prepare_review(request, **inputs)
    assert changed.sha256 != original.sha256
    assert changed.execution_sha256 == original.execution_sha256
    assert changed.review_status(review, now="2026-10-04T00:00:00Z") == "approved"
    exported = original.to_dict()
    exported["body"]["payload"]["sha256"] = "e" * 64
    inputs["calibration"]["sha256"] = "f" * 64
    assert original.to_dict()["body"]["calibration"]["sha256"] == "c" * 64
    assert original.sha256 == review.dossier_sha256


def test_operator_review_dossiers_02() -> None:
    """Deny and exclusive expiry remain distinct from approval or source refusal."""
    request, inputs = review_inputs()
    dossier = ExecuteActionHandler().prepare_review(request, **inputs)
    approved = dossier.record_review(
        "approved", reviewer_ref="local-human", recorded_at="2026-10-04T00:00:01Z"
    )
    denied = dossier.record_review(
        "denied", reviewer_ref="local-human", recorded_at="2026-10-04T00:00:01Z"
    )
    assert dossier.review_status(None, now="2026-10-03T23:59:59Z") == "not_yet_valid"
    assert dossier.review_status(approved, now="2026-10-04T00:00:00Z") == "not_yet_valid"
    assert dossier.review_status(approved, now="2026-10-04T23:59:59Z") == "approved"
    assert dossier.review_status(approved, now="2026-10-05T00:00:00Z") == "expired"
    assert dossier.review_status(None, now="2026-10-05T00:00:00Z") == "expired"
    assert dossier.review_status(denied, now="2026-10-05T00:00:00Z") == "denied"
    with pytest.raises(PermissionError):
        dossier.record_review(
            "approved", reviewer_ref="local-human", recorded_at="2026-10-05T00:00:00Z"
        )
    inputs["policy_decision"] = replace(inputs["policy_decision"], estimate=None)
    reassess(inputs)
    refused = ExecuteActionHandler().prepare_review(request, **inputs)
    assert refused.review_status(None, now="2026-10-04T00:00:00Z") == "refused"
    assert "price_unknown" in refused.to_dict()["body"]["policy_decision"]["reasons"]
    with pytest.raises(PermissionError):
        refused.record_review(
            "approved", reviewer_ref="local-human", recorded_at="2026-10-04T00:00:00Z"
        )
    denial = refused.record_review(
        "denied", reviewer_ref="local-human", recorded_at="2026-10-04T00:00:00Z"
    )
    assert refused.review_status(denial, now="2026-10-04T00:00:00Z") == "denied"


@pytest.mark.parametrize(
    "fault",
    [
        "extra",
        "wrong_verb",
        "trim",
        "empty_payload",
        "text_payload",
        "oversized_payload",
        "payload_digest",
        "workload",
        "profile",
        "provider",
        "target",
        "shots",
        "region",
        "policy_route",
        "policy_ref",
        "policy_schema",
        "created_before_assessment",
        "empty_interval",
        "beyond_policy",
        "beyond_price",
        "calibration_missing",
        "calibration_extra",
        "calibration_ref",
        "calibration_target",
        "calibration_future",
        "calibration_expired",
        "beyond_calibration",
        "absolute_ref",
        "parent_ref",
        "url_ref",
        "oversized_document",
    ],
)
def test_native_review_refusal_preserves_previous_source(fault: str) -> None:
    """Malformed source cannot replace an existing accepted dossier.

    Parameters
    ----------
    fault
        One real source identity, resource, date or provenance conflict.

    """
    request, inputs = review_inputs()
    prior = ExecuteActionHandler().prepare_review(request, **inputs)
    original = prior.text
    parameters = dict(request.parameters)
    if fault == "extra":
        parameters["credentials"] = "unsupported-field"
    elif fault == "wrong_verb":
        request = replace(request, verb="compile")
    elif fault == "trim":
        parameters["provider"] += " "
    elif fault == "empty_payload":
        inputs["compiled_payload"] = b""
    elif fault == "text_payload":
        inputs["compiled_payload"] = "unsupported"
    elif fault == "oversized_payload":
        inputs["compiled_payload"] = b"x" * (1024 * 1024 + 1)
    elif fault == "payload_digest":
        parameters["circuit_digest"] = "sha256:" + "0" * 64
    elif fault == "workload":
        inputs["workload"] = replace(inputs["workload"], workload_id="changed")
    elif fault == "profile":
        inputs["profile"] = replace(inputs["profile"], backend_id="different")
    elif fault == "provider":
        parameters["provider"] = "another"
    elif fault == "target":
        parameters["endpoint"] = "device-B"
    elif fault == "shots":
        parameters["shots"] = 512
    elif fault == "region":
        inputs["profile"] = replace(inputs["profile"], region="us-east1")
    elif fault == "policy_route":
        inputs["policy_decision"] = replace(
            inputs["policy_decision"],
            policy=replace(inputs["policy_decision"].policy, backend_id="different"),
        )
    elif fault in ("policy_ref", "policy_schema"):
        body = inputs["settings"].to_dict()["body"]
        body["policy_ref"]["sha256" if fault == "policy_ref" else "schema"] = (
            "f" * 64 if fault == "policy_ref" else "another.v1"
        )
        inputs["settings"] = ResolvedSettings(body)
    elif fault == "created_before_assessment":
        inputs["created_at"] = "2026-10-03T23:59:59Z"
    elif fault == "empty_interval":
        inputs["expires_at"] = inputs["created_at"]
    elif fault == "beyond_policy":
        inputs["expires_at"] = "2026-10-05T00:00:01Z"
    elif fault == "beyond_price":
        inputs["policy_decision"] = replace(
            inputs["policy_decision"],
            estimate=replace(
                inputs["policy_decision"].estimate, expires_at="2026-10-04T23:59:59Z"
            ),
        )
    elif fault == "calibration_missing":
        inputs["calibration"] = None
    elif fault == "calibration_extra":
        inputs["calibration"]["extra"] = "unsupported"
    elif fault == "calibration_ref":
        inputs["calibration"]["reference"] = "synthetic/another"
    elif fault == "calibration_target":
        inputs["calibration"]["target"] = "device-B"
    elif fault == "calibration_future":
        inputs["calibration"]["observed_at"] = "2026-10-04T00:00:01Z"
    elif fault == "calibration_expired":
        inputs["calibration"]["expires_at"] = "2026-10-04T00:00:00Z"
    elif fault == "beyond_calibration":
        inputs["calibration"]["expires_at"] = "2026-10-04T23:59:59Z"
    elif fault == "absolute_ref":
        parameters["circuit_ref"] = "/unowned/payload"
    elif fault == "parent_ref":
        parameters["circuit_ref"] = "synthetic/../payload"
    elif fault == "url_ref":
        parameters["circuit_ref"] = "https://unowned.invalid/payload"
    else:
        inputs["profile"] = replace(inputs["profile"], notes=("x" * (1024 * 1024),))
    with pytest.raises(ValueError):
        ExecuteActionHandler().prepare_review(replace(request, parameters=parameters), **inputs)
    assert prior.text == original


def test_explicit_unknown_calibration_and_price_stay_unknown() -> None:
    """Absence never generates a calibration or substitutes a price source."""
    request, inputs = review_inputs()
    parameters = dict(request.parameters)
    parameters.pop("calibration_ref")
    inputs["calibration"] = None
    inputs["policy_decision"] = replace(inputs["policy_decision"], estimate=None)
    reassess(inputs)
    dossier = ExecuteActionHandler().prepare_review(
        replace(request, parameters=parameters), **inputs
    )
    body = dossier.to_dict()["body"]
    assert body["calibration"] is None and body["policy_decision"]["estimate"] is None
    assert dossier.review_status(None, now="2026-10-04T00:00:00Z") == "refused"


def test_only_original_execute_no_submit_projection_can_make_a_dossier() -> None:
    """Actual native plan/result substitution cannot acquire review identity."""
    from scpn_quantum_control.studio.executive import ExecutionResult, resolve_verb_contract

    request, inputs = review_inputs()
    handler = ExecuteActionHandler()
    plan = handler.plan(request, resolve_verb_contract("execute"))
    deployment = handler.execute(plan)
    for candidate in (
        ExecutionResult("failed", {}, "no projection"),
        ExecutionResult("succeeded", {**deployment.outputs, "submitted": True}),
        ExecutionResult("succeeded", {**deployment.outputs, "shots": 512}),
    ):
        with pytest.raises(ValueError):
            OperatorReviewDossier(plan, candidate, **inputs)
    altered = replace(plan, contract=replace(plan.contract, requires_approval=False))
    with pytest.raises(ValueError):
        OperatorReviewDossier(altered, deployment, **inputs)


@pytest.mark.parametrize(
    "field,value",
    [
        ("dossier_sha256", "bad"),
        ("execution_sha256", "bad"),
        ("choice", "submit"),
        ("reviewer_ref", ""),
        ("recorded_at", "2026-10-04T00:00:00+00:00"),
    ],
)
def test_review_record_rejects_malformed_identity(field: str, value: str) -> None:
    """A human record cannot silently coerce malformed source references.

    Parameters
    ----------
    field, value
        One malformed field in an otherwise independent review receipt.

    """
    values: dict[str, Any] = {
        "dossier_sha256": "0" * 64,
        "execution_sha256": "1" * 64,
        "choice": "approved",
        "reviewer_ref": "local-human",
        "recorded_at": "2026-10-04T00:00:00Z",
    }
    values[field] = value
    with pytest.raises(ValueError):
        OperatorReviewDecision(**values)


@pytest.mark.parametrize("choice", ["approved", "denied"])
def test_human_review_is_not_visible_before_recorded_time(choice: Any) -> None:
    """Neither human choice can appear current before its own recorded instant.

    Parameters
    ----------
    choice
        Either human review choice, independently of provider admission.

    """
    request, inputs = review_inputs()
    dossier = ExecuteActionHandler().prepare_review(request, **inputs)
    review = dossier.record_review(
        choice, reviewer_ref="local-human", recorded_at="2026-10-04T00:00:01Z"
    )
    assert dossier.review_status(review, now="2026-10-04T00:00:00Z") == "not_yet_valid"


def test_human_denial_cannot_predate_its_source() -> None:
    """A future source cannot acquire an invented earlier human denial."""
    request, inputs = review_inputs()
    dossier = ExecuteActionHandler().prepare_review(request, **inputs)
    with pytest.raises(ValueError):
        dossier.record_review(
            "denied", reviewer_ref="local-human", recorded_at="2026-10-03T23:59:59Z"
        )


@pytest.mark.parametrize(
    "reasons,substitutions",
    [
        (list(), ()),
        (tuple(f"r{i}" for i in range(257)), ()),
        (("duplicate", "duplicate"), ()),
        ((), ("unowned",)),
    ],
)
def test_source_review_refuses_malformed_native_reason_sequences(
    reasons: Any, substitutions: Any
) -> None:
    """Malformed native verdict reasons cannot become review source evidence.

    Parameters
    ----------
    reasons, substitutions
        Invalid mutable, unbounded, repeated or unowned native refusal identities.

    """
    request, inputs = review_inputs()
    inputs["policy_decision"] = replace(
        inputs["policy_decision"], reasons=reasons, rejected_substitutions=substitutions
    )
    with pytest.raises(ValueError):
        ExecuteActionHandler().prepare_review(request, **inputs)
