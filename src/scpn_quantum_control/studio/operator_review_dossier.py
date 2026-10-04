# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — immutable native operator review references
"""Seal source-owned deployment evidence and separate human review decisions."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any, Literal, cast

from ..canonical_encoding import canonical_digest
from ..hardware.operator_policy_contracts import _hash, _text, utc_second
from ..studio_workspace.json_transport import read_json, write_json
from ..studio_workspace.operator_policy import operator_request_from_settings
from ..studio_workspace.settings import SEMANTIC_FIELDS, settings_plan_digest

if TYPE_CHECKING:
    from ..hardware.hal import BackendProfile, QuantumWorkload
    from ..hardware.operator_policy_contracts import OperatorPolicyDecision
    from ..studio_workspace.contracts import ResolvedSettings
    from .executive import ExecutionPlan, ExecutionResult, GeneratedScript

DOSSIER_SCHEMA = "studio.operator-review-dossier.v1"
"""Readonly source envelope; never a provider submission or credential contract."""
REVIEW_SCHEMA = "studio.operator-review.v1"
"""Separate human decision retaining its original dossier reference."""
REVIEW_EXPORT_SCHEMA = "studio.operator-review-export.v1"
"""Native dossier text and its native verification script, without circular identity."""
MAX_REVIEW_BYTES = 1024 * 1024
"""Product payload and UTF-8 document bound, not an observed provider capacity."""
DOSSIER_BODY_FIELDS = frozenset(
    {
        "no_submit",
        "claim_boundary",
        "producer_identity",
        "plan",
        "plan_sha256",
        "profile",
        "profile_sha256",
        "workload_sha256",
        "payload",
        "settings",
        "settings_sha256",
        "semantic_settings",
        "semantic_settings_sha256",
        "policy_decision",
        "policy_decision_sha256",
        "calibration",
        "created_at",
        "expires_at",
        "execution_sha256",
    }
)
"""Exact native dossier fields; unsupported additions never become executable source."""
ReviewChoice = Literal["approved", "denied"]
ReviewStatus = Literal[
    "pending", "approved", "denied", "expired", "invalidated", "refused", "not_yet_valid"
]


def _reference(value: object) -> str:
    """Admit bounded relative artifact labels without dereferencing a path."""
    text = _text(value)
    if text.startswith("/") or "\\" in text or "://" in text or "?" in text:
        raise ValueError("review artifact references must be relative opaque labels")
    if ".." in PurePosixPath(text).parts:
        raise ValueError("review artifact references cannot traverse a parent")
    return text


def _semantic_settings(settings: ResolvedSettings) -> dict[str, object]:
    """Retain semantic requested/effective/origin and governing source references."""
    body = settings.body
    result: dict[str, object] = {
        "policy_ref": body["policy_ref"],
        "environment_ref": body["environment_ref"],
        "settings_plan_sha256": settings_plan_digest(settings),
    }
    for name in ("requested", "effective", "origins"):
        values = cast(Mapping[str, object], body[name])
        result[name] = {key: value for key, value in values.items() if key in SEMANTIC_FIELDS}
    return result


def _calibration(
    value: Mapping[str, object] | None, *, reference: object, target: str | None, created_at: str
) -> dict[str, object] | None:
    """Detach dated target evidence without inventing unknown calibration."""
    if value is None:
        if reference is not None:
            raise ValueError("declared calibration requires its original dated evidence")
        return None
    if set(value) != {"reference", "sha256", "target", "observed_at", "expires_at"}:
        raise ValueError("calibration evidence fields differ")
    result: dict[str, object] = {
        "reference": _reference(value["reference"]),
        "sha256": _hash(value["sha256"]),
        "target": _text(value["target"]),
        "observed_at": _text(value["observed_at"]),
        "expires_at": _text(value["expires_at"]),
    }
    if result["reference"] != reference or result["target"] != target:
        raise ValueError("calibration belongs to another deployment or target")
    observed = utc_second(cast(str, result["observed_at"]))
    expiry = utc_second(cast(str, result["expires_at"]))
    if not observed <= utc_second(created_at) < expiry:
        raise ValueError("calibration is not valid at dossier creation")
    return result


def _execution_identity(body: Mapping[str, object]) -> str:
    """Bind every execution input while excluding display-only settings."""
    keys = (
        "plan_sha256",
        "profile_sha256",
        "workload_sha256",
        "payload",
        "semantic_settings_sha256",
        "policy_decision_sha256",
        "calibration",
        "created_at",
        "expires_at",
    )
    return canonical_digest("studio.operator-review-execution.v1", {k: body[k] for k in keys})


def _source_body(
    plan: ExecutionPlan,
    deployment: ExecutionResult,
    *,
    profile: BackendProfile,
    workload: QuantumWorkload,
    compiled_payload: bytes,
    settings: ResolvedSettings,
    policy_decision: OperatorPolicyDecision,
    calibration: Mapping[str, object] | None,
    created_at: str,
    expires_at: str,
) -> dict[str, object]:
    """Validate actual native source coupling before creating detached metadata."""
    from .executive import resolve_verb_contract

    if plan.verb != "execute" or plan.contract != resolve_verb_contract("execute"):
        raise ValueError("review preserves the authoritative execute plan contract")
    if deployment.status != "succeeded" or deployment.outputs.get("submitted") is not False:
        raise ValueError("review requires the original no-submit deployment projection")
    if type(compiled_payload) is not bytes or not 1 <= len(compiled_payload) <= MAX_REVIEW_BYTES:
        raise ValueError("compiled payload must contain 1..1048576 original bytes")
    digest = hashlib.sha256(compiled_payload).hexdigest()
    parameters = plan.parameters
    if parameters["circuit_digest"] != "sha256:" + digest:
        raise ValueError("compiled payload differs from the original plan digest")
    request = operator_request_from_settings(workload, settings)
    for reasons in (policy_decision.reasons, policy_decision.rejected_substitutions):
        if type(reasons) is not tuple or len(reasons) > 256:
            raise ValueError("native decision reasons must be bounded immutable sequences")
        validated = tuple(_text(reason) for reason in reasons)
        if len(set(validated)) != len(validated):
            raise ValueError("native decision reasons must be distinct")
    if not set(policy_decision.rejected_substitutions) <= set(policy_decision.reasons):
        raise ValueError("native substitutions must retain their refusal reasons")
    if request != policy_decision.request or request.backend_id != profile.backend_id:
        raise ValueError("policy verdict belongs to another original workload or settings")
    if (
        parameters["provider"] != profile.provider
        or parameters["endpoint"] != request.target
        or parameters["shots"] != request.shots
        or workload.shots != request.shots
        or request.region != profile.region
        or policy_decision.policy.backend_id != profile.backend_id
    ):
        raise ValueError("review refuses substituted provider, target, region or shots")
    for key in (
        "provider",
        "endpoint",
        "circuit_digest",
        "circuit_ref",
        "shots",
        "calibration_ref",
    ):
        if deployment.outputs.get(key) != parameters.get(key):
            raise ValueError("deployment projection differs from its original plan")
    reference = cast(Mapping[str, object], settings.body["policy_ref"])
    if reference["schema"] != "operator_policy.v1" or reference["sha256"] != canonical_digest(
        "operator_policy.v1", policy_decision.policy.to_dict()
    ):
        raise ValueError("settings refer to another governing policy")
    start, expiry = utc_second(created_at), utc_second(expires_at)
    if start < utc_second(policy_decision.assessed_at) or start >= expiry:
        raise ValueError("review creation or exclusive expiry is inconsistent")
    dates = [policy_decision.policy.expires_at]
    if policy_decision.estimate is not None:
        dates.append(policy_decision.estimate.expires_at)
    calibration_copy = _calibration(
        calibration,
        reference=parameters.get("calibration_ref"),
        target=request.target,
        created_at=created_at,
    )
    if calibration_copy is not None:
        dates.append(cast(str, calibration_copy["expires_at"]))
    if any(expiry > utc_second(date) for date in dates):
        raise ValueError("review expiry cannot extend supplied evidence validity")
    semantic = _semantic_settings(settings)
    plan_copy, profile_copy, verdict = (
        plan.to_dict(),
        profile.to_semantic_source(),
        policy_decision.to_dict(),
    )
    if profile.capabilities.max_shots is not None:
        cast(dict[str, object], profile_copy["capabilities"])["max_shots"] = (
            profile.capabilities.max_shots
        )
    body: dict[str, object] = {
        "no_submit": True,
        "claim_boundary": "human_review_only",
        "producer_identity": "scpn_quantum_control.studio.executive_execute.ExecuteActionHandler",
        "plan": plan_copy,
        "plan_sha256": canonical_digest("execution_plan.v1", plan_copy),
        "profile": profile_copy,
        "profile_sha256": canonical_digest("backend_profile.v1", profile_copy),
        "workload_sha256": request.workload_sha256,
        "payload": {
            "reference": _reference(parameters["circuit_ref"]),
            "sha256": digest,
            "size_bytes": len(compiled_payload),
        },
        "settings": settings.to_dict(),
        "settings_sha256": settings.digest,
        "semantic_settings": semantic,
        "semantic_settings_sha256": canonical_digest("operator_review_settings.v1", semantic),
        "policy_decision": verdict,
        "policy_decision_sha256": canonical_digest("operator_policy_decision.v1", verdict),
        "calibration": calibration_copy,
        "created_at": created_at,
        "expires_at": expires_at,
    }
    body["execution_sha256"] = _execution_identity(body)
    return body


@dataclass(frozen=True)
class OperatorReviewDecision:
    """Immutable human review; neither an account credential nor a run approval.

    Parameters
    ----------
    dossier_sha256, execution_sha256
        Original admitted dossier and complete semantic execution identities.
    choice
        Human review choice, approved or denied; never provider authority.
    reviewer_ref
        Bounded human record label without a credential or key claim.
    recorded_at
        Exact UTC-second time of this review record.

    """

    dossier_sha256: str
    execution_sha256: str
    choice: ReviewChoice
    reviewer_ref: str
    recorded_at: str

    def __post_init__(self) -> None:
        """Reject malformed identities, choices and dates without persistence."""
        _hash(self.dossier_sha256)
        _hash(self.execution_sha256)
        _reference(self.reviewer_ref)
        utc_second(self.recorded_at)
        if self.choice not in ("approved", "denied"):
            raise ValueError("review choice must be approved or denied")

    def to_dict(self) -> dict[str, object]:
        """Return a detached sealed review record with its original source reference."""
        wire: dict[str, object] = {
            "schema": REVIEW_SCHEMA,
            "body": {
                "no_submit": True,
                "claim_boundary": "human_review_only",
                "dossier_sha256": self.dossier_sha256,
                "execution_sha256": self.execution_sha256,
                "choice": self.choice,
                "reviewer_ref": self.reviewer_ref,
                "recorded_at": self.recorded_at,
            },
            "extensions": {},
        }
        wire["sha256"] = canonical_digest(REVIEW_SCHEMA, wire)
        return wire


@dataclass(frozen=True, init=False)
class OperatorReviewDossier:
    """Detached source-only dossier, constructed solely from validated native inputs.

    Parameters
    ----------
    plan, deployment
        Existing original execute plan and its no-submit projection.
    profile, workload, compiled_payload
        Native route, logical source and exact original compiled byte artifact.
    settings, policy_decision
        Complete immutable settings and original dated native verdict.
    calibration
        Original dated target calibration reference, or explicit unknown None.
    created_at, expires_at
        Exact UTC-second review interval within supplied evidence validity.

    """

    text: str

    def __init__(
        self,
        plan: ExecutionPlan,
        deployment: ExecutionResult,
        *,
        profile: BackendProfile,
        workload: QuantumWorkload,
        compiled_payload: bytes,
        settings: ResolvedSettings,
        policy_decision: OperatorPolicyDecision,
        calibration: Mapping[str, object] | None,
        created_at: str,
        expires_at: str,
    ) -> None:
        """Validate and seal all original source fields into immutable JSON text."""
        body = _source_body(
            plan,
            deployment,
            profile=profile,
            workload=workload,
            compiled_payload=compiled_payload,
            settings=settings,
            policy_decision=policy_decision,
            calibration=calibration,
            created_at=created_at,
            expires_at=expires_at,
        )
        wire: dict[str, object] = {"schema": DOSSIER_SCHEMA, "body": body, "extensions": {}}
        wire["sha256"] = canonical_digest(DOSSIER_SCHEMA, wire)
        text = write_json(wire) + "\n"
        if len(text.encode("utf-8")) > MAX_REVIEW_BYTES:
            raise ValueError("review dossier exceeds its UTF-8 document bound")
        object.__setattr__(self, "text", text)

    def to_dict(self) -> dict[str, Any]:
        """Return fresh decoded source values, preserving original numeric types."""
        return cast(dict[str, Any], read_json(self.text))

    def export_bundle(self) -> dict[str, object]:
        """Bundle original source text and its native script without reserializing evidence.

        Returns
        -------
        dict[str, object]
            Detached sealed export; its original dossier and script remain separately
            identifiable. A consumer never needs to generate executable source.

        """
        wire: dict[str, object] = {
            "schema": REVIEW_EXPORT_SCHEMA,
            "body": {
                "no_submit": True,
                "dossier_text": self.text,
                "dossier_sha256": self.sha256,
                "script": self.script.to_dict(),
            },
            "extensions": {},
        }
        wire["sha256"] = canonical_digest(REVIEW_EXPORT_SCHEMA, wire)
        return wire

    @property
    def sha256(self) -> str:
        """Exact complete source envelope identity, including original display metadata."""
        return cast(str, self.to_dict()["sha256"])

    @property
    def execution_sha256(self) -> str:
        """Complete execution identity excluding display-only settings."""
        return cast(str, self.to_dict()["body"]["execution_sha256"])

    @property
    def script(self) -> GeneratedScript:
        """Usable no-submit verification script bound to the exact source dossier."""
        from .operator_review_script import build_review_script

        return build_review_script(self.text)

    def review_status(self, review: OperatorReviewDecision | None, *, now: str) -> ReviewStatus:
        """Resolve a human record against exact current inputs and exclusive expiry.

        Parameters
        ----------
        review
            Earlier immutable human decision, or no decision.
        now
            Explicit current UTC-second time; no source clock is substituted.

        Returns
        -------
        str
            Pending, approved, denied, expired, invalidated, refused or not_yet_valid.
            A matching record grants no submission authority.

        """
        current = utc_second(now)
        body = self.to_dict()["body"]
        if review is not None and review.execution_sha256 != self.execution_sha256:
            return "invalidated"
        if current < utc_second(body["created_at"]) or (
            review is not None and utc_second(review.recorded_at) > current
        ):
            return "not_yet_valid"
        if review is not None and review.choice == "denied":
            return "denied"
        if current >= utc_second(body["expires_at"]):
            return "expired"
        if not body["policy_decision"]["allowed"]:
            return "refused"
        if review is None:
            return "pending"
        return "approved"

    def record_review(
        self, choice: ReviewChoice, *, reviewer_ref: str, recorded_at: str
    ) -> OperatorReviewDecision:
        """Create a separate human decision without mutating the source or requesting a run.

        Parameters
        ----------
        choice
            Approved or denied human review choice.
        reviewer_ref
            Opaque human record label; no credentials are accepted.
        recorded_at
            Exact current UTC seconds, retained on the separate immutable record.

        Returns
        -------
        OperatorReviewDecision
            Original dossier and execution references plus the recorded human choice.

        Raises
        ------
        PermissionError
            If approval would accept expired, future or source-refused inputs.
        ValueError
            If review choice, reference, date or identity is malformed.

        """
        review = OperatorReviewDecision(
            self.sha256, self.execution_sha256, choice, reviewer_ref, recorded_at
        )
        if utc_second(recorded_at) < utc_second(self.to_dict()["body"]["created_at"]):
            raise ValueError("human review cannot predate its original source")
        if choice == "approved" and self.review_status(None, now=recorded_at) != "pending":
            raise PermissionError("current source cannot be approved for review")
        return review
