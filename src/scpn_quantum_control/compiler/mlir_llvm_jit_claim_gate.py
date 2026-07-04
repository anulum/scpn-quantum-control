# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- LLVM/JIT promotion claim gate
"""Promotion gate for native LLVM/JIT differentiable-programming claims.

The native LLVM/JIT execution evidence records prove bounded local execution.
This module keeps the promotion decision separate: a JIT claim is not
promotion-ready until executable lowering, correctness tests, crash-safety
tests, isolated benchmark artefacts, rollback policy and fallback policy are
all attached as explicit evidence.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .mlir_native_execution_evidence import NativeWholeProgramADExecutionEvidence

LLVM_JIT_CLAIM_GATE_BOUNDARY = (
    "Bounded native LLVM/JIT claim gate: executable lowering evidence alone is "
    "not a promoted LLVM/JIT claim. Promotion requires verified beyond-scalar "
    "native execution, correctness test identifiers, crash-safety test "
    "identifiers, isolated benchmark artifact identifiers, rollback policy, and "
    "fallback policy; until every requirement is attached there is no LLVM/JIT "
    "promotion, provider, hardware, GPU, or performance claim."
)

_REQUIREMENT_ORDER = (
    "executable_lowering",
    "correctness_tests",
    "crash_safety_tests",
    "benchmark_artifact_ids",
    "rollback_policy",
    "fallback_policy",
)


def _clean_optional_text(value: str | None, field_name: str) -> str | None:
    """Return stripped optional text, rejecting blank policy bodies."""

    if value is None:
        return None
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} must be non-empty when provided")
    return cleaned


def _clean_text_tuple(values: Sequence[str], field_name: str) -> tuple[str, ...]:
    """Return a tuple of stripped non-empty strings from a public evidence list."""

    cleaned = tuple(value.strip() for value in values)
    if any(not value for value in cleaned):
        raise ValueError(f"{field_name} must contain non-empty identifiers")
    return cleaned


def _payload_text_tuple(value: object, field_name: str) -> tuple[str, ...]:
    """Decode a JSON payload field into a string tuple."""

    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field_name} must be a list of strings")
    rows: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError(f"{field_name} must be a list of strings")
        rows.append(item)
    return _clean_text_tuple(rows, field_name)


def _payload_str(value: object, field_name: str) -> str:
    """Decode a required JSON string field."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _payload_optional_str(value: object, field_name: str) -> str | None:
    """Decode an optional JSON string field."""

    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string when provided")
    return _clean_optional_text(value, field_name)


def _payload_bool(value: object, field_name: str) -> bool:
    """Decode a required JSON boolean field."""

    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a bool")
    return value


@dataclass(frozen=True)
class LLVMJITClaimGate:
    """Fail-closed promotion gate for native LLVM/JIT claims.

    Parameters
    ----------
    artifact_id:
        Stable identifier for the serialized gate evidence.
    executable_lowering_evidence_id:
        Artifact identifier for verified native execution evidence, or ``None``
        when no executable lowering evidence is attached.
    executable_lowering_verified:
        Whether the attached execution evidence verifies a beyond-scalar native
        LLVM/JIT path with gradient parity inside its declared tolerance.
    correctness_test_ids:
        Focused test identifiers that exercise native value/gradient parity.
    crash_safety_test_ids:
        Focused test identifiers that exercise fail-closed behavior and crash
        boundaries for native LLVM/JIT lowering.
    benchmark_artifact_ids:
        Isolated benchmark artifact identifiers backing any performance or
        comparative promotion language.
    rollback_policy:
        Operator policy for withdrawing LLVM/JIT promotion when evidence,
        runtime health or dependency checks regress.
    fallback_policy:
        Operator policy for routing callers to interpreted Program AD or another
        declared fallback when the native gate is not ready.
    claim_boundary:
        Public-safe boundary text that travels with docs and evidence bundles.
    """

    artifact_id: str
    executable_lowering_evidence_id: str | None
    executable_lowering_verified: bool
    correctness_test_ids: Sequence[str]
    crash_safety_test_ids: Sequence[str]
    benchmark_artifact_ids: Sequence[str]
    rollback_policy: str | None
    fallback_policy: str | None
    claim_boundary: str = LLVM_JIT_CLAIM_GATE_BOUNDARY

    def __post_init__(self) -> None:
        artifact_id = self.artifact_id.strip()
        if not artifact_id:
            raise ValueError("artifact_id must be non-empty")
        executable_lowering_evidence_id = _clean_optional_text(
            self.executable_lowering_evidence_id,
            "executable_lowering_evidence_id",
        )
        if self.executable_lowering_verified and executable_lowering_evidence_id is None:
            raise ValueError(
                "executable_lowering_evidence_id is required when lowering is verified"
            )
        correctness_test_ids = _clean_text_tuple(
            self.correctness_test_ids,
            "correctness_test_ids",
        )
        crash_safety_test_ids = _clean_text_tuple(
            self.crash_safety_test_ids,
            "crash_safety_test_ids",
        )
        benchmark_artifact_ids = _clean_text_tuple(
            self.benchmark_artifact_ids,
            "benchmark_artifact_ids",
        )
        rollback_policy = _clean_optional_text(self.rollback_policy, "rollback_policy")
        fallback_policy = _clean_optional_text(self.fallback_policy, "fallback_policy")
        claim_boundary = self.claim_boundary.strip()
        if not claim_boundary:
            raise ValueError("claim_boundary must be non-empty")
        object.__setattr__(self, "artifact_id", artifact_id)
        object.__setattr__(
            self,
            "executable_lowering_evidence_id",
            executable_lowering_evidence_id,
        )
        object.__setattr__(self, "correctness_test_ids", correctness_test_ids)
        object.__setattr__(self, "crash_safety_test_ids", crash_safety_test_ids)
        object.__setattr__(self, "benchmark_artifact_ids", benchmark_artifact_ids)
        object.__setattr__(self, "rollback_policy", rollback_policy)
        object.__setattr__(self, "fallback_policy", fallback_policy)
        object.__setattr__(self, "claim_boundary", claim_boundary)

    @property
    def missing_requirements(self) -> tuple[str, ...]:
        """Return the ordered requirement keys blocking LLVM/JIT promotion."""

        missing: list[str] = []
        if not self.executable_lowering_verified or self.executable_lowering_evidence_id is None:
            missing.append("executable_lowering")
        if not self.correctness_test_ids:
            missing.append("correctness_tests")
        if not self.crash_safety_test_ids:
            missing.append("crash_safety_tests")
        if not self.benchmark_artifact_ids:
            missing.append("benchmark_artifact_ids")
        if self.rollback_policy is None:
            missing.append("rollback_policy")
        if self.fallback_policy is None:
            missing.append("fallback_policy")
        return tuple(requirement for requirement in _REQUIREMENT_ORDER if requirement in missing)

    @property
    def promotion_ready(self) -> bool:
        """Return whether every LLVM/JIT promotion requirement is attached."""

        return not self.missing_requirements

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready claim gate record with derived readiness fields."""

        return {
            "artifact_id": self.artifact_id,
            "executable_lowering_evidence_id": self.executable_lowering_evidence_id,
            "executable_lowering_verified": self.executable_lowering_verified,
            "correctness_test_ids": list(self.correctness_test_ids),
            "crash_safety_test_ids": list(self.crash_safety_test_ids),
            "benchmark_artifact_ids": list(self.benchmark_artifact_ids),
            "rollback_policy": self.rollback_policy,
            "fallback_policy": self.fallback_policy,
            "missing_requirements": list(self.missing_requirements),
            "promotion_ready": self.promotion_ready,
            "claim_boundary": self.claim_boundary,
        }


def _native_evidence_verifies_executable_lowering(
    evidence: NativeWholeProgramADExecutionEvidence,
) -> bool:
    """Return whether native execution evidence satisfies the lowering gate."""

    return (
        evidence.beyond_scalar_executed
        and bool(evidence.executed_operation_families)
        and evidence.max_gradient_error <= evidence.gradient_parity_tolerance
    )


def build_llvm_jit_claim_gate(
    *,
    artifact_id: str,
    native_execution_evidence: NativeWholeProgramADExecutionEvidence | None,
    correctness_test_ids: Sequence[str] = (),
    crash_safety_test_ids: Sequence[str] = (),
    benchmark_artifact_ids: Sequence[str] = (),
    rollback_policy: str | None = None,
    fallback_policy: str | None = None,
    claim_boundary: str = LLVM_JIT_CLAIM_GATE_BOUNDARY,
) -> LLVMJITClaimGate:
    """Build a promotion gate from native execution evidence and policy artefacts.

    Native execution evidence is deliberately necessary but insufficient. The
    returned gate derives ``promotion_ready`` from all required evidence classes
    so docs and claim ledgers cannot accidentally promote LLVM/JIT support from
    a single successful native execution record.
    """

    executable_lowering_evidence_id: str | None = None
    executable_lowering_verified = False
    if native_execution_evidence is not None:
        if not isinstance(native_execution_evidence, NativeWholeProgramADExecutionEvidence):
            raise ValueError(
                "native_execution_evidence must be NativeWholeProgramADExecutionEvidence"
            )
        executable_lowering_evidence_id = native_execution_evidence.artifact_id
        executable_lowering_verified = _native_evidence_verifies_executable_lowering(
            native_execution_evidence,
        )
    return LLVMJITClaimGate(
        artifact_id=artifact_id,
        executable_lowering_evidence_id=executable_lowering_evidence_id,
        executable_lowering_verified=executable_lowering_verified,
        correctness_test_ids=correctness_test_ids,
        crash_safety_test_ids=crash_safety_test_ids,
        benchmark_artifact_ids=benchmark_artifact_ids,
        rollback_policy=rollback_policy,
        fallback_policy=fallback_policy,
        claim_boundary=claim_boundary,
    )


def llvm_jit_claim_gate_from_dict(payload: Mapping[str, object]) -> LLVMJITClaimGate:
    """Rebuild and validate a serialized LLVM/JIT claim gate."""

    gate = LLVMJITClaimGate(
        artifact_id=_payload_str(payload.get("artifact_id"), "artifact_id"),
        executable_lowering_evidence_id=_payload_optional_str(
            payload.get("executable_lowering_evidence_id"),
            "executable_lowering_evidence_id",
        ),
        executable_lowering_verified=_payload_bool(
            payload.get("executable_lowering_verified"),
            "executable_lowering_verified",
        ),
        correctness_test_ids=_payload_text_tuple(
            payload.get("correctness_test_ids"),
            "correctness_test_ids",
        ),
        crash_safety_test_ids=_payload_text_tuple(
            payload.get("crash_safety_test_ids"),
            "crash_safety_test_ids",
        ),
        benchmark_artifact_ids=_payload_text_tuple(
            payload.get("benchmark_artifact_ids"),
            "benchmark_artifact_ids",
        ),
        rollback_policy=_payload_optional_str(payload.get("rollback_policy"), "rollback_policy"),
        fallback_policy=_payload_optional_str(payload.get("fallback_policy"), "fallback_policy"),
        claim_boundary=_payload_str(payload.get("claim_boundary"), "claim_boundary"),
    )
    if "missing_requirements" in payload:
        missing = _payload_text_tuple(payload["missing_requirements"], "missing_requirements")
        if missing != gate.missing_requirements:
            raise ValueError("missing_requirements must match derived gate requirements")
    if "promotion_ready" in payload:
        ready = _payload_bool(payload["promotion_ready"], "promotion_ready")
        if ready is not gate.promotion_ready:
            raise ValueError("promotion_ready must match derived gate readiness")
    return gate


def render_llvm_jit_claim_gate_markdown(gate: LLVMJITClaimGate) -> str:
    """Render a concise reviewer-facing Markdown summary for the claim gate."""

    if not isinstance(gate, LLVMJITClaimGate):
        raise ValueError("gate must be LLVMJITClaimGate")
    rows = [
        (
            "executable_lowering",
            "ready" if "executable_lowering" not in gate.missing_requirements else "blocked",
        ),
        ("correctness_tests", "ready" if gate.correctness_test_ids else "blocked"),
        ("crash_safety_tests", "ready" if gate.crash_safety_test_ids else "blocked"),
        ("benchmark_artifact_ids", "ready" if gate.benchmark_artifact_ids else "blocked"),
        ("rollback_policy", "ready" if gate.rollback_policy is not None else "blocked"),
        ("fallback_policy", "ready" if gate.fallback_policy is not None else "blocked"),
    ]
    markdown = [
        "# LLVM/JIT claim gate",
        "",
        f"- artifact_id: `{gate.artifact_id}`",
        f"- promotion_ready: {gate.promotion_ready}",
        f"- executable_lowering_evidence_id: `{gate.executable_lowering_evidence_id}`",
        f"- benchmark_artifact_ids: {', '.join(gate.benchmark_artifact_ids) or 'none'}",
        "",
        "| requirement | status |",
        "|-------------|--------|",
    ]
    markdown.extend(f"| {name} | {status} |" for name, status in rows)
    markdown.extend(("", f"Claim boundary: {gate.claim_boundary}", ""))
    return "\n".join(markdown)


__all__ = [
    "LLVM_JIT_CLAIM_GATE_BOUNDARY",
    "LLVMJITClaimGate",
    "build_llvm_jit_claim_gate",
    "llvm_jit_claim_gate_from_dict",
    "render_llvm_jit_claim_gate_markdown",
]
