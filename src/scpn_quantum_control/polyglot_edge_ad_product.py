# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — polyglot edge Program-AD product
"""Fail-closed polyglot edge Program-AD product.

This module governs the real bounded Rust replay and browser WASM kernel while
keeping the optional Julia tier honest. The committed rational replay is the
only browser bit-exact claim. Julia acceleration currently belongs to the
Kuramoto numerical tier; it is not a Program-AD implementation and is therefore
an explicit unsupported boundary. Edge requests never fall back silently to a
host Python or native Rust path.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, NoReturn, cast

from .differentiable_claim_ledger import REPO_ROOT
from .polyglot_parity_certificate import build_sample_certificate, verify_certificate
from .studio.program_ad_replay_artifact import (
    DEFAULT_PROGRAM_AD_REPLAY_JSON_PATH,
    MAX_PROGRAM_AD_REPLAY_INPUTS,
    MAX_PROGRAM_AD_REPLAY_IR_BYTES,
    PROGRAM_AD_REPLAY_ARTIFACT_ID,
    PROGRAM_AD_REPLAY_SCHEMA,
    inspect_program_ad_replay_artifact,
)

RuntimeId = Literal["rust_native_replay", "browser_wasm_replay", "julia_program_ad"]
"""Stable identifiers for the governed edge Program-AD runtimes."""

RuntimeKind = Literal["native_rust", "browser_wasm", "julia"]
"""Runtime implementation families."""

SupportPosture = Literal[
    "bounded_authority",
    "committed_sample_bitexact",
    "boundary_unsupported",
]
"""Support posture for one edge Program-AD runtime."""

PathOutcome = Literal["allowed", "refused"]
"""Structured edge-path decision outcomes."""

POLYGLOT_EDGE_AD_PRODUCT_SCHEMA: Final[str] = "polyglot_edge_ad_product.v2"
"""Schema identifier for the product registry."""

POLYGLOT_EDGE_AD_CERTIFICATE_SCHEMA: Final[str] = "polyglot_edge_ad_certificate.v2"
"""Schema identifier for the composed committed-sample certificate."""

POLYGLOT_EDGE_AD_CLAIM_BOUNDARY: Final[str] = (
    "Bounded edge Program-AD runtime contract: native Rust is the replay authority; "
    "browser WASM is bit-exact only for the committed rational value-and-gradient "
    "artefact and wasm-safe bounded replay; arbitrary programs, transcendentals, "
    "general linear algebra, live edge execution, and performance are not claimed; "
    "Julia Program-AD is unsupported because the existing Julia optional tier "
    "accelerates Kuramoto numerics only; no silent host fallback"
)
"""Shared non-promotional claim boundary."""

_WASM_SAFE_PARITY_FAMILY: Final[str] = "value_and_gradient_replay"
_COMMITTED_ARTIFACT_PATH: Final[Path] = REPO_ROOT / DEFAULT_PROGRAM_AD_REPLAY_JSON_PATH


@dataclass(frozen=True, slots=True)
class EdgeADRuntimeRow:
    """One governed runtime capability row.

    Parameters
    ----------
    runtime_id
        Stable runtime identifier.
    runtime_kind
        Runtime implementation family.
    title
        Human-readable title.
    summary
        Honest support summary.
    support_posture
        Bounded authority, committed sample, or unsupported boundary.
    authority_pointer
        Repository authority for the row.
    studio_verb_ids
        Existing Studio executive verbs that route this surface.
    wasm_safe_operations
        Operations admitted by the committed browser sample.
    max_ir_bytes
        Maximum accepted effect-IR bytes, zero when unsupported.
    max_inputs
        Maximum scalar input arity, zero when unsupported.
    silent_host_fallback
        Must always be false.
    general_program_ad
        Whether general Program AD is claimed; always false here.
    claim_boundary
        Product claim boundary.

    """

    runtime_id: RuntimeId
    runtime_kind: RuntimeKind
    title: str
    summary: str
    support_posture: SupportPosture
    authority_pointer: str
    studio_verb_ids: tuple[str, ...]
    wasm_safe_operations: tuple[str, ...]
    max_ir_bytes: int
    max_inputs: int
    silent_host_fallback: bool = False
    general_program_ad: bool = False
    claim_boundary: str = POLYGLOT_EDGE_AD_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate capability-row invariants."""
        if self.runtime_id not in {
            "rust_native_replay",
            "browser_wasm_replay",
            "julia_program_ad",
        }:
            raise ValueError(f"unknown runtime_id: {self.runtime_id!r}")
        if self.runtime_kind not in {"native_rust", "browser_wasm", "julia"}:
            raise ValueError(f"unknown runtime_kind: {self.runtime_kind!r}")
        if not self.title or not self.title.strip():
            raise ValueError("title must be non-empty")
        if not self.summary or not self.summary.strip():
            raise ValueError("summary must be non-empty")
        if self.support_posture not in {
            "bounded_authority",
            "committed_sample_bitexact",
            "boundary_unsupported",
        }:
            raise ValueError(f"unknown support_posture: {self.support_posture!r}")
        if not self.authority_pointer or not self.authority_pointer.strip():
            raise ValueError("authority_pointer must be non-empty")
        if not self.studio_verb_ids or any(not item.strip() for item in self.studio_verb_ids):
            raise ValueError("studio_verb_ids must contain non-empty entries")
        if len(set(self.studio_verb_ids)) != len(self.studio_verb_ids):
            raise ValueError("studio_verb_ids must be unique")
        if self.max_ir_bytes < 0 or self.max_inputs < 0:
            raise ValueError("runtime bounds must be non-negative")
        if self.silent_host_fallback:
            raise ValueError("silent_host_fallback must be False")
        if self.general_program_ad:
            raise ValueError("general_program_ad must be False")
        if self.support_posture == "boundary_unsupported":
            if self.max_ir_bytes != 0 or self.max_inputs != 0:
                raise ValueError("unsupported runtimes must use zero bounds")
            if self.wasm_safe_operations:
                raise ValueError("unsupported runtimes cannot list wasm-safe operations")
        elif self.max_ir_bytes <= 0 or self.max_inputs <= 0:
            raise ValueError("supported bounded runtimes require positive bounds")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready capability row."""
        return {
            "runtime_id": self.runtime_id,
            "runtime_kind": self.runtime_kind,
            "title": self.title,
            "summary": self.summary,
            "support_posture": self.support_posture,
            "authority_pointer": self.authority_pointer,
            "studio_verb_ids": list(self.studio_verb_ids),
            "wasm_safe_operations": list(self.wasm_safe_operations),
            "max_ir_bytes": self.max_ir_bytes,
            "max_inputs": self.max_inputs,
            "silent_host_fallback": self.silent_host_fallback,
            "general_program_ad": self.general_program_ad,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True, slots=True)
class EdgeADPathDecision:
    """Fail-closed decision for one requested runtime path."""

    runtime_id: str
    outcome: PathOutcome
    allowed: bool
    reason: str
    blockers: tuple[str, ...]
    host_fallback_used: bool = False
    claim_boundary: str = POLYGLOT_EDGE_AD_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate decision invariants."""
        if not self.runtime_id or not self.runtime_id.strip():
            raise ValueError("runtime_id must be non-empty")
        if self.outcome not in {"allowed", "refused"}:
            raise ValueError(f"unknown outcome: {self.outcome!r}")
        if not self.reason or not self.reason.strip():
            raise ValueError("reason must be non-empty")
        if self.allowed != (self.outcome == "allowed"):
            raise ValueError("allowed must agree with outcome")
        if self.allowed and self.blockers:
            raise ValueError("allowed decisions cannot list blockers")
        if not self.allowed and not self.blockers:
            raise ValueError("refused decisions require blockers")
        if any(not item or not item.strip() for item in self.blockers):
            raise ValueError("blockers entries must be non-empty")
        if self.host_fallback_used:
            raise ValueError("host_fallback_used must be False")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready decision."""
        return {
            "runtime_id": self.runtime_id,
            "outcome": self.outcome,
            "allowed": self.allowed,
            "reason": self.reason,
            "blockers": list(self.blockers),
            "host_fallback_used": self.host_fallback_used,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True, slots=True)
class CommittedWasmReplayCertificate:
    """Composed value-and-gradient parity certificate for browser WASM."""

    schema: str
    artifact_id: str
    artifact_schema: str
    input_sha256: str
    expected_value: float | None
    expected_gradient: tuple[float, ...]
    parity_family_id: str
    artifact_verified: bool
    parity_verified: bool
    supported: bool
    blockers: tuple[str, ...]
    claim_boundary: str = POLYGLOT_EDGE_AD_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate certificate invariants."""
        if self.schema != POLYGLOT_EDGE_AD_CERTIFICATE_SCHEMA:
            raise ValueError("unknown edge certificate schema")
        if not self.parity_family_id or not self.parity_family_id.strip():
            raise ValueError("parity_family_id must be non-empty")
        if any(not item or not item.strip() for item in self.blockers):
            raise ValueError("blockers entries must be non-empty")
        if self.supported:
            if not self.artifact_verified or not self.parity_verified:
                raise ValueError("supported certificates require both verifications")
            if self.blockers:
                raise ValueError("supported certificates cannot list blockers")
            if self.artifact_id != PROGRAM_AD_REPLAY_ARTIFACT_ID:
                raise ValueError("supported certificate artifact_id drift")
            if self.artifact_schema != PROGRAM_AD_REPLAY_SCHEMA:
                raise ValueError("supported certificate artifact_schema drift")
            if not self.input_sha256.startswith("sha256:") or len(self.input_sha256) != 71:
                raise ValueError("supported certificate requires a SHA-256 input digest")
            if self.expected_value is None or not self.expected_gradient:
                raise ValueError("supported certificate requires expected value and gradient")
        elif not self.blockers:
            raise ValueError("unsupported certificates require blockers")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready certificate."""
        return {
            "schema": self.schema,
            "artifact_id": self.artifact_id,
            "artifact_schema": self.artifact_schema,
            "input_sha256": self.input_sha256,
            "expected_value": self.expected_value,
            "expected_gradient": list(self.expected_gradient),
            "parity_family_id": self.parity_family_id,
            "artifact_verified": self.artifact_verified,
            "parity_verified": self.parity_verified,
            "supported": self.supported,
            "blockers": list(self.blockers),
            "claim_boundary": self.claim_boundary,
        }


_RUNTIMES: Final[tuple[EdgeADRuntimeRow, ...]] = (
    EdgeADRuntimeRow(
        runtime_id="rust_native_replay",
        runtime_kind="native_rust",
        title="Bounded native Rust Program-AD replay",
        summary="Reference authority shared by the Python engine and browser WASM crate.",
        support_posture="bounded_authority",
        authority_pointer="scpn_quantum_engine/program_ad_replay",
        studio_verb_ids=("differentiate", "replay"),
        wasm_safe_operations=("parameter", "add", "sub", "mul", "div", "neg"),
        max_ir_bytes=MAX_PROGRAM_AD_REPLAY_IR_BYTES,
        max_inputs=MAX_PROGRAM_AD_REPLAY_INPUTS,
    ),
    EdgeADRuntimeRow(
        runtime_id="browser_wasm_replay",
        runtime_kind="browser_wasm",
        title="Browser WASM committed rational replay",
        summary=(
            "Standalone WASM build of the same bounded Rust replay; product support is "
            "limited to the committed rational value+gradient artefact."
        ),
        support_posture="committed_sample_bitexact",
        authority_pointer="scpn_quantum_engine/studio_program_ad_wasm",
        studio_verb_ids=("replay",),
        wasm_safe_operations=("parameter", "add", "mul"),
        max_ir_bytes=MAX_PROGRAM_AD_REPLAY_IR_BYTES,
        max_inputs=MAX_PROGRAM_AD_REPLAY_INPUTS,
    ),
    EdgeADRuntimeRow(
        runtime_id="julia_program_ad",
        runtime_kind="julia",
        title="Julia Program-AD boundary",
        summary=(
            "The julia optional extra and oscillatools Julia sources accelerate Kuramoto "
            "numerics; no Julia Program-AD replay authority exists."
        ),
        support_posture="boundary_unsupported",
        authority_pointer="oscillatools.accel.julia",
        studio_verb_ids=("differentiate",),
        wasm_safe_operations=(),
        max_ir_bytes=0,
        max_inputs=0,
    ),
)


def _runtime_map() -> dict[str, EdgeADRuntimeRow]:
    """Build the runtime map while refusing blank or duplicate identifiers."""
    result: dict[str, EdgeADRuntimeRow] = {}
    for row in _RUNTIMES:
        key = row.runtime_id.strip()
        if not key:
            raise RuntimeError("edge runtime catalogue contains blank runtime_id")
        if key in result:
            raise RuntimeError(f"duplicate runtime_id in edge catalogue: {key!r}")
        result[key] = row
    if not result:
        raise RuntimeError("edge runtime catalogue must be non-empty")
    return result


_RUNTIME_BY_ID: Final[Mapping[str, EdgeADRuntimeRow]] = _runtime_map()


def list_edge_ad_runtime_ids() -> tuple[str, ...]:
    """Return runtime identifiers in stable catalogue order."""
    return tuple(row.runtime_id for row in _RUNTIMES)


def get_edge_ad_runtime(runtime_id: str) -> EdgeADRuntimeRow:
    """Return one runtime row and fail closed on blank or unknown ids."""
    if not runtime_id or not str(runtime_id).strip():
        raise ValueError("runtime_id must be non-empty")
    key = str(runtime_id).strip()
    try:
        return _RUNTIME_BY_ID[key]
    except KeyError as exc:
        raise ValueError(f"unknown runtime_id: {key!r}") from exc


def iter_edge_ad_runtimes(
    *, support_posture: SupportPosture | None = None
) -> tuple[EdgeADRuntimeRow, ...]:
    """Return runtime rows, optionally filtered by support posture."""
    rows: Sequence[EdgeADRuntimeRow] = _RUNTIMES
    if support_posture is not None:
        rows = tuple(row for row in rows if row.support_posture == support_posture)
    return tuple(rows)


def decide_edge_ad_path(
    runtime_id: str,
    *,
    studio_verb_id: str,
    artifact_payload: Mapping[str, object] | None = None,
    committed_sample_only: bool = True,
    request_general_program_ad: bool = False,
    request_host_fallback: bool = False,
) -> EdgeADPathDecision:
    """Decide whether a bounded edge Program-AD path may proceed.

    Browser WASM is admitted only for the verified committed sample. Julia is
    always refused until a real Julia Program-AD authority exists. A requested
    host fallback is always a blocker, including when the native path could run.
    """
    row = get_edge_ad_runtime(runtime_id)
    blockers: list[str] = []
    if not studio_verb_id or not studio_verb_id.strip():
        blockers.append("studio_verb_id must be non-empty")
    elif studio_verb_id not in row.studio_verb_ids:
        blockers.append(
            f"Studio verb {studio_verb_id!r} is not routed for runtime {row.runtime_id!r}"
        )
    if request_host_fallback:
        blockers.append("silent or requested host fallback is forbidden for an edge path")
    if request_general_program_ad:
        blockers.append("general Program-AD execution is outside the bounded runtime contract")
    if row.support_posture == "boundary_unsupported":
        blockers.append(
            "Julia Program-AD is unsupported; the existing Julia tier is Kuramoto-only"
        )
    if row.runtime_kind == "browser_wasm":
        certificate = materialise_wasm_replay_certificate(artifact_payload)
        if not certificate.supported:
            blockers.append("browser WASM requires a verified committed replay artefact")
            blockers.extend(f"artefact: {item}" for item in certificate.blockers)
        if not committed_sample_only:
            blockers.append("browser WASM support is limited to the committed rational sample")
    if blockers:
        return EdgeADPathDecision(
            runtime_id=row.runtime_id,
            outcome="refused",
            allowed=False,
            reason="edge Program-AD path refused under the fail-closed runtime policy",
            blockers=tuple(blockers),
        )
    return EdgeADPathDecision(
        runtime_id=row.runtime_id,
        outcome="allowed",
        allowed=True,
        reason=(
            f"bounded edge Program-AD path allowed for {row.runtime_id!r} via "
            f"Studio verb {studio_verb_id!r}"
        ),
        blockers=(),
    )


def _reject_json_constant(value: str) -> NoReturn:
    """Reject non-standard JSON constants."""
    raise ValueError(f"non-standard JSON constant {value!r} is forbidden")


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Build a JSON object while rejecting duplicate keys."""
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r} is forbidden")
        result[key] = value
    return result


def load_committed_wasm_replay_payload(path: Path = _COMMITTED_ARTIFACT_PATH) -> dict[str, object]:
    """Load the committed replay artefact as strict object-root JSON."""
    decoded = cast(
        object,
        json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        ),
    )
    if not isinstance(decoded, dict):
        raise ValueError("committed WASM replay artefact must be a JSON object")
    return cast(dict[str, object], decoded)


def _trusted_artifact_fields(
    payload: Mapping[str, object],
) -> tuple[str, str, str, float, tuple[float, ...]]:
    """Extract trusted certificate fields after artifact validation."""
    artifact_id = payload.get("artifact_id")
    artifact_schema = payload.get("schema")
    input_sha256 = payload.get("input_sha256")
    expected = payload.get("expected")
    if not isinstance(artifact_id, str) or not isinstance(artifact_schema, str):
        raise ValueError("verified artefact identifiers must be strings")
    if not isinstance(input_sha256, str):
        raise ValueError("verified artefact input_sha256 must be a string")
    if not isinstance(expected, Mapping):
        raise ValueError("verified artefact expected field must be a mapping")
    value = expected.get("value")
    gradient = expected.get("gradient")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("verified artefact expected value must be numeric")
    if not isinstance(gradient, list) or not gradient:
        raise ValueError("verified artefact expected gradient must be non-empty")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in gradient):
        raise ValueError("verified artefact expected gradient must be numeric")
    return (
        artifact_id,
        artifact_schema,
        input_sha256,
        float(value),
        tuple(float(cast(int | float, item)) for item in gradient),
    )


def materialise_wasm_replay_certificate(
    payload: Mapping[str, object] | None = None,
) -> CommittedWasmReplayCertificate:
    """Compose committed replay validation with value-and-gradient parity evidence."""
    blockers: list[str] = []
    candidate: Mapping[str, object]
    try:
        candidate = payload if payload is not None else load_committed_wasm_replay_payload()
        artifact_validation = inspect_program_ad_replay_artifact(candidate)
    except (OSError, UnicodeError, ValueError, RuntimeError) as exc:
        candidate = payload or {}
        artifact_validation = None
        blockers.append(f"committed artefact could not be verified: {type(exc).__name__}: {exc}")

    artifact_verified = artifact_validation is not None and artifact_validation.passed
    if artifact_validation is not None and not artifact_validation.passed:
        blockers.extend(artifact_validation.errors)

    parity = build_sample_certificate(_WASM_SAFE_PARITY_FAMILY)
    parity_decision = verify_certificate(parity, expect_supported=True)
    parity_verified = parity_decision.passed
    if not parity_verified:
        blockers.extend(parity_decision.blockers or (parity_decision.reason,))

    artifact_id = ""
    artifact_schema = ""
    input_sha256 = ""
    expected_value: float | None = None
    expected_gradient: tuple[float, ...] = ()
    if artifact_verified:
        try:
            (
                artifact_id,
                artifact_schema,
                input_sha256,
                expected_value,
                expected_gradient,
            ) = _trusted_artifact_fields(candidate)
        except ValueError as exc:
            artifact_verified = False
            blockers.append(str(exc))

    supported = artifact_verified and parity_verified and not blockers
    if not supported and not blockers:
        blockers.append("composed WASM replay certificate is unsupported")
    return CommittedWasmReplayCertificate(
        schema=POLYGLOT_EDGE_AD_CERTIFICATE_SCHEMA,
        artifact_id=artifact_id,
        artifact_schema=artifact_schema,
        input_sha256=input_sha256,
        expected_value=expected_value,
        expected_gradient=expected_gradient,
        parity_family_id=_WASM_SAFE_PARITY_FAMILY,
        artifact_verified=artifact_verified,
        parity_verified=parity_verified,
        supported=supported,
        blockers=tuple(blockers),
    )


def map_polyglot_edge_ad_public_surfaces() -> tuple[dict[str, object], ...]:
    """Return the bounded edge Program-AD public surface map."""
    return (
        {
            "module_path": "scpn_quantum_control.polyglot_edge_ad_product",
            "role": "polyglot_edge_ad_product_surface",
            "runtime_ids": list(list_edge_ad_runtime_ids()),
            "claim_boundary": POLYGLOT_EDGE_AD_CLAIM_BOUNDARY,
        },
        {
            "module_path": "scpn_quantum_control.studio.program_ad_replay_artifact",
            "role": "committed_wasm_replay_artifact_authority",
            "artifact_id": PROGRAM_AD_REPLAY_ARTIFACT_ID,
            "claim_boundary": POLYGLOT_EDGE_AD_CLAIM_BOUNDARY,
        },
        {
            "module_path": "scpn_quantum_control.polyglot_parity_certificate",
            "role": "parity_certificate_subset_authority",
            "family_id": _WASM_SAFE_PARITY_FAMILY,
            "claim_boundary": POLYGLOT_EDGE_AD_CLAIM_BOUNDARY,
        },
    )


def build_polyglot_edge_ad_product_registry() -> dict[str, object]:
    """Build the deterministic edge Program-AD runtime registry."""
    runtimes = [row.to_dict() for row in _RUNTIMES]
    return {
        "schema": POLYGLOT_EDGE_AD_PRODUCT_SCHEMA,
        "claim_boundary": POLYGLOT_EDGE_AD_CLAIM_BOUNDARY,
        "runtime_count": len(runtimes),
        "blank_entry_count": 0,
        "default_runtime_id": "browser_wasm_replay",
        "silent_host_fallback_policy": False,
        "general_program_ad_policy": False,
        "wasm_artifact_id": PROGRAM_AD_REPLAY_ARTIFACT_ID,
        "wasm_artifact_schema": PROGRAM_AD_REPLAY_SCHEMA,
        "wasm_safe_parity_family": _WASM_SAFE_PARITY_FAMILY,
        "public_surfaces": list(map_polyglot_edge_ad_public_surfaces()),
        "runtimes": runtimes,
    }


def assert_polyglot_edge_ad_product_integrity(
    payload: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Validate the edge Program-AD registry and return a plain dictionary."""
    registry = dict(payload) if payload is not None else build_polyglot_edge_ad_product_registry()
    if registry.get("schema") != POLYGLOT_EDGE_AD_PRODUCT_SCHEMA:
        raise ValueError("unknown edge Program-AD product schema")
    if registry.get("claim_boundary") != POLYGLOT_EDGE_AD_CLAIM_BOUNDARY:
        raise ValueError("edge Program-AD claim boundary drift")
    runtimes_obj = registry.get("runtimes")
    if not isinstance(runtimes_obj, list) or not runtimes_obj:
        raise ValueError("edge product registry requires non-empty runtimes")
    seen: set[str] = set()
    blank = 0
    wasm_found = False
    julia_found = False
    for raw in runtimes_obj:
        if not isinstance(raw, Mapping):
            raise ValueError("edge product runtime row must be a mapping")
        runtime_id = raw.get("runtime_id")
        if not isinstance(runtime_id, str) or not runtime_id.strip():
            blank += 1
            continue
        if runtime_id in seen:
            raise ValueError(f"duplicate runtime_id in registry: {runtime_id!r}")
        seen.add(runtime_id)
        if raw.get("silent_host_fallback") is not False:
            raise ValueError("runtime silent_host_fallback must be False")
        if raw.get("general_program_ad") is not False:
            raise ValueError("runtime general_program_ad must be False")
        if raw.get("claim_boundary") != POLYGLOT_EDGE_AD_CLAIM_BOUNDARY:
            raise ValueError("runtime claim boundary drift")
        if runtime_id == "browser_wasm_replay":
            wasm_found = True
            if raw.get("support_posture") != "committed_sample_bitexact":
                raise ValueError("browser WASM support posture drift")
        if runtime_id == "julia_program_ad":
            julia_found = True
            if raw.get("support_posture") != "boundary_unsupported":
                raise ValueError("Julia Program-AD must remain boundary_unsupported")
    if blank:
        raise ValueError(f"edge product registry has {blank} blank runtime row(s)")
    if not wasm_found or not julia_found:
        raise ValueError("edge product registry must include browser WASM and Julia boundary rows")
    if registry.get("runtime_count") != len(runtimes_obj):
        raise ValueError("runtime_count does not match runtimes length")
    if registry.get("blank_entry_count") != 0:
        raise ValueError("blank_entry_count must be zero")
    if registry.get("silent_host_fallback_policy") is not False:
        raise ValueError("silent_host_fallback_policy must be False")
    if registry.get("general_program_ad_policy") is not False:
        raise ValueError("general_program_ad_policy must be False")
    if seen != set(list_edge_ad_runtime_ids()):
        raise ValueError("runtime catalogue drift vs canonical product")
    if registry.get("public_surfaces") != list(map_polyglot_edge_ad_public_surfaces()):
        raise ValueError("edge Program-AD public surface map drift")
    return registry


__all__ = [
    "POLYGLOT_EDGE_AD_CERTIFICATE_SCHEMA",
    "POLYGLOT_EDGE_AD_CLAIM_BOUNDARY",
    "POLYGLOT_EDGE_AD_PRODUCT_SCHEMA",
    "CommittedWasmReplayCertificate",
    "EdgeADPathDecision",
    "EdgeADRuntimeRow",
    "assert_polyglot_edge_ad_product_integrity",
    "build_polyglot_edge_ad_product_registry",
    "decide_edge_ad_path",
    "get_edge_ad_runtime",
    "iter_edge_ad_runtimes",
    "list_edge_ad_runtime_ids",
    "load_committed_wasm_replay_payload",
    "map_polyglot_edge_ad_public_surfaces",
    "materialise_wasm_replay_certificate",
]
