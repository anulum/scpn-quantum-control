# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — native semantic source custody
"""Bind native producer metadata to exact, detached semantic source records.

This module only proves source identity, version and content custody. It does
not qualify physical units, circuit execution, hardware attestation or a
scientific verdict; those remain separate companion and domain decisions.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final, Literal, cast

from .differentiable_parameter_contracts import Parameter
from .differentiable_result_contracts import GradientResult, StochasticGradientResult
from .hardware.hal import BackendProfile, QuantumJobRef, QuantumJobResult, QuantumWorkload
from .phase.qnode_circuit_contracts import PhaseQNodeClassicalFisherResult
from .program_ad_registry import PrimitiveContract
from .stable_core_product import canonical_json_bytes, digest_stable_core_payload

NativeSource = (
    PrimitiveContract
    | GradientResult
    | StochasticGradientResult
    | Parameter
    | BackendProfile
    | QuantumWorkload
    | QuantumJobRef
    | QuantumJobResult
    | PhaseQNodeClassicalFisherResult
)
"""Current typed owners that publish a detached semantic source projection."""

NATIVE_SOURCE_SCHEMAS: Final[Mapping[type[object], str]] = {
    PrimitiveContract: "program_ad.primitive_contract.v1",
    GradientResult: "differentiable.gradient_result.v1",
    StochasticGradientResult: "differentiable.stochastic_gradient_result.v1",
    Parameter: "differentiable.parameter.v1",
    BackendProfile: "hal.backend_profile.v1",
    QuantumWorkload: "hal.quantum_workload.v1",
    QuantumJobRef: "hal.quantum_job_ref.v1",
    QuantumJobResult: "hal.quantum_job_result.v1",
    PhaseQNodeClassicalFisherResult: "phase_qnode.classical_fisher_result.v1",
}
"""Source-specific metadata versions; none changes stable-core v2."""

STUDIO_EXECUTION_PLAN_SCHEMA: Final = "studio.execution_plan.v1"
"""Optional Studio plan schema, resolved without importing Studio at core load."""

SYNTHETIC_UNIT_DECLARATION_ORIGIN: Final = "caller_declared_synthetic"
"""Closed provenance label for synthetic dimensionless caller declarations."""

NativeRefusalCode = Literal[
    "source_schema_mismatch",
    "source_identity_mismatch",
    "source_digest_mismatch",
    "source_content_mismatch",
    "source_record_malformed",
]
"""Exact reasons a retained source cannot be bound to its actual owner."""


@dataclass(frozen=True, slots=True)
class NativeSourceBinding:
    """Result of comparing retained metadata with the actual native object.

    Attributes
    ----------
    source_identity
        Module-qualified type of the original native owner object.
    source_digest
        Digest of the actual owner's detached projection.
    reasons
        Named source-custody refusals; empty only for an exact match.

    """

    source_identity: str
    source_digest: str
    reasons: tuple[NativeRefusalCode, ...]

    @property
    def matched(self) -> bool:
        """Whether the retained version, identity, digest and content all match."""
        return not self.reasons


def capture_native_source(source: object) -> dict[str, Any]:
    """Capture one supported owner's exact JSON-ready metadata projection.

    Parameters
    ----------
    source
        Actual typed Program-AD, derivative, HAL or Studio plan owner.

    Returns
    -------
    dict[str, Any]
        Detached source-specific version, owner identity, payload and digest.

    Raises
    ------
    ValueError
        If the owner type is unsupported or its projection is not JSON-ready.

    """
    identity = f"{type(source).__module__}.{type(source).__qualname__}"
    schema = NATIVE_SOURCE_SCHEMAS.get(type(source))
    if schema is None:
        if identity != "scpn_quantum_control.studio.executive.ExecutionPlan":
            raise ValueError(f"unsupported native semantic source: {type(source)!r}")
        try:
            from .studio.executive import ExecutionPlan
        except ImportError as exc:
            raise ValueError("optional Studio plan source is unavailable") from exc
        if type(source) is not ExecutionPlan:
            raise ValueError("unsupported native semantic source impersonating Studio plan")
        schema = STUDIO_EXECUTION_PLAN_SCHEMA
        projected = source.to_semantic_source()
    else:
        projected = cast(NativeSource, source).to_semantic_source()
    if projected.get("producer_identity") != identity:
        raise ValueError("native semantic source projection changed producer identity")
    detached = json.loads(canonical_json_bytes(projected))
    return {
        "schema": schema,
        "producer_identity": identity,
        "record": detached,
        "record_sha256": digest_stable_core_payload(detached),
    }


def validate_native_source_record(source: object, retained: object) -> NativeSourceBinding:
    """Compare retained source bytes to the actual typed owner's projection.

    Parameters
    ----------
    source
        Actual typed owner; no adapter, device or numerical engine is invoked.
    retained
        Previously captured source record to verify.

    Returns
    -------
    NativeSourceBinding
        Exact-match decision with independent version, identity, digest and
        content refusals. A retained self-hash alone never proves the source.

    Raises
    ------
    ValueError
        If the actual source is an unsupported Python type.

    """
    actual = capture_native_source(source)
    reasons: list[NativeRefusalCode] = []
    identity = str(actual["producer_identity"])
    digest = str(actual["record_sha256"])
    if not isinstance(retained, Mapping):
        return NativeSourceBinding(identity, digest, ("source_record_malformed",))
    if set(retained) != set(actual):
        reasons.append("source_record_malformed")
    if retained.get("schema") != actual["schema"]:
        reasons.append("source_schema_mismatch")
    if retained.get("producer_identity") != identity:
        reasons.append("source_identity_mismatch")
    if retained.get("record_sha256") != digest:
        reasons.append("source_digest_mismatch")
    record = retained.get("record")
    if not isinstance(record, Mapping):
        reasons.append("source_content_mismatch")
    else:
        try:
            retained_bytes = canonical_json_bytes(record)
        except (TypeError, ValueError):
            reasons.append("source_content_mismatch")
        else:
            if retained_bytes != canonical_json_bytes(actual["record"]):
                reasons.append("source_content_mismatch")
    return NativeSourceBinding(identity, digest, tuple(reasons))


def validate_stochastic_result_companion(
    companion: Mapping[str, Any],
    raw_record: Mapping[str, Any],
    raw_digest: str,
    native_sources: Mapping[str, object],
    claim_boundary: str,
) -> list[tuple[str, str]]:
    """Bind synthetic derivative metadata to a real result and unchanged v2 bytes.

    The unit declaration is explicitly caller supplied for a synthetic,
    dimensionless objective. This check verifies source custody and exact
    numerical components; it does not establish physical units or hardware
    execution.

    Parameters
    ----------
    companion
        Versioned companion payload to check.
    raw_record
        Original stable-core result envelope.
    raw_digest
        Digest measured from the original envelope.
    native_sources
        Actual typed source owners.
    claim_boundary
        Accepted no-hardware claim for this synthetic result.

    Returns
    -------
    list[tuple[str, str]]
        Field paths and exact reasons for any unsupported assertion.

    """
    issues: list[tuple[str, str]] = []
    binding = companion.get("source_binding")
    binding = binding if isinstance(binding, Mapping) else {}
    ref = binding.get("native_source_ref")
    retained_sources = companion.get("source_records")
    retained_sources = retained_sources if isinstance(retained_sources, Mapping) else {}
    retained = retained_sources.get(ref) if isinstance(ref, str) else None
    owner = native_sources.get(ref) if isinstance(ref, str) else None
    if not isinstance(owner, StochasticGradientResult) or not isinstance(retained, Mapping):
        return [("source_binding", "a retained, actual stochastic gradient owner is required")]
    if set(retained_sources) != {ref} or set(native_sources) != {ref}:
        issues.append(("source_records", "synthetic result must have exactly one native owner"))
    source_binding = validate_native_source_record(owner, retained)
    if not source_binding.matched:
        issues.append(("source_records", f"native source differs: {source_binding.reasons!r}"))
    projected = owner.to_semantic_source()
    body = raw_record.get("body")
    body = body if isinstance(body, Mapping) else {}
    metadata = body.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    expected_binding = {
        "native_source_ref": ref,
        "raw_field": "body.metadata.native_source_record_sha256",
    }
    if (
        binding != expected_binding
        or metadata.get("native_source_record_sha256") != source_binding.source_digest
    ):
        issues.append(("source_binding", "raw result does not reference this exact native source"))
    if body.get("status") != "succeeded" or body.get("observables") != {"objective": owner.value}:
        issues.append(
            ("raw_record.body.observables", "raw objective does not equal the native result")
        )
    if body.get("backend_id") != "caller-supplied":
        issues.append(("raw_record.body.backend_id", "synthetic source is caller-supplied only"))
    expected_backend = {
        "source_record": "raw_record",
        "field_path": "body.backend_id",
        "record_digest": raw_digest,
        "backend_id": body.get("backend_id"),
        "stage": "result",
    }
    if companion.get("backend_reference") != expected_backend:
        issues.append(("backend_reference", "backend reference differs from the raw result"))
    if companion.get("producer_identity") != projected["producer_identity"]:
        issues.append(("producer_identity", "native result identity differs"))
    if companion.get("modality") != "stochastic_derivative_result":
        issues.append(
            ("modality", "native result is a stochastic derivative, not another modality")
        )
    if companion.get("parameter_order") != projected["parameter_names"]:
        issues.append(("parameter_order", "native parameter order differs"))
    if companion.get("trainable_mask") != projected["trainable"]:
        issues.append(("trainable_mask", "native trainability differs"))
    if companion.get("tangent_convention") != "forward_real":
        issues.append(
            ("tangent_convention", "the native shift result has a real forward convention")
        )
    expected_field = {
        "gradient": {
            "dtype": str(owner.gradient.dtype),
            "shape": list(owner.gradient.shape),
            "unit": "1",
        }
    }
    if companion.get("fields") != expected_field:
        issues.append(("fields.gradient", "gradient layout or caller-declared unit differs"))
    if companion.get("claim_boundary") != claim_boundary:
        issues.append(
            ("claim_boundary", "synthetic source cannot claim hardware or physical units")
        )
    if companion.get("calibration_reference") is not None:
        issues.append(("calibration_reference", "native result has no calibration evidence"))
    required_unavailable = {
        "hardware_execution",
        "calibration_reference",
        "unit_conversion",
        "supported_transform_composition",
        "native_parameter_units",
    }
    unavailable = companion.get("unavailable")
    if not isinstance(unavailable, list) or not required_unavailable.issubset(
        {item for item in unavailable if isinstance(item, str)}
    ):
        issues.append(("unavailable", "missing native-unit or hardware limitation"))
    expected_units = {
        "origin": SYNTHETIC_UNIT_DECLARATION_ORIGIN,
        "objective": "1",
        "parameters": dict.fromkeys(owner.parameter_names, "1"),
    }
    if companion.get("fidelity_unit_declaration") != expected_units:
        issues.append(("fidelity_unit_declaration", "synthetic caller unit declaration differs"))
    settings = companion.get("settings")
    expected_settings = {
        "stage": "observation",
        "requested": {},
        "effective": {},
        "origins": {},
        "rejected_fields": [],
    }
    if settings != expected_settings:
        issues.append(
            ("settings", "native result has no settings to infer from this raw envelope")
        )
    components = companion.get("fidelity_components")
    if not isinstance(components, list) or len(components) != 2:
        return issues + [
            ("fidelity_components", "both native uncertainty descriptions are required")
        ]
    expected_assumptions = [
        "caller declares this synthetic objective and both parameters dimensionless",
        "supplied shifted estimates; not hardware observations",
        "radius and standard error describe the same covariance, not independent errors",
    ]
    seen: set[str] = set()
    for index, component in enumerate(components):
        path = f"fidelity_components.{index}"
        if not isinstance(component, Mapping):
            issues.append((path, "component must be a mapping"))
            continue
        kind = component.get("kind")
        if (
            not isinstance(kind, str)
            or kind
            not in {
                "standard_error",
                "confidence_radius",
            }
            or kind in seen
        ):
            issues.append((path, "unsupported or duplicate uncertainty kind"))
            continue
        seen.add(kind)
        expected_ref = {
            "source": ref,
            "sha256": source_binding.source_digest,
            "field_path": f"record.{kind}",
            "covariance_path": "record.covariance",
            "confidence_level_path": "record.confidence_level",
            "confidence_z_path": "record.confidence_interval.confidence_z",
        }
        if (
            component.get("value") != projected[kind]
            or component.get("method") != projected["method"]
            or component.get("estimand") != "gradient in native parameter_names order"
            or component.get("unit") != "1"
            or component.get("assumptions") != expected_assumptions
            or component.get("evidence_ref") != expected_ref
        ):
            issues.append(
                (path, "component differs from the actual native uncertainty and provenance")
            )
    if seen != {"standard_error", "confidence_radius"}:
        issues.append(("fidelity_components", "one native uncertainty description is missing"))
    return issues


__all__ = [
    "NATIVE_SOURCE_SCHEMAS",
    "STUDIO_EXECUTION_PLAN_SCHEMA",
    "SYNTHETIC_UNIT_DECLARATION_ORIGIN",
    "NativeRefusalCode",
    "NativeSource",
    "NativeSourceBinding",
    "capture_native_source",
    "validate_native_source_record",
    "validate_stochastic_result_companion",
]
