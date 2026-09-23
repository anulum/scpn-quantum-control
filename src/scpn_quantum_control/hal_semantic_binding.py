# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — HAL result semantic binding
"""Bind HAL result metadata without conferring unsupported count semantics."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

from .hardware.hal import QuantumJobResult
from .native_semantic_binding import validate_native_source_record

HAL_METADATA_CLAIM_BOUNDARY: Final = (
    "provider-neutral HAL result metadata bound to an actual typed job and unchanged "
    "raw result; native counts are retained but bit/measurement mapping is unknown, "
    "so count semantics remain unqualified; no statevector, hardware attestation, "
    "calibration, unit conversion or uncertainty claim"
)
"""Narrow metadata qualification without measurement-wiring authority."""


def validate_hal_result_companion(
    companion: Mapping[str, Any],
    raw_record: Mapping[str, Any],
    raw_digest: str,
    native_sources: Mapping[str, object],
) -> list[tuple[str, str]]:
    """Check an actual HAL job/result and preserve its unqualified native counts.

    No count bit mapping or statevector is inferred from a count-shaped result.
    The caller must supply the typed result from its HAL route; this validator
    neither invokes a backend nor promotes a provider or hardware claim.
    """
    issues: list[tuple[str, str]] = []
    binding = companion.get("source_binding")
    binding = binding if isinstance(binding, Mapping) else {}
    ref = binding.get("native_source_ref")
    retained_sources = companion.get("source_records")
    retained_sources = retained_sources if isinstance(retained_sources, Mapping) else {}
    retained = retained_sources.get(ref) if isinstance(ref, str) else None
    owner = native_sources.get(ref) if isinstance(ref, str) else None
    if type(owner) is not QuantumJobResult or not isinstance(retained, Mapping):
        return [("source_binding", "a retained, actual HAL job result is required")]
    if set(retained_sources) != {ref} or set(native_sources) != {ref}:
        issues.append(("source_records", "HAL result must have exactly one native owner"))
    source_binding = validate_native_source_record(owner, retained)
    if not source_binding.matched:
        issues.append(("source_records", f"native HAL result differs: {source_binding.reasons!r}"))
    projected = owner.to_semantic_source()
    body = raw_record.get("body")
    body = body if isinstance(body, Mapping) else {}
    metadata = body.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    if set(metadata) != {"native_source_record_sha256", "source_job_id"}:
        issues.append(("raw_record.body.metadata", "unbound HAL metadata cannot be qualified"))
    expected_binding = {
        "native_source_ref": ref,
        "raw_field": "body.metadata.native_source_record_sha256",
    }
    if (
        binding != expected_binding
        or metadata.get("native_source_record_sha256") != source_binding.source_digest
    ):
        issues.append(("source_binding", "raw result does not reference this HAL source"))
    if metadata.get("source_job_id") != owner.job.job_id:
        issues.append(("raw_record.body.metadata.source_job_id", "HAL job identity differs"))
    if (
        body.get("status") != "succeeded"
        or body.get("experiment_id") != owner.job.workload_id
        or body.get("backend_id") != owner.job.backend_id
        or body.get("observables") != {"shots": float(owner.shots)}
    ):
        issues.append(("raw_record.body", "raw result differs from native HAL job metadata"))
    if body.get("artifacts") not in ([], ()) or body.get("blockers") not in ([], ()):
        issues.append(("raw_record.body", "unbound HAL artifacts or blockers cannot qualify"))
    if (
        owner.status != "completed"
        or owner.job.status != "completed"
        or owner.shots <= 0
        or not owner.counts
    ):
        issues.append(("source_records", "HAL result has no completed positive-shot observation"))
    expected_backend = {
        "source_record": "raw_record",
        "field_path": "body.backend_id",
        "record_digest": raw_digest,
        "backend_id": owner.job.backend_id,
        "stage": "result",
    }
    if companion.get("backend_reference") != expected_backend:
        issues.append(("backend_reference", "backend reference differs from the HAL result"))
    if companion.get("producer_identity") != projected["producer_identity"]:
        issues.append(("producer_identity", "HAL producer identity differs"))
    if companion.get("modality") != "hal_result_metadata_only":
        issues.append(("modality", "HAL counts do not establish a qualified count modality"))
    expected_mapping = {
        "kind": "not_applicable",
        "reason": "HAL bit mapping is unavailable; counts are not qualified",
    }
    if companion.get("measurement_mapping") != expected_mapping:
        issues.append(("measurement_mapping", "HAL result has no native bit mapping"))
    if companion.get("parameter_order") != [] or companion.get("trainable_mask") != []:
        issues.append(("parameter_order", "HAL job result has no derivative parameters"))
    if companion.get("tangent_convention") != "not_applicable":
        issues.append(("tangent_convention", "HAL job result has no tangent convention"))
    if companion.get("fields") != {"shots": {"dtype": "int", "shape": [], "unit": "shots"}}:
        issues.append(("fields", "HAL result qualifies shot count metadata only"))
    expected_settings = {
        "stage": "observation",
        "requested": {},
        "effective": {"shots": owner.shots},
        "origins": {"shots": "native_hal_result"},
        "rejected_fields": [],
    }
    if companion.get("settings") != expected_settings:
        issues.append(("settings", "observed shots differ from native HAL result"))
    if companion.get("claim_boundary") != HAL_METADATA_CLAIM_BOUNDARY:
        issues.append(("claim_boundary", "HAL result cannot claim unsupported evidence"))
    if companion.get("calibration_reference") is not None:
        issues.append(("calibration_reference", "HAL result has no verified calibration"))
    if companion.get("fidelity_components") != []:
        issues.append(("fidelity_components", "HAL result has no uncertainty components"))
    if companion.get("fidelity_unit_declaration") != {
        "origin": "native_shot_count",
        "shots": "shots",
    }:
        issues.append(("fidelity_unit_declaration", "HAL shot unit provenance differs"))
    required_unavailable = {
        "count_bit_mapping",
        "count_semantics",
        "requested_shots",
        "statevector_amplitudes",
        "hardware_attestation",
        "calibration_reference",
        "fidelity_components",
        "unit_conversion",
        "supported_transform_composition",
    }
    unavailable = companion.get("unavailable")
    if not isinstance(unavailable, list) or not required_unavailable.issubset(
        {item for item in unavailable if isinstance(item, str)}
    ):
        issues.append(("unavailable", "missing HAL count, hardware or fidelity limits"))
    return issues


__all__ = ["HAL_METADATA_CLAIM_BOUNDARY", "validate_hal_result_companion"]
