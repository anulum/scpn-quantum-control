# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Phase-QNode Fisher semantic binding
"""Qualify local Fisher result metadata against its complete native evidence."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

from .native_semantic_binding import validate_native_source_record
from .phase.qnode_circuit_contracts import PhaseQNodeClassicalFisherResult

FISHER_RESULT_CLAIM_BOUNDARY: Final = (
    "local statevector computational-basis Fisher reference and finite-shot "
    "multinomial analysis; observed counts are caller-supplied replay, expected "
    "counts are modelled, never acquired hardware observations; no backend "
    "calibration, physical-unit, unit-conversion or hardware-execution claim"
)
"""Closed claim accepted for native local Fisher evidence."""


def validate_fisher_result_companion(
    companion: Mapping[str, Any],
    raw_record: Mapping[str, Any],
    raw_digest: str,
    native_sources: Mapping[str, object],
) -> list[tuple[str, str]]:
    """Compare every count, uncertainty and setting claim with a typed Fisher owner.

    The source object is the sole owner of Fisher matrices, count records and
    uncertainty arrays. Stable-core v2 carries only its scalar trace and digest.
    """
    issues: list[tuple[str, str]] = []
    binding = companion.get("source_binding")
    binding = binding if isinstance(binding, Mapping) else {}
    ref = binding.get("native_source_ref")
    retained_sources = companion.get("source_records")
    retained_sources = retained_sources if isinstance(retained_sources, Mapping) else {}
    retained = retained_sources.get(ref) if isinstance(ref, str) else None
    owner = native_sources.get(ref) if isinstance(ref, str) else None
    if type(owner) is not PhaseQNodeClassicalFisherResult or not isinstance(retained, Mapping):
        return [("source_binding", "a retained, actual Phase-QNode Fisher owner is required")]
    if set(retained_sources) != {ref} or set(native_sources) != {ref}:
        issues.append(("source_records", "Fisher result must have exactly one native owner"))
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
        issues.append(("source_binding", "raw result does not reference this Fisher source"))
    native_shape = owner.classical_fisher_information.shape
    expected_observables = {
        "classical_fisher_trace": (
            float(owner.classical_fisher_information.trace()) if len(native_shape) == 2 else None
        )
    }
    if body.get("status") != "succeeded" or body.get("observables") != expected_observables:
        issues.append(("raw_record.body.observables", "raw trace differs from native Fisher"))
    if body.get("backend_id") != "local_statevector":
        issues.append(("raw_record.body.backend_id", "Fisher source is local statevector only"))
    expected_backend = {
        "source_record": "raw_record",
        "field_path": "body.backend_id",
        "record_digest": raw_digest,
        "backend_id": "local_statevector",
        "stage": "result",
    }
    if companion.get("backend_reference") != expected_backend:
        issues.append(("backend_reference", "backend reference differs from the raw result"))
    if companion.get("producer_identity") != projected["producer_identity"]:
        issues.append(("producer_identity", "Fisher producer identity differs"))
    observed = owner.sampling_model == "multinomial_delta_method_raw_count_replay"
    expected = owner.sampling_model == "multinomial_delta_method_expected_counts"
    if not (observed or expected) or owner.shot_count is None:
        issues.append(("modality", "this companion requires a finite-shot Fisher route"))
    reference_shape = native_shape
    estimate = owner.finite_shot_classical_fisher_information
    error = owner.fisher_standard_error
    radius = owner.fisher_confidence_radius
    if (
        len(reference_shape) != 2
        or reference_shape[0] != reference_shape[1]
        or estimate is None
        or error is None
        or radius is None
        or estimate.shape != reference_shape
        or error.shape != reference_shape
        or radius.shape != reference_shape
        or owner.confidence_level is None
        or owner.confidence_z is None
    ):
        issues.append(("fields", "native Fisher estimate and uncertainty shapes are incomplete"))
    modality = "fisher_observed_count_replay" if observed else "fisher_expected_count_model"
    if companion.get("modality") != modality:
        issues.append(("modality", "observed and expected Fisher routes must stay distinct"))
    mapping = companion.get("measurement_mapping")
    if observed:
        native_mapping = projected.get("count_mapping")
        if native_mapping is None or owner.count_record is None:
            issues.append(("measurement_mapping", "observed route lacks native count mapping"))
        if (
            owner.count_mapping is not None
            and owner.count_record is not None
            and (
                owner.count_mapping.count_vector != owner.count_record
                or sum(owner.count_record) != owner.shot_count
                or sum(count for _, count in owner.count_mapping.raw_counts) != owner.shot_count
            )
        ):
            issues.append(
                ("measurement_mapping", "native replay counts contradict mapping or shots")
            )
        expected_mapping = {
            "kind": "native_bitstring_count_replay",
            "source": ref,
            "sha256": source_binding.source_digest,
            "field_path": "record.count_mapping",
            "mapping": native_mapping,
            "count_record_path": "record.count_record",
        }
    else:
        if owner.count_record is not None or owner.count_mapping is not None:
            issues.append(("measurement_mapping", "expected route cannot own observed counts"))
        expected_mapping = {"kind": "not_applicable", "reason": "expected counts are modelled"}
    if mapping != expected_mapping:
        issues.append(("measurement_mapping", "count mapping differs from native route"))
    parameter_count = reference_shape[0] if len(reference_shape) == 2 else 0
    if companion.get("parameter_order") != list(range(parameter_count)):
        issues.append(("parameter_order", "Fisher matrix parameter order differs"))
    differentiable = set(owner.support_report.differentiable_parameters)
    if companion.get("trainable_mask") != [
        index in differentiable for index in range(parameter_count)
    ]:
        issues.append(("trainable_mask", "Fisher parameter mask differs"))
    if companion.get("tangent_convention") != "local_statevector_real":
        issues.append(("tangent_convention", "Fisher source uses real statevector derivatives"))
    expected_fields = {
        "classical_fisher_information": {
            "dtype": str(owner.classical_fisher_information.dtype),
            "shape": list(owner.classical_fisher_information.shape),
            "unit": None,
        },
        "finite_shot_classical_fisher_information": {
            "dtype": str(owner.finite_shot_classical_fisher_information.dtype)
            if owner.finite_shot_classical_fisher_information is not None
            else None,
            "shape": list(owner.finite_shot_classical_fisher_information.shape)
            if owner.finite_shot_classical_fisher_information is not None
            else None,
            "unit": None,
        },
    }
    if companion.get("fields") != expected_fields:
        issues.append(("fields", "Fisher matrix layout or undeclared unit differs"))
    expected_settings = {
        "stage": "observation",
        "requested": {},
        "effective": {"shot_count": owner.shot_count},
        "origins": {"shot_count": "native_fisher_result"},
        "rejected_fields": [],
    }
    if companion.get("settings") != expected_settings:
        issues.append(("settings", "shot setting differs from native Fisher result"))
    if companion.get("claim_boundary") != FISHER_RESULT_CLAIM_BOUNDARY:
        issues.append(("claim_boundary", "Fisher analysis cannot claim hardware observation"))
    if companion.get("calibration_reference") is not None:
        issues.append(("calibration_reference", "local Fisher source has no calibration"))
    if companion.get("fidelity_unit_declaration") != {
        "origin": "native_undeclared",
        "fisher": None,
    }:
        issues.append(("fidelity_unit_declaration", "Fisher units are not natively declared"))
    required_unavailable = {
        "hardware_execution",
        "calibration_reference",
        "physical_fisher_units",
        "unit_conversion",
        "supported_transform_composition",
    }
    unavailable = companion.get("unavailable")
    if not isinstance(unavailable, list) or not required_unavailable.issubset(
        {item for item in unavailable if isinstance(item, str)}
    ):
        issues.append(("unavailable", "missing local Fisher evidence limitations"))
    assumptions = [
        "finite-shot multinomial delta-method on computational-basis outcomes",
        (
            "caller-supplied raw counts replayed locally, not hardware observations"
            if observed
            else "expected counts modelled from local statevector, not observed counts"
        ),
        "radius and standard error describe the same uncertainty, not independent errors",
    ]
    expected_components = [
        {
            "kind": "standard_error",
            "value": projected["fisher_standard_error"],
            "method": owner.sampling_model,
            "estimand": "finite_shot_classical_fisher_information",
            "unit": None,
            "assumptions": assumptions,
            "evidence_ref": {
                "source": ref,
                "sha256": source_binding.source_digest,
                "field_path": "record.fisher_standard_error",
                "confidence_level_path": "record.confidence_level",
                "confidence_z_path": "record.confidence_z",
            },
        },
        {
            "kind": "confidence_radius",
            "value": projected["fisher_confidence_radius"],
            "method": owner.sampling_model,
            "estimand": "finite_shot_classical_fisher_information",
            "unit": None,
            "assumptions": assumptions,
            "evidence_ref": {
                "source": ref,
                "sha256": source_binding.source_digest,
                "field_path": "record.fisher_confidence_radius",
                "confidence_level_path": "record.confidence_level",
                "confidence_z_path": "record.confidence_z",
            },
        },
    ]
    if companion.get("fidelity_components") != expected_components:
        issues.append(("fidelity_components", "native Fisher uncertainties or route differ"))
    return issues


__all__ = ["FISHER_RESULT_CLAIM_BOUNDARY", "validate_fisher_result_companion"]
