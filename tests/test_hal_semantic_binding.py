# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — HAL semantic binding tests
"""Bind a real offline HAL result through the public stable-core result reader."""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from scpn_quantum_control import stable_core_product as scp
from scpn_quantum_control.hal_semantic_binding import HAL_METADATA_CLAIM_BOUNDARY
from scpn_quantum_control.hardware.hal import (
    BackendProfile,
    HardwareAbstractionLayer,
    LocalDeterministicSimulator,
    QuantumJobResult,
    QuantumWorkload,
)
from scpn_quantum_control.native_semantic_binding import capture_native_source
from scpn_quantum_control.semantic_operations import qualify_native_modality
from scpn_quantum_control.stable_core import Result

CORPUS = Path(__file__).parent / "data" / "contract_custody_corpus"


def _hal_case() -> tuple[dict[str, Any], dict[str, Any], QuantumJobResult, BackendProfile]:
    """Produce local HAL evidence and a separate metadata-only v1 companion."""
    hal = HardwareAbstractionLayer.with_builtin_profiles()
    profile = hal.profile("local_statevector")
    backend = LocalDeterministicSimulator(profile)
    hal.register_backend(backend)
    workload = QuantumWorkload("semantic-hal-result", "mlir", "module {}", 2, shots=16)
    job = hal.submit(backend.backend_id, workload)
    owner = hal.result(job)
    retained = capture_native_source(owner)
    raw = scp.serialise_result(
        Result(
            experiment_id=job.workload_id,
            backend_id=job.backend_id,
            status="succeeded",
            observables={"shots": float(owner.shots)},
            metadata={
                "native_source_record_sha256": retained["record_sha256"],
                "source_job_id": job.job_id,
            },
        )
    )
    companion = json.loads((CORPUS / "companion_positive_base.json").read_text())
    raw_digest = scp.digest_stable_core_payload(raw)
    companion.update(
        {
            "record_reference": {
                "schema": raw["schema_version"],
                "kind": "result",
                "digest": raw_digest,
            },
            "source_binding": {
                "native_source_ref": "hal_result",
                "raw_field": "body.metadata.native_source_record_sha256",
            },
            "source_records": {"hal_result": retained},
            "backend_reference": {
                "source_record": "raw_record",
                "field_path": "body.backend_id",
                "record_digest": raw_digest,
                "backend_id": job.backend_id,
                "stage": "result",
            },
            "producer_identity": retained["producer_identity"],
            "modality": "hal_result_metadata_only",
            "measurement_mapping": {
                "kind": "not_applicable",
                "reason": "HAL bit mapping is unavailable; counts are not qualified",
            },
            "parameter_order": [],
            "trainable_mask": [],
            "tangent_convention": "not_applicable",
            "fields": {"shots": {"dtype": "int", "shape": [], "unit": "shots"}},
            "settings": {
                "stage": "observation",
                "requested": {},
                "effective": {"shots": owner.shots},
                "origins": {"shots": "native_hal_result"},
                "rejected_fields": [],
            },
            "claim_boundary": HAL_METADATA_CLAIM_BOUNDARY,
            "calibration_reference": None,
            "fidelity_components": [],
            "fidelity_unit_declaration": {"origin": "native_shot_count", "shots": "shots"},
            "unavailable": [
                "count_bit_mapping",
                "count_semantics",
                "requested_shots",
                "statevector_amplitudes",
                "hardware_attestation",
                "calibration_reference",
                "fidelity_components",
                "unit_conversion",
                "supported_transform_composition",
            ],
        }
    )
    return raw, companion, owner, profile


def test_actual_hal_result_binds_metadata_without_qualifying_counts() -> None:
    """Keep native counts and job identity while qualifying metadata alone."""
    raw, companion, owner, profile = _hal_case()
    before = scp.canonical_json_bytes(raw)

    result, binding = scp.read_result_with_semantics(
        raw, companion, native_sources={"hal_result": owner}
    )

    assert binding.qualified, binding.refusals
    assert result == scp.deserialise_result(raw)
    assert scp.canonical_json_bytes(raw) == before
    assert binding.semantics is not None
    assert binding.semantics.payload["source_records"]["hal_result"]["record"]["counts"] == dict(
        owner.counts
    )
    assert binding.semantics.payload["measurement_mapping"]["kind"] == "not_applicable"
    amplitude = qualify_native_modality(
        owner.to_semantic_source(),
        "statevector_amplitudes",
        profile=profile.to_semantic_source(),
    )
    assert amplitude.qualification == "unavailable"


def test_hal_result_discloses_missing_requested_shots() -> None:
    """A result alone cannot reconstruct the submitted workload's request."""
    raw, companion, owner, _ = _hal_case()
    companion["unavailable"].remove("requested_shots")

    _, binding = scp.read_result_with_semantics(
        raw, companion, native_sources={"hal_result": owner}
    )

    assert binding.raw_readable
    assert "hal_result_mismatch" in binding.reasons


@pytest.mark.parametrize(
    "field", ["modality", "measurement_mapping", "claim_boundary", "unavailable"]
)
def test_hal_result_refuses_promoted_count_or_hardware_claim(field: str) -> None:
    """A count-shaped source does not confer bit wiring or hardware authority."""
    raw, companion, owner, _ = _hal_case()
    changed = copy.deepcopy(companion)
    changed[field] = (
        {"kind": "native_bitstring_count_replay", "bit_wires": [0, 1]}
        if field == "measurement_mapping"
        else "hardware_counts"
        if field != "unavailable"
        else []
    )

    _, binding = scp.read_result_with_semantics(raw, changed, native_sources={"hal_result": owner})

    assert binding.raw_readable
    assert not binding.qualified


@pytest.mark.parametrize(
    ("field", "wrong"),
    [
        ("backend_reference", {}),
        ("producer_identity", "unrelated.Result"),
        ("parameter_order", ["theta"]),
        ("trainable_mask", [True]),
        ("tangent_convention", "forward_real"),
        ("fields", {"counts": {"dtype": "int", "shape": [2], "unit": "shots"}}),
        ("settings", {"stage": "planning"}),
        ("calibration_reference", "unverified"),
        ("fidelity_components", [{"kind": "zero_error"}]),
        ("fidelity_unit_declaration", {"shots": "Hz"}),
    ],
)
def test_hal_result_refuses_unbacked_metadata(field: str, wrong: object) -> None:
    """Changing a claimed unit, setting or fidelity fact loses qualification."""
    raw, companion, owner, _ = _hal_case()
    companion[field] = wrong

    _, binding = scp.read_result_with_semantics(
        raw, companion, native_sources={"hal_result": owner}
    )

    assert binding.raw_readable
    assert not binding.qualified


def test_hal_result_refuses_absent_extra_and_rehashed_sources() -> None:
    """The native job result, rather than a retained self-hash, owns counts."""
    raw, companion, owner, _ = _hal_case()
    absent = copy.deepcopy(companion)
    del absent["source_records"]["hal_result"]
    _, missing = scp.read_result_with_semantics(raw, absent, native_sources={"hal_result": owner})
    assert not missing.qualified

    extra = copy.deepcopy(companion)
    extra["source_records"]["other"] = copy.deepcopy(extra["source_records"]["hal_result"])
    _, unbound = scp.read_result_with_semantics(raw, extra, native_sources={"hal_result": owner})
    assert not unbound.qualified

    substituted = copy.deepcopy(companion)
    record = substituted["source_records"]["hal_result"]["record"]
    first_key = next(iter(record["counts"]))
    record["counts"][first_key] += 1
    substituted["source_records"]["hal_result"]["record_sha256"] = scp.digest_stable_core_payload(
        record
    )
    _, rebinding = scp.read_result_with_semantics(
        raw, substituted, native_sources={"hal_result": owner}
    )
    assert not rebinding.qualified
    assert "source_record_not_reproduced" in rebinding.reasons


@pytest.mark.parametrize(
    "field",
    [
        "native_source_record_sha256",
        "source_job_id",
        "unbound_hardware_claim",
        "artifacts",
        "experiment_id",
        "backend_id",
        "shots",
    ],
)
def test_hal_result_refuses_rehashed_raw_rebinding(field: str) -> None:
    """Changing the raw job identity or shot summary cannot reuse typed evidence."""
    raw, companion, owner, _ = _hal_case()
    if field == "unbound_hardware_claim":
        raw["body"]["metadata"]["hardware_execution"] = True
    elif field == "artifacts":
        raw["body"]["artifacts"] = ["unverified-artifact"]
    elif field in {"native_source_record_sha256", "source_job_id"}:
        raw["body"]["metadata"][field] = "substituted"
    elif field == "shots":
        raw["body"]["observables"][field] = float(owner.shots + 1)
    else:
        raw["body"][field] = "substituted"
    digest = scp.digest_stable_core_payload(raw)
    companion["record_reference"]["digest"] = digest
    companion["backend_reference"]["record_digest"] = digest
    if field == "backend_id":
        companion["backend_reference"]["backend_id"] = "substituted"

    _, binding = scp.read_result_with_semantics(
        raw, companion, native_sources={"hal_result": owner}
    )

    assert binding.raw_readable
    assert not binding.qualified


@pytest.mark.parametrize("case", ["failed_status", "cancelled_job", "unknown_shots"])
def test_hal_result_refuses_incomplete_native_observation(case: str) -> None:
    """A typed failed or unknown-shot result cannot claim completed metadata."""
    raw, companion, owner, _ = _hal_case()
    if case == "failed_status":
        altered = replace(owner, status="failed")
    elif case == "cancelled_job":
        altered = replace(owner, job=replace(owner.job, status="cancelled"))
    elif case == "unknown_shots":
        altered = replace(owner, shots=0)
    _, binding = scp.read_result_with_semantics(
        raw, companion, native_sources={"hal_result": altered}
    )

    assert binding.raw_readable
    assert not binding.qualified


def test_hal_result_refuses_missing_native_counts_at_construction() -> None:
    """Reject incomplete native counts before they can enter semantic binding."""
    _, _, owner, _ = _hal_case()
    with pytest.raises(ValueError, match="counts must sum to shots"):
        replace(owner, counts={})
