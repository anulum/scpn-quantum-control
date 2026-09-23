# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Fisher semantic binding tests
"""Exercise observed and expected Fisher routes through the public v2 reader."""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scpn_quantum_control import stable_core_product as scp
from scpn_quantum_control.fisher_semantic_binding import FISHER_RESULT_CLAIM_BOUNDARY
from scpn_quantum_control.native_semantic_binding import capture_native_source
from scpn_quantum_control.phase.qnode_circuit_contracts import (
    PauliTerm,
    PhaseQNodeCircuit,
    PhaseQNodeClassicalFisherResult,
)
from scpn_quantum_control.phase.qnode_circuit_differentiation import (
    phase_qnode_computational_basis_fisher_information,
)
from scpn_quantum_control.stable_core import Result

CORPUS = Path(__file__).parent / "data" / "contract_custody_corpus"


def _fisher_case(
    route: str,
) -> tuple[dict[str, Any], dict[str, Any], PhaseQNodeClassicalFisherResult]:
    """Replay the frozen input through the public producer and v2 reader input."""
    frozen = json.loads(
        (CORPUS / "fisher_observed_and_expected_routes_stay_distinct.json").read_text(
            encoding="utf-8"
        )
    )
    spec = frozen["circuit"]
    observable = PauliTerm(
        spec["observable"]["coefficient"],
        tuple(tuple(item) for item in spec["observable"]["factors"]),
    )
    circuit = PhaseQNodeCircuit(
        n_qubits=spec["n_qubits"],
        operations=tuple((item[0], tuple(item[1]), item[2]) for item in spec["operations"]),
        observable=observable,
    )
    owner = phase_qnode_computational_basis_fisher_information(
        circuit, **frozen["routes"][route]["inputs"]
    )
    actual = owner.to_dict()
    expected = copy.deepcopy(frozen["routes"][route]["result"])
    for field in ("fisher_standard_error", "fisher_confidence_radius"):
        actual_uncertainty = actual.pop(field)
        expected_uncertainty = expected.pop(field)
        assert isinstance(actual_uncertainty, list)
        assert isinstance(expected_uncertainty, list)
        actual_array = np.asarray(actual_uncertainty, dtype=float)
        expected_array = np.asarray(expected_uncertainty, dtype=float)
        assert actual_array.shape == expected_array.shape == (1, 1)
        np.testing.assert_allclose(actual_array, expected_array, rtol=1e-12, atol=0.0)
    assert actual == expected
    retained = capture_native_source(owner)
    raw = scp.serialise_result(
        Result(
            experiment_id=f"fisher-{route}",
            backend_id="local_statevector",
            status="succeeded",
            observables={
                "classical_fisher_trace": float(owner.classical_fisher_information.trace())
            },
            metadata={"native_source_record_sha256": retained["record_sha256"]},
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
                "native_source_ref": "fisher",
                "raw_field": "body.metadata.native_source_record_sha256",
            },
            "source_records": {"fisher": retained},
            "backend_reference": {
                "source_record": "raw_record",
                "field_path": "body.backend_id",
                "record_digest": raw_digest,
                "backend_id": "local_statevector",
                "stage": "result",
            },
            "producer_identity": retained["producer_identity"],
            "modality": (
                "fisher_observed_count_replay"
                if route == "observed"
                else "fisher_expected_count_model"
            ),
            "measurement_mapping": (
                {
                    "kind": "native_bitstring_count_replay",
                    "source": "fisher",
                    "sha256": retained["record_sha256"],
                    "field_path": "record.count_mapping",
                    "mapping": retained["record"]["count_mapping"],
                    "count_record_path": "record.count_record",
                }
                if route == "observed"
                else {"kind": "not_applicable", "reason": "expected counts are modelled"}
            ),
            "parameter_order": [0],
            "trainable_mask": [True],
            "tangent_convention": "local_statevector_real",
            "fields": {
                name: {"dtype": "float64", "shape": [1, 1], "unit": None}
                for name in (
                    "classical_fisher_information",
                    "finite_shot_classical_fisher_information",
                )
            },
            "settings": {
                "stage": "observation",
                "requested": {},
                "effective": {"shot_count": 512},
                "origins": {"shot_count": "native_fisher_result"},
                "rejected_fields": [],
            },
            "claim_boundary": FISHER_RESULT_CLAIM_BOUNDARY,
            "calibration_reference": None,
            "fidelity_unit_declaration": {"origin": "native_undeclared", "fisher": None},
            "unavailable": [
                "hardware_execution",
                "calibration_reference",
                "physical_fisher_units",
                "unit_conversion",
                "supported_transform_composition",
            ],
            "fidelity_components": [
                {
                    "kind": kind,
                    "value": retained["record"][f"fisher_{kind}"],
                    "method": owner.sampling_model,
                    "estimand": "finite_shot_classical_fisher_information",
                    "unit": None,
                    "assumptions": [
                        "finite-shot multinomial delta-method on computational-basis outcomes",
                        (
                            "caller-supplied raw counts replayed locally, not hardware observations"
                            if route == "observed"
                            else "expected counts modelled from local statevector, not observed counts"
                        ),
                        "radius and standard error describe the same uncertainty, not independent errors",
                    ],
                    "evidence_ref": {
                        "source": "fisher",
                        "sha256": retained["record_sha256"],
                        "field_path": f"record.fisher_{kind}",
                        "confidence_level_path": "record.confidence_level",
                        "confidence_z_path": "record.confidence_z",
                    },
                }
                for kind in ("standard_error", "confidence_radius")
            ],
        }
    )
    return raw, companion, owner


@pytest.mark.parametrize("route", ["observed", "expected"])
def test_native_fisher_routes_qualify_without_mutating_raw_evidence(route: str) -> None:
    """Both true routes retain distinct uncertainty and count custody."""
    raw, companion, owner = _fisher_case(route)
    original_bytes = scp.canonical_json_bytes(raw)

    result, binding = scp.read_result_with_semantics(
        raw, companion, native_sources={"fisher": owner}
    )

    assert binding.qualified, binding.refusals
    assert result == scp.deserialise_result(raw)
    assert scp.canonical_json_bytes(raw) == original_bytes
    assert binding.semantics is not None
    assert binding.semantics.payload["fidelity_components"] == companion["fidelity_components"]
    if route == "observed":
        assert binding.semantics.payload["measurement_mapping"]["mapping"]["raw_counts"] == {
            "0": 300,
            "1": 212,
        }
    else:
        assert "mapping" not in binding.semantics.payload["measurement_mapping"]


@pytest.mark.parametrize("route", ["observed", "expected"])
@pytest.mark.parametrize(
    "field",
    ["measurement_mapping", "modality", "settings", "fidelity_components", "claim_boundary"],
)
def test_fisher_route_refuses_unbacked_semantic_changes(route: str, field: str) -> None:
    """Rebinding a changed route, count map or uncertainty never qualifies."""
    raw, companion, owner = _fisher_case(route)
    changed = copy.deepcopy(companion)
    changed[field] = {} if field not in {"modality", "claim_boundary"} else "hardware"

    _, binding = scp.read_result_with_semantics(raw, changed, native_sources={"fisher": owner})

    assert not binding.qualified
    assert binding.raw_readable


def test_fisher_observed_counts_cannot_be_claimed_by_expected_source() -> None:
    """An expected route cannot borrow the observed route's native bit map."""
    raw, companion, owner = _fisher_case("expected")
    _, observed, _ = _fisher_case("observed")
    companion["measurement_mapping"] = observed["measurement_mapping"]

    _, binding = scp.read_result_with_semantics(raw, companion, native_sources={"fisher": owner})

    assert not binding.qualified
    assert "fisher_result_mismatch" in binding.reasons


def test_fisher_uncertainty_assumption_cannot_be_promoted_to_hardware() -> None:
    """The route's delta-method uncertainty cannot inherit QPU provenance."""
    raw, companion, owner = _fisher_case("observed")
    companion["fidelity_components"][0]["assumptions"][1] = "observed on QPU"

    _, binding = scp.read_result_with_semantics(raw, companion, native_sources={"fisher": owner})

    assert not binding.qualified
    assert "fidelity_source_unverifiable" in binding.reasons


@pytest.mark.parametrize(
    ("field", "wrong"),
    [
        ("backend_reference", {}),
        ("producer_identity", "unrelated.Fisher"),
        ("parameter_order", [1]),
        ("trainable_mask", [False]),
        ("tangent_convention", "reverse_holomorphic"),
        ("fields", {}),
        ("calibration_reference", "unverified"),
        ("fidelity_unit_declaration", {"fisher": "Hz"}),
        ("unavailable", []),
    ],
)
def test_fisher_refuses_unbacked_result_metadata(field: str, wrong: object) -> None:
    """A typed source cannot lend authority to changed units or metadata."""
    raw, companion, owner = _fisher_case("observed")
    companion[field] = wrong

    _, binding = scp.read_result_with_semantics(raw, companion, native_sources={"fisher": owner})

    assert binding.raw_readable
    assert not binding.qualified


def test_fisher_refuses_absent_or_rehashed_native_evidence() -> None:
    """A retained self-hash and extra source cannot replace exact typed custody."""
    raw, companion, owner = _fisher_case("observed")
    absent = copy.deepcopy(companion)
    del absent["source_records"]["fisher"]
    _, missing = scp.read_result_with_semantics(raw, absent, native_sources={"fisher": owner})
    assert not missing.qualified

    extra = copy.deepcopy(companion)
    extra["source_records"]["other"] = copy.deepcopy(extra["source_records"]["fisher"])
    _, unbound = scp.read_result_with_semantics(raw, extra, native_sources={"fisher": owner})
    assert not unbound.qualified

    substituted = copy.deepcopy(companion)
    record = substituted["source_records"]["fisher"]["record"]
    record["count_mapping"]["raw_counts"]["0"] = 299
    substituted["source_records"]["fisher"]["record_sha256"] = scp.digest_stable_core_payload(
        record
    )
    _, rebinding = scp.read_result_with_semantics(
        raw, substituted, native_sources={"fisher": owner}
    )
    assert not rebinding.qualified
    assert "source_record_not_reproduced" in rebinding.reasons


def test_fisher_subclass_cannot_borrow_the_registered_native_source_schema() -> None:
    """The exact source-specific version cannot be inherited by another type."""
    raw, companion, owner = _fisher_case("observed")
    foreign_type = type("ForeignFisher", (PhaseQNodeClassicalFisherResult,), {})
    foreign = foreign_type(**{name: getattr(owner, name) for name in owner.__dataclass_fields__})

    _, binding = scp.read_result_with_semantics(raw, companion, native_sources={"fisher": foreign})

    assert not binding.qualified
    assert "fisher_result_mismatch" in binding.reasons


@pytest.mark.parametrize(
    "field", ["native_source_record_sha256", "classical_fisher_trace", "backend_id"]
)
def test_fisher_refuses_rehashed_raw_result_changes(field: str) -> None:
    """Rehashing changed v2 bytes cannot establish a native Fisher result."""
    raw, companion, owner = _fisher_case("observed")
    if field == "native_source_record_sha256":
        raw["body"]["metadata"][field] = "0" * 64
    elif field == "classical_fisher_trace":
        raw["body"]["observables"][field] = 999.0
    else:
        raw["body"][field] = "ibm_brisbane"
    digest = scp.digest_stable_core_payload(raw)
    companion["record_reference"]["digest"] = digest
    companion["backend_reference"]["record_digest"] = digest
    if field == "backend_id":
        companion["backend_reference"]["backend_id"] = "ibm_brisbane"

    _, binding = scp.read_result_with_semantics(raw, companion, native_sources={"fisher": owner})

    assert binding.raw_readable
    assert not binding.qualified


@pytest.mark.parametrize(
    "case",
    [
        "missing_mapping",
        "inconsistent_counts",
        "invented_counts",
        "no_shots",
        "no_matrix",
        "no_uncertainty",
        "wrong_reference_shape",
    ],
)
def test_fisher_refuses_internally_inconsistent_native_routes(
    case: str,
) -> None:
    """A typed object with stripped route evidence cannot qualify old semantics."""
    route = "observed" if case in {"missing_mapping", "inconsistent_counts"} else "expected"
    raw, companion, owner = _fisher_case(route)
    if case == "missing_mapping":
        altered = replace(owner, count_record=None, count_mapping=None)
    elif case == "inconsistent_counts":
        altered = replace(owner, count_record=(299, 212))
    elif case == "invented_counts":
        altered = replace(owner, count_record=(300, 212))
    elif case == "no_shots":
        altered = replace(owner, sampling_model=None, shot_count=None)
    elif case == "no_uncertainty":
        altered = replace(owner, fisher_standard_error=None)
    elif case == "wrong_reference_shape":
        altered = replace(owner, classical_fisher_information=np.array([1.0]))
    else:
        altered = replace(owner, finite_shot_classical_fisher_information=None)

    _, binding = scp.read_result_with_semantics(raw, companion, native_sources={"fisher": altered})

    assert binding.raw_readable
    assert not binding.qualified
