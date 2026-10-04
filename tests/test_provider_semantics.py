# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — native provider companion contract tests
"""Keep request wiring and native observations immutable and source-bound."""

from __future__ import annotations

import base64
import hashlib
from dataclasses import replace
from typing import Literal, cast

import pytest

from scpn_quantum_control.hardware.hal import QuantumWorkload
from scpn_quantum_control.hardware.provider_modalities import ModalitySemantics
from scpn_quantum_control.hardware.provider_semantics import (
    GateModelObservation,
    NativeRegisterSamples,
    SubmissionSemantics,
    WorkloadSemantics,
    modality_submission_semantics,
)


def _request() -> WorkloadSemantics:
    """Declare the independent three-qubit partial-permutation fixture."""
    return WorkloadSemantics(
        program_sha256=hashlib.sha256(b"native-program").hexdigest(),
        n_qubits=3,
        n_clbits=2,
        measurement_map=((2, 0), (0, 1)),
        classical_registers=(("alpha", (0,)), ("beta", (1,))),
    )


def test_request_binds_exact_original_program_and_width() -> None:
    """The public workload refuses a companion for a different native source."""
    request = _request()
    workload = QuantumWorkload("bound", "qiskit_qpy", "native-program", 3, semantics=request)
    assert workload.semantics is request
    for change in ({"program": "other-program"}, {"n_qubits": 2}):
        with pytest.raises(ValueError, match="semantics.*source"):
            replace(workload, **change)


def test_gate_observation_preserves_register_raw_output() -> None:
    """Flatten only the declared register separator while retaining the raw map."""
    raw = {"0 1": 8}
    observation = GateModelObservation(request=_request(), raw_counts=raw, shots=8)
    assert observation.counts == {"01": 8}
    assert observation.raw_counts == {"0 1": 8}
    assert observation.measurement_map == ((2, 0), (0, 1))
    raw["0 1"] = 2
    assert observation.shots == 8
    assert observation.raw_counts == {"0 1": 8}


@pytest.mark.parametrize("raw", [{"01": 7}, {"001": 8}, {"0\t1": 8}, {"01": 4, "0 1": 4}])
def test_observation_refuses_lossy_or_ambiguous_counts(raw: dict[str, int]) -> None:
    """Bad width, separators, collisions or totals never qualify a gate result."""
    with pytest.raises(ValueError):
        GateModelObservation(request=_request(), raw_counts=raw, shots=8)


def test_binding_refuses_numeric_overflow_as_a_contract_error() -> None:
    """An unrepresentable native numeric binding has a stable refusal category."""
    request = replace(_request(), parameters=(("theta", "native-theta", ((0, 0),)),))
    with pytest.raises(ValueError, match="finite"):
        replace(request, parameter_values=(("native-theta", 10**400),))


def test_native_count_companion_refuses_string_coercion() -> None:
    """Native count custody cannot relabel a numeric string as an SDK integer."""
    with pytest.raises(ValueError, match="integer"):
        GateModelObservation(
            request=_request(),
            shots=8,
            raw_counts=cast(dict[str, int], {"01": "8"}),
        )


@pytest.mark.parametrize(
    "changes,diagnostic",
    [
        ({"schema": "provider_semantics.v2"}, "schema"),
        ({"program_sha256": "A" * 64}, "digest"),
        ({"program_sha256": 123}, "digest"),
        ({"n_qubits": 0}, "integer"),
        ({"n_qubits": True}, "integer"),
        ({"n_clbits": -1}, "integer"),
        ({"measurement_map": ((3, 0),)}, "measurement map"),
        ({"measurement_map": ((0, 2),)}, "measurement map"),
        ({"measurement_map": ((0, 0), (1, 0))}, "measurement map"),
        ({"measurement_map": ((-1, 0),)}, "integer"),
        ({"measurement_map": ((0, False),)}, "integer"),
        ({"classical_registers": (("", (0, 1)),)}, "register names"),
        ({"classical_registers": ((1, (0, 1)),)}, "register names"),
        ({"classical_registers": (("c", (0,)), ("c", (1,)))}, "register names"),
        ({"classical_registers": (("c", ()),)}, "register names"),
        ({"classical_registers": (("c", (0, 2)),)}, "indices"),
        ({"classical_registers": (("c", (0, 0)),)}, "indices"),
        ({"classical_registers": (("c", (False, 1)),)}, "integer"),
        ({"classical_registers": (("c", (0,)),)}, "cover"),
        ({"parameters": (("", "theta-id", ((0, 0),)),)}, "parameter identity"),
        ({"parameters": ((1, "theta-id", ((0, 0),)),)}, "parameter identity"),
        ({"parameters": (("theta", "", ((0, 0),)),)}, "parameter identity"),
        ({"parameters": (("theta", 1, ((0, 0),)),)}, "parameter identity"),
        ({"parameters": (("theta", "theta-id", ()),)}, "parameter identity"),
        (
            {"parameters": (("a", "same", ((0, 0),)), ("b", "same", ((1, 0),)))},
            "parameter identity",
        ),
        ({"parameters": (("theta", "theta-id", ((-1, 0),)),)}, "integer"),
        ({"parameters": (("theta", "theta-id", ((0, -1),)),)}, "integer"),
        ({"global_phase_parameters": ("",)}, "global phase"),
        ({"global_phase_parameters": (1,)}, "global phase"),
        ({"global_phase_parameters": ("missing", "missing")}, "global phase"),
        ({"global_phase_parameters": ("missing",)}, "original native"),
        ({"requested_target": ""}, "target"),
        ({"requested_target": " padded"}, "target"),
        ({"requested_target": "native\x00target"}, "target"),
        ({"requested_target": "native\x7ftarget"}, "target"),
        ({"requested_target": 123}, "target"),
        ({"count_bit_order": "unknown"}, "bit order"),
    ],
)
def test_native_request_refuses_contradictory_declarations(
    changes: dict[str, object],
    diagnostic: str,
) -> None:
    """Malformed source, indices, identity, version and target stay observable."""
    request = _request()
    # Cast each deliberately malformed external field at its public boundary.
    with pytest.raises(ValueError, match=diagnostic):
        replace(
            request,
            schema=cast(str, changes.get("schema", request.schema)),
            program_sha256=cast(str, changes.get("program_sha256", request.program_sha256)),
            n_qubits=cast(int, changes.get("n_qubits", request.n_qubits)),
            n_clbits=cast(int, changes.get("n_clbits", request.n_clbits)),
            measurement_map=cast(
                tuple[tuple[int, int], ...],
                changes.get("measurement_map", request.measurement_map),
            ),
            classical_registers=cast(
                tuple[tuple[str, tuple[int, ...]], ...],
                changes.get("classical_registers", request.classical_registers),
            ),
            parameters=cast(
                tuple[tuple[str, str, tuple[tuple[int, int], ...]], ...],
                changes.get("parameters", request.parameters),
            ),
            global_phase_parameters=cast(
                tuple[str, ...],
                changes.get("global_phase_parameters", request.global_phase_parameters),
            ),
            requested_target=cast(
                str | None, changes.get("requested_target", request.requested_target)
            ),
            count_bit_order=cast(
                Literal["classical_lsb_right", "classical_lsb_left"],
                changes.get("count_bit_order", request.count_bit_order),
            ),
        )


@pytest.mark.parametrize(
    "bindings",
    [
        (("foreign-id", 1.0),),
        (("theta-id", 1.0), ("theta-id", 2.0)),
        (("theta-id", True),),
        (("theta-id", "1.0"),),
        (("theta-id", float("inf")),),
        (("theta-id", float("nan")),),
    ],
)
def test_bindings_require_original_unique_finite_native_values(
    bindings: tuple[tuple[str, object], ...],
) -> None:
    """Binding coercion and duplicate or foreign identities never qualify."""
    request = replace(_request(), parameters=(("theta", "theta-id", ((0, 0),)),))
    with pytest.raises(ValueError, match="binding"):
        replace(request, parameter_values=cast(tuple[tuple[str, float], ...], bindings))


def test_native_request_payload_detaches_input_and_global_phase_identity() -> None:
    """Ordered mutable declarations detach, including phase-only parameters."""
    measurements = [[2, 0], [0, 1]]
    registers = [["alpha", [0]], ["beta", [1]]]
    request = replace(
        _request(),
        measurement_map=cast(tuple[tuple[int, int], ...], measurements),
        classical_registers=cast(tuple[tuple[str, tuple[int, ...]], ...], registers),
        parameters=(("theta", "phase-id", ()),),
        global_phase_parameters=("phase-id",),
        parameter_values=(("phase-id", 0.25),),
        requested_target="native-backend",
    )
    measurements[0][0] = 1
    registers.clear()
    assert request.measurement_map == ((2, 0), (0, 1))
    payload = request.to_payload()
    assert payload == {
        "schema": "provider_semantics.v1",
        "program_sha256": request.program_sha256,
        "n_qubits": 3,
        "n_clbits": 2,
        "measurement_map": [[2, 0], [0, 1]],
        "classical_registers": [["alpha", [0]], ["beta", [1]]],
        "parameters": [["theta", "phase-id", []]],
        "global_phase_parameters": ["phase-id"],
        "parameter_values": [["phase-id", 0.25]],
        "requested_target": "native-backend",
        "count_bit_order": "classical_lsb_right",
    }
    unmeasured = WorkloadSemantics(request.program_sha256, 1, 0)
    assert unmeasured.to_payload()["measurement_map"] == []


def _submission() -> SubmissionSemantics:
    """Bind the independent source to its explicitly pinned execution target."""
    return SubmissionSemantics(
        replace(_request(), requested_target="native-backend"),
        "native-program",
        "qiskit_qpy",
        8,
        8,
        "native-backend",
        compilation="targeted",
        compiled_program_sha256="a" * 64,
    )


@pytest.mark.parametrize(
    "changes,diagnostic",
    [
        ({"requested_shots": 0}, "integer"),
        ({"effective_shots": False}, "integer"),
        ({"effective_shots": 7}, "shots differ"),
        ({"ir_format": ""}, "nonempty"),
        ({"target_name": ""}, "nonempty"),
        ({"target_name": "other-backend"}, "target differs"),
        ({"compilation": "guessed"}, "provenance"),
        ({"compiled_program_sha256": "bad-digest"}, "digest"),
        ({"compiled_program_sha256": None}, "actual native"),
        ({"target_origin": "inferred"}, "origin"),
    ],
)
def test_submission_refuses_settings_or_provenance_drift(
    changes: dict[str, object],
    diagnostic: str,
) -> None:
    """Exact original source and execution settings cannot silently diverge."""
    submission = _submission()
    with pytest.raises(ValueError, match=diagnostic):
        replace(
            submission,
            requested_shots=cast(int, changes.get("requested_shots", submission.requested_shots)),
            effective_shots=cast(int, changes.get("effective_shots", submission.effective_shots)),
            ir_format=cast(str, changes.get("ir_format", submission.ir_format)),
            target_name=cast(str, changes.get("target_name", submission.target_name)),
            compilation=cast(
                Literal["targeted", "native_provider", "caller_precompiled"],
                changes.get("compilation", submission.compilation),
            ),
            compiled_program_sha256=cast(
                str | None,
                changes.get("compiled_program_sha256", submission.compiled_program_sha256),
            ),
            target_origin=cast(
                Literal["native_sdk", "adapter_selector"],
                changes.get("target_origin", submission.target_origin),
            ),
        )


def test_submission_payload_preserves_original_and_actual_compilation_separately() -> None:
    """Target compilation keeps its digest without replacing original source."""
    submission = _submission()
    payload = submission.to_payload()
    assert submission.require_gate_request() is submission.request
    assert payload["original_program"] == "native-program"
    assert payload["compiled_program_sha256"] == "a" * 64
    assert payload["target_origin"] == "native_sdk"
    assert payload["requested_shots"] == payload["effective_shots"] == 8
    assert payload["request"] == submission.request.to_payload()


@pytest.mark.parametrize("precompiled", [False, True])
def test_modality_submission_retains_native_axes_and_selector_provenance(
    precompiled: bool,
) -> None:
    """Non-gate source binding labels a selector without claiming an SDK target."""
    request = ModalitySemantics(
        hashlib.sha256(b"plan").hexdigest(),
        "analog",
        ("b", "a"),
        requested_target="declared",
    )
    submission = modality_submission_semantics(
        request,
        program="plan",
        ir_format="native-json",
        native_axes=("b", "a"),
        modality="analog",
        target_name="declared",
        shots=8,
        caller_precompiled=precompiled,
    )
    assert submission is not None
    assert submission.request is request
    assert submission.target_origin == "adapter_selector"
    assert submission.compilation == ("caller_precompiled" if precompiled else "native_provider")
    with pytest.raises(ValueError, match="gate-model"):
        submission.require_gate_request()
    with pytest.raises(ValueError, match="non-gate"):
        modality_submission_semantics(
            _request(),
            program="plan",
            ir_format="native-json",
            native_axes=("b", "a"),
            modality="analog",
            target_name="declared",
            shots=8,
        )
    with pytest.raises(ValueError, match="ordered axes"):
        modality_submission_semantics(
            request,
            program="plan",
            ir_format="native-json",
            native_axes=("a", "b"),
            modality="analog",
            target_name="declared",
            shots=8,
        )
    assert (
        modality_submission_semantics(
            None,
            program="plan",
            ir_format="native-json",
            native_axes=("b", "a"),
            modality="analog",
            target_name="declared",
            shots=8,
        )
        is None
    )


@pytest.mark.parametrize(
    "changes,diagnostic",
    [
        ({"name": ""}, "name"),
        ({"name": 3}, "name"),
        ({"num_bits": 0}, "integer"),
        ({"shape": (8,)}, "two-dimensional"),
        ({"shape": (0, 1)}, "integer"),
        ({"shape": (8, False)}, "integer"),
        ({"num_bits": 9}, "packed width"),
        ({"data": b"short"}, "byte length"),
        ({"data": bytearray(8)}, "byte length"),
    ],
)
def test_native_packed_samples_refuse_incompatible_storage(
    changes: dict[str, object],
    diagnostic: str,
) -> None:
    """Register bytes require the exact declared native storage shape and dtype."""
    sample = NativeRegisterSamples("alpha", 1, (8, 1), b"\xff" * 8)
    with pytest.raises(ValueError, match=diagnostic):
        replace(
            sample,
            name=cast(str, changes.get("name", sample.name)),
            num_bits=cast(int, changes.get("num_bits", sample.num_bits)),
            shape=cast(tuple[int, int], changes.get("shape", sample.shape)),
            data=cast(bytes, changes.get("data", sample.data)),
        )


def test_native_register_bytes_and_observation_payload_preserve_padding() -> None:
    """Original packed padding bytes remain intact rather than regenerated."""
    alpha = NativeRegisterSamples("alpha", 1, (8, 1), b"\xff" * 8)
    beta = NativeRegisterSamples("beta", 1, (8, 1), b"\xfe" * 8)
    observation = GateModelObservation(_request(), {"0 1": 8}, 8, (alpha, beta))
    payload = observation.to_payload()
    native = alpha.to_payload()
    assert base64.b64decode(str(native["data_base64"])) == b"\xff" * 8
    assert native["data_sha256"] == hashlib.sha256(b"\xff" * 8).hexdigest()
    assert native["shape"] == [8, 1]
    assert payload["raw_counts"] == {"0 1": 8}
    assert payload["counts"] == {"01": 8}
    assert payload["register_samples"] == [alpha.to_payload(), beta.to_payload()]
    for samples, diagnostic in (
        ((beta, alpha), "order"),
        ((replace(alpha, num_bits=2), beta), "width or shots"),
        ((replace(alpha, shape=(4, 1), data=b"\xff" * 4), beta), "width or shots"),
    ):
        with pytest.raises(ValueError, match=diagnostic):
            replace(observation, register_samples=samples)


@pytest.mark.parametrize(
    "raw,diagnostic",
    [
        ({"": 8}, "separator"),
        ({" 01": 8}, "separator"),
        ({"0  1": 8}, "separator"),
        ({1: 8}, "separator"),
        ({"0 01": 8}, "group widths"),
        ({"01": True}, "integer"),
        ({"01": -1}, "non-negative"),
        ({}, "shot total"),
    ],
)
def test_gate_native_output_refuses_invalid_keys_and_native_values(
    raw: dict[object, object],
    diagnostic: str,
) -> None:
    """Native decoding refuses bad register groups, coercion and missing evidence."""
    with pytest.raises(ValueError, match=diagnostic):
        GateModelObservation(_request(), cast(dict[str, int], raw), 8)
