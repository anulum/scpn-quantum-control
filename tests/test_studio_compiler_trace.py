# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — source-bound compiler trace boundary tests
"""Exercise native compiler records, independent mapped observables and export."""

from __future__ import annotations

import copy
import json
import math
import re
import runpy
from collections.abc import Callable
from dataclasses import FrozenInstanceError, fields, replace
from pathlib import Path
from typing import Any, cast

import pytest
from qiskit import QuantumCircuit, qasm2
from qiskit.quantum_info import Pauli, Statevector

from scpn_quantum_control.compiler import (
    CircuitPassRecord,
    CircuitPassRefused,
    qualify_circuit_pass,
)
from scpn_quantum_control.studio.compiler_trace import (
    MAX_TRACE_BYTES,
    CompilerTrace,
    build_compiler_trace,
    compiler_backend_snapshot,
    project_compiler_passes,
)
from scpn_quantum_control.studio_workspace.canonical import canonical_digest

SOURCE = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\ncreg c[2];\nry(0.41) q[0];\ncx q[0],q[1];\nmeasure q[1] -> c[0];\nmeasure q[0] -> c[1];\n'


_MEASURED = ("global_phase_delta", "operator_error")
_MEASUREMENT_NOISE = 1e-14


def _measurements(document: dict[str, Any]) -> list[tuple[float, ...]]:
    """Return the measured phase delta and operator error of every pass."""
    return [tuple(row["record"][name] for name in _MEASURED) for row in document["body"]["passes"]]


def _measurement_digests(document: dict[str, Any]) -> list[str]:
    """Return the digests that bind a measured value, in document order."""
    digests = [row["native_record_sha256"] for row in document["body"]["passes"]]
    emitted = document["body"].get("emitted_ir")
    if emitted is not None:
        digests.append(emitted["metadata"]["pass_sha256"])
    return [*digests, document["sha256"]]


def _without_measurement(document: dict[str, Any]) -> dict[str, Any]:
    """Return a copy without the measured values and the digests that bind them."""
    stripped = copy.deepcopy(document)
    del stripped["sha256"]
    for row in stripped["body"]["passes"]:
        del row["native_record_sha256"]
        for name in _MEASURED:
            del row["record"][name]
    emitted = stripped["body"].get("emitted_ir")
    if emitted is not None:
        del emitted["metadata"]["pass_sha256"]
    return stripped


def _assert_reproduced(produced: dict[str, Any], committed: dict[str, Any]) -> None:
    """Require the committed trace, up to the last bits of its measured values.

    The phase delta and the operator error are measured through the reference
    linear algebra, whose last bits differ between platforms; the record
    digest, the lowering's pass digest and the document digest bind them. A
    bit-equal measurement requires equal documents. Any other measurement must
    agree within the measurement noise, and everything except the measured
    values and exactly those digests must still be equal.
    """
    measured, expected = _measurements(produced), _measurements(committed)
    if measured == expected:
        assert produced == committed
        return
    assert len(measured) == len(expected)
    for values, references in zip(measured, expected, strict=True):
        for value, reference in zip(values, references, strict=True):
            assert abs(value - reference) <= _MEASUREMENT_NOISE
    assert _without_measurement(produced) == _without_measurement(committed)
    for digest in _measurement_digests(produced):
        assert re.fullmatch(r"[0-9a-f]{64}", digest) is not None


def mapped_circuits() -> tuple[QuantumCircuit, QuantumCircuit, CircuitPassRecord]:
    """Produce an independently known logical-to-physical swap through native admission."""
    before = QuantumCircuit(2, 2)
    before.ry(0.41, 0)
    before.cx(0, 1)
    before.rx(0.27, 1)
    before.measure([0, 1], [1, 0])
    after = QuantumCircuit(2, 2)
    after.ry(0.41, 1)
    after.cx(1, 0)
    after.rx(0.27, 0)
    after.measure([1, 0], [1, 0])
    record = qualify_circuit_pass(before, after, pass_name="physical_swap", output_layout=(1, 0))
    return before, after, record


def test_swapped_layout_preserves_logical_expectation() -> None:
    """The public projection retains the mapping proved against an analytic observable."""
    before, after, swapped = mapped_circuits()
    physical = Statevector.from_instruction(after.remove_final_measurements(inplace=False))
    expected = math.cos(0.41)
    assert physical.expectation_value(Pauli("ZI")).real == pytest.approx(expected, abs=1e-12)
    assert physical.expectation_value(Pauli("IZ")).real == pytest.approx(
        expected * math.cos(0.27), abs=1e-12
    )
    assert physical.expectation_value(Pauli("ZI")).real != pytest.approx(
        physical.expectation_value(Pauli("IZ")).real
    )
    trace = project_compiler_passes([swapped], backend_snapshot=compiler_backend_snapshot())
    wire = trace.to_dict()
    body = wire["body"]
    assert isinstance(body, dict)
    rows = body["passes"]
    assert isinstance(rows, list)
    assert rows[0]["record"]["output_layout"] == [1, 0]
    assert rows[0]["record"]["observable_map"] == [[0, 1, 1, 1], [1, 0, 0, 0]]
    assert rows[0]["metrics"]["input"]["gate_counts"] == {"ry": 1, "cx": 1, "rx": 1}
    assert rows[0]["metrics"]["input"]["depth"] == before.depth()
    assert rows[0]["metrics"]["delta"]["depth"] == 0
    with pytest.raises(CircuitPassRefused, match="measurement mapping|unitary semantics"):
        qualify_circuit_pass(before, after, pass_name="undeclared_swap")
    assert before.data[0].qubits[0] == before.qubits[0]


def test_native_lowering_export_binds_full_snapshot() -> None:
    """Actual lowering emits original text and exact local backend/config identity."""
    trace = build_compiler_trace(SOURCE, optimisation_level=2)
    wire = trace.to_dict()
    body = wire["body"]
    assert isinstance(body, dict)
    assert body["source"] == SOURCE
    assert body["execution_status"] == "emitted_not_executed"
    assert body["emitted_ir"]["execution_status"] == "textual_ir"
    assert body["backend_snapshot"]["settings"]["optimisation_level"] == 2
    assert body["backend_snapshot"]["compiler_version"]
    assert body["passes"][0]["record"]["pass_name"] == "qiskit_basis_lowering"
    envelope = {key: wire[key] for key in ("schema", "body", "extensions")}
    assert canonical_digest("studio.compiler-trace.v1", envelope) == wire["sha256"]
    assert json.loads(trace.to_json()) == wire
    body["source"] = "mutated detached export"
    assert trace.to_dict()["body"]["source"] == SOURCE
    for field in fields(trace):
        with pytest.raises(FrozenInstanceError):
            setattr(trace, field.name, "changed")


def test_missing_pass_remains_explicit() -> None:
    """Absent intermediate evidence cannot be projected as an empty qualified pass."""
    _, after, swapped = mapped_circuits()
    identity = qualify_circuit_pass(
        after, after, pass_name="physical_identity", input_layout=(1, 0), output_layout=(1, 0)
    )
    trace = project_compiler_passes(
        [swapped, None, identity], backend_snapshot=compiler_backend_snapshot()
    )
    body = trace.to_dict()["body"]
    assert body["complete"] is False
    assert body["passes"][1] == {
        "state": "missing",
        "reason": "Native pass artifact was not supplied.",
    }
    assert body["emitted_ir"] is None
    assert body["passes"][2]["record"]["input_layout"] == [1, 0]


def test_disconnected_trace_cannot_rebind_source_or_layout() -> None:
    """A subsequent native record must consume the exact preceding snapshot and layout."""
    before, after, swapped = mapped_circuits()
    for circuit, layout in [(before, (0, 1)), (after, (0, 1))]:
        other = qualify_circuit_pass(
            circuit, circuit, pass_name="disconnected", input_layout=layout, output_layout=layout
        )
        with pytest.raises(ValueError, match="continuity"):
            project_compiler_passes([swapped, other], backend_snapshot=compiler_backend_snapshot())


@pytest.mark.parametrize("level", [True, -1, 4, "2", 2.0])
def test_lowering_settings_refuse_without_source_mutation(level: object) -> None:
    """Native configuration retains exact integer admission and the authored source."""
    boundary: Callable[..., CompilerTrace] = build_compiler_trace
    with pytest.raises((ValueError, CircuitPassRefused), match="optimisation"):
        boundary(SOURCE, optimisation_level=level)
    assert SOURCE.endswith("measure q[0] -> c[1];\n")


@pytest.mark.parametrize(
    "statement", ["reset q[0];", "measure q[0] -> c[0];\nx q[0];", "if(c==1) x q[0];"]
)
def test_unsupported_lowering_is_a_located_refusal(statement: str) -> None:
    """Effectful source cannot be silently converted to a static qualified trace."""
    source = SOURCE.split("ry(0.41)")[0] + statement + "\n"
    with pytest.raises(CircuitPassRefused):
        build_compiler_trace(source)


def test_projection_refuses_unbound_or_oversized_pass_lists() -> None:
    """Projection needs a source-bearing first native record within its pass ceiling."""
    _, _, swapped = mapped_circuits()
    invalid: list[list[CircuitPassRecord | None]] = [[], [None], [swapped] * 33]
    for rows in invalid:
        with pytest.raises(ValueError, match="pass"):
            project_compiler_passes(rows, backend_snapshot=compiler_backend_snapshot())


@pytest.mark.parametrize("level", [0, 1, 2, 3])
def test_each_native_optimisation_level_preserves_its_requested_snapshot(level: int) -> None:
    """Real backend settings and source identity remain explicit at each supported level."""
    wire = build_compiler_trace(SOURCE, optimisation_level=level).to_dict()
    assert wire["body"]["backend_snapshot"]["settings"]["optimisation_level"] == level
    assert wire["body"]["passes"][0]["parameters"]["optimisation_level"] == level


def test_real_cli_exports_trace_and_reproduction_script(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The public CLI and generated script reproduce the same complete native artifact."""
    from scpn_quantum_control.studio.executive_cli import run

    params = json.dumps(
        {"program_source": SOURCE, "compiler_trace": True, "optimisation_level": 1}
    )
    assert (
        run(
            [
                "compile",
                "--action-id",
                "native-trace",
                "--params",
                params,
                "--script-dir",
                str(tmp_path),
            ]
        )
        == 0
    )
    wire = json.loads(capsys.readouterr().out)
    trace = wire["result"]["outputs"]["compiler_trace"]
    assert "studio.compiler-trace.v1" in wire["produced_schemas"]
    assert "studio.program-source.v1" in wire["produced_schemas"]
    assert "native static operator qualification" in wire["plan"]["claim_boundary"]
    assert trace["body"]["execution_status"] == "emitted_not_executed"
    assert trace["body"]["emitted_ir"]["execution_status"] == "textual_ir"
    assert list(tmp_path.iterdir()) == [tmp_path / "compile_native_trace.py"]
    with pytest.raises(SystemExit) as outcome:
        runpy.run_path(str(tmp_path / "compile_native_trace.py"), run_name="__main__")
    assert outcome.value.code == 0
    assert json.loads(capsys.readouterr().out) == trace


@pytest.mark.parametrize(
    "changes",
    [
        {"compiler_trace": False},
        {"compiler_trace": 1},
        {"compiler_trace": "true"},
        {"optimisation_level": 0},
        {"compiler_trace": True, "optimisation_level": 4},
        {"compiler_trace": True, "optimisation_level": True},
    ],
)
def test_cli_refuses_unsupported_trace_requests_without_writing_script(
    changes: dict[str, object], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Malformed trace settings preserve source and create no runnable artifact."""
    from scpn_quantum_control.studio.executive_cli import run

    params = json.dumps({"program_source": SOURCE, **changes})
    assert (
        run(
            [
                "compile",
                "--action-id",
                "refused-trace",
                "--params",
                params,
                "--script-dir",
                str(tmp_path),
            ]
        )
        == 2
    )
    assert not list(tmp_path.iterdir())
    assert "error" in capsys.readouterr().err


def test_barriers_empty_circuits_and_declared_dense_payload_are_metadata_only() -> None:
    """Native identity traces retain barriers/readout and correctly bound empty gate sets."""
    for circuit in [QuantumCircuit(2), QuantumCircuit(2, 1)]:
        if circuit.num_clbits:
            circuit.h(0)
            circuit.barrier(0, 1)
            circuit.measure(1, 0)
        record = qualify_circuit_pass(circuit, circuit, pass_name="identity")
        metrics = project_compiler_passes(
            [record], backend_snapshot=compiler_backend_snapshot()
        ).to_dict()["body"]["passes"][0]["metrics"]
        assert metrics["input"]["depth"] == circuit.depth()
        assert metrics["input"]["statevector_bytes"] == 64
        assert metrics["input"]["operator_bytes"] == 256
        assert metrics["delta"] == {
            "depth": 0,
            "operation_count": 0,
            "statevector_bytes": 0,
            "operator_bytes": 0,
        }


@pytest.mark.parametrize(
    "changes",
    [
        {"compiler": "other"},
        {"target": "device"},
        {"compiler_version": ""},
        {"compiler_version": 123},
        {"settings": []},
        {"reference_backend": "other"},
        {"basis_convention": "big_endian"},
        {"extra": "undeclared"},
    ],
)
def test_backend_snapshot_refuses_substitution(changes: dict[str, object]) -> None:
    """A projection cannot imply a backend/device unsupported by its native records."""
    _, _, record = mapped_circuits()
    with pytest.raises(ValueError, match="backend snapshot"):
        project_compiler_passes(
            [record], backend_snapshot={**compiler_backend_snapshot(), **changes}
        )


def test_pass_parameter_count_is_not_silently_filled() -> None:
    """Caller-supplied pass settings must align exactly with original pass slots."""
    _, _, record = mapped_circuits()
    with pytest.raises(ValueError, match="parameter count"):
        project_compiler_passes(
            [record], backend_snapshot=compiler_backend_snapshot(), pass_parameters=[]
        )


@pytest.mark.parametrize(
    "text",
    [
        "[]",
        '{"schema":"future","body":{},"extensions":{},"sha256":""}',
        '{"schema":"studio.compiler-trace.v1","body":[],"extensions":{},"sha256":""}',
        '{"schema":"studio.compiler-trace.v1","body":{},"extensions":[],"sha256":""}',
        '{"schema":"studio.compiler-trace.v1","body":{},"extensions":{},"sha256":"changed"}',
        '{"schema":"studio.compiler-trace.v1","schema":"studio.compiler-trace.v1"}',
    ],
)
def test_trace_container_refuses_unbound_envelopes(text: str) -> None:
    """Reject unsupported, duplicated or unbound JSON without changing retained bytes."""
    with pytest.raises(ValueError):
        CompilerTrace(text)


def test_json_and_utf8_limits_are_distinct_public_boundaries() -> None:
    """Both ASCII size and multibyte size are bounded before envelope parsing."""
    for text in [" " * (MAX_TRACE_BYTES + 1), "α" * (MAX_TRACE_BYTES // 2 + 1)]:
        with pytest.raises(ValueError, match="ceiling"):
            CompilerTrace(text)
    boundary: Callable[..., CompilerTrace] = CompilerTrace
    with pytest.raises(ValueError, match="ceiling"):
        boundary(None)


def test_projection_refuses_wrong_record_kind_and_oversized_source() -> None:
    """Restored native data cannot bypass the original record/source boundaries."""
    _, _, record = mapped_circuits()
    invalid = cast(CircuitPassRecord, object())
    with pytest.raises(ValueError, match="original native record"):
        project_compiler_passes([record, invalid], backend_snapshot=compiler_backend_snapshot())
    long_ir = replace(record.input_ir, source=record.input_ir.source + " " * (1024 * 1024))
    long_record = replace(record, input_ir=long_ir)
    with pytest.raises(ValueError, match="one MiB|1 MiB"):
        project_compiler_passes([long_record], backend_snapshot=compiler_backend_snapshot())


def test_metadata_export_limit_preserves_native_input() -> None:
    """Opaque settings cannot make an unbounded export or mutate the original record."""
    _, _, record = mapped_circuits()
    original = record.sha256
    with pytest.raises(ValueError, match="ceiling"):
        project_compiler_passes(
            [record],
            backend_snapshot=compiler_backend_snapshot(),
            pass_parameters=[{"explanation": "x" * MAX_TRACE_BYTES}],
        )
    assert record.sha256 == original


def test_browser_fixtures_reproduce_through_the_original_native_producers() -> None:
    """Committed browser examples retain actual native qualification and full bindings."""
    data = Path(__file__).resolve().parents[1] / "data/studio"
    original = json.loads((data / "compiler_trace_demo.json").read_text())
    records = []
    parameters = []
    for row in original["body"]["passes"]:
        record = row["record"]
        records.append(
            qualify_circuit_pass(
                qasm2.loads(row["input"]["ir"]["source"]),
                qasm2.loads(row["output"]["ir"]["source"]),
                source=row["input"]["ir"]["source"],
                pass_name=record["pass_name"],
                input_layout=tuple(record["input_layout"]),
                output_layout=tuple(record["output_layout"]),
                output_classical_layout=tuple(record["output_classical_layout"]),
                allow_global_phase=record["allow_global_phase"],
                tolerance=record["tolerance"],
            )
        )
        parameters.append(row["parameters"])
    _assert_reproduced(
        project_compiler_passes(
            records, backend_snapshot=compiler_backend_snapshot(), pass_parameters=parameters
        ).to_dict(),
        original,
    )
    cases = json.loads((data / "compiler_trace_cases.json").read_text())
    _assert_reproduced(
        build_compiler_trace(cases["lowering"]["body"]["source"]).to_dict(), cases["lowering"]
    )
    original_crlf = cases["crlf"]["body"]["source"]
    circuit = qasm2.loads(original_crlf)
    crlf_record = qualify_circuit_pass(
        circuit, circuit, source=original_crlf, pass_name="crlf", allow_global_phase=False
    )
    _assert_reproduced(
        project_compiler_passes(
            [crlf_record], backend_snapshot=compiler_backend_snapshot()
        ).to_dict(),
        cases["crlf"],
    )
    for name, circuit in [
        ("empty", QuantumCircuit(1)),
        ("readout", QuantumCircuit(1, 1)),
        ("negative_zero", QuantumCircuit(1)),
    ]:
        if name == "readout":
            circuit.measure(0, 0)
        if name == "negative_zero":
            circuit.rx(-0.0, 0)
        record = qualify_circuit_pass(circuit, circuit, pass_name=name, allow_global_phase=False)
        _assert_reproduced(
            project_compiler_passes(
                [record], backend_snapshot=compiler_backend_snapshot()
            ).to_dict(),
            cases[name],
        )


def test_fixture_reproduction_admits_only_measurement_noise_and_its_digests() -> None:
    """Another platform's last bits are admitted; any other difference still fails."""
    data = Path(__file__).resolve().parents[1] / "data/studio"
    committed = json.loads((data / "compiler_trace_cases.json").read_text())["lowering"]
    measured = committed["body"]["passes"][0]["record"]
    assert measured["global_phase_delta"] != 0.0

    elsewhere = copy.deepcopy(committed)
    record = elsewhere["body"]["passes"][0]["record"]
    record["global_phase_delta"] = math.nextafter(measured["global_phase_delta"], 0.0)
    record["operator_error"] = math.nextafter(measured["operator_error"], 1.0)
    elsewhere["body"]["passes"][0]["native_record_sha256"] = "a" * 64
    elsewhere["body"]["emitted_ir"]["metadata"]["pass_sha256"] = "a" * 64
    elsewhere["sha256"] = "b" * 64
    _assert_reproduced(elsewhere, committed)

    stale_digest = copy.deepcopy(committed)
    stale_digest["sha256"] = "b" * 64
    beyond_noise = copy.deepcopy(elsewhere)
    beyond_noise["body"]["passes"][0]["record"]["global_phase_delta"] = 1e-13
    other_content = copy.deepcopy(elsewhere)
    other_content["body"]["emitted_ir"]["sha256"] = "c" * 64
    malformed_digest = copy.deepcopy(elsewhere)
    malformed_digest["sha256"] = "unbound"
    missing_pass = copy.deepcopy(elsewhere)
    missing_pass["body"]["passes"] = []
    for changed in (stale_digest, beyond_noise, other_content, malformed_digest, missing_pass):
        with pytest.raises(AssertionError):
            _assert_reproduced(changed, committed)


@pytest.mark.parametrize(
    "statement", ["reset q[0];", "measure q[0] -> c[0];\nx q[0];", "if(c==1) x q[0];"]
)
def test_real_cli_unsupported_lowering_creates_no_script(
    statement: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Public CLI returns an explicit static qualification refusal with no output writes."""
    from scpn_quantum_control.studio.executive_cli import run

    source = SOURCE.split("ry(0.41)")[0] + statement + "\n"
    params = json.dumps({"program_source": source, "compiler_trace": True})
    assert (
        run(
            [
                "compile",
                "--action-id",
                "effectful-trace",
                "--params",
                params,
                "--script-dir",
                str(tmp_path),
            ]
        )
        == 1
    )
    output = capsys.readouterr()
    failure = json.loads(output.out)
    assert failure["result"]["status"] == "failed"
    assert "CircuitPassRefused" in failure["result"]["error"]
    assert failure["result"]["outputs"] == {}
    assert failure["script"] is None
    assert failure["request"]["parameters"]["program_source"] == source
    assert "failed: CircuitPassRefused" in output.err
    assert not list(tmp_path.iterdir())
