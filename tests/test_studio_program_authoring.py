# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — original supported source and numerical roundtrip
"""Native program imports retain readout, classical conditions and phase semantics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import ClassicalRegister, Gate, Instruction, Parameter, QuantumRegister
from qiskit.quantum_info import Operator

from scpn_quantum_control.studio.program_authoring import (
    compile_program_source,
    export_program_source,
    import_program_source,
)
from scpn_quantum_control.studio.program_authoring_contracts import ProgramSourceRefused

HEADER = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\ncreg c[2];\n'

_CORPUS = json.loads((Path(__file__).parent / "data/program_authoring/corpus.json").read_text())[
    "cases"
]


@pytest.mark.parametrize("case", _CORPUS, ids=[case["name"] for case in _CORPUS])
def test_native_shared_source_corpus(case: dict[str, Any]) -> None:
    """Use independently declared cases shared with Rust and actual browser WASM."""
    source = case["source"]
    if case["ok"]:
        record = compile_program_source(source).to_dict()
        assert [op["name"] for op in record["operations"]] == case["operations"]
        assert record["measurements"] == case["measurements"]
        if "parameters" in case:
            assert [op["parameters"] for op in record["operations"]] == case["parameters"]
        if "conditions" in case:
            assert [op["condition"] for op in record["operations"]] == case["conditions"]
    else:
        with pytest.raises(ProgramSourceRefused) as caught:
            compile_program_source(source)
        diagnostic = caught.value.diagnostic
        assert diagnostic.code == case["code"]
        start = source.index(case["token"], case["token_after"])
        assert (diagnostic.source_span.start, diagnostic.source_span.end) == (
            start,
            start + len(case["token"]),
        )


def test_program_authoring_01() -> None:
    """Source export restores exact measurements and a native unitary oracle."""
    source = HEADER + "h q[0];cx q[0],q[1];measure q[1] -> c[0];measure q[0] -> c[1];"
    plan = compile_program_source(source)
    restored = compile_program_source(plan.source)
    assert plan == restored
    assert plan.measurements == ((1, 0), (0, 1))
    native = import_program_source(plan.source)
    unitary = native.remove_final_measurements(inplace=False)
    expected = np.array(
        [[1, 1, 0, 0], [0, 0, 1, -1], [0, 0, 1, 1], [1, -1, 0, 0]],
        dtype=np.complex128,
    ) / np.sqrt(2)
    np.testing.assert_allclose(Operator(unitary).data, expected, rtol=0, atol=1e-12)
    assert plan.execution_status == "emitted_not_executed"


def test_program_authoring_02() -> None:
    """The original unsupported token retains its exact source coordinates."""
    source = HEADER + "  unexpected q[0];"
    with pytest.raises(ProgramSourceRefused) as caught:
        compile_program_source(source)
    diagnostic = caught.value.diagnostic
    assert diagnostic.code == "unsupported_operation"
    assert (diagnostic.source_span.line, diagnostic.source_span.column) == (5, 3)
    assert source[diagnostic.source_span.start : diagnostic.source_span.end] == "unexpected"


@pytest.mark.parametrize(
    "source",
    [
        "import os; os.remove('file');",
        'OPENQASM 2.0; include "/etc/passwd"; qreg q[1];',
    ],
)
def test_program_authoring_03(source: str) -> None:
    """Python and external includes are refused at the public import boundary."""
    with pytest.raises(ProgramSourceRefused):
        import_program_source(source)


def test_conditional_rotation_and_mid_circuit_readout_roundtrip() -> None:
    """Native conditional blocks preserve exact phase and original readout."""
    angle = -0.7853981633974492
    source = (
        HEADER + f"h q[0];measure q[0] -> c[0];if(c==1) rz({angle}) q[1];measure q[1] -> c[1];"
    )
    plan = compile_program_source(source)
    native = import_program_source(plan.source)
    conditional = native.data[2].operation
    assert conditional.name == "if_else"
    assert conditional.condition == (native.cregs[0], 1)
    phase = conditional.blocks[0]
    reference = np.diag([np.exp(-0.5j * angle), np.exp(0.5j * angle)])
    np.testing.assert_allclose(Operator(phase).data, reference, rtol=0, atol=1e-12)
    assert plan.operations[2].condition is not None
    assert plan.operations[2].condition.value == "1"
    assert plan.operations[2].parameters == ("bfe921fb54442d20",)
    assert plan.measurements == ((0, 0), (1, 1))
    # Export the actual native circuit through the public exact-decimal producer.
    reparsed = compile_program_source(export_program_source(native))
    assert reparsed.measurements == plan.measurements
    assert [(op.name, op.parameters, op.condition) for op in reparsed.operations] == [
        (op.name, op.parameters, op.condition) for op in plan.operations
    ]


def test_original_static_qualifier_still_refuses_classical_effects() -> None:
    """Source emission does not broaden the original static-unitary admission."""
    from scpn_quantum_control.compiler import CircuitPassRefused, import_circuit_source

    source = HEADER + "measure q[0] -> c[0];if(c==1) x q[1];"
    assert compile_program_source(source).operations[1].condition is not None
    with pytest.raises(CircuitPassRefused):
        import_circuit_source(source)


def test_native_rotation_has_the_same_original_gate_phase() -> None:
    """Exact native parameter bits preserve a phase oracle beyond readout counts."""
    angle = 0.25
    source = HEADER + "rz(0.25) q[0];"
    native = import_program_source(source)
    expected = QuantumCircuit(2)
    expected.rz(angle, 0)
    np.testing.assert_allclose(Operator(native).data, Operator(expected).data, rtol=0, atol=1e-12)
    assert compile_program_source(source).operations[0].parameters == ("3fd0000000000000",)


def test_native_export_retains_reordered_conditional_operands() -> None:
    """Map actual inner block operands through the outer native instruction."""
    native = QuantumCircuit(2, 2)
    body = QuantumCircuit(2)
    body.cx(1, 0)
    native.if_else((native.cregs[0], 1), body, None, [0, 1], [])
    record = compile_program_source(export_program_source(native))
    assert record.operations[0].qubits == (1, 0)
    assert record.operations[0].condition is not None
    assert record.operations[0].condition.value == "1"
    assert native.data[0].operation.blocks[0].data[0].qubits == (body.qubits[1], body.qubits[0])


def test_native_export_gate_only_and_effectful_readout_roundtrip() -> None:
    """Retain every original public native operation and signed rotation zero."""
    native = QuantumCircuit(2, 2)
    native.rx(-0.0, 0)
    native.barrier(1, 0)
    native.reset(1)
    native.measure(1, 0)
    native.measure(0, 1)
    record = compile_program_source(export_program_source(native))
    assert record.operations[0].parameters == ("8000000000000000",)
    assert record.operations[1].qubits == (1, 0)
    assert record.measurements == ((1, 0), (0, 1))
    gate_only = QuantumCircuit(1)
    gate_only.h(0)
    assert compile_program_source(export_program_source(gate_only)).num_clbits == 0


@pytest.mark.parametrize(
    "kind",
    [
        "empty",
        "width",
        "classical_width",
        "name",
        "classical_name",
        "registers",
        "unbound",
        "phase",
        "custom",
        "complex",
        "infinite",
        "else",
        "multi_gate",
        "bit_condition",
        "block_phase",
        "operation_budget",
    ],
)
def test_native_export_refuses_unsupported_circuits_without_mutation(kind: str) -> None:
    """No exact export drops a native phase, register, condition or effect."""
    native = QuantumCircuit(1, 1)
    if kind == "empty":
        native = QuantumCircuit()
    elif kind == "width":
        native = QuantumCircuit(9)
    elif kind == "classical_width":
        native = QuantumCircuit(1, 65)
    elif kind == "name":
        native = QuantumCircuit(QuantumRegister(1, "other"))
    elif kind == "classical_name":
        native = QuantumCircuit(QuantumRegister(1, "q"), ClassicalRegister(1, "other"))
    elif kind == "registers":
        native = QuantumCircuit(QuantumRegister(1, "q"), QuantumRegister(1, "other"))
    elif kind == "unbound":
        native.rx(Parameter("angle"), 0)
    elif kind == "phase":
        native.global_phase = 0.25
    elif kind == "custom":
        native.append(Gate("custom", 1, []), [0])
    elif kind == "complex":
        native.append(Instruction("rx", 1, 0, [1j]), [0])
    elif kind == "infinite":
        native.append(Gate("rx", 1, [float("inf")]), [0])
    elif kind == "operation_budget":
        for _ in range(4097):
            native.x(0)
    else:
        body = QuantumCircuit(1)
        body.x(0)
        if kind == "multi_gate":
            body.h(0)
        if kind == "block_phase":
            body.global_phase = 0.25
        native.if_else(
            (native.clbits[0] if kind == "bit_condition" else native.cregs[0], 1),
            body,
            body if kind == "else" else None,
            [0],
            [],
        )
    original = native.copy()
    with pytest.raises(ProgramSourceRefused) as caught:
        export_program_source(native)
    assert caught.value.diagnostic.code == "unsupported_export"
    assert native == original


@pytest.mark.parametrize(
    "source,code",
    [
        ("", "invalid_source"),
        (" \n", "invalid_source"),
        ("\ud800", "invalid_source"),
        ("x" * 1_048_577, "source_budget"),
        (";" * 65_537, "source_budget"),
        (HEADER + "x q[0];" * 4097, "circuit_budget"),
    ],
)
def test_original_source_transport_and_operation_limits(source: str, code: str) -> None:
    """Reject source size, UTF8 and token/operation budgets at the public boundary."""
    with pytest.raises(ProgramSourceRefused) as caught:
        compile_program_source(source)
    assert caught.value.diagnostic.code == code


@pytest.mark.parametrize("suffix", ["h q[0]", "if(c==", "barrier q", "u3(0.25,0.5,1.0) q[0]"])
def test_missing_tokens_retain_original_eof_location(suffix: str) -> None:
    """Locate absent grammar tokens at the end of the original scalar source."""
    source = HEADER + suffix
    with pytest.raises(ProgramSourceRefused) as caught:
        compile_program_source(source)
    span = caught.value.diagnostic.source_span
    assert span.start == span.end == len(source)


@pytest.mark.parametrize(
    "change,code",
    [
        ("added_gate", "source_mismatch"),
        ("changed_gate", "source_mismatch"),
        ("bit_condition", "unsupported_operation"),
        ("parser_error", "invalid_source"),
    ],
)
def test_native_sdk_disagreement_never_produces_an_emitted_record(
    change: str, code: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject real native circuits that no longer represent the admitted source."""
    from qiskit import qasm2

    source = HEADER + "if(c==1) x q[0];"
    native = import_program_source(source)
    if change == "added_gate":
        native.x(1)
    elif change == "changed_gate":
        native = QuantumCircuit(2, 2)
        with native.if_test((native.cregs[0], 1)):
            native.h(0)
    elif change == "bit_condition":
        native = QuantumCircuit(2, 2)
        with native.if_test((native.clbits[0], 1)):
            native.x(0)
    original_loads = qasm2.loads

    def incompatible_sdk(_source: str, **_kwargs: Any) -> QuantumCircuit:
        if change == "parser_error":
            return original_loads("OPENQASM future;")
        return native

    monkeypatch.setattr(qasm2, "loads", incompatible_sdk)
    with pytest.raises(ProgramSourceRefused) as caught:
        compile_program_source(source)
    assert caught.value.diagnostic.code == code
    assert "<input>" not in caught.value.diagnostic.message


@pytest.mark.parametrize("source", [None, 7, {}, ["import os"]])
def test_nontext_source_transport_receives_authored_refusal(source: object) -> None:
    """Reject nontext external parameters without treating them as source code."""
    with pytest.raises(ProgramSourceRefused) as caught:
        compile_program_source(cast(str, source))
    assert caught.value.diagnostic.code == "invalid_source"


@pytest.mark.parametrize(
    "gate,pauli",
    [
        ("rxx", [[0, 1], [1, 0]]),
        ("ryy", [[0, -1j], [1j, 0]]),
        ("rzz", [[1, 0], [0, -1]]),
    ],
)
def test_native_two_qubit_phase_matches_independent_pauli_oracle(
    gate: str, pauli: list[list[complex]]
) -> None:
    """Preserve the exact angle and full phase through actual native import/export."""
    angle = -0.7853981633974492
    source = 'OPENQASM 2.0; include "qelib1.inc"; qreg q[2]; ' + f"{gate}({angle}) q[1],q[0];"
    native = import_program_source(source)
    generator = np.kron(np.asarray(pauli), np.asarray(pauli))
    expected = np.cos(angle / 2) * np.eye(4) - 1j * np.sin(angle / 2) * generator
    np.testing.assert_allclose(Operator(native).data, expected, rtol=0, atol=1e-12)
    restored = import_program_source(export_program_source(native))
    np.testing.assert_allclose(Operator(restored).data, expected, rtol=0, atol=1e-12)
    assert compile_program_source(export_program_source(native)).operations[0].parameters == (
        "bfe921fb54442d20",
    )
