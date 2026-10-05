# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Located circuit source tests
"""Actual native source parsing, original spans and bounded refusal contracts."""

from __future__ import annotations

from collections.abc import Callable

import pytest
from qiskit import QuantumCircuit, qasm2
from qiskit.circuit import Parameter

from scpn_quantum_control.compiler import (
    CircuitPassRefused,
    import_circuit_source,
    snapshot_circuit,
)

HEADER = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n'


def test_broadcast_operations_retain_original_statement_locations() -> None:
    """Native register expansion binds every expanded operand to its real source."""
    source = HEADER + "qreg q[2];\ncreg c[2];\n// preserved α\n  h q;\nmeasure q -> c;"
    circuit = import_circuit_source(source)
    record = snapshot_circuit(circuit, source=source)
    assert [(op.name, op.qubits) for op in record.operations] == [
        ("h", (0,)),
        ("h", (1,)),
        ("measure", (0,)),
        ("measure", (1,)),
    ]
    for operation in record.operations[:2]:
        span = operation.source_span
        assert (span.line, span.column) == (6, 3)
        assert source[span.start : span.end] == "h q;"
    assert record.measurements == ((0, 0), (1, 1))
    assert record.source == source


@pytest.mark.parametrize(
    "statement", ["reset q[0];", "if(c==1) x q[0];", "opaque foo q;", "gate foo a { x a; }"]
)
def test_effectful_and_custom_source_is_located(statement: str) -> None:
    """Effects and unregistered gate definitions cannot enter static lowering."""
    source = HEADER + "qreg q[1];\ncreg c[1];\n  " + statement
    with pytest.raises(CircuitPassRefused) as caught:
        import_circuit_source(source)
    assert (caught.value.diagnostic.line, caught.value.diagnostic.column) == (5, 3)


@pytest.mark.parametrize("source", ["", " ", None, 12])
def test_empty_or_nontext_source_is_refused(source: object) -> None:
    """The actual source boundary rejects malformed transport values."""
    boundary: Callable[..., QuantumCircuit] = import_circuit_source
    with pytest.raises(CircuitPassRefused, match="non-empty text"):
        boundary(source)


def test_external_include_is_refused_before_filesystem_lookup() -> None:
    """Includes do not read an arbitrary existing host file."""
    source = 'OPENQASM 2.0;\n include "/etc/passwd";\nqreg q[1];'
    with pytest.raises(CircuitPassRefused) as caught:
        import_circuit_source(source)
    assert caught.value.diagnostic.code == "unsupported_include"
    assert (caught.value.diagnostic.line, caught.value.diagnostic.column) == (2, 2)


@pytest.mark.parametrize(
    "source, code",
    [
        (HEADER + "qreg q[9];", "circuit_budget"),
        (HEADER + "qreg a[5];qreg b[4];", "circuit_budget"),
        (HEADER + "qreg q[1];creg c[65];", "circuit_budget"),
        (HEADER + "qreg q[1000000000];", "circuit_budget"),
        (HEADER + "qreg q[ 1000000000 ];", "circuit_budget"),
        pytest.param(
            HEADER + "qreg q[" + "9" * 5000 + "];",
            "circuit_budget",
            id="5000-digit-register-width",
        ),
        pytest.param(
            HEADER + "qreg q[1];" + "x q[0];" * 4097, "circuit_budget", id="4097-operations"
        ),
        pytest.param("//" + "x" * (1024 * 1024), "source_budget", id="mebibyte-comment"),
    ],
)
def test_source_budgets_refuse_large_native_allocations(source: str, code: str) -> None:
    """Source widths and counts are checked before reference allocation."""
    with pytest.raises(CircuitPassRefused) as caught:
        import_circuit_source(source)
    assert caught.value.diagnostic.code == code


def test_register_broadcast_operation_budget_is_checked() -> None:
    """Native expanded operation count also obeys the declared budget."""
    with pytest.raises(CircuitPassRefused, match="budget"):
        import_circuit_source(HEADER + "qreg q[8];" + "h q;" * 513)


@pytest.mark.parametrize(
    "tail, line, column",
    [
        ("\n   x q[0]", 4, 4),
        ("\n ;", 4, 2),
        ("\n  unknown q[0];", 4, 3),
    ],
)
def test_native_and_statement_syntax_errors_are_authored_and_located(
    tail: str, line: int, column: int
) -> None:
    """Refusals preserve real locations and omit native exception text."""
    with pytest.raises(CircuitPassRefused) as caught:
        import_circuit_source(HEADER + "qreg q[1];" + tail)
    assert (caught.value.diagnostic.line, caught.value.diagnostic.column) == (line, column)
    assert "not defined" not in str(caught.value)
    assert "<input>" not in str(caught.value)


def test_unitary_after_readout_is_not_silently_dropped() -> None:
    """A measured circuit cannot acquire a static-unitary certificate."""
    source = HEADER + "qreg q[1];creg c[1];\nmeasure q[0] -> c[0];\nx q[0];"
    with pytest.raises(CircuitPassRefused) as caught:
        import_circuit_source(source)
    assert caught.value.diagnostic.code == "unsupported_effect"
    assert caught.value.diagnostic.line == 5


def test_unbound_native_circuit_is_explicit() -> None:
    """Unbound parameters are refused before textual export or transpilation."""
    circuit = QuantumCircuit(1)
    circuit.rx(Parameter("angle"), 0)
    with pytest.raises(CircuitPassRefused, match="bound"):
        snapshot_circuit(circuit)


@pytest.mark.parametrize(
    "value", [None, QuantumCircuit(), QuantumCircuit(9), QuantumCircuit(1, 65)]
)
def test_native_circuit_admission_checks_type_and_width(value: object) -> None:
    """Malformed/native out-of-budget inputs do not reach source export."""
    boundary: Callable[..., object] = snapshot_circuit
    with pytest.raises(CircuitPassRefused):
        boundary(value)


def test_exact_source_mismatch_is_refused_without_input_mutation() -> None:
    """Unrelated source cannot provide a circuit's purported provenance."""
    circuit = QuantumCircuit(1)
    circuit.h(0)
    sources = [HEADER + "qreg q[1];", HEADER + "qreg q[1];x q[0];", HEADER + "qreg q[2];h q[0];"]
    for source in sources:
        with pytest.raises(CircuitPassRefused, match="source does not match"):
            snapshot_circuit(circuit, source=source)
    assert circuit.count_ops() == {"h": 1}


def test_empty_unitary_has_native_source_and_no_operations() -> None:
    """An identity circuit has a source record without invented gate locations."""
    record = snapshot_circuit(QuantumCircuit(1))
    assert record.operations == ()
    assert record.measurements == ()


def test_native_dynamic_circuit_has_explicit_export_refusal() -> None:
    """Actual native control-flow objects cannot be advertised as static source."""
    circuit = QuantumCircuit(1, 1)
    with circuit.if_test((circuit.clbits[0], True)):
        circuit.x(0)
    with pytest.raises(CircuitPassRefused) as caught:
        snapshot_circuit(circuit)
    assert caught.value.diagnostic.code == "unsupported_export"


@pytest.mark.parametrize("phase", [float("nan"), float("inf")])
def test_nonfinite_native_phase_is_not_qualified(phase: float) -> None:
    """Native circuit phase admission does not accept Qiskit's nonfinite values."""
    with pytest.raises(CircuitPassRefused, match="phase must be finite"):
        snapshot_circuit(QuantumCircuit(1, global_phase=phase))


def test_nonfinite_source_rotation_is_located() -> None:
    """A parsed overflowing real parameter fails before unitary reference work."""
    source = HEADER + "qreg q[1];\nrx(1.0e309) q[0];"
    with pytest.raises(CircuitPassRefused) as caught:
        import_circuit_source(source)
    assert caught.value.diagnostic.code == "nonfinite_parameters"
    assert caught.value.diagnostic.line == 4


def test_unregistered_braced_syntax_is_located() -> None:
    """Braced statement syntax cannot enter the admitted native subset."""
    with pytest.raises(CircuitPassRefused) as caught:
        import_circuit_source(HEADER + "qreg q[1];\n x {q[0]};")
    assert caught.value.diagnostic.code == "unsupported_operation"
    assert (caught.value.diagnostic.line, caught.value.diagnostic.column) == (4, 2)


def test_native_parser_budget_fault_is_an_authored_refusal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A native boundary budget exception supplements real parser conformance."""
    from qiskit import qasm2

    def exhausted(
        source: str,
        *,
        include_path: tuple[()],
        strict: bool,
        custom_instructions: tuple[qasm2.CustomInstruction, ...],
    ) -> QuantumCircuit:
        """Inject the documented native parser recursion failure at its boundary."""
        raise RecursionError("native interpreter detail")

    monkeypatch.setattr(qasm2, "loads", exhausted)
    source = HEADER + "qreg q[1];x q[0];"
    with pytest.raises(CircuitPassRefused) as caught:
        import_circuit_source(source)
    assert caught.value.diagnostic.code == "expression_budget"
    assert "interpreter detail" not in str(caught.value)


def test_native_parser_unlocated_fault_has_safe_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing SDK coordinate never exposes arbitrary generated exception text."""
    from qiskit import qasm2

    def unlocated(
        source: str,
        *,
        include_path: tuple[()],
        strict: bool,
        custom_instructions: tuple[qasm2.CustomInstruction, ...],
    ) -> QuantumCircuit:
        """Inject a native SDK error without a source-coordinate prefix."""
        raise qasm2.QASM2ParseError("native interpreter detail")

    monkeypatch.setattr(qasm2, "loads", unlocated)
    with pytest.raises(CircuitPassRefused) as caught:
        import_circuit_source(HEADER + "qreg q[1];x q[0];")
    assert (caught.value.diagnostic.line, caught.value.diagnostic.column) == (1, 1)
    assert "interpreter detail" not in str(caught.value)


def test_generated_source_retains_exact_native_parameters() -> None:
    """Canonical source uses round-trip decimals instead of approximate pi aliases."""
    circuit = QuantumCircuit(1)
    angle = -0.7853981633974492
    circuit.rx(angle, 0)
    record = snapshot_circuit(circuit)
    assert record.operations[0].parameters == (angle,)
    restored = qasm2.loads(record.source, include_path=(), strict=True)
    assert restored.data[0].operation.params == [angle]
    recovered = import_circuit_source(record.source)
    assert recovered.data[0].operation.params == [angle]


def test_invalid_utf8_source_is_an_authored_refusal() -> None:
    """Malformed Unicode transport values cannot reach a native source parser."""
    with pytest.raises(CircuitPassRefused, match="UTF-8"):
        import_circuit_source(HEADER + "\ud800")
